"""
Celery task execution tests — runs kazi.run and kazi.ingest tasks
directly in-process (task_always_eager=True) against a real Kazi instance
backed by a real OpenAI LLM.

Requires OPENAI_API_KEY and Redis on localhost:6379/15.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
REDIS_URL = "redis://localhost:6379/15"
pytestmark = pytest.mark.skipif(not OPENAI_KEY, reason="OPENAI_API_KEY not set")


def _config():
    from kazi.core.config import KaziConfig, LLMConfig, LLMProvider
    return KaziConfig(llm=LLMConfig(
        provider=LLMProvider.OPENAI, model="gpt-4o-mini", api_key=OPENAI_KEY
    ))


def _make_app(config=None, **kwargs):
    from kazi.queue.celery_worker import build_celery_app
    app = build_celery_app(config or _config(), broker=REDIS_URL, **kwargs)
    app.conf.task_always_eager = True
    return app


@pytest.fixture
def real_kazi():
    """
    Wire a real config so _get_kazi() creates a real Kazi instance lazily
    inside the worker's persistent event loop (same loop used by tasks).
    """
    import kazi.queue.celery_worker as cw

    orig_instance = cw._kazi_instance
    orig_config = cw._kazi_config
    cw._kazi_instance = None
    cw._kazi_config = _config()
    yield
    # Close the kazi instance that was created during the test
    created = cw._kazi_instance
    cw._kazi_instance = orig_instance
    cw._kazi_config = orig_config
    if created and created is not orig_instance:
        try:
            cw._run_async(created.close())
        except Exception:
            pass


# ── _create_kazi / _get_kazi ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_kazi_returns_kazi_instance():
    from kazi.core.orchestrator import Kazi
    from kazi.queue.celery_worker import _create_kazi

    kazi = await _create_kazi(_config())
    assert isinstance(kazi, Kazi)
    await kazi.close()


def test_get_kazi_creates_and_caches_instance():
    import kazi.queue.celery_worker as cw
    from kazi.core.orchestrator import Kazi

    orig_instance = cw._kazi_instance
    orig_config = cw._kazi_config
    try:
        cw._kazi_instance = None
        cw._kazi_config = _config()
        instance = cw._get_kazi()
        assert isinstance(instance, Kazi)
        assert cw._get_kazi() is instance
    finally:
        created = cw._kazi_instance
        cw._kazi_instance = orig_instance
        cw._kazi_config = orig_config
        if created and created is not orig_instance:
            try:
                cw._run_async(created.close())
            except Exception:
                pass


# ── kazi.run task ────────────────────────────────────────────────────────────

def test_kazi_run_task_executes_and_returns_reply(real_kazi):
    """Task executes kazi.run() with a real LLM and returns a non-empty reply."""
    app = _make_app()
    result = app.tasks["kazi.run"].apply(
        args=["Say exactly: TASK_OK"], kwargs={"thread_id": "celery-run-1"}
    )
    payload = result.result
    assert "reply" in payload
    assert "TASK_OK" in payload["reply"]


def test_kazi_run_task_with_all_options(real_kazi):
    app = _make_app()
    result = app.tasks["kazi.run"].apply(
        args=["Say exactly: OPTIONS_OK"],
        kwargs={
            "thread_id": "celery-opts",
            "system_prompt": "Be brief.",
            "max_tool_calls": 5,
            "track_cost": False,
        },
    )
    payload = result.result
    assert "OPTIONS_OK" in payload["reply"]


def test_kazi_run_task_with_run_result_cost_tracking(real_kazi):
    """With track_cost=True the task serialises input/output tokens and cost_usd."""
    app = _make_app()
    result = app.tasks["kazi.run"].apply(
        args=["Say hi"], kwargs={"thread_id": "celery-cost", "track_cost": True}
    )
    payload = result.result
    assert "reply" in payload
    assert isinstance(payload.get("input_tokens"), int)
    assert isinstance(payload.get("output_tokens"), int)
    assert isinstance(payload.get("cost_usd"), float)
    assert payload["input_tokens"] > 0
    assert payload["output_tokens"] > 0


def test_kazi_run_task_fails_on_bad_api_key():
    """Task marks itself failed when the LLM raises (bad API key → 401)."""
    import kazi.queue.celery_worker as cw
    from kazi.core.config import KaziConfig, LLMConfig, LLMProvider

    bad_config = KaziConfig(llm=LLMConfig(
        provider=LLMProvider.OPENAI, model="gpt-4o-mini", api_key="sk-invalid-key"
    ))
    orig_instance = cw._kazi_instance
    orig_config = cw._kazi_config
    try:
        cw._kazi_instance = None
        cw._kazi_config = bad_config
        app = _make_app(config=bad_config, max_retries=0)
        result = app.tasks["kazi.run"].apply(args=["hello"])
        assert result.failed()
    finally:
        created = cw._kazi_instance
        cw._kazi_instance = orig_instance
        cw._kazi_config = orig_config
        if created and created is not orig_instance:
            try:
                cw._run_async(created.close())
            except Exception:
                pass


# ── kazi.ingest task ─────────────────────────────────────────────────────────

def test_kazi_ingest_task_returns_ok(real_kazi):
    """Ingest task on a real directory succeeds and returns path + index name."""
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir).joinpath("doc.txt").write_text("Kazi is a real AI framework.")

        app = _make_app()
        result = app.tasks["kazi.ingest"].apply(
            args=[tmpdir], kwargs={"index_name": "celery-ingest"}
        )
        payload = result.result
        assert payload["status"] == "ok"
        assert payload["path"] == tmpdir
        assert payload["index"] == "celery-ingest"


def test_kazi_ingest_task_default_index_name(real_kazi):
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir).joinpath("file.txt").write_text("hello")

        app = _make_app()
        result = app.tasks["kazi.ingest"].apply(args=[tmpdir])
        assert result.result["index"] == "default"


def test_kazi_ingest_task_fails_on_bad_path(real_kazi):
    """Ingest task fails cleanly when the path does not exist."""
    app = _make_app(max_retries=0)
    result = app.tasks["kazi.ingest"].apply(args=["/nonexistent/path/xyz"])
    assert result.failed()


# ── App configuration ─────────────────────────────────────────────────────────

def test_build_celery_app_custom_dlq_queue():
    from kazi.queue.celery_worker import build_celery_app
    app = build_celery_app(_config(), broker=REDIS_URL, dlq_queue="my-dlq")
    assert app.conf.task_routes.get("kazi.dead_letter", {}).get("queue") == "my-dlq"


def test_build_celery_app_custom_time_limits():
    from kazi.queue.celery_worker import build_celery_app
    app = build_celery_app(
        _config(), broker=REDIS_URL,
        task_time_limit=600, task_soft_time_limit=550,
    )
    assert app.conf.task_time_limit == 600
    assert app.conf.task_soft_time_limit == 550
