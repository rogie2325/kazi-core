"""
Celery task execution tests — runs kazi.run, kazi.ingest, and kazi.dead_letter
tasks directly in-process without a broker, covering the task function bodies.

Uses task_always_eager=True so tasks execute synchronously.  _kazi_instance is
patched with a stub so no LLM API calls are made (we already test real LLM calls
elsewhere).  Tests for _get_kazi() and _create_kazi() are included.
"""
from __future__ import annotations

import asyncio
import os

import pytest

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
REDIS_URL = "redis://localhost:6379/15"


def _config(api_key: str = "fake"):
    from kazi.core.config import KaziConfig, LLMConfig, LLMProvider
    return KaziConfig(llm=LLMConfig(
        provider=LLMProvider.OPENAI, model="gpt-4o-mini", api_key=api_key
    ))


def _make_app(config=None, **kwargs):
    from kazi.queue.celery_worker import build_celery_app
    app = build_celery_app(config or _config(), broker=REDIS_URL, **kwargs)
    app.conf.task_always_eager = True
    return app


# ── _create_kazi ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.skipif(not OPENAI_KEY, reason="OPENAI_API_KEY not set")
async def test_create_kazi_returns_kazi_instance():
    from kazi.core.orchestrator import Kazi
    from kazi.queue.celery_worker import _create_kazi

    kazi = await _create_kazi(_config(api_key=OPENAI_KEY))
    assert isinstance(kazi, Kazi)
    await kazi.close()


# ── _get_kazi creates instance when none exists ──────────────────────────────

@pytest.mark.skipif(not OPENAI_KEY, reason="OPENAI_API_KEY not set")
def test_get_kazi_creates_instance_when_none():
    import kazi.queue.celery_worker as cw
    from kazi.core.orchestrator import Kazi

    orig_instance = cw._kazi_instance
    orig_config = cw._kazi_config
    try:
        cw._kazi_instance = None
        cw._kazi_config = _config(api_key=OPENAI_KEY)
        instance = cw._get_kazi()
        assert isinstance(instance, Kazi)
        # Calling again returns cached instance
        assert cw._get_kazi() is instance
    finally:
        if cw._kazi_instance and cw._kazi_instance is not orig_instance:
            asyncio.run(cw._kazi_instance.close())
        cw._kazi_instance = orig_instance
        cw._kazi_config = orig_config


# ── kazi.run task ────────────────────────────────────────────────────────────

class _StubKazi:
    async def run(self, message, *, thread_id="default", system_prompt=None,
                  max_tool_calls=25, track_cost=False, **kwargs):
        return f"echo: {message}"

    async def ingest(self, path, *, index_name="default"):
        pass


def test_kazi_run_task_executes_and_returns_reply():
    import kazi.queue.celery_worker as cw

    orig = cw._kazi_instance
    try:
        cw._kazi_instance = _StubKazi()
        app = _make_app()
        result = app.tasks["kazi.run"].apply(
            args=["hello"], kwargs={"thread_id": "test"}
        )
        payload = result.result
        assert "reply" in payload
        assert "echo: hello" in payload["reply"]
    finally:
        cw._kazi_instance = orig


def test_kazi_run_task_with_all_options():
    import kazi.queue.celery_worker as cw

    orig = cw._kazi_instance
    try:
        cw._kazi_instance = _StubKazi()
        app = _make_app()
        result = app.tasks["kazi.run"].apply(
            args=["test message"],
            kwargs={
                "thread_id": "t1",
                "system_prompt": "Be brief.",
                "max_tool_calls": 5,
                "track_cost": False,
            },
        )
        assert result.result["reply"] == "echo: test message"
    finally:
        cw._kazi_instance = orig


def test_kazi_run_task_with_run_result():
    """When kazi.run returns a RunResult, the task serializes it correctly."""
    import kazi.queue.celery_worker as cw
    from kazi.core.cost import RunCost, RunResult

    class _CostKazi:
        async def run(self, message, *, track_cost=False, **kwargs):
            if track_cost:
                return RunResult(
                    reply="cost reply",
                    cost=RunCost(input_tokens=10, output_tokens=5, cost_usd=0.001),
                )
            return "plain reply"

    orig = cw._kazi_instance
    try:
        cw._kazi_instance = _CostKazi()
        app = _make_app()
        result = app.tasks["kazi.run"].apply(
            args=["hello"], kwargs={"track_cost": True}
        )
        payload = result.result
        assert payload["reply"] == "cost reply"
        assert payload["cost_usd"] == pytest.approx(0.001)
        assert payload["input_tokens"] == 10
        assert payload["output_tokens"] == 5
    finally:
        cw._kazi_instance = orig


def test_kazi_run_task_retry_on_failure():
    """On failure below max_retries, task stores exception in result (celery eager mode)."""
    import kazi.queue.celery_worker as cw

    class _FailKazi:
        async def run(self, *args, **kwargs):
            raise RuntimeError("transient failure")

    orig = cw._kazi_instance
    try:
        cw._kazi_instance = _FailKazi()
        app = _make_app(max_retries=2)
        result = app.tasks["kazi.run"].apply(args=["fail"])
        assert result.failed()
        with pytest.raises(RuntimeError, match="transient failure"):
            result.get(propagate=True)
    finally:
        cw._kazi_instance = orig


# ── kazi.ingest task ─────────────────────────────────────────────────────────

def test_kazi_ingest_task_returns_ok():
    import kazi.queue.celery_worker as cw

    orig = cw._kazi_instance
    try:
        cw._kazi_instance = _StubKazi()
        app = _make_app()
        result = app.tasks["kazi.ingest"].apply(
            args=["/docs/report.pdf"], kwargs={"index_name": "q3"}
        )
        payload = result.result
        assert payload["status"] == "ok"
        assert payload["path"] == "/docs/report.pdf"
        assert payload["index"] == "q3"
    finally:
        cw._kazi_instance = orig


def test_kazi_ingest_task_default_index_name():
    import kazi.queue.celery_worker as cw

    orig = cw._kazi_instance
    try:
        cw._kazi_instance = _StubKazi()
        app = _make_app()
        result = app.tasks["kazi.ingest"].apply(args=["/path/file.pdf"])
        assert result.result["index"] == "default"
    finally:
        cw._kazi_instance = orig


def test_kazi_ingest_task_retry_on_failure():
    import kazi.queue.celery_worker as cw

    class _FailIngest:
        async def ingest(self, *args, **kwargs):
            raise OSError("disk error")

    orig = cw._kazi_instance
    try:
        cw._kazi_instance = _FailIngest()
        app = _make_app(max_retries=2)
        result = app.tasks["kazi.ingest"].apply(args=["/bad/path"])
        assert result.failed()
        with pytest.raises(OSError, match="disk error"):
            result.get(propagate=True)
    finally:
        cw._kazi_instance = orig


# ── kazi.dead_letter task (already covered, but test custom dlq_queue) ───────

def test_build_celery_app_custom_dlq_queue():
    from kazi.queue.celery_worker import build_celery_app

    app = build_celery_app(_config(), broker=REDIS_URL, dlq_queue="my-dlq")
    routes = app.conf.task_routes
    assert routes.get("kazi.dead_letter", {}).get("queue") == "my-dlq"


def test_build_celery_app_custom_time_limits():
    from kazi.queue.celery_worker import build_celery_app

    app = build_celery_app(
        _config(), broker=REDIS_URL,
        task_time_limit=600, task_soft_time_limit=550,
    )
    assert app.conf.task_time_limit == 600
    assert app.conf.task_soft_time_limit == 550
