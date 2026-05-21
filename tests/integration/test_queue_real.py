"""
Real queue integration tests — ARQ and Celery workers against a live Redis.

Requires:
  - Redis running on localhost:6379
  - arq and celery installed
  - OPENAI_API_KEY set (for tests that actually run Kazi)

Structure
---------
Section 1: ARQ — KaziQueue client, DLQ helpers, _kazi_run task, build_worker_settings
Section 2: Celery — build_celery_app, task registration, _get_kazi, dead_letter task
Section 3: Webhook — _sign, dispatch_webhook with a real local HTTP server
"""
from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

REDIS_URL = "redis://localhost:6379/15"   # DB 15 keeps tests isolated from default DB 0
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
needs_llm = pytest.mark.skipif(not OPENAI_KEY, reason="OPENAI_API_KEY not set")

# ── Redis availability guard ───────────────────────────────────────────────────

def _redis_available() -> bool:
    try:
        import redis
        r = redis.Redis.from_url(REDIS_URL)
        r.ping()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _redis_available(),
    reason="Redis not available on localhost:6379",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
async def _flush_test_db():
    """Flush DB 15 before and after each test so tests don't bleed state."""
    from urllib.parse import urlparse

    from arq.connections import RedisSettings, create_pool
    parsed = urlparse(REDIS_URL)
    settings = RedisSettings(
        host=parsed.hostname or "localhost",
        port=parsed.port or 6379,
        database=int(parsed.path.lstrip("/") or 0),
    )
    pool = await create_pool(settings)
    await pool.flushdb()
    yield
    await pool.flushdb()
    await pool.aclose()


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: ARQ
# ═══════════════════════════════════════════════════════════════════════════════

# ── KaziQueue.connect / close / context manager ──────────────────────────────

@pytest.mark.asyncio
async def test_kazi_queue_connect_and_close():
    from kazi.queue.arq_worker import KaziQueue
    queue = await KaziQueue.connect(REDIS_URL)
    assert queue._pool is not None
    await queue.close()


@pytest.mark.asyncio
async def test_kazi_queue_context_manager():
    from kazi.queue.arq_worker import KaziQueue
    async with await KaziQueue.connect(REDIS_URL) as queue:
        assert queue._pool is not None


# ── enqueue ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_enqueue_returns_job_handle():
    from kazi.queue.arq_worker import KaziQueue
    async with await KaziQueue.connect(REDIS_URL) as queue:
        job = await queue.enqueue("hello", thread_id="t1")
        assert job is not None
        assert hasattr(job, "job_id")


@pytest.mark.asyncio
async def test_enqueue_with_all_options():
    from kazi.queue.arq_worker import KaziQueue
    async with await KaziQueue.connect(REDIS_URL) as queue:
        job = await queue.enqueue(
            "hello",
            thread_id="t1",
            system_prompt="Be brief.",
            max_tool_calls=5,
            track_cost=True,
            webhook_url=None,
            webhook_secret="",
        )
        assert job is not None


# ── enqueue_ingest ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_enqueue_ingest_returns_job_handle():
    from kazi.queue.arq_worker import KaziQueue
    async with await KaziQueue.connect(REDIS_URL) as queue:
        job = await queue.enqueue_ingest("/some/path", index_name="docs")
        assert job is not None


# ── DLQ helpers ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_dlq_jobs_empty_initially():
    from kazi.queue.arq_worker import KaziQueue
    async with await KaziQueue.connect(REDIS_URL) as queue:
        jobs = await queue.get_dlq_jobs()
        assert jobs == []


@pytest.mark.asyncio
async def test_push_dlq_and_retrieve():
    from kazi.queue.arq_worker import KaziQueue, _push_dlq
    async with await KaziQueue.connect(REDIS_URL) as queue:
        await _push_dlq(
            queue._pool,
            job_id="job-1",
            function="kazi_run",
            message="hello",
            error=RuntimeError("test error"),
            attempt=3,
        )
        jobs = await queue.get_dlq_jobs()
        assert len(jobs) == 1
        assert jobs[0]["job_id"] == "job-1"
        assert jobs[0]["function"] == "kazi_run"
        assert "test error" in jobs[0]["error"]
        assert jobs[0]["attempt"] == 3
        assert "failed_at" in jobs[0]


@pytest.mark.asyncio
async def test_push_dlq_truncates_long_message():
    from kazi.queue.arq_worker import KaziQueue, _push_dlq
    async with await KaziQueue.connect(REDIS_URL) as queue:
        long_msg = "x" * 5000
        await _push_dlq(
            queue._pool,
            job_id="job-long",
            function="kazi_run",
            message=long_msg,
            error=RuntimeError("err"),
            attempt=1,
        )
        jobs = await queue.get_dlq_jobs()
        assert len(jobs[0]["message"]) <= 2000


@pytest.mark.asyncio
async def test_clear_dlq_removes_all_entries():
    from kazi.queue.arq_worker import KaziQueue, _push_dlq
    async with await KaziQueue.connect(REDIS_URL) as queue:
        for i in range(3):
            await _push_dlq(
                queue._pool,
                job_id=f"job-{i}",
                function="kazi_run",
                message="msg",
                error=RuntimeError("err"),
                attempt=1,
            )
        count = await queue.clear_dlq()
        assert count == 3
        assert await queue.get_dlq_jobs() == []


@pytest.mark.asyncio
async def test_requeue_dlq_only_requeues_kazi_run():
    from kazi.queue.arq_worker import KaziQueue, _push_dlq
    async with await KaziQueue.connect(REDIS_URL) as queue:
        await _push_dlq(
            queue._pool,
            job_id="run-job",
            function="kazi_run",
            message="retry me",
            error=RuntimeError("err"),
            attempt=3,
        )
        await _push_dlq(
            queue._pool,
            job_id="ingest-job",
            function="kazi_ingest",
            message="/some/path",
            error=RuntimeError("err"),
            attempt=3,
        )
        requeued = await queue.requeue_dlq()
        assert requeued == 1  # only kazi_run is requeued


@pytest.mark.asyncio
async def test_requeue_dlq_empty_is_zero():
    from kazi.queue.arq_worker import KaziQueue
    async with await KaziQueue.connect(REDIS_URL) as queue:
        assert await queue.requeue_dlq() == 0


@pytest.mark.asyncio
async def test_get_dlq_jobs_limit_respected():
    from kazi.queue.arq_worker import KaziQueue, _push_dlq
    async with await KaziQueue.connect(REDIS_URL) as queue:
        for i in range(10):
            await _push_dlq(
                queue._pool,
                job_id=f"job-{i}",
                function="kazi_run",
                message="msg",
                error=RuntimeError("e"),
                attempt=1,
            )
        jobs = await queue.get_dlq_jobs(limit=3)
        assert len(jobs) == 3


# ── _kazi_run task function ──────────────────────────────────────────────────

@pytest.mark.asyncio
@needs_llm
async def test_kazi_run_task_returns_reply():
    """Execute _kazi_run directly with a real Kazi in ctx."""
    from kazi import Kazi
    from kazi.core.config import KaziConfig, LLMConfig, LLMProvider
    from kazi.queue.arq_worker import KaziQueue, _kazi_run

    config = KaziConfig(llm=LLMConfig(
        provider=LLMProvider.OPENAI, model="gpt-4o-mini", api_key=OPENAI_KEY
    ))
    async with await Kazi.create(config) as kazi:
        async with await KaziQueue.connect(REDIS_URL) as queue:
            ctx = {
                "kazi": kazi,
                "redis": queue._pool,
                "max_task_retries": 3,
                "job_try": 1,
                "job_id": "test-job",
            }
            result = await _kazi_run(ctx, "Say: TASK_OK", thread_id="arq-test")
    assert "reply" in result
    assert isinstance(result["reply"], str)
    assert len(result["reply"]) > 0


@pytest.mark.asyncio
@needs_llm
async def test_kazi_run_task_with_cost_tracking():
    from kazi import Kazi
    from kazi.core.config import KaziConfig, LLMConfig, LLMProvider
    from kazi.queue.arq_worker import KaziQueue, _kazi_run

    config = KaziConfig(llm=LLMConfig(
        provider=LLMProvider.OPENAI, model="gpt-4o-mini", api_key=OPENAI_KEY
    ))
    async with await Kazi.create(config) as kazi:
        async with await KaziQueue.connect(REDIS_URL) as queue:
            ctx = {
                "kazi": kazi,
                "redis": queue._pool,
                "max_task_retries": 3,
                "job_try": 1,
                "job_id": "cost-job",
            }
            result = await _kazi_run(ctx, "Say: OK", track_cost=True)
    assert "reply" in result


@pytest.mark.asyncio
async def test_kazi_run_task_pushes_dlq_on_final_attempt():
    """On the last retry, a failing task pushes to DLQ and re-raises."""
    from kazi.queue.arq_worker import KaziQueue, _kazi_run

    class _FailingKazi:
        async def run(self, *args, **kwargs):
            raise RuntimeError("always fails")

    async with await KaziQueue.connect(REDIS_URL) as queue:
        ctx = {
            "kazi": _FailingKazi(),
            "redis": queue._pool,
            "max_task_retries": 3,
            "job_try": 3,  # final attempt
            "job_id": "fail-job",
        }
        with pytest.raises(RuntimeError, match="always fails"):
            await _kazi_run(ctx, "fail me", thread_id="t")

        jobs = await queue.get_dlq_jobs()
        assert len(jobs) == 1
        assert jobs[0]["job_id"] == "fail-job"


@pytest.mark.asyncio
async def test_kazi_run_task_warns_on_non_final_attempt():
    """On a non-final retry, re-raises without pushing to DLQ."""
    from kazi.queue.arq_worker import KaziQueue, _kazi_run

    class _FailingKazi:
        async def run(self, *args, **kwargs):
            raise RuntimeError("transient")

    async with await KaziQueue.connect(REDIS_URL) as queue:
        ctx = {
            "kazi": _FailingKazi(),
            "redis": queue._pool,
            "max_task_retries": 3,
            "job_try": 1,  # not final
            "job_id": "warn-job",
        }
        with pytest.raises(RuntimeError):
            await _kazi_run(ctx, "fail", thread_id="t")

        # DLQ should be empty — only warns on non-final
        assert await queue.get_dlq_jobs() == []


# ── _kazi_ingest task function ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_kazi_ingest_task_returns_ok():
    from kazi.queue.arq_worker import KaziQueue, _kazi_ingest

    class _StubKazi:
        async def ingest(self, path, *, index_name="default"):
            pass

    async with await KaziQueue.connect(REDIS_URL) as queue:
        ctx = {
            "kazi": _StubKazi(),
            "redis": queue._pool,
            "max_task_retries": 3,
            "job_try": 1,
            "job_id": "ingest-job",
        }
        result = await _kazi_ingest(ctx, "/docs/report.pdf", index_name="q3")
    assert result["status"] == "ok"
    assert result["path"] == "/docs/report.pdf"
    assert result["index"] == "q3"


@pytest.mark.asyncio
async def test_kazi_ingest_task_pushes_dlq_on_final_failure():
    from kazi.queue.arq_worker import KaziQueue, _kazi_ingest

    class _FailIngest:
        async def ingest(self, *args, **kwargs):
            raise OSError("disk full")

    async with await KaziQueue.connect(REDIS_URL) as queue:
        ctx = {
            "kazi": _FailIngest(),
            "redis": queue._pool,
            "max_task_retries": 3,
            "job_try": 3,
            "job_id": "ingest-fail",
        }
        with pytest.raises(OSError):
            await _kazi_ingest(ctx, "/bad/path")

        jobs = await queue.get_dlq_jobs()
        assert len(jobs) == 1
        assert "disk full" in jobs[0]["error"]


# ── build_worker_settings ─────────────────────────────────────────────────────

def test_build_worker_settings_returns_class():
    from kazi.core.config import KaziConfig, LLMConfig, LLMProvider
    from kazi.queue.arq_worker import build_worker_settings

    config = KaziConfig(llm=LLMConfig(
        provider=LLMProvider.OPENAI, model="gpt-4o-mini", api_key="fake"
    ))
    WorkerSettings = build_worker_settings(config, redis_url=REDIS_URL)
    assert hasattr(WorkerSettings, "functions")
    assert hasattr(WorkerSettings, "on_startup")
    assert hasattr(WorkerSettings, "on_shutdown")
    assert hasattr(WorkerSettings, "redis_settings")
    assert WorkerSettings.max_jobs == 10


def test_build_worker_settings_custom_params():
    from kazi.core.config import KaziConfig, LLMConfig, LLMProvider
    from kazi.queue.arq_worker import build_worker_settings

    config = KaziConfig(llm=LLMConfig(
        provider=LLMProvider.OPENAI, model="gpt-4o-mini", api_key="fake"
    ))
    WS = build_worker_settings(
        config, redis_url=REDIS_URL, max_jobs=5, max_task_retries=2
    )
    assert WS.max_jobs == 5
    assert WS.max_tries == 2


def test_build_worker_settings_raises_without_arq(monkeypatch):
    import builtins
    real = builtins.__import__

    def _block(name, *args, **kwargs):
        if name == "arq.connections":
            raise ImportError("no arq")
        return real(name, *args, **kwargs)

    from kazi.core.config import KaziConfig, LLMConfig, LLMProvider
    from kazi.queue.arq_worker import build_worker_settings

    config = KaziConfig(llm=LLMConfig(
        provider=LLMProvider.OPENAI, model="gpt-4o-mini", api_key="fake"
    ))
    monkeypatch.setattr(builtins, "__import__", _block)
    with pytest.raises(ImportError, match="arq is required"):
        build_worker_settings(config, redis_url=REDIS_URL)


@pytest.mark.asyncio
async def test_worker_on_startup_and_shutdown():
    """on_startup creates a Kazi; on_shutdown closes it cleanly."""
    from kazi.core.config import KaziConfig, LLMConfig, LLMProvider
    from kazi.queue.arq_worker import KaziQueue, build_worker_settings

    config = KaziConfig(llm=LLMConfig(
        provider=LLMProvider.OPENAI, model="gpt-4o-mini", api_key=OPENAI_KEY or "fake"
    ))
    WS = build_worker_settings(config, redis_url=REDIS_URL)

    async with await KaziQueue.connect(REDIS_URL) as queue:
        ctx: dict = {"redis": queue._pool}
        await WS.on_startup(ctx)
        assert "kazi" in ctx
        assert ctx.get("max_task_retries") == 3
        await WS.on_shutdown(ctx)


@pytest.mark.asyncio
async def test_worker_on_shutdown_handles_missing_kazi():
    """on_shutdown is safe even if kazi was never created."""
    from kazi.core.config import KaziConfig, LLMConfig, LLMProvider
    from kazi.queue.arq_worker import build_worker_settings

    config = KaziConfig(llm=LLMConfig(
        provider=LLMProvider.OPENAI, model="gpt-4o-mini", api_key="fake"
    ))
    WS = build_worker_settings(config, redis_url=REDIS_URL)
    await WS.on_shutdown({})  # no "kazi" key — must not raise


# ── get_result polling ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_result_raises_timeout_for_unexecuted_job():
    from kazi.queue.arq_worker import KaziQueue
    async with await KaziQueue.connect(REDIS_URL) as queue:
        job = await queue.enqueue("hello")
        with pytest.raises((TimeoutError, RuntimeError)):
            await queue.get_result(job.job_id, timeout=1)


@pytest.mark.asyncio
async def test_get_result_raises_runtime_error_for_nonexistent_job():
    """A completely non-existent job_id immediately returns not_found."""
    from kazi.queue.arq_worker import KaziQueue
    async with await KaziQueue.connect(REDIS_URL) as queue:
        with pytest.raises(RuntimeError, match="not found"):
            await queue.get_result("nonexistent-job-id-abc123", timeout=5)


@pytest.mark.asyncio
async def test_get_result_returns_job_result_for_completed_job():
    """Inject a pre-computed pickled result into Redis then poll get_result."""
    import pickle

    from kazi.queue.arq_worker import JobResult, KaziQueue

    async with await KaziQueue.connect(REDIS_URL) as queue:
        fake_job_id = "test-complete-job-xyz"
        payload = {"reply": "done", "cost_usd": None, "input_tokens": None, "output_tokens": None}
        result_key = f"arq:result:{fake_job_id}"
        now_ms = int(time.time() * 1000)
        # arq deserializes via pickle.loads with these keys
        arq_result = {
            "t": 1, "f": "_kazi_run", "a": [], "k": {}, "et": now_ms - 2000,
            "s": True, "r": payload, "st": now_ms - 1000, "ft": now_ms,
            "q": "arq:queue", "id": fake_job_id,
        }
        await queue._pool.set(result_key, pickle.dumps(arq_result))

        result = await queue.get_result(fake_job_id, timeout=5)
        assert isinstance(result, JobResult)
        assert result.reply == "done"


@pytest.mark.asyncio
async def test_get_dlq_jobs_handles_corrupt_entry():
    """A non-JSON DLQ entry is returned as {"raw": ...} without crashing."""
    from kazi.queue.arq_worker import _DLQ_KEY, KaziQueue
    async with await KaziQueue.connect(REDIS_URL) as queue:
        await queue._pool.rpush(_DLQ_KEY, b"this is not valid JSON!!!")
        jobs = await queue.get_dlq_jobs()
        corrupt = [j for j in jobs if "raw" in j]
        assert len(corrupt) >= 1


@pytest.mark.asyncio
async def test_requeue_dlq_re_enqueues_kazi_run_jobs():
    """kazi_run entries are re-enqueued; ingest entries are skipped."""
    import json

    from kazi.queue.arq_worker import _DLQ_KEY, KaziQueue

    async with await KaziQueue.connect(REDIS_URL) as queue:
        # Push a kazi_run DLQ entry and an ingest entry
        await queue._pool.rpush(_DLQ_KEY, json.dumps({
            "function": "kazi_run", "message": "retry this", "job_id": "j1",
            "error": "boom", "attempt": 3, "failed_at": "2026-01-01T00:00:00",
        }))
        await queue._pool.rpush(_DLQ_KEY, json.dumps({
            "function": "kazi_ingest", "message": "/some/path", "job_id": "j2",
            "error": "boom", "attempt": 3, "failed_at": "2026-01-01T00:00:00",
        }))

        requeued = await queue.requeue_dlq()
        assert requeued == 1


@pytest.mark.asyncio
async def test_kazi_run_task_dispatches_webhook():
    """When webhook_url is set, _kazi_run dispatches the result to the URL."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    received: list[bytes] = []

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            received.append(self.rfile.read(length))
            self.send_response(200)
            self.end_headers()
        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()

    class _StubKazi:
        async def run(self, *a, **kw):
            return "webhook reply"

    from kazi.queue.arq_worker import _kazi_run
    ctx = {"kazi": _StubKazi(), "job_id": "wh-test", "job_try": 1, "max_task_retries": 3}
    payload = await _kazi_run(
        ctx,
        "hello",
        webhook_url=f"http://127.0.0.1:{port}/webhook",
        webhook_secret="",
    )
    server.shutdown()

    assert payload["reply"] == "webhook reply"
    assert len(received) == 1
    import json
    body = json.loads(received[0])
    assert body["result"]["reply"] == "webhook reply"
    assert "event" in body


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Celery
# ═══════════════════════════════════════════════════════════════════════════════

def test_build_celery_app_returns_celery_instance():
    from kazi.core.config import KaziConfig, LLMConfig, LLMProvider
    from kazi.queue.celery_worker import build_celery_app

    config = KaziConfig(llm=LLMConfig(
        provider=LLMProvider.OPENAI, model="gpt-4o-mini", api_key="fake"
    ))
    app = build_celery_app(config, broker=REDIS_URL)
    from celery import Celery
    assert isinstance(app, Celery)


def test_build_celery_app_registers_tasks():
    from kazi.core.config import KaziConfig, LLMConfig, LLMProvider
    from kazi.queue.celery_worker import build_celery_app

    config = KaziConfig(llm=LLMConfig(
        provider=LLMProvider.OPENAI, model="gpt-4o-mini", api_key="fake"
    ))
    app = build_celery_app(config, broker=REDIS_URL)
    assert "kazi.run" in app.tasks
    assert "kazi.ingest" in app.tasks
    assert "kazi.dead_letter" in app.tasks


def test_build_celery_app_raises_without_celery(monkeypatch):
    import builtins
    real = builtins.__import__

    def _block(name, *args, **kwargs):
        if name == "celery":
            raise ImportError("no celery")
        return real(name, *args, **kwargs)

    from kazi.core.config import KaziConfig, LLMConfig, LLMProvider
    from kazi.queue.celery_worker import build_celery_app

    config = KaziConfig(llm=LLMConfig(
        provider=LLMProvider.OPENAI, model="gpt-4o-mini", api_key="fake"
    ))
    monkeypatch.setattr(builtins, "__import__", _block)
    with pytest.raises(ImportError, match="celery is required"):
        build_celery_app(config, broker=REDIS_URL)


def test_celery_dead_letter_task_returns_dict():
    from kazi.core.config import KaziConfig, LLMConfig, LLMProvider
    from kazi.queue.celery_worker import build_celery_app

    config = KaziConfig(llm=LLMConfig(
        provider=LLMProvider.OPENAI, model="gpt-4o-mini", api_key="fake"
    ))
    app = build_celery_app(config, broker=REDIS_URL)
    task = app.tasks["kazi.dead_letter"]

    # Call underlying function directly (no broker needed)
    result = task.run(
        task_name="kazi.run",
        message="hello",
        error="something broke",
        attempts=3,
    )
    assert result["status"] == "dead_lettered"
    assert result["task_name"] == "kazi.run"
    assert result["error"] == "something broke"
    assert result["attempts"] == 3
    assert "received_at" in result


def test_get_kazi_raises_when_config_not_set():
    """_get_kazi() must raise RuntimeError if build_celery_app hasn't been called."""
    import kazi.queue.celery_worker as cw
    original = cw._kazi_config
    original_instance = cw._kazi_instance
    try:
        cw._kazi_config = None
        cw._kazi_instance = None
        with pytest.raises(RuntimeError, match="KaziConfig not set"):
            cw._get_kazi()
    finally:
        cw._kazi_config = original
        cw._kazi_instance = original_instance


def test_get_kazi_returns_cached_instance():
    """_get_kazi() returns the same instance on repeated calls."""
    import kazi.queue.celery_worker as cw

    sentinel = object()
    original = cw._kazi_instance
    try:
        cw._kazi_instance = sentinel
        assert cw._get_kazi() is sentinel
    finally:
        cw._kazi_instance = original


def test_celery_app_conf_applied():
    from kazi.core.config import KaziConfig, LLMConfig, LLMProvider
    from kazi.queue.celery_worker import build_celery_app

    config = KaziConfig(llm=LLMConfig(
        provider=LLMProvider.OPENAI, model="gpt-4o-mini", api_key="fake"
    ))
    app = build_celery_app(config, broker=REDIS_URL, task_time_limit=600)
    assert app.conf.task_time_limit == 600
    assert app.conf.task_serializer == "json"


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: Webhook
# ═══════════════════════════════════════════════════════════════════════════════

def test_sign_produces_hex_digest():
    from kazi.queue.webhook import _sign
    sig = _sign(b"hello world", "my-secret")
    assert len(sig) == 64  # SHA-256 hex = 64 chars
    assert all(c in "0123456789abcdef" for c in sig)


def test_sign_is_deterministic():
    from kazi.queue.webhook import _sign
    assert _sign(b"data", "secret") == _sign(b"data", "secret")


def test_sign_differs_with_different_secret():
    from kazi.queue.webhook import _sign
    assert _sign(b"data", "secret1") != _sign(b"data", "secret2")


# ── Local HTTP server for webhook tests ───────────────────────────────────────

class _WebhookCapture(BaseHTTPRequestHandler):
    captured: list[dict] = []
    respond_status: int = 200

    def log_message(self, *_):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        _WebhookCapture.captured.append({
            "body": json.loads(body),
            "headers": dict(self.headers),
        })
        self.send_response(_WebhookCapture.respond_status)
        self.end_headers()


@contextmanager
def _webhook_server(status: int = 200):
    _WebhookCapture.captured = []
    _WebhookCapture.respond_status = status
    server = HTTPServer(("127.0.0.1", 0), _WebhookCapture)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        yield f"http://localhost:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


@pytest.mark.asyncio
async def test_dispatch_webhook_posts_to_url():
    from kazi.queue.webhook import WebhookConfig, dispatch_webhook

    with _webhook_server() as url:
        config = WebhookConfig(url=url)
        ok = await dispatch_webhook(config, job_id="j1", result={"reply": "hello"})
    assert ok is True
    assert len(_WebhookCapture.captured) == 1
    payload = _WebhookCapture.captured[0]["body"]
    assert payload["job_id"] == "j1"
    assert payload["event"] == "job.complete"
    assert payload["result"]["reply"] == "hello"


@pytest.mark.asyncio
async def test_dispatch_webhook_signs_request():
    from kazi.queue.webhook import WebhookConfig, dispatch_webhook

    with _webhook_server() as url:
        config = WebhookConfig(url=url, secret="my-secret")
        await dispatch_webhook(config, job_id="j2", result={"reply": "ok"})

    headers = _WebhookCapture.captured[0]["headers"]
    assert "X-Kazi-Signature" in headers or "x-kazi-signature" in headers


@pytest.mark.asyncio
async def test_dispatch_webhook_no_signature_when_no_secret():
    from kazi.queue.webhook import WebhookConfig, dispatch_webhook

    with _webhook_server() as url:
        config = WebhookConfig(url=url, secret="")
        await dispatch_webhook(config, job_id="j3", result={"reply": "ok"})

    headers = _WebhookCapture.captured[0]["headers"]
    assert "X-Kazi-Signature" not in headers
    assert "x-kazi-signature" not in headers


@pytest.mark.asyncio
async def test_dispatch_webhook_excludes_reply_when_include_reply_false():
    from kazi.queue.webhook import WebhookConfig, dispatch_webhook

    with _webhook_server() as url:
        config = WebhookConfig(url=url, include_reply=False)
        await dispatch_webhook(
            config,
            job_id="j4",
            result={"reply": "secret text", "cost_usd": 0.001},
        )

    payload = _WebhookCapture.captured[0]["body"]
    assert "reply" not in payload["result"]
    assert "cost_usd" in payload["result"]


@pytest.mark.asyncio
async def test_dispatch_webhook_returns_false_on_4xx():
    from kazi.queue.webhook import WebhookConfig, dispatch_webhook

    with _webhook_server(status=400) as url:
        config = WebhookConfig(url=url, retry_attempts=1)
        ok = await dispatch_webhook(config, job_id="j5", result={"reply": "x"})
    assert ok is False


@pytest.mark.asyncio
async def test_dispatch_webhook_retries_on_failure():
    from kazi.queue.webhook import WebhookConfig, dispatch_webhook

    with _webhook_server(status=500) as url:
        config = WebhookConfig(url=url, retry_attempts=3)
        ok = await dispatch_webhook(config, job_id="j6", result={"reply": "x"})

    assert ok is False
    # 3 retry attempts → server should have received 3 requests
    assert len(_WebhookCapture.captured) == 3


@pytest.mark.asyncio
async def test_dispatch_webhook_custom_event():
    from kazi.queue.webhook import WebhookConfig, dispatch_webhook

    with _webhook_server() as url:
        config = WebhookConfig(url=url)
        await dispatch_webhook(config, job_id="j7", result={}, event="job.failed")

    assert _WebhookCapture.captured[0]["body"]["event"] == "job.failed"


@pytest.mark.asyncio
async def test_dispatch_webhook_returns_false_without_aiohttp(monkeypatch):
    import builtins
    real = builtins.__import__

    def _block(name, *args, **kwargs):
        if name == "aiohttp":
            raise ImportError("no aiohttp")
        return real(name, *args, **kwargs)

    from kazi.queue.webhook import WebhookConfig, dispatch_webhook
    monkeypatch.setattr(builtins, "__import__", _block)

    config = WebhookConfig(url="http://example.com")
    ok = await dispatch_webhook(config, job_id="j8", result={})
    assert ok is False
