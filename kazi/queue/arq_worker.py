"""
ARQ async queue integration for kazi.

ARQ is a lightweight async task queue backed by Redis, purpose-built for
asyncio applications.  It is the recommended queue for kazi because the
entire library is async-native — no sync wrappers, no event loop gymnastics.

Install::

    pip install kazi-core[arq]

Quick start::

    # worker.py
    from kazi import KaziConfig, LLMConfig, LLMProvider
    from kazi.queue.arq_worker import build_worker_settings

    config = KaziConfig(llm=LLMConfig(provider=LLMProvider.ANTHROPIC, model="claude-sonnet-4-6"))
    WorkerSettings = build_worker_settings(
        config,
        redis_url="redis://localhost:6379",
        max_task_retries=3,      # retry 3× before sending to DLQ
    )

    # Run the worker:
    #   python -m arq worker.WorkerSettings

    # Enqueue from your app:
    from kazi.queue.arq_worker import KaziQueue

    async def main():
        queue = await KaziQueue.connect("redis://localhost:6379")
        job = await queue.enqueue("Analyse Q3 expenses", thread_id="user:123")
        result = await queue.get_result(job.job_id, timeout=120)
        print(result.reply)

        # Inspect the dead-letter queue for failed jobs
        failed = await queue.get_dlq_jobs(limit=10)
        for entry in failed:
            print(entry["error"])

Dead-letter queue
-----------------
Failed jobs (after max_task_retries) are pushed to the Redis list
``kazi:dlq``.  Each entry is a JSON object::

    {
        "job_id": "...",
        "function": "kazi_run",
        "message": "<truncated input>",
        "error": "<exception message>",
        "attempt": 3,
        "failed_at": 1715000000.0
    }

Use ``KaziQueue.get_dlq_jobs()`` to inspect and ``KaziQueue.requeue_dlq()``
to reprocess entries after fixing the underlying issue.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Redis key for the dead-letter queue
_DLQ_KEY = "kazi:dlq"
_DLQ_MAX_SIZE = 10_000  # cap to prevent unbounded growth


# ── Dead-letter queue helpers ─────────────────────────────────────────────────

async def _push_dlq(
    redis,
    *,
    job_id: str,
    function: str,
    message: str,
    error: Exception,
    attempt: int,
) -> None:
    """Push a failed job record to the dead-letter queue Redis list."""
    entry = json.dumps({
        "job_id": job_id,
        "function": function,
        "message": message[:2000],
        "error": str(error)[:1000],
        "attempt": attempt,
        "failed_at": time.time(),
    }, separators=(",", ":"))
    try:
        await redis.rpush(_DLQ_KEY, entry)
        # Trim to max size so the list never grows unbounded
        await redis.ltrim(_DLQ_KEY, -_DLQ_MAX_SIZE, -1)
        logger.error(
            "Job %s pushed to DLQ after %d attempt(s): %s", job_id, attempt, error
        )
    except Exception as exc:
        logger.error("Failed to push job %s to DLQ: %s", job_id, exc)


# ── Task functions (run inside the ARQ worker process) ────────────────────────

async def _kazi_run(
    ctx: dict,
    message: str,
    *,
    thread_id: str = "default",
    system_prompt: str | None = None,
    max_tool_calls: int = 25,
    track_cost: bool = False,
    webhook_url: str | None = None,
    webhook_secret: str = "",
) -> dict:
    """
    ARQ task function: run a kazi agent turn in the background.

    ctx["kazi"] is the shared Kazi instance created in on_startup.
    Returns a dict so ARQ can serialize the result to Redis.

    Retry behaviour
    ---------------
    ARQ automatically retries up to ``max_tries`` times (set in WorkerSettings).
    ``ctx["job_try"]`` gives the current attempt number (1-indexed).
    On the final failed attempt, the job is pushed to the dead-letter queue at
    ``kazi:dlq`` for manual inspection and reprocessing.

    When ``webhook_url`` is provided, POSTs the result to that URL after
    completion (HMAC-signed when ``webhook_secret`` is non-empty).
    """
    kazi = ctx["kazi"]
    max_retries: int = ctx.get("max_task_retries", 3)
    job_try: int = ctx.get("job_try", 1)
    job_id: str = ctx.get("job_id", "unknown")

    from kazi.core.cost import RunResult
    try:
        result = await kazi.run(
            message,
            thread_id=thread_id,
            system_prompt=system_prompt,
            max_tool_calls=max_tool_calls,
            track_cost=track_cost,
        )
    except Exception as exc:
        if job_try >= max_retries:
            await _push_dlq(
                ctx["redis"],
                job_id=job_id,
                function="kazi_run",
                message=message,
                error=exc,
                attempt=job_try,
            )
        else:
            logger.warning(
                "kazi_run job %s attempt %d/%d failed — will retry: %s",
                job_id, job_try, max_retries, exc,
            )
        raise

    if isinstance(result, RunResult):
        payload = {
            "reply": result.reply,
            "cost_usd": result.cost.cost_usd,
            "input_tokens": result.cost.input_tokens,
            "output_tokens": result.cost.output_tokens,
        }
    else:
        payload = {"reply": result}

    if webhook_url:
        from kazi.queue.webhook import WebhookConfig, dispatch_webhook
        wh_config = WebhookConfig(url=webhook_url, secret=webhook_secret)
        await dispatch_webhook(wh_config, job_id=job_id, result=payload)

    return payload


async def _kazi_ingest(ctx: dict, path: str, *, index_name: str = "default") -> dict:
    """ARQ task function: ingest documents in the background."""
    kazi = ctx["kazi"]
    job_id: str = ctx.get("job_id", "unknown")
    job_try: int = ctx.get("job_try", 1)
    max_retries: int = ctx.get("max_task_retries", 3)
    try:
        await kazi.ingest(path, index_name=index_name)
    except Exception as exc:
        if job_try >= max_retries:
            await _push_dlq(
                ctx["redis"],
                job_id=job_id,
                function="kazi_ingest",
                message=path,
                error=exc,
                attempt=job_try,
            )
        raise
    return {"status": "ok", "path": path, "index": index_name}


# ── Worker settings builder ───────────────────────────────────────────────────

def build_worker_settings(
    config,
    *,
    redis_url: str = "redis://localhost:6379",
    max_jobs: int = 10,
    job_timeout: int = 300,
    keep_result: int = 3600,
    max_task_retries: int = 3,
):
    """
    Build an ARQ WorkerSettings class for the given KaziConfig.

    Usage::

        WorkerSettings = build_worker_settings(config)

        # Run worker:
        #   python -m arq your_module.WorkerSettings

    Parameters
    ----------
    config              KaziConfig for this worker.
    redis_url           Redis connection string.
    max_jobs            Max concurrent jobs per worker process.
    job_timeout         Seconds before a job is considered timed out.
    keep_result         Seconds to keep job results in Redis.
    max_task_retries    Max attempts before a job is sent to the dead-letter
                        queue (``kazi:dlq``).  ARQ will retry the task up to
                        this many times with its default backoff before giving up.
    """
    try:
        from arq.connections import RedisSettings
    except ImportError:
        raise ImportError(
            "arq is required for the async queue integration. "
            "Install: pip install kazi-core[arq]"
        )

    _config = config
    _max_task_retries = max_task_retries

    async def on_startup(ctx: dict) -> None:
        from kazi.core.orchestrator import Kazi
        logger.info("ARQ worker starting — initialising Kazi...")
        ctx["kazi"] = await Kazi.create(_config)
        ctx["max_task_retries"] = _max_task_retries
        logger.info("ARQ worker ready (max_task_retries=%d)", _max_task_retries)

    async def on_shutdown(ctx: dict) -> None:
        kazi = ctx.get("kazi")
        if kazi:
            await kazi.close()
        logger.info("ARQ worker shut down")

    from urllib.parse import urlparse
    parsed = urlparse(redis_url)
    redis_settings = RedisSettings(
        host=parsed.hostname or "localhost",
        port=parsed.port or 6379,
        password=parsed.password,
        database=int(parsed.path.lstrip("/") or 0),
    )

    class WorkerSettings:
        functions = [_kazi_run, _kazi_ingest]

    # Class body can't close over enclosing function locals — set after creation
    WorkerSettings.on_startup = staticmethod(on_startup)
    WorkerSettings.on_shutdown = staticmethod(on_shutdown)
    WorkerSettings.redis_settings = redis_settings
    WorkerSettings.max_jobs = max_jobs
    WorkerSettings.job_timeout = job_timeout
    WorkerSettings.keep_result_ms = keep_result * 1000
    WorkerSettings.max_tries = max_task_retries

    return WorkerSettings


# ── Client-side queue interface ───────────────────────────────────────────────

@dataclass
class JobResult:
    job_id: str
    reply: str
    cost_usd: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class KaziQueue:
    """
    Client-side interface for enqueueing kazi tasks via ARQ.

    Usage::

        queue = await KaziQueue.connect("redis://localhost:6379")

        job = await queue.enqueue("Summarise the Q3 report", thread_id="user:42")
        result = await queue.get_result(job.job_id, timeout=120)
        print(result.reply)

        # Fire-and-forget (no result polling)
        await queue.enqueue_nowait("Ingest new documents", thread_id="internal")

        await queue.close()

    Or as an async context manager::

        async with await KaziQueue.connect(redis_url) as queue:
            job = await queue.enqueue("Hello", thread_id="user:1")
    """

    def __init__(self, pool) -> None:
        self._pool = pool

    @classmethod
    async def connect(cls, redis_url: str = "redis://localhost:6379") -> KaziQueue:
        try:
            from arq.connections import RedisSettings, create_pool
        except ImportError:
            raise ImportError(
                "arq is required. Install: pip install kazi-core[arq]"
            )
        from urllib.parse import urlparse
        parsed = urlparse(redis_url)
        settings = RedisSettings(
            host=parsed.hostname or "localhost",
            port=parsed.port or 6379,
            password=parsed.password,
            database=int(parsed.path.lstrip("/") or 0),
        )
        pool = await create_pool(settings)
        return cls(pool)

    async def enqueue(
        self,
        message: str,
        *,
        thread_id: str = "default",
        system_prompt: str | None = None,
        max_tool_calls: int = 25,
        track_cost: bool = False,
        defer_by: float | None = None,
        webhook_url: str | None = None,
        webhook_secret: str = "",
    ):
        """
        Enqueue a kazi.run() call and return a job handle.

        defer_by        Seconds to wait before running (scheduled jobs).
        webhook_url     When set, POST the result here after completion.
        webhook_secret  HMAC-SHA256 signing key for the webhook.
        """
        kwargs: dict[str, Any] = dict(
            thread_id=thread_id,
            system_prompt=system_prompt,
            max_tool_calls=max_tool_calls,
            track_cost=track_cost,
            webhook_url=webhook_url,
            webhook_secret=webhook_secret,
        )
        job = await self._pool.enqueue_job(
            "_kazi_run",
            message,
            **kwargs,
            _defer_by=defer_by,
        )
        return job

    async def enqueue_ingest(self, path: str, *, index_name: str = "default"):
        """Enqueue a background document ingestion job."""
        return await self._pool.enqueue_job("_kazi_ingest", path, index_name=index_name)

    async def get_result(self, job_id: str, *, timeout: int = 60) -> JobResult:
        """
        Poll until the job completes and return its result.
        Raises TimeoutError if the job doesn't finish within `timeout` seconds.
        """
        import asyncio

        from arq.jobs import Job, JobStatus
        job = Job(job_id, self._pool)
        elapsed = 0.0
        interval = 0.5
        while elapsed < timeout:
            status = await job.status()
            if status == JobStatus.complete:
                raw = await job.result()
                return JobResult(
                    job_id=job_id,
                    reply=raw.get("reply", ""),
                    cost_usd=raw.get("cost_usd"),
                    input_tokens=raw.get("input_tokens"),
                    output_tokens=raw.get("output_tokens"),
                )
            if status == JobStatus.not_found:
                raise RuntimeError(f"Job {job_id} not found or expired")
            await asyncio.sleep(interval)
            elapsed += interval
        raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")

    async def get_dlq_jobs(self, *, limit: int = 100) -> list[dict]:
        """
        Return up to ``limit`` entries from the dead-letter queue.

        Each entry is a dict with keys:
          job_id, function, message, error, attempt, failed_at

        Use this to inspect failed jobs before deciding whether to requeue or
        discard them.
        """
        raw = await self._pool.lrange(_DLQ_KEY, -limit, -1)
        results = []
        for item in raw:
            try:
                results.append(json.loads(item))
            except json.JSONDecodeError:
                results.append({"raw": item})
        return results

    async def requeue_dlq(self, *, limit: int = 100) -> int:
        """
        Re-enqueue up to ``limit`` jobs from the dead-letter queue.

        Only re-enqueues ``kazi_run`` jobs (ingest jobs must be re-triggered
        manually as their source paths may no longer be valid).

        Returns the number of jobs re-enqueued.
        """
        entries = await self.get_dlq_jobs(limit=limit)
        requeued = 0
        for entry in entries:
            if entry.get("function") != "kazi_run":
                continue
            message = entry.get("message", "")
            if not message:
                continue
            await self._pool.enqueue_job("_kazi_run", message)
            requeued += 1
        if requeued:
            # Trim the requeued entries from the DLQ
            await self._pool.ltrim(_DLQ_KEY, requeued, -1)
            logger.info("Requeued %d job(s) from DLQ", requeued)
        return requeued

    async def clear_dlq(self) -> int:
        """Delete all entries from the dead-letter queue. Returns the count removed."""
        count = await self._pool.llen(_DLQ_KEY)
        await self._pool.delete(_DLQ_KEY)
        logger.info("Cleared %d DLQ entry/entries", count)
        return count

    async def close(self) -> None:
        await self._pool.aclose()

    async def __aenter__(self) -> KaziQueue:
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()
