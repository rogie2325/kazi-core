"""
Celery queue integration for kazi.

Celery is the dominant task queue in Python enterprise stacks.  Because kazi
is fully async, Celery tasks use asyncio.run() to bridge into the async world.
Each worker process creates and caches a single Kazi instance.

Install::

    pip install kazi-core[celery]

Quick start::

    # tasks.py
    from kazi import KaziConfig, LLMConfig, LLMProvider
    from kazi.queue.celery_worker import build_celery_app

    config = KaziConfig(llm=LLMConfig(provider=LLMProvider.ANTHROPIC, model="claude-sonnet-4-6"))
    celery_app = build_celery_app(
        config,
        broker="redis://localhost:6379",
        max_retries=3,   # retry 3× with exponential backoff before DLQ
    )

    # Run worker:
    #   celery -A tasks worker --loglevel=info

    # Enqueue from your app:
    from tasks import celery_app
    job = celery_app.send_task("kazi.run", args=["Analyse expenses"], kwargs={"thread_id": "user:1"})
    result = job.get(timeout=120)
    print(result["reply"])

Retry policy
------------
Failed tasks are retried up to ``max_retries`` times with exponential backoff
(2^attempt seconds + jitter).  After exhausting retries, the task is routed to
the ``kazi-dlq`` queue (dead-letter) and raises the original exception.

To process the dead-letter queue::

    # Run a second worker that consumes the DLQ
    celery -A tasks worker --queues=kazi-dlq --loglevel=warning
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import random
import threading

logger = logging.getLogger(__name__)

# A single event loop that lives for the lifetime of the worker process.
# All async work (Kazi creation + every task call) runs on this loop so that
# httpx/aiohttp connection pools remain valid across calls — asyncio.run()
# opens *and closes* a loop each time, invalidating any clients created in a
# prior call.
_worker_loop: asyncio.AbstractEventLoop | None = None
_worker_loop_lock = threading.Lock()

# Module-level Kazi instance — shared across tasks in the same worker process.
_kazi_instance = None
_kazi_lock = threading.Lock()
_kazi_config = None


def _get_worker_loop() -> asyncio.AbstractEventLoop:
    global _worker_loop
    if _worker_loop is not None and not _worker_loop.is_closed():
        return _worker_loop
    with _worker_loop_lock:
        if _worker_loop is None or _worker_loop.is_closed():
            _worker_loop = asyncio.new_event_loop()
            t = threading.Thread(target=_worker_loop.run_forever, daemon=True)
            t.start()
    return _worker_loop


def _run_async(coro, timeout: int = 300):
    """Submit a coroutine to the persistent worker loop and block until done."""
    future = asyncio.run_coroutine_threadsafe(coro, _get_worker_loop())
    return future.result(timeout=timeout)


def _get_kazi():
    global _kazi_instance
    if _kazi_instance is not None:
        return _kazi_instance
    with _kazi_lock:
        if _kazi_instance is None:
            if _kazi_config is None:
                raise RuntimeError(
                    "KaziConfig not set. Call build_celery_app(config, ...) before using tasks."
                )
            _kazi_instance = _run_async(_create_kazi(_kazi_config))
    return _kazi_instance


async def _create_kazi(config):
    from kazi.core.orchestrator import Kazi
    return await Kazi.create(config)


def build_celery_app(
    config,
    *,
    broker: str = "redis://localhost:6379",
    backend: str | None = None,
    task_time_limit: int = 300,
    task_soft_time_limit: int = 270,
    max_retries: int = 3,
    dlq_queue: str = "kazi-dlq",
):
    """
    Build and return a Celery app with kazi tasks registered.

    config                KaziConfig for this worker.
    broker                Celery broker URL (Redis or RabbitMQ).
    backend               Result backend URL. Defaults to the broker URL.
    task_time_limit       Hard task timeout in seconds.
    task_soft_time_limit  Soft timeout — raises SoftTimeLimitExceeded before hard kill.
    max_retries           Max automatic retries with exponential backoff before
                          routing the task to ``dlq_queue``.
    dlq_queue             Celery queue name for dead-lettered tasks.
                          Run a dedicated worker for this queue to reprocess manually.

    Registered tasks:
      kazi.run         — run a conversation turn
      kazi.ingest      — ingest documents in the background
      kazi.dead_letter — sink task routed to dlq_queue for failed jobs
    """
    try:
        from celery import Celery
    except ImportError:
        raise ImportError(
            "celery is required. Install: pip install kazi-core[celery]"
        )

    global _kazi_config
    _kazi_config = config
    _max_retries = max_retries

    app = Celery(
        "kazi",
        broker=broker,
        backend=backend or broker,
    )
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        task_time_limit=task_time_limit,
        task_soft_time_limit=task_soft_time_limit,
        worker_prefetch_multiplier=1,   # one task at a time per worker thread
        # Route dead-lettered tasks to a dedicated queue
        task_routes={
            "kazi.dead_letter": {"queue": dlq_queue},
        },
    )

    @app.task(name="kazi.run", bind=True, max_retries=_max_retries)
    def kazi_run(
        self,
        message: str,
        *,
        thread_id: str = "default",
        system_prompt: str | None = None,
        max_tool_calls: int = 25,
        track_cost: bool = False,
    ) -> dict:
        """
        Run a kazi agent turn and return the reply as a JSON-serialisable dict.

        Retries automatically on failure with exponential backoff.
        After ``max_retries`` attempts, routes to the dead-letter queue.
        """
        kazi = _get_kazi()
        from kazi.core.cost import RunResult

        try:
            result = _run_async(
                kazi.run(
                    message,
                    thread_id=thread_id,
                    system_prompt=system_prompt,
                    max_tool_calls=max_tool_calls,
                    track_cost=track_cost,
                )
            )
        except Exception as exc:
            attempt = self.request.retries
            if attempt < _max_retries:
                backoff = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(
                    "kazi.run attempt %d/%d failed — retrying in %.1fs: %s",
                    attempt + 1, _max_retries, backoff, exc,
                )
                raise self.retry(exc=exc, countdown=backoff)
            # All retries exhausted — route to dead-letter queue
            logger.error(
                "kazi.run exhausted %d retries — routing to DLQ: %s",
                _max_retries, exc,
            )
            app.send_task(
                "kazi.dead_letter",
                kwargs={
                    "task_name": "kazi.run",
                    "message": message,
                    "error": str(exc),
                    "attempts": attempt + 1,
                },
            )
            raise

        if isinstance(result, RunResult):
            return {
                "reply": result.reply,
                "cost_usd": result.cost.cost_usd,
                "input_tokens": result.cost.input_tokens,
                "output_tokens": result.cost.output_tokens,
            }
        return {"reply": result}

    @app.task(name="kazi.ingest", bind=True, max_retries=_max_retries)
    def kazi_ingest(self, path: str, *, index_name: str = "default") -> dict:
        """Ingest documents into the vector store in the background."""
        kazi = _get_kazi()
        try:
            _run_async(kazi.ingest(path, index_name=index_name))
        except Exception as exc:
            attempt = self.request.retries
            if attempt < _max_retries:
                backoff = (2 ** attempt) + random.uniform(0, 1)
                raise self.retry(exc=exc, countdown=backoff)
            app.send_task(
                "kazi.dead_letter",
                kwargs={
                    "task_name": "kazi.ingest",
                    "message": path,
                    "error": str(exc),
                    "attempts": attempt + 1,
                },
            )
            raise
        return {"status": "ok", "path": path, "index": index_name}

    @app.task(name="kazi.dead_letter", queue=dlq_queue)
    def kazi_dead_letter(
        *,
        task_name: str,
        message: str,
        error: str,
        attempts: int,
    ) -> dict:
        """
        Dead-letter sink — receives tasks that exhausted their retries.

        This task is routed to the ``{dlq_queue}`` Celery queue.
        Run a dedicated worker to inspect, alert on, or reprocess these::

            celery -A tasks worker --queues=kazi-dlq --loglevel=warning

        To re-run a dead-lettered task, call the original task directly::

            celery_app.send_task("kazi.run", args=[message], kwargs={...})
        """
        import time
        logger.error(
            "DLQ: task=%s attempts=%d error=%s message_preview=%.200s",
            task_name, attempts, error, message,
        )
        return {
            "status": "dead_lettered",
            "task_name": task_name,
            "error": error,
            "attempts": attempts,
            "received_at": time.time(),
        }

    return app
