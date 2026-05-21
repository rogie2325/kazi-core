"""
Example 9: Background job queues (ARQ + Celery)

Off-load long-running agent tasks to a background worker.
Use this when your API needs to return immediately and the agent
result is delivered asynchronously (webhook, polling, email).

── ARQ (async-native, recommended) ──────────────────────────────────────────

Install::

    pip install kazi-core[arq]
    pip install redis

Start worker::

    python examples/09_background_jobs.py worker

Enqueue from your API::

    python examples/09_background_jobs.py enqueue

── Celery (enterprise, Django/Flask compatible) ──────────────────────────────

Install::

    pip install kazi-core[celery]

Start worker::

    celery -A examples.09_background_jobs.celery_app worker --loglevel=info

Enqueue::

    python examples/09_background_jobs.py celery-enqueue
"""
import asyncio
import sys

from kazi import KaziConfig, LLMConfig, LLMProvider, MemoryConfig, MemoryBackend

REDIS_URL = "redis://localhost:6379"


# ── Shared config ──────────────────────────────────────────────────────────────

def make_config() -> KaziConfig:
    return KaziConfig(
        llm=LLMConfig(provider=LLMProvider.ANTHROPIC, model="claude-sonnet-4-6"),
        memory=MemoryConfig(
            backend=MemoryBackend.REDIS,
            connection_string=REDIS_URL,
        ),
    )


# ── ARQ worker ─────────────────────────────────────────────────────────────────

from kazi.queue.arq_worker import KaziQueue, build_worker_settings

config = make_config()
WorkerSettings = build_worker_settings(config, redis_url=REDIS_URL)


async def arq_enqueue_example():
    """Enqueue a job and wait for the result."""
    queue = KaziQueue(redis_url=REDIS_URL)

    job_id = await queue.enqueue(
        "Analyse the current state of AI agent frameworks and list the top 5",
        thread_id="research-session-1",
    )
    print(f"Job enqueued: {job_id}")

    # Poll for result (in production, use a webhook instead)
    result = await queue.get_result(job_id, timeout=120)
    print(f"Result: {result['reply'][:300]}…")

    await queue.close()


# ── Celery app ─────────────────────────────────────────────────────────────────

from kazi.queue.celery_worker import build_celery_app

celery_app = build_celery_app(make_config(), broker=REDIS_URL)


def celery_enqueue_example():
    """Enqueue a Celery task and block for the result."""
    job = celery_app.send_task(
        "kazi.run",
        args=["Write a haiku about distributed systems"],
        kwargs={"thread_id": "haiku-session", "track_cost": False},
    )
    result = job.get(timeout=120)
    print(f"Celery result: {result['reply']}")


# ── Background document ingestion ─────────────────────────────────────────────

async def background_ingest_example():
    """Ingest documents in the background while the API stays responsive."""
    queue = KaziQueue(redis_url=REDIS_URL)

    job_id = await queue.enqueue_ingest(
        "./company_docs",
        index_name="company",
    )
    print(f"Ingest job enqueued: {job_id}")
    print("Documents are being indexed in the background — your API is unblocked.")

    await queue.close()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"

    if cmd == "worker":
        import arq
        arq.run_worker(WorkerSettings)

    elif cmd == "enqueue":
        asyncio.run(arq_enqueue_example())

    elif cmd == "ingest":
        asyncio.run(background_ingest_example())

    elif cmd == "celery-enqueue":
        celery_enqueue_example()

    else:
        print(__doc__)
