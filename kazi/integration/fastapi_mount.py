"""
Mount kazi HTTP routes onto a client's existing FastAPI app.

Use case
========
Your client already runs a FastAPI service.  They don't want a second
process / port / load-balancer config just for the AI layer.  ``mount_to``
adds kazi's routes to their existing app::

    # client_app/main.py — their existing code
    from fastapi import FastAPI
    app = FastAPI()

    @app.get("/their/existing/route")
    def existing_endpoint(): ...

    # client_app/ai_layer.py — what you add
    from kazi import Kazi, KaziConfig
    from kazi.integration import mount_to

    async def install_ai_layer():
        kazi = await Kazi.create(KaziConfig())
        mount_to(app, kazi, prefix="/ai", api_key="...")

After install, the client's app has:
  POST /ai/run        — single-turn chat
  POST /ai/stream     — token stream
  POST /ai/events     — typed event stream
  POST /ai/ingest     — document ingestion
  GET  /ai/health     — health probe
  GET  /ai/metrics    — usage metrics
  GET  /ai/prometheus — Prometheus scrape

…alongside every route the host app already owned.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kazi.core.orchestrator import Kazi

logger = logging.getLogger(__name__)


def mount_to(
    existing_app: Any,
    kazi: Kazi,
    *,
    prefix: str = "/ai",
    api_key: str | None = None,
    cors_origins: list[str] | None = None,
    rate_limit_per_minute: int = 0,
    rate_limit_redis_url: str | None = None,
    max_body_bytes: int = 1 * 1024 * 1024,
    max_concurrent_runs: int = 50,
    request_timeout_seconds: int = 120,
    enable_audit_log: bool = True,
) -> Any:
    """
    Attach kazi routes onto an existing FastAPI app at ``prefix``.

    Parameters mirror ``kazi.as_app()`` — the same hardening (rate limit,
    body size, concurrency cap, audit log) is applied to the mounted routes.

    Returns the existing app for chaining.

    Notes
    -----
    - The host app's lifespan is NOT replaced.  ``Kazi.close()`` is
      the caller's responsibility — wire it into the host's shutdown hook.
    - CORS / security headers added by the host app are preserved.
    - If the host app already has a route at ``{prefix}/run`` (etc.), FastAPI
      will raise a RouteAlreadyExistsError — change ``prefix`` to avoid.
    """
    try:
        from fastapi import FastAPI
    except ImportError:
        raise ImportError(
            "fastapi is required. Install: pip install kazi-core[serve]"
        )

    if not isinstance(existing_app, FastAPI):
        raise TypeError(
            f"mount_to() expected a FastAPI instance, got {type(existing_app).__name__}.  "
            "If you're using Starlette or another ASGI app, use kazi.as_app() and "
            "mount it as a sub-app instead."
        )

    # Build a standalone kazi app at no prefix, then include its routes
    # under ``prefix`` into the host app.  This way the host's middleware /
    # lifespan stay in charge while kazi contributes routes.
    kazi_app = kazi.as_app(
        prefix="",
        api_key=api_key,
        cors_origins=cors_origins,
        rate_limit_per_minute=rate_limit_per_minute,
        max_body_bytes=max_body_bytes,
        enable_audit_log=enable_audit_log,
        max_concurrent_runs=max_concurrent_runs,
        request_timeout_seconds=request_timeout_seconds,
    )

    # FastAPI's include_router can splice in routes — wrap kazi_app.router
    existing_app.include_router(kazi_app.router, prefix=prefix.rstrip("/"))

    logger.info(
        "Mounted kazi routes at prefix=%s (rate_limit=%d/min, concurrency=%d)",
        prefix, rate_limit_per_minute, max_concurrent_runs,
    )
    return existing_app
