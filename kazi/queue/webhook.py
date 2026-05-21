"""
Webhook callbacks for kazi background jobs.

When a job completes (via ARQ or Celery), kazi can POST the result to any
HTTPS endpoint.  Payloads are signed with HMAC-SHA256 so the receiver can
verify the request actually came from kazi.

Security
--------
Every request includes an ``X-Kazi-Signature`` header:

    X-Kazi-Signature: sha256=<hex_digest>

The digest is computed over the raw JSON body using the ``secret`` you
configure.  Verify it on the receiving end::

    import hashlib, hmac
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    assert hmac.compare_digest(expected, received_sig)

Usage
-----
Pass a ``WebhookConfig`` to ``build_worker_settings`` or to the Celery app
builder.  Results are dispatched automatically after every completed job.

    from kazi.queue.webhook import WebhookConfig

    config = WebhookConfig(
        url="https://yourapp.com/webhooks/kazi",
        secret="my-signing-secret",
        retry_attempts=3,
        timeout_seconds=10,
    )
    WorkerSettings = build_worker_settings(kazi_config, webhook=config, ...)
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class WebhookConfig:
    """
    Configuration for job-completion webhook notifications.

    url              Target HTTPS endpoint.  Must be reachable from the worker.
    secret           HMAC-SHA256 signing key.  Empty string = no signature header.
    retry_attempts   How many times to retry a failed POST (non-2xx or network error).
    timeout_seconds  Per-attempt HTTP timeout.
    headers          Extra headers to include in every request (e.g. Authorization).
    include_reply    When True, the LLM reply text is sent in the payload.
                     Set False when the reply may be large or sensitive.
    """
    url: str
    secret: str = ""
    retry_attempts: int = 3
    timeout_seconds: int = 10
    headers: dict[str, str] = field(default_factory=dict)
    include_reply: bool = True


def _sign(body: bytes, secret: str) -> str:
    """Return the hex HMAC-SHA256 digest of ``body`` using ``secret``."""
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


async def dispatch_webhook(
    config: WebhookConfig,
    job_id: str,
    result: dict[str, Any],
    *,
    event: str = "job.complete",
) -> bool:
    """
    POST a signed JSON payload to ``config.url``.

    Returns True on success, False if all retry attempts fail.
    Never raises — failures are logged only.
    """
    try:
        import aiohttp
    except ImportError:
        logger.warning(
            "aiohttp is required for webhooks: pip install aiohttp.  "
            "Webhook for job %s not dispatched.", job_id,
        )
        return False

    payload: dict[str, Any] = {
        "event": event,
        "job_id": job_id,
    }
    if config.include_reply:
        payload["result"] = result
    else:
        # Send cost metadata but omit the reply text
        payload["result"] = {
            k: v for k, v in result.items() if k != "reply"
        }

    body = json.dumps(payload, separators=(",", ":")).encode()

    headers: dict[str, str] = {
        "Content-Type": "application/json",
        **config.headers,
    }
    if config.secret:
        headers["X-Kazi-Signature"] = f"sha256={_sign(body, config.secret)}"

    timeout = aiohttp.ClientTimeout(total=config.timeout_seconds)

    for attempt in range(max(1, config.retry_attempts)):
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    config.url, data=body, headers=headers, ssl=True
                ) as resp:
                    if resp.status < 300:
                        logger.info(
                            "Webhook dispatched: job=%s event=%s status=%d",
                            job_id, event, resp.status,
                        )
                        return True
                    logger.warning(
                        "Webhook attempt %d/%d returned HTTP %d for job %s",
                        attempt + 1, config.retry_attempts, resp.status, job_id,
                    )
        except Exception as exc:
            logger.warning(
                "Webhook attempt %d/%d failed for job %s: %s",
                attempt + 1, config.retry_attempts, job_id, exc,
            )

    logger.error("Webhook failed after %d attempts for job %s", config.retry_attempts, job_id)
    return False
