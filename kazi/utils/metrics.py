"""
Optional Prometheus metrics for kazi.

Install: pip install prometheus-client

Metrics exposed:
  kazi_requests_total{route, status}        — HTTP request count
  kazi_request_duration_seconds{route}      — HTTP request latency histogram
  kazi_active_requests                      — In-flight request gauge
  kazi_tokens_total{model, token_type}      — Token usage (input | output)
  kazi_cost_usd_total{model, tenant_id}     — Estimated cost in USD
  kazi_cache_hits_total{cache_type}         — Cache hits (semantic | tool)
  kazi_cache_misses_total{cache_type}       — Cache misses
  kazi_tool_calls_total{tool_name, status}  — Tool execution count (ok | error)

All functions are no-ops when prometheus-client is not installed.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Generator
from contextlib import contextmanager

logger = logging.getLogger(__name__)

_METRICS: dict = {}
_metrics_ready = False


def _ensure_metrics() -> bool:
    global _metrics_ready, _METRICS
    if _metrics_ready:
        return True
    try:
        from prometheus_client import Counter, Gauge, Histogram

        def _safe(cls, name: str, doc: str, labels: list[str] | None = None):
            """Create a metric, returning the existing one if already registered."""
            try:
                return cls(name, doc, labels or []) if labels else cls(name, doc)
            except ValueError:
                # Already registered (e.g. during hot-reload or test reruns)
                from prometheus_client import REGISTRY
                return REGISTRY._names_to_collectors.get(name)

        _METRICS["requests_total"] = _safe(
            Counter, "kazi_requests_total", "Total HTTP requests", ["route", "status"]
        )
        _METRICS["request_duration"] = _safe(
            Histogram,
            "kazi_request_duration_seconds",
            "HTTP request latency in seconds",
            ["route"],
        )
        _METRICS["active_requests"] = _safe(
            Gauge, "kazi_active_requests", "In-flight requests"
        )
        _METRICS["tokens_total"] = _safe(
            Counter, "kazi_tokens_total", "Total tokens processed", ["model", "token_type"]
        )
        _METRICS["cost_usd_total"] = _safe(
            Counter, "kazi_cost_usd_total", "Estimated cost USD", ["model", "tenant_id"]
        )
        _METRICS["cache_hits"] = _safe(
            Counter, "kazi_cache_hits_total", "Cache hits", ["cache_type"]
        )
        _METRICS["cache_misses"] = _safe(
            Counter, "kazi_cache_misses_total", "Cache misses", ["cache_type"]
        )
        _METRICS["tool_calls"] = _safe(
            Counter, "kazi_tool_calls_total", "Tool executions", ["tool_name", "status"]
        )
        _metrics_ready = True
        return True
    except ImportError:
        return False


@contextmanager
def track_request(route: str) -> Generator[None, None, None]:
    """
    Context manager that increments active_requests and records duration + status.

    Usage (inside a FastAPI middleware)::

        async with track_request("POST /run"):
            response = await call_next(request)
    """
    if not _ensure_metrics():
        yield
        return

    _METRICS["active_requests"].inc()
    start = time.perf_counter()
    status = "200"
    try:
        yield
    except Exception:
        status = "500"
        raise
    finally:
        duration = time.perf_counter() - start
        _METRICS["active_requests"].dec()
        _METRICS["requests_total"].labels(route=route, status=status).inc()
        _METRICS["request_duration"].labels(route=route).observe(duration)


def record_request(route: str, status: int, duration_seconds: float) -> None:
    """Record a completed request with explicit status and duration."""
    if not _ensure_metrics():
        return
    _METRICS["requests_total"].labels(route=route, status=str(status)).inc()
    _METRICS["request_duration"].labels(route=route).observe(duration_seconds)


def record_tokens(model: str, input_tokens: int, output_tokens: int) -> None:
    """Record token usage from an LLM response."""
    if not _ensure_metrics():
        return
    if input_tokens:
        _METRICS["tokens_total"].labels(model=model, token_type="input").inc(input_tokens)
    if output_tokens:
        _METRICS["tokens_total"].labels(model=model, token_type="output").inc(output_tokens)


def record_cost(model: str, cost_usd: float, tenant_id: str = "") -> None:
    """Record estimated cost for a run, labelled by model and tenant."""
    if not _ensure_metrics():
        return
    _METRICS["cost_usd_total"].labels(model=model, tenant_id=tenant_id).inc(cost_usd)


def record_cache_hit(cache_type: str = "semantic") -> None:
    if not _ensure_metrics():
        return
    _METRICS["cache_hits"].labels(cache_type=cache_type).inc()


def record_cache_miss(cache_type: str = "semantic") -> None:
    if not _ensure_metrics():
        return
    _METRICS["cache_misses"].labels(cache_type=cache_type).inc()


def record_tool_call(tool_name: str, *, success: bool = True) -> None:
    if not _ensure_metrics():
        return
    _METRICS["tool_calls"].labels(
        tool_name=tool_name, status="ok" if success else "error"
    ).inc()


def prometheus_output() -> tuple[bytes, str]:
    """
    Return ``(body_bytes, content_type)`` suitable for a /prometheus scrape endpoint.

    Returns a plain-text fallback when prometheus-client is not installed.
    """
    try:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
        _ensure_metrics()
        return generate_latest(), CONTENT_TYPE_LATEST
    except ImportError:
        return (
            b"# prometheus-client not installed\n"
            b"# pip install kazi-core[prometheus]\n",
            "text/plain; version=0.0.4",
        )
