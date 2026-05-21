"""
Optional OpenTelemetry tracing for Kazi operations.

Tracing is a no-op when telemetry_enabled=False or when
opentelemetry packages are not installed.
"""
from __future__ import annotations

import contextlib
import functools
from typing import Any, Generator, Optional


_tracer = None


def _get_tracer():
    global _tracer
    if _tracer is not None:
        return _tracer
    try:
        from opentelemetry import trace
        _tracer = trace.get_tracer("kazi")
    except ImportError:
        _tracer = _NoOpTracer()
    return _tracer


@contextlib.contextmanager
def span(name: str, attributes: Optional[dict] = None) -> Generator:
    """Context manager that creates an OTel span (no-op if telemetry disabled)."""
    tracer = _get_tracer()
    if isinstance(tracer, _NoOpTracer):
        yield
        return
    with tracer.start_as_current_span(name) as s:
        if attributes:
            for k, v in attributes.items():
                s.set_attribute(k, str(v))
        yield s


def instrument_tool_call(func):
    """Decorator that wraps an async tool handler with an OTel span."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        with span(f"tool.{func.__name__}", {"tool.args": str(kwargs)}):
            return await func(*args, **kwargs)
    return wrapper


class _NoOpTracer:
    """Stub tracer used when opentelemetry is not installed."""
    @contextlib.contextmanager
    def start_as_current_span(self, *args, **kwargs):
        yield self

    def set_attribute(self, *args, **kwargs):
        pass
