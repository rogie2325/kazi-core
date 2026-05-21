"""
Optional OpenTelemetry tracing for Kazi operations.

Tracing is a no-op when telemetry_enabled=False or when
opentelemetry packages are not installed.

Quick start — export traces to any OTLP collector::

    from kazi.utils.telemetry import configure_telemetry

    # Jaeger, Grafana Tempo, Honeycomb, Datadog OTLP endpoint, etc.
    configure_telemetry(
        endpoint="http://localhost:4317",   # gRPC OTLP
        service_name="my-kazi-app",
    )

    # Or via environment variables (the OTel SDK reads these automatically):
    #   OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
    #   OTEL_SERVICE_NAME=my-kazi-app
"""
from __future__ import annotations

import contextlib
import functools
import logging
from collections.abc import Generator

logger = logging.getLogger(__name__)

_tracer = None


def configure_telemetry(
    endpoint: str,
    *,
    service_name: str = "kazi",
    headers: dict[str, str] | None = None,
    insecure: bool = False,
    use_http: bool = False,
) -> bool:
    """
    Configure and register an OTLP trace exporter.

    Call this once at application startup, before creating your Kazi instance.
    Spans created by ``span()`` and ``instrument_tool_call()`` will be exported
    to the given endpoint.

    Parameters
    ----------
    endpoint     OTLP collector endpoint.
                 gRPC (default): ``http://localhost:4317``
                 HTTP/protobuf:  ``http://localhost:4318``
    service_name Resource attribute ``service.name`` attached to every span.
    headers      Extra headers sent with every export request
                 (e.g. ``{"x-honeycomb-team": "my-api-key"}``).
    insecure     Skip TLS verification (dev/local only).
    use_http     Use OTLP/HTTP instead of OTLP/gRPC.

    Returns True on success, False when opentelemetry packages are missing.
    """
    global _tracer
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.warning(
            "opentelemetry-sdk not installed — tracing disabled. "
            "Install: pip install kazi-core[telemetry]"
        )
        return False

    try:
        if use_http:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            exporter = OTLPSpanExporter(
                endpoint=endpoint.rstrip("/") + "/v1/traces",
                headers=headers or {},
            )
        else:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            exporter = OTLPSpanExporter(
                endpoint=endpoint,
                headers=headers or {},
                insecure=insecure,
            )
    except ImportError as exc:
        logger.warning(
            "OTLP exporter not installed — tracing disabled: %s. "
            "Install: pip install opentelemetry-exporter-otlp",
            exc,
        )
        return False

    resource = Resource.create({SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    _tracer = trace.get_tracer("kazi")
    logger.info(
        "OTel tracing configured — service=%s endpoint=%s transport=%s",
        service_name, endpoint, "http" if use_http else "grpc",
    )
    return True


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


def current_trace_id() -> str | None:
    """
    Return the current OTel trace ID as a hex string, or None.

    Useful for injecting correlation IDs into structured logs so a log line
    can be linked back to the trace in Jaeger / Tempo / Honeycomb::

        logger.info("Processing request trace_id=%s", current_trace_id())
    """
    try:
        from opentelemetry import trace
        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx and ctx.is_valid:
            return format(ctx.trace_id, "032x")
    except Exception:
        pass
    return None


@contextlib.contextmanager
def span(name: str, attributes: dict | None = None) -> Generator:
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
