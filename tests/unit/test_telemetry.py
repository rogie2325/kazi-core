"""
Unit tests for kazi.utils.telemetry.

Covers configure_telemetry (gRPC and HTTP paths), current_trace_id,
span(), and instrument_tool_call().  The OTLP exporters are installed
in the dev environment; no live collector is required for the tests
since we only exercise span creation, not export.
"""
from __future__ import annotations

import pytest


def _reset_tracer():
    """Reset the module-level _tracer so each test starts fresh."""
    import kazi.utils.telemetry as t
    t._tracer = None


# ── configure_telemetry — gRPC path ──────────────────────────────────────────

def test_configure_telemetry_grpc_returns_true():
    _reset_tracer()
    from kazi.utils.telemetry import configure_telemetry
    result = configure_telemetry("http://localhost:4317", service_name="test-svc")
    assert result is True


def test_configure_telemetry_grpc_sets_tracer():
    _reset_tracer()
    import kazi.utils.telemetry as t
    t.configure_telemetry("http://localhost:4317", service_name="test-grpc")
    assert t._tracer is not None


def test_configure_telemetry_with_headers():
    _reset_tracer()
    from kazi.utils.telemetry import configure_telemetry
    result = configure_telemetry(
        "http://localhost:4317",
        service_name="test-headers",
        headers={"x-token": "secret"},
        insecure=True,
    )
    assert result is True


# ── configure_telemetry — HTTP path ──────────────────────────────────────────

def test_configure_telemetry_http_returns_true():
    _reset_tracer()
    from kazi.utils.telemetry import configure_telemetry
    result = configure_telemetry(
        "http://localhost:4318",
        service_name="test-http",
        use_http=True,
    )
    assert result is True


def test_configure_telemetry_http_sets_tracer():
    _reset_tracer()
    import kazi.utils.telemetry as t
    t.configure_telemetry("http://localhost:4318", service_name="test-http2", use_http=True)
    assert t._tracer is not None


# ── current_trace_id ─────────────────────────────────────────────────────────

def test_current_trace_id_returns_none_outside_span():
    _reset_tracer()
    from kazi.utils.telemetry import current_trace_id
    assert current_trace_id() is None


def test_current_trace_id_returns_hex_inside_span():
    _reset_tracer()
    from kazi.utils.telemetry import configure_telemetry, current_trace_id, span
    configure_telemetry("http://localhost:4317", service_name="trace-id-test")
    with span("test-span"):
        tid = current_trace_id()
    assert tid is not None
    assert len(tid) == 32
    int(tid, 16)  # must be valid hex


# ── span() context manager ───────────────────────────────────────────────────

def test_span_with_real_tracer_yields_span():
    _reset_tracer()
    from kazi.utils.telemetry import configure_telemetry, span
    configure_telemetry("http://localhost:4317", service_name="span-test")
    result = []
    with span("my-span", {"key": "val"}) as s:
        result.append(s)
    assert len(result) == 1


def test_span_noop_when_tracer_not_configured():
    _reset_tracer()
    import kazi.utils.telemetry as t
    t._tracer = t._NoOpTracer()
    from kazi.utils.telemetry import span
    ran = []
    with span("noop-span"):
        ran.append(True)
    assert ran


# ── instrument_tool_call decorator ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_instrument_tool_call_wraps_async_function():
    _reset_tracer()
    from kazi.utils.telemetry import configure_telemetry, instrument_tool_call
    configure_telemetry("http://localhost:4317", service_name="tool-test")

    @instrument_tool_call
    async def my_tool(x: int) -> int:
        return x * 2

    result = await my_tool(x=5)
    assert result == 10


@pytest.mark.asyncio
async def test_instrument_tool_call_without_tracer():
    _reset_tracer()
    from kazi.utils.telemetry import instrument_tool_call

    @instrument_tool_call
    async def plain_tool() -> str:
        return "done"

    assert await plain_tool() == "done"


# ── configure_telemetry — OTLP exporter missing ───────────────────────────────

def test_configure_telemetry_returns_false_when_exporter_missing(monkeypatch):
    """Simulates the OTLP exporter packages not being installed."""
    import sys
    _reset_tracer()

    # Remove the grpc exporter from sys.modules so import fails
    saved = {}
    to_hide = [k for k in list(sys.modules) if "otlp.proto.grpc" in k]
    for key in to_hide:
        saved[key] = sys.modules.pop(key)
    monkeypatch.setitem(sys.modules, "opentelemetry.exporter.otlp.proto.grpc.trace_exporter",
                        None)  # type: ignore

    from kazi.utils.telemetry import configure_telemetry
    result = configure_telemetry("http://localhost:4317", service_name="no-exporter")

    # Restore
    for key, val in saved.items():
        sys.modules[key] = val

    assert result is False


# ── current_trace_id — exception path ────────────────────────────────────────

def test_current_trace_id_returns_none_on_otel_error(monkeypatch):
    """When OTel itself raises, current_trace_id() returns None gracefully."""
    import kazi.utils.telemetry as t
    _reset_tracer()

    def _bad_import(name, *args, **kwargs):
        if "opentelemetry" in name:
            raise RuntimeError("OTel broken")
        return orig_import(name, *args, **kwargs)

    import builtins
    orig_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", _bad_import)
    result = t.current_trace_id()
    assert result is None
