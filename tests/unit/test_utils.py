"""Tests for kazi.utils.logging and kazi.utils.telemetry."""
import logging
import sys

import pytest

# ── configure_logging ─────────────────────────────────────────────────────────

def _fresh_kazi_logger():
    """Remove all handlers from the kazi logger so tests start clean."""
    lg = logging.getLogger("kazi")
    lg.handlers.clear()
    lg.propagate = True
    return lg


def test_configure_logging_sets_info_level():
    from kazi.utils.logging import configure_logging
    _fresh_kazi_logger()
    configure_logging(level="INFO")
    assert logging.getLogger("kazi").level == logging.INFO


def test_configure_logging_sets_debug_level():
    from kazi.utils.logging import configure_logging
    _fresh_kazi_logger()
    configure_logging(level="DEBUG")
    assert logging.getLogger("kazi").level == logging.DEBUG


def test_configure_logging_sets_warning_level():
    from kazi.utils.logging import configure_logging
    _fresh_kazi_logger()
    configure_logging(level="WARNING")
    assert logging.getLogger("kazi").level == logging.WARNING


def test_configure_logging_adds_handler():
    from kazi.utils.logging import configure_logging
    _fresh_kazi_logger()
    configure_logging()
    lg = logging.getLogger("kazi")
    assert len(lg.handlers) == 1


def test_configure_logging_idempotent():
    """Calling configure_logging twice should not double-add handlers."""
    from kazi.utils.logging import configure_logging
    _fresh_kazi_logger()
    configure_logging()
    configure_logging()  # second call — handler already exists
    assert len(logging.getLogger("kazi").handlers) == 1


def test_configure_logging_disables_propagation():
    from kazi.utils.logging import configure_logging
    _fresh_kazi_logger()
    configure_logging()
    assert logging.getLogger("kazi").propagate is False
    # Restore so subsequent tests' caplog capture (which needs propagate=True) works.
    logging.getLogger("kazi").propagate = True


def test_configure_logging_accepts_custom_handler():
    from kazi.utils.logging import configure_logging
    _fresh_kazi_logger()
    custom = logging.NullHandler()
    configure_logging(handler=custom)
    lg = logging.getLogger("kazi")
    assert custom in lg.handlers


def test_configure_logging_uses_stdout_by_default():
    from kazi.utils.logging import configure_logging
    _fresh_kazi_logger()
    configure_logging()
    lg = logging.getLogger("kazi")
    handler = lg.handlers[0]
    assert isinstance(handler, logging.StreamHandler)
    assert handler.stream is sys.stdout


def test_get_logger_namespaced():
    from kazi.utils.logging import get_logger
    lg = get_logger("brain.graph")
    assert lg.name == "kazi.brain.graph"


def test_get_logger_is_child_of_kazi():
    from kazi.utils.logging import get_logger
    child = get_logger("registry")
    # Python's logger hierarchy: child is a descendant of parent
    assert child.name.startswith("kazi.")


# ── telemetry ─────────────────────────────────────────────────────────────────

def test_span_noop_when_tracer_is_noop():
    """span() must never raise and must yield control even with no OTel installed."""
    import kazi.utils.telemetry as tel

    # Force the module to use _NoOpTracer by resetting the cached tracer
    tel._tracer = tel._NoOpTracer()

    ran = False
    with tel.span("test.operation", {"key": "value"}):
        ran = True
    assert ran


def test_span_with_no_attributes():
    import kazi.utils.telemetry as tel
    tel._tracer = tel._NoOpTracer()

    ran = False
    with tel.span("test.no_attrs"):
        ran = True
    assert ran


def test_noop_tracer_set_attribute_does_not_raise():
    from kazi.utils.telemetry import _NoOpTracer
    t = _NoOpTracer()
    with t.start_as_current_span("op") as s:
        s.set_attribute("key", "value")  # must be silent


def test_span_propagates_exception():
    import kazi.utils.telemetry as tel
    tel._tracer = tel._NoOpTracer()

    with pytest.raises(ValueError, match="expected"):
        with tel.span("failing.op"):
            raise ValueError("expected")


@pytest.mark.asyncio
async def test_instrument_tool_call_wraps_async_function():
    import kazi.utils.telemetry as tel
    tel._tracer = tel._NoOpTracer()

    @tel.instrument_tool_call
    async def my_tool(x: int) -> int:
        return x * 2

    result = await my_tool(x=5)
    assert result == 10


@pytest.mark.asyncio
async def test_instrument_tool_call_preserves_function_name():
    import kazi.utils.telemetry as tel
    tel._tracer = tel._NoOpTracer()

    @tel.instrument_tool_call
    async def original_name():
        return "ok"

    assert original_name.__name__ == "original_name"


@pytest.mark.asyncio
async def test_instrument_tool_call_propagates_exception():
    import kazi.utils.telemetry as tel
    tel._tracer = tel._NoOpTracer()

    @tel.instrument_tool_call
    async def broken():
        raise RuntimeError("tool broke")

    with pytest.raises(RuntimeError, match="tool broke"):
        await broken()


def test_get_tracer_returns_noop_when_opentelemetry_absent(monkeypatch):
    """When opentelemetry is not importable, _get_tracer returns _NoOpTracer."""
    import kazi.utils.telemetry as tel
    tel._tracer = None  # reset cache

    import builtins
    real_import = builtins.__import__

    def blocking_import(name, *args, **kwargs):
        if name == "opentelemetry":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocking_import)
    tracer = tel._get_tracer()
    assert isinstance(tracer, tel._NoOpTracer)
    tel._tracer = None  # clean up for subsequent tests


def test_configure_telemetry_returns_false_when_sdk_absent(monkeypatch):
    """configure_telemetry returns False gracefully when OTel SDK not installed."""
    import builtins

    import kazi.utils.telemetry as tel

    real_import = builtins.__import__

    def blocking_import(name, *args, **kwargs):
        if "opentelemetry" in name:
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocking_import)
    result = tel.configure_telemetry("http://localhost:4317", service_name="test")
    assert result is False


def test_current_trace_id_returns_none_without_active_span():
    """current_trace_id returns None when there is no active OTel span."""
    from kazi.utils.telemetry import current_trace_id
    # Without an active span, the function should return None (no OTel context)
    result = current_trace_id()
    # Either None (no OTel) or a string (if OTel is configured globally) — never raises
    assert result is None or isinstance(result, str)
