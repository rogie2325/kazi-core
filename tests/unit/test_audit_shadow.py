"""
Tests for kazi.core.audit (RunAudit, AuditRecorder, shadow mode) and the
ToolRegistry.execute() instrumentation that ties them together.
"""
from __future__ import annotations

import pytest

from kazi.core.audit import (
    AuditRecorder,
    RunAudit,
    RunAuditResult,
    ToolCallRecord,
    get_recorder,
    is_shadow,
    run_context,
)
from kazi.core.registry import ToolRegistry

# ── ToolCallRecord ────────────────────────────────────────────────────────────

def test_tool_call_record_str_ok():
    rec = ToolCallRecord(
        name="get_user", args={"id": 42}, result="alice", duration_ms=12.3, status="ok",
    )
    s = str(rec)
    assert "[OK]" in s
    assert "get_user" in s
    assert "id=42" in s
    assert "alice" in s
    assert "12.3" in s


def test_tool_call_record_str_error():
    rec = ToolCallRecord(
        name="fail", args={}, result=None, duration_ms=1.0, status="error",
        error="boom",
    )
    s = str(rec)
    assert "[ERROR]" in s
    assert "boom" in s


def test_tool_call_record_str_long_result_truncates():
    rec = ToolCallRecord(
        name="t", args={}, result="x" * 500, duration_ms=0, status="ok",
    )
    s = str(rec)
    # Result preview truncates to 80 chars + ellipsis
    assert "…" in s


# ── RunAudit ──────────────────────────────────────────────────────────────────

def test_run_audit_counts_and_summary():
    audit = RunAudit()
    audit.tool_calls.append(ToolCallRecord("a", {}, "ok", 5.0, "ok"))
    audit.tool_calls.append(ToolCallRecord("b", {}, None, 2.0, "error", "boom"))
    audit.tool_calls.append(ToolCallRecord("c", {}, "stub", 1.0, "shadow"))
    audit.duration_ms = 100.0

    assert audit.tool_call_count == 3
    assert audit.error_count == 1
    assert audit.total_tool_ms == 8.0
    summary = audit.summary()
    assert "3 tool call" in summary
    assert "1 error" in summary


def test_run_audit_summary_marks_shadow():
    audit = RunAudit(shadow=True)
    assert audit.summary().startswith("SHADOW")


# ── Recorder + context manager ────────────────────────────────────────────────

def test_run_context_audit_only_binds_recorder():
    assert get_recorder() is None
    assert is_shadow() is False

    with run_context(audit=True, shadow=False) as ctx:
        assert isinstance(get_recorder(), AuditRecorder)
        assert ctx.recorder is get_recorder()
        assert is_shadow() is False

    # Resets after context
    assert get_recorder() is None
    assert is_shadow() is False


def test_run_context_shadow_only_sets_flag():
    with run_context(audit=False, shadow=True) as ctx:
        assert ctx.recorder is None
        assert get_recorder() is None
        assert is_shadow() is True
    assert is_shadow() is False


def test_run_context_both_sets_both():
    with run_context(audit=True, shadow=True) as ctx:
        assert isinstance(get_recorder(), AuditRecorder)
        assert is_shadow() is True
        assert ctx.recorder is not None and ctx.recorder.audit.shadow is True


def test_recorder_finalize_sets_duration_and_error():
    rec = AuditRecorder()
    rec.record_tool_call(
        name="x", args={"a": 1}, result="r", duration_ms=5.0, status="ok"
    )
    audit = rec.finalize(error="something failed")
    assert audit.duration_ms > 0
    assert audit.error == "something failed"
    assert audit.tool_call_count == 1


def test_recorder_args_are_defensively_copied():
    rec = AuditRecorder()
    args = {"k": "v"}
    rec.record_tool_call(name="t", args=args, result=None, duration_ms=0, status="ok")
    args["k"] = "mutated"
    assert rec.audit.tool_calls[0].args["k"] == "v"


# ── Registry execute() with audit + shadow ────────────────────────────────────

@pytest.mark.asyncio
async def test_registry_execute_records_when_audit_enabled():
    registry = ToolRegistry()

    async def my_tool(name: str) -> str:
        return f"hello, {name}"

    registry.register_function(my_tool, name="greet")

    with run_context(audit=True, shadow=False) as ctx:
        result = await registry.execute("greet", name="world")

    assert result == "hello, world"
    assert ctx.recorder is not None
    audit = ctx.recorder.finalize()
    assert audit.tool_call_count == 1
    call = audit.tool_calls[0]
    assert call.name == "greet"
    assert call.args == {"name": "world"}
    assert call.status == "ok"
    assert "hello, world" in (call.result or "")


@pytest.mark.asyncio
async def test_registry_execute_shadow_skips_handler_and_returns_stub():
    registry = ToolRegistry()
    side_effects: list[str] = []

    async def dangerous(x: int) -> str:
        side_effects.append("BOOM")
        return f"deleted {x}"

    registry.register_function(dangerous, name="delete_user")

    with run_context(audit=True, shadow=True) as ctx:
        result = await registry.execute("delete_user", x=99)

    # Real handler must not run
    assert side_effects == []
    # Stub identifies tool + args
    assert "[SHADOW]" in result
    assert "delete_user" in result
    assert "99" in result
    # Audit captures it as shadow
    audit = ctx.recorder.finalize()
    assert audit.tool_call_count == 1
    assert audit.tool_calls[0].status == "shadow"


@pytest.mark.asyncio
async def test_registry_execute_records_error_in_audit():
    registry = ToolRegistry()

    async def broken() -> str:
        raise RuntimeError("nope")

    registry.register_function(broken, name="broken")

    with run_context(audit=True, shadow=False) as ctx:
        with pytest.raises(Exception):
            await registry.execute("broken")

    audit = ctx.recorder.finalize()
    assert audit.error_count == 1
    assert audit.tool_calls[0].status == "error"
    assert "nope" in (audit.tool_calls[0].error or "")


@pytest.mark.asyncio
async def test_registry_execute_without_context_does_not_raise():
    """Calling tools outside a run_context must not crash — recorder is None."""
    registry = ToolRegistry()

    async def echo(msg: str) -> str:
        return msg

    registry.register_function(echo)
    result = await registry.execute("echo", msg="hi")
    assert result == "hi"


# ── RunAuditResult ────────────────────────────────────────────────────────────

def test_run_audit_result_dataclass():
    audit = RunAudit()
    result = RunAuditResult(reply="hello", audit=audit, cost=None)
    assert result.reply == "hello"
    assert result.audit is audit
    assert result.cost is None
