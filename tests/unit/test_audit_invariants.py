"""
Property-based invariant suite for audit / shadow / registry contracts.

These tests fuzz the public surface and assert that core contracts hold
across every input.  They catch the failure modes hand-written cases miss.

Invariants verified
===================

1. **Fingerprint determinism** — identical RunAudits always produce the same
   fingerprint; dict-key ordering must not leak into the hash.

2. **Fingerprint sensitivity** — adding, removing, or reordering tool calls
   changes the fingerprint.

3. **Fingerprint stability across irrelevant noise** — timestamps, durations,
   and freeform result strings must NOT affect the fingerprint.

4. **JSON round-trip** — audit.to_json → from_json preserves fingerprint.

5. **Shadow guarantee** — when shadow=True, the real handler is NEVER invoked,
   no matter how many parallel runs are issued.

6. **Audit isolation** — concurrent runs in independent contexts must not
   record into each other's recorders.
"""
from __future__ import annotations

import asyncio
import string

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from kazi.core.audit import (
    RunAudit,
    ToolCallRecord,
    get_recorder,
    is_shadow,
    run_context,
)
from kazi.core.registry import ToolRegistry

# ── Strategies ────────────────────────────────────────────────────────────────

tool_name_st = st.text(
    alphabet=string.ascii_letters + string.digits + "_-",
    min_size=1,
    max_size=40,
).filter(lambda s: s.strip())

json_value_st = st.recursive(
    st.one_of(
        st.none(),
        st.booleans(),
        st.integers(min_value=-1000, max_value=1000),
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
        st.text(max_size=50),
    ),
    lambda children: st.one_of(
        st.lists(children, max_size=5),
        st.dictionaries(st.text(max_size=20), children, max_size=5),
    ),
    max_leaves=10,
)

args_st = st.dictionaries(
    keys=st.text(alphabet=string.ascii_letters + "_", min_size=1, max_size=20),
    values=json_value_st,
    max_size=5,
)

status_st = st.sampled_from(["ok", "error", "shadow"])


def tool_call_record_st():
    return st.builds(
        ToolCallRecord,
        name=tool_name_st,
        args=args_st,
        result=st.one_of(st.none(), st.text(max_size=200)),
        duration_ms=st.floats(min_value=0, max_value=10_000, allow_nan=False),
        status=status_st,
        error=st.one_of(st.none(), st.text(max_size=100)),
    )


def run_audit_st():
    return st.builds(
        RunAudit,
        tool_calls=st.lists(tool_call_record_st(), max_size=8),
        started_at=st.floats(min_value=0, max_value=2e9, allow_nan=False),
        duration_ms=st.floats(min_value=0, max_value=1e6, allow_nan=False),
        shadow=st.booleans(),
        thread_id=st.text(max_size=64),
        tenant_id=st.text(max_size=64),
        user_id=st.text(max_size=64),
        error=st.one_of(st.none(), st.text(max_size=100)),
    )


settings.register_profile("invariants", database=None, max_examples=50, deadline=None)
settings.load_profile("invariants")


# ── 1. Fingerprint determinism ────────────────────────────────────────────────

@given(audit=run_audit_st())
def test_fingerprint_is_deterministic(audit: RunAudit):
    """Same audit, computed twice, must produce identical fingerprints."""
    assert audit.fingerprint() == audit.fingerprint()


@given(audit=run_audit_st())
def test_fingerprint_independent_of_dict_key_order(audit: RunAudit):
    """Rebuilding tool-call args with shuffled keys must not change the hash."""
    # Build a clone with each args dict reconstructed in reverse key order
    shuffled_calls = []
    for c in audit.tool_calls:
        shuffled = {k: c.args[k] for k in reversed(list(c.args.keys()))}
        shuffled_calls.append(ToolCallRecord(
            name=c.name,
            args=shuffled,
            result=c.result,
            duration_ms=c.duration_ms,
            status=c.status,
            error=c.error,
        ))
    clone = RunAudit(
        tool_calls=shuffled_calls,
        started_at=audit.started_at,
        duration_ms=audit.duration_ms,
        shadow=audit.shadow,
        thread_id=audit.thread_id,
        tenant_id=audit.tenant_id,
        user_id=audit.user_id,
        error=audit.error,
    )
    assert audit.fingerprint() == clone.fingerprint()


# ── 2. Fingerprint sensitivity ────────────────────────────────────────────────

@given(
    base=run_audit_st(),
    extra=tool_call_record_st(),
)
def test_fingerprint_changes_when_tool_call_appended(base: RunAudit, extra: ToolCallRecord):
    """Adding a tool call must change the fingerprint."""
    before = base.fingerprint()
    base.tool_calls.append(extra)
    after = base.fingerprint()
    assert before != after


@given(audit=run_audit_st().filter(lambda a: len(a.tool_calls) >= 2))
def test_fingerprint_changes_when_order_swapped(audit: RunAudit):
    """Swapping the first two tool calls must change the fingerprint."""
    before = audit.fingerprint()
    audit.tool_calls[0], audit.tool_calls[1] = audit.tool_calls[1], audit.tool_calls[0]
    after = audit.fingerprint()
    # If the swapped pair were identical in name+args+status, fingerprint
    # is allowed to stay the same — semantic, not syntactic, change.
    a, b = audit.tool_calls[0], audit.tool_calls[1]
    if (a.name, a.args, a.status) == (b.name, b.args, b.status):
        assert before == after
    else:
        assert before != after


# ── 3. Fingerprint stability across irrelevant noise ──────────────────────────

@given(
    audit=run_audit_st(),
    new_started=st.floats(min_value=0, max_value=2e9, allow_nan=False),
    new_duration=st.floats(min_value=0, max_value=1e6, allow_nan=False),
)
def test_fingerprint_ignores_timing_noise(audit, new_started, new_duration):
    """Wall-clock and duration must NOT change the fingerprint."""
    before = audit.fingerprint()
    audit.started_at = new_started
    audit.duration_ms = new_duration
    for c in audit.tool_calls:
        c.duration_ms += 17.5
    after = audit.fingerprint()
    assert before == after


@given(audit=run_audit_st(), new_result=st.text(max_size=300))
def test_fingerprint_ignores_freeform_result(audit, new_result):
    """Tool result strings must NOT affect the fingerprint."""
    before = audit.fingerprint()
    for c in audit.tool_calls:
        c.result = new_result
    after = audit.fingerprint()
    assert before == after


# ── 4. JSON round-trip preserves fingerprint ──────────────────────────────────

@given(audit=run_audit_st())
def test_json_round_trip_preserves_fingerprint(audit: RunAudit):
    payload = audit.to_json()
    restored = RunAudit.from_json(payload)
    assert audit.fingerprint() == restored.fingerprint()


# ── 5. Shadow guarantee ───────────────────────────────────────────────────────

@pytest.mark.asyncio
@settings(max_examples=20, deadline=None)
@given(arg=json_value_st)
async def test_shadow_never_invokes_real_handler(arg):
    """In shadow mode the real handler must NEVER run, regardless of args."""
    registry = ToolRegistry()
    invocations: list = []

    async def destructive(payload):
        invocations.append(payload)
        return "ran live"

    registry.register_function(destructive, name="destructive")

    with run_context(audit=True, shadow=True):
        result = await registry.execute("destructive", payload=arg)

    assert invocations == []
    assert isinstance(result, str)
    assert "[SHADOW]" in result


# ── 6. Concurrent audit isolation ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_concurrent_runs_do_not_share_recorder():
    """
    Many parallel runs in independent contexts must each get their own
    AuditRecorder.  No cross-talk.
    """
    registry = ToolRegistry()

    async def echo(payload: str) -> str:
        return f"echo:{payload}"

    registry.register_function(echo, name="echo")

    async def one(label: str):
        # Each call runs in its own asyncio task, so contextvars are copied
        # at task creation — modifications inside this coroutine are isolated.
        with run_context(audit=True, shadow=False) as ctx:
            await registry.execute("echo", payload=label)
            return ctx.recorder.finalize()

    audits = await asyncio.gather(*[one(f"task-{i}") for i in range(20)])

    # Each audit must contain exactly one call, with its own label
    seen = set()
    for i, a in enumerate(audits):
        assert a.tool_call_count == 1
        call = a.tool_calls[0]
        assert call.name == "echo"
        assert call.args == {"payload": f"task-{i}"}
        seen.add(call.args["payload"])
    assert len(seen) == 20  # no two audits share input


@pytest.mark.asyncio
async def test_no_recorder_outside_context():
    """get_recorder() must return None / is_shadow() must be False at module scope."""
    assert get_recorder() is None
    assert is_shadow() is False
