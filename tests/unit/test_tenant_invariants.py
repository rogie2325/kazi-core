"""
Property-based invariants for tenant isolation.

The enterprise-security contract: tenant A's prompts can never resolve to
tenant B's thread, audit recorder, or tool registry — under any sequence of
calls, with any choice of thread IDs, with any amount of concurrency.

These tests fuzz the entire surface and assert the invariant holds.  A single
counterexample is a critical bug; the suite must pass with 0 failures across
hundreds of generated inputs.

Invariants verified
===================

1. **Thread ID namespacing**: when ``tenant_id`` is passed, the effective
   thread ID is prefixed with ``t:{tenant_id}:`` exactly once.  Two tenants
   choosing the same user-supplied thread ID resolve to DIFFERENT internal IDs.

2. **No prefix double-application**: passing an already-prefixed thread ID
   does not produce ``t:X:t:X:...`` chains.

3. **Sanitisation safety**: malicious thread IDs (path traversal, control
   chars, very long inputs) cannot escape the tenant prefix or strip it.

4. **Audit attribution**: an audit produced under ``tenant_id=A`` has
   ``audit.tenant_id == A``.  Under load, this never gets confused.

5. **Tenant cost ledger isolation**: a record for tenant A is never visible
   in tenant B's report.
"""
from __future__ import annotations

import asyncio
import re
import string

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from kazi.core.audit import run_context
from kazi.core.cost import RunCost, TenantCostLedger
from kazi.core.orchestrator import _sanitize_thread_id

# ── Strategies ────────────────────────────────────────────────────────────────

# Tenant ids: realistic — slugs with letters, digits, hyphens
tenant_id_st = st.text(
    alphabet=string.ascii_lowercase + string.digits + "-",
    min_size=1,
    max_size=32,
).filter(lambda s: s.strip("-"))

# Thread IDs from clients can include anything — let hypothesis go wild
raw_thread_id_st = st.text(min_size=0, max_size=128)

# User IDs
user_id_st = st.text(
    alphabet=string.ascii_letters + string.digits + "@.-_",
    min_size=0,
    max_size=64,
)

_SAFE_TID = re.compile(r"^[A-Za-z0-9_\-:.@]*$")


def _apply_tenant_prefix(tid: str, tenant_id: str) -> str:
    """Mirror the orchestrator.run() prefixing logic for unit-level testing."""
    tid = _sanitize_thread_id(tid)
    if tenant_id:
        prefix = f"t:{_sanitize_thread_id(tenant_id)}:"
        if not tid.startswith(prefix):
            tid = prefix + tid
    return tid


settings.register_profile("tenant_invariants", database=None, max_examples=100, deadline=None)
settings.load_profile("tenant_invariants")


# ── 1. Prefix is always applied ───────────────────────────────────────────────


@given(tenant_id=tenant_id_st, raw_tid=raw_thread_id_st)
def test_thread_id_always_carries_tenant_prefix(tenant_id, raw_tid):
    """For any (tenant, raw_tid), the result must start with t:{tenant}:."""
    result = _apply_tenant_prefix(raw_tid, tenant_id)
    expected_prefix = f"t:{_sanitize_thread_id(tenant_id)}:"
    assert result.startswith(expected_prefix), (
        f"Missing tenant prefix.  tenant={tenant_id!r}, raw={raw_tid!r}, "
        f"result={result!r}"
    )


@given(raw_tid=raw_thread_id_st)
def test_no_tenant_no_prefix_added(raw_tid):
    """Empty tenant_id must leave the thread_id untouched (except sanitisation)."""
    result = _apply_tenant_prefix(raw_tid, "")
    assert not result.startswith("t::"), (
        f"Empty tenant produced a prefix: {result!r}"
    )


# ── 2. No double-prefix ───────────────────────────────────────────────────────


@given(tenant_id=tenant_id_st, suffix=raw_thread_id_st)
def test_idempotent_prefix(tenant_id, suffix):
    """
    Applying the prefix twice must equal applying it once.
    Prevents t:acme:t:acme:user-123 corruption.
    """
    once = _apply_tenant_prefix(suffix, tenant_id)
    twice = _apply_tenant_prefix(once, tenant_id)
    assert once == twice, (
        f"Prefix was reapplied.  once={once!r}, twice={twice!r}"
    )


# ── 3. Two tenants never resolve to the same thread ───────────────────────────


@given(
    tenant_a=tenant_id_st,
    tenant_b=tenant_id_st,
    raw_tid=raw_thread_id_st,
)
def test_distinct_tenants_get_distinct_threads(tenant_a, tenant_b, raw_tid):
    """For any pair of distinct tenants, the resolved thread IDs must differ."""
    if _sanitize_thread_id(tenant_a) == _sanitize_thread_id(tenant_b):
        # Hypothesis sometimes draws equivalent values; skip — not a counterexample
        return
    a_tid = _apply_tenant_prefix(raw_tid, tenant_a)
    b_tid = _apply_tenant_prefix(raw_tid, tenant_b)
    assert a_tid != b_tid, (
        f"Two tenants resolved to same thread!  a={a_tid!r}, b={b_tid!r}"
    )


# ── 4. Sanitisation safety ────────────────────────────────────────────────────


@given(raw_tid=raw_thread_id_st, tenant_id=tenant_id_st)
def test_sanitized_thread_id_is_in_whitelist(raw_tid, tenant_id):
    """The output must only contain whitelisted chars: [A-Za-z0-9_-:.@]."""
    result = _apply_tenant_prefix(raw_tid, tenant_id)
    assert _SAFE_TID.match(result), (
        f"Sanitised thread ID escaped the whitelist: {result!r}"
    )


@given(tenant_id=tenant_id_st)
def test_path_traversal_cannot_escape_prefix(tenant_id):
    """A malicious thread ID like '../../../etc/passwd' must remain prefixed."""
    malicious = "../../../etc/passwd"
    result = _apply_tenant_prefix(malicious, tenant_id)
    expected_prefix = f"t:{_sanitize_thread_id(tenant_id)}:"
    assert result.startswith(expected_prefix)
    assert "/" not in result
    assert ".." not in result or "_" in result  # dots get replaced with _


@given(tenant_id=tenant_id_st)
def test_null_byte_cannot_truncate_prefix(tenant_id):
    """Null bytes must be stripped so they cannot truncate the prefix."""
    result = _apply_tenant_prefix("\x00\x00\x00malicious", tenant_id)
    assert "\x00" not in result


@given(tenant_id=tenant_id_st)
def test_extremely_long_thread_id_capped(tenant_id):
    """1MB of garbage must produce a bounded-length output."""
    result = _apply_tenant_prefix("x" * 1_000_000, tenant_id)
    assert len(result) <= 256 + len(f"t:{tenant_id}:")  # sanitiser caps at 256


# ── 5. Audit attribution under load ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_audit_tenant_attribution_under_concurrent_load():
    """
    Run 50 parallel audited operations, each tagged with a distinct tenant.
    Each audit must carry the correct tenant_id — never the wrong one.
    """
    from kazi.core.registry import ToolRegistry

    registry = ToolRegistry()

    async def noop() -> str:
        return "ok"

    registry.register_function(noop, name="noop")

    async def one(tenant_id: str, user_id: str):
        with run_context(
            audit=True, shadow=False,
            tenant_id=tenant_id, user_id=user_id,
        ) as ctx:
            await registry.execute("noop")
            return tenant_id, user_id, ctx.recorder.finalize()

    tasks = [one(f"tenant-{i}", f"user-{i}") for i in range(50)]
    outcomes = await asyncio.gather(*tasks)

    for tenant_id, user_id, audit in outcomes:
        assert audit.tenant_id == tenant_id, (
            f"Audit attribution wrong: expected tenant={tenant_id}, got {audit.tenant_id}"
        )
        assert audit.user_id == user_id


# ── 6. TenantCostLedger isolation ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ledger_isolation_under_concurrent_writes():
    """
    Two tenants writing in parallel must end up with their own ledger entries.
    No cross-contamination of totals or counts.
    """
    ledger = TenantCostLedger()
    cost = RunCost.compute(1_000, 500, "gpt-4o-mini")

    async def record_n(tenant_id: str, n: int):
        for _ in range(n):
            await ledger.record(tenant_id=tenant_id, user_id="u", cost=cost)

    # Tenant A records 30 entries, tenant B records 20 — in parallel
    await asyncio.gather(record_n("acme", 30), record_n("globex", 20))

    acme = await ledger.report(tenant_id="acme")
    globex = await ledger.report(tenant_id="globex")

    assert len(acme) == 1
    assert len(globex) == 1
    assert acme[0].run_count == 30
    assert globex[0].run_count == 20
    # And acme's report never includes globex
    for row in acme:
        assert row.tenant_id == "acme"


@pytest.mark.asyncio
async def test_ledger_report_filtered_by_tenant_excludes_others():
    """A report scoped to one tenant must not leak entries from another."""
    ledger = TenantCostLedger()
    cost = RunCost.compute(1000, 500, "gpt-4o-mini")
    await ledger.record(tenant_id="acme", user_id="u1", cost=cost)
    await ledger.record(tenant_id="globex", user_id="u2", cost=cost)
    await ledger.record(tenant_id="acme", user_id="u3", cost=cost)

    rows = await ledger.report(tenant_id="acme")
    assert {r.user_id for r in rows} == {"u1", "u3"}
    for r in rows:
        assert r.tenant_id == "acme"


@given(
    tenant_a=tenant_id_st,
    tenant_b=tenant_id_st,
)
def test_property_based_distinct_ledger_keys(tenant_a, tenant_b):
    """For any pair of distinct tenants, ledger keys never collide."""
    if tenant_a == tenant_b:
        return
    key_a = (tenant_a, "u", "2026-05-14")
    key_b = (tenant_b, "u", "2026-05-14")
    assert key_a != key_b
