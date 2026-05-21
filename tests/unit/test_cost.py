"""Unit tests for kazi.core.cost — RunCost, CostAccumulator, TenantCostLedger."""
from __future__ import annotations

import pytest

from kazi.core.cost import (
    CostAccumulator,
    CostReport,
    RunCost,
    TenantCostLedger,
    _model_pricing,
)

# ── RunCost ───────────────────────────────────────────────────────────────────

def test_run_cost_compute_known_model():
    cost = RunCost.compute(1_000_000, 500_000, "gpt-4o")
    assert cost.input_tokens == 1_000_000
    assert cost.output_tokens == 500_000
    assert cost.cost_usd == pytest.approx(2.50 + 5.00, rel=1e-6)


def test_run_cost_compute_unknown_model():
    cost = RunCost.compute(100, 50, "unknown-model-xyz")
    assert cost.cost_usd == 0.0


def test_run_cost_add():
    a = RunCost.compute(100_000, 50_000, "gpt-4o-mini")
    b = RunCost.compute(200_000, 100_000, "gpt-4o-mini")
    total = a + b
    assert total.input_tokens == 300_000
    assert total.output_tokens == 150_000
    assert total.cost_usd == pytest.approx(a.cost_usd + b.cost_usd, rel=1e-6)


def test_run_cost_str():
    cost = RunCost.compute(1000, 500, "gpt-4o")
    s = str(cost)
    assert "1,000" in s
    assert "500" in s
    assert "$" in s


def test_run_cost_str_zero():
    cost = RunCost()
    s = str(cost)
    assert "tokens" in s


def test_model_pricing_prefix_match():
    # "claude-sonnet-4-6-20260501" should match "claude-sonnet-4-6"
    pricing = _model_pricing("claude-sonnet-4-6-20260501")
    assert pricing is not None
    assert pricing["input"] == 3.00


def test_model_pricing_exact_match():
    pricing = _model_pricing("claude-opus-4-7")
    assert pricing is not None
    assert pricing["output"] == 75.00


def test_model_pricing_unknown_returns_none():
    assert _model_pricing("not-a-real-model-abc123") is None


# ── CostAccumulator ───────────────────────────────────────────────────────────

def test_cost_accumulator_record_and_compute():
    acc = CostAccumulator("gpt-4o-mini")
    acc.record(10_000, 5_000)
    acc.record(20_000, 10_000)
    cost = acc.to_run_cost()
    assert cost.input_tokens == 30_000
    assert cost.output_tokens == 15_000
    assert cost.model == "gpt-4o-mini"
    assert cost.cost_usd > 0


# ── TenantCostLedger ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ledger_record_and_report():
    ledger = TenantCostLedger()
    cost = RunCost.compute(100_000, 50_000, "gpt-4o-mini")
    await ledger.record(tenant_id="acme", user_id="u1", cost=cost)

    rows = await ledger.report(tenant_id="acme")
    assert len(rows) == 1
    assert rows[0].tenant_id == "acme"
    assert rows[0].user_id == "u1"
    assert rows[0].total_usd == pytest.approx(cost.cost_usd, rel=1e-6)
    assert rows[0].run_count == 1


@pytest.mark.asyncio
async def test_ledger_accumulates_across_runs():
    ledger = TenantCostLedger()
    cost = RunCost.compute(100_000, 50_000, "gpt-4o-mini")
    await ledger.record(tenant_id="acme", user_id="u1", cost=cost)
    await ledger.record(tenant_id="acme", user_id="u1", cost=cost)

    rows = await ledger.report(tenant_id="acme")
    assert rows[0].run_count == 2
    assert rows[0].total_usd == pytest.approx(cost.cost_usd * 2, rel=1e-6)


@pytest.mark.asyncio
async def test_ledger_isolates_tenants():
    ledger = TenantCostLedger()
    cost = RunCost.compute(100_000, 50_000, "gpt-4o-mini")
    await ledger.record(tenant_id="acme", user_id="u1", cost=cost)
    await ledger.record(tenant_id="globex", user_id="u2", cost=cost)

    acme_rows = await ledger.report(tenant_id="acme")
    globex_rows = await ledger.report(tenant_id="globex")
    assert len(acme_rows) == 1
    assert acme_rows[0].tenant_id == "acme"
    assert len(globex_rows) == 1
    assert globex_rows[0].tenant_id == "globex"


@pytest.mark.asyncio
async def test_ledger_budget_check_passes_when_under():
    ledger = TenantCostLedger()
    # Should not raise
    await ledger.check_budget(tenant_id="acme", user_id="u1", limit_usd=10.0)


@pytest.mark.asyncio
async def test_ledger_budget_check_raises_when_over():
    from kazi.core.exceptions import BudgetExceededError
    ledger = TenantCostLedger()
    cost = RunCost.compute(10_000_000, 5_000_000, "gpt-4o")  # expensive run
    await ledger.record(tenant_id="acme", user_id="u1", cost=cost)

    with pytest.raises(BudgetExceededError):
        await ledger.check_budget(tenant_id="acme", user_id="u1", limit_usd=0.01)


@pytest.mark.asyncio
async def test_ledger_budget_check_disabled_when_zero():
    ledger = TenantCostLedger()
    cost = RunCost.compute(10_000_000, 5_000_000, "gpt-4o")
    await ledger.record(tenant_id="acme", user_id="u1", cost=cost)
    # limit_usd=0.0 means disabled — should not raise
    await ledger.check_budget(tenant_id="acme", user_id="u1", limit_usd=0.0)


@pytest.mark.asyncio
async def test_ledger_report_empty_when_no_matching_tenant():
    ledger = TenantCostLedger()
    cost = RunCost.compute(1000, 500, "gpt-4o-mini")
    await ledger.record(tenant_id="acme", user_id="u1", cost=cost)

    rows = await ledger.report(tenant_id="does-not-exist")
    assert rows == []


@pytest.mark.asyncio
async def test_ledger_report_sorted_by_cost_descending():
    ledger = TenantCostLedger()
    cheap = RunCost.compute(100, 50, "gpt-4o-mini")
    expensive = RunCost.compute(1_000_000, 500_000, "gpt-4o")
    await ledger.record(tenant_id="acme", user_id="u1", cost=cheap)
    await ledger.record(tenant_id="acme", user_id="u2", cost=expensive)

    rows = await ledger.report(tenant_id="acme")
    assert rows[0].total_usd >= rows[1].total_usd


# ── CostReport __str__ ────────────────────────────────────────────────────────

def test_cost_report_str_with_tenant():
    r = CostReport(
        tenant_id="acme", user_id="", date="2026-05-13",
        total_usd=0.042, input_tokens=10000, output_tokens=5000, run_count=3,
    )
    s = str(r)
    assert "acme" in s
    assert "2026-05-13" in s
    assert "3" in s
