"""Cost tracking — estimates $ per run based on model pricing."""
from __future__ import annotations

import asyncio
import datetime
from dataclasses import dataclass

# Pricing in USD per 1M tokens (input / output).
# Updated May 2026 — check provider docs for current rates.
_PRICING: dict[str, dict[str, float]] = {
    # Anthropic
    "claude-opus-4-7":           {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-6":         {"input":  3.00, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input":  0.80, "output":  4.00},
    # Aliases without date suffix
    "claude-opus-4":             {"input": 15.00, "output": 75.00},
    "claude-sonnet-4":           {"input":  3.00, "output": 15.00},
    "claude-haiku-4":            {"input":  0.80, "output":  4.00},
    # OpenAI
    "gpt-4o":                    {"input":  2.50, "output": 10.00},
    "gpt-4o-mini":               {"input":  0.15, "output":  0.60},
    "gpt-4-turbo":               {"input": 10.00, "output": 30.00},
    "gpt-4":                     {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo":             {"input":  0.50, "output":  1.50},
    "o1":                        {"input": 15.00, "output": 60.00},
    "o1-mini":                   {"input":  3.00, "output": 12.00},
    "o3-mini":                   {"input":  1.10, "output":  4.40},
    # Google
    "gemini-1.5-pro":            {"input":  1.25, "output":  5.00},
    "gemini-1.5-flash":          {"input":  0.075,"output":  0.30},
    "gemini-2.0-flash":          {"input":  0.10, "output":  0.40},
}


@dataclass
class RunCost:
    """Token usage and estimated cost for a single kazi.run() call."""
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    cost_usd: float = 0.0

    @classmethod
    def compute(cls, input_tokens: int, output_tokens: int, model: str) -> RunCost:
        pricing = _model_pricing(model)
        if pricing:
            cost = (
                input_tokens  / 1_000_000 * pricing["input"] +
                output_tokens / 1_000_000 * pricing["output"]
            )
        else:
            cost = 0.0
        return cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            cost_usd=round(cost, 8),
        )

    def __add__(self, other: RunCost) -> RunCost:
        return RunCost(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            model=self.model or other.model,
            cost_usd=round(self.cost_usd + other.cost_usd, 8),
        )

    def __str__(self) -> str:
        if self.cost_usd == 0 and not self.model:
            return f"{self.input_tokens + self.output_tokens} tokens"
        return (
            f"{self.input_tokens:,} in + {self.output_tokens:,} out = "
            f"${self.cost_usd:.6f} ({self.model})"
        )


@dataclass
class RunResult:
    """
    Return type for kazi.run() when cost tracking is enabled.

    reply   The agent's text response.
    cost    Token usage and estimated cost for this run.
    """
    reply: str
    cost: RunCost


def _model_pricing(model: str) -> dict[str, float] | None:
    """Return pricing for `model`, trying progressively shorter prefixes."""
    if model in _PRICING:
        return _PRICING[model]
    # Try prefix match (e.g. "claude-sonnet-4-6-20260501" → "claude-sonnet-4-6")
    for key in _PRICING:
        if model.startswith(key):
            return _PRICING[key]
    return None


class CostAccumulator:
    """
    Accumulates token usage across multiple LLM calls within one kazi.run().

    Hooks into the LangChain callback system via on_llm_end to capture
    usage metadata returned by the provider without requiring extra API calls.
    """

    def __init__(self, model: str) -> None:
        self.model = model
        self.input_tokens = 0
        self.output_tokens = 0

    def record(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def to_run_cost(self) -> RunCost:
        return RunCost.compute(self.input_tokens, self.output_tokens, self.model)


# ── Per-tenant cost ledger ────────────────────────────────────────────────────


@dataclass
class CostReport:
    """
    Spending summary for one tenant/user on a given date.

    Returned by ``TenantCostLedger.report()`` — use for dashboards,
    chargeback reports, and per-team budget alerts.
    """
    tenant_id: str
    user_id: str
    date: str
    total_usd: float
    input_tokens: int
    output_tokens: int
    run_count: int

    def __str__(self) -> str:
        scope = f"tenant={self.tenant_id}" if self.tenant_id else f"user={self.user_id}"
        return (
            f"[{self.date}] {scope} — "
            f"${self.total_usd:.6f} | "
            f"{self.input_tokens:,} in + {self.output_tokens:,} out | "
            f"{self.run_count} runs"
        )


class TenantCostLedger:
    """
    Thread-safe in-process ledger for per-tenant and per-user cost tracking.

    Replaces the simple ``_daily_spend`` dict in Kazi for multi-tenant
    deployments.  Key scheme: (tenant_id, user_id, YYYY-MM-DD).

    Stale entries (prior days) are pruned on each write, so memory stays
    bounded to today's active tenants/users.

    For multi-process deployments (multiple Kazi workers), use a Redis-backed
    store instead — wire ``record()`` to increment a Redis hash and expose
    ``report()`` via a separate management API.

    Usage::

        ledger = TenantCostLedger()

        # Check before running (raises BudgetExceededError when over limit)
        await ledger.check_budget(tenant_id="acme", user_id="u1", limit_usd=5.0)

        # Record after running
        await ledger.record(tenant_id="acme", user_id="u1", cost=run_cost)

        # Report for billing / dashboards
        for row in await ledger.report(tenant_id="acme"):
            print(row)
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        # (tenant_id, user_id, date) → {usd, input_tokens, output_tokens, runs}
        self._data: dict[tuple[str, str, str], dict] = {}

    def _today(self) -> str:
        return datetime.date.today().isoformat()

    async def check_budget(
        self,
        *,
        tenant_id: str = "",
        user_id: str = "",
        limit_usd: float,
    ) -> None:
        """Raise BudgetExceededError if the spend for today already meets the limit."""
        if limit_usd <= 0:
            return
        today = self._today()
        key = (tenant_id, user_id, today)
        async with self._lock:
            entry = self._data.get(key, {})
        spent = entry.get("usd", 0.0)
        if spent >= limit_usd:
            from kazi.core.exceptions import BudgetExceededError
            label = f"tenant '{tenant_id}'" if tenant_id else f"user '{user_id}'"
            raise BudgetExceededError(
                f"{label} has reached their daily spending limit "
                f"(${spent:.4f} / ${limit_usd:.4f})."
            )

    async def record(
        self,
        *,
        tenant_id: str = "",
        user_id: str = "",
        cost: RunCost,
    ) -> None:
        """Add a completed run's cost to the ledger."""
        if cost.cost_usd <= 0 and cost.input_tokens == 0:
            return
        today = self._today()
        key = (tenant_id, user_id, today)
        async with self._lock:
            # Prune entries from previous days
            stale = [k for k in self._data if k[2] != today]
            for k in stale:
                del self._data[k]
            entry = self._data.setdefault(
                key, {"usd": 0.0, "input_tokens": 0, "output_tokens": 0, "runs": 0}
            )
            entry["usd"] += cost.cost_usd
            entry["input_tokens"] += cost.input_tokens
            entry["output_tokens"] += cost.output_tokens
            entry["runs"] += 1

    async def report(
        self,
        *,
        tenant_id: str | None = None,
        user_id: str | None = None,
        date: str | None = None,
    ) -> list[CostReport]:
        """
        Return cost summaries filtered by tenant, user, or date.

        Leave all parameters as None to get today's full report.
        """
        target_date = date or self._today()
        async with self._lock:
            rows = []
            for (tid, uid, d), entry in self._data.items():
                if d != target_date:
                    continue
                if tenant_id is not None and tid != tenant_id:
                    continue
                if user_id is not None and uid != user_id:
                    continue
                rows.append(CostReport(
                    tenant_id=tid,
                    user_id=uid,
                    date=d,
                    total_usd=round(entry["usd"], 8),
                    input_tokens=entry["input_tokens"],
                    output_tokens=entry["output_tokens"],
                    run_count=entry["runs"],
                ))
        rows.sort(key=lambda r: r.total_usd, reverse=True)
        return rows
