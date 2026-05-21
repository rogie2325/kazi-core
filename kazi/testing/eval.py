"""
Declarative scenario harness for go/no-go validation of a kazi integration.

A senior engineer reviewing a generated integration runs the eval suite and
gets a per-scenario pass/fail in seconds.  Each scenario is a small dataclass
describing:

  - a prompt to feed the agent
  - the expected shape of the resulting audit (tool names, count, status)
  - optional fingerprint expectation for replay-grade determinism

Example::

    from kazi.testing.eval import Scenario, run_scenarios

    scenarios = [
        Scenario(
            name="reads-invoice-once",
            prompt="Fetch invoice 4521",
            expect_tools=["get_invoice"],
            expect_no_errors=True,
        ),
        Scenario(
            name="never-touches-deletion",
            prompt="Clean up old invoices",
            forbid_tools=["delete_invoice", "purge_all"],
        ),
    ]
    report = await run_scenarios(kazi, scenarios)
    print(report.summary())     # 47/50 passed
    assert report.all_passed, report.failures_text()
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from kazi.core.audit import RunAudit

if TYPE_CHECKING:
    from kazi.core.orchestrator import Kazi

logger = logging.getLogger(__name__)


@dataclass
class Scenario:
    """
    One declarative validation scenario.

    name              Stable identifier — used in reports and to look up failures.
    prompt            User message to send to the agent.
    expect_tools      Tools that MUST appear in the audit (order ignored).
    forbid_tools      Tools that MUST NOT appear (e.g. destructive ops).
    expect_exact      When set, the audit's tool-call names (in order) must
                      match this list exactly.  Stricter than expect_tools.
    expect_min_calls  Lower bound on tool call count.
    expect_max_calls  Upper bound on tool call count.
    expect_no_errors  When True, no tool call may have status == "error".
    expect_fingerprint  Optional fingerprint (or fixture path) for replay-grade match.
    shadow            Run in shadow mode (no side effects).  Default True.
    """
    name: str
    prompt: str
    expect_tools: list[str] = field(default_factory=list)
    forbid_tools: list[str] = field(default_factory=list)
    expect_exact: list[str] | None = None
    expect_min_calls: int | None = None
    expect_max_calls: int | None = None
    expect_no_errors: bool = True
    expect_fingerprint: str | None = None
    shadow: bool = True


@dataclass
class ScenarioResult:
    name: str
    passed: bool
    audit: RunAudit | None
    failures: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        calls = len(self.audit.tool_calls) if self.audit else 0
        return f"[{status}] {self.name}  ({calls} tool call(s))"


@dataclass
class EvalReport:
    """Aggregated outcome of a scenario suite."""
    results: list[ScenarioResult]

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def total_count(self) -> int:
        return len(self.results)

    @property
    def all_passed(self) -> bool:
        return self.passed_count == self.total_count

    def summary(self) -> str:
        return f"{self.passed_count}/{self.total_count} scenario(s) passed"

    def failures_text(self) -> str:
        lines = []
        for r in self.results:
            if r.passed:
                continue
            lines.append(f"\n❌ {r.name}")
            for f in r.failures:
                lines.append(f"   - {f}")
        return "\n".join(lines) or "(no failures)"


# ── Runner ────────────────────────────────────────────────────────────────────


async def run_scenarios(kazi: Kazi, scenarios: list[Scenario]) -> EvalReport:
    """
    Run every scenario against ``kazi`` and return a structured report.

    Scenarios run sequentially so audit recorders never overlap.  For parallel
    execution across independent threads, group scenarios by thread_id and
    invoke run_scenarios on each group concurrently.
    """
    results: list[ScenarioResult] = []
    for s in scenarios:
        result = await _run_one(kazi, s)
        results.append(result)
        logger.info("%s", result)
    return EvalReport(results=results)


async def _run_one(kazi: Kazi, s: Scenario) -> ScenarioResult:
    failures: list[str] = []
    audit: RunAudit | None = None
    try:
        run_result = await kazi.run(
            s.prompt,
            thread_id=f"eval:{s.name}",
            audit=True,
            shadow=s.shadow,
        )
        audit = run_result.audit if hasattr(run_result, "audit") else None
    except Exception as exc:
        failures.append(f"run raised {type(exc).__name__}: {exc}")
        return ScenarioResult(name=s.name, passed=False, audit=None, failures=failures)

    if audit is None:
        failures.append("kazi.run returned no audit — was audit=True forwarded?")
        return ScenarioResult(name=s.name, passed=False, audit=None, failures=failures)

    called = [c.name for c in audit.tool_calls]

    # expect_exact (strictest — beats expect_tools / forbid_tools)
    if s.expect_exact is not None:
        if called != s.expect_exact:
            failures.append(
                f"tool-call sequence mismatch: expected {s.expect_exact}, got {called}"
            )
    else:
        for required in s.expect_tools:
            if required not in called:
                failures.append(f"required tool '{required}' was not called")
        for forbidden in s.forbid_tools:
            if forbidden in called:
                failures.append(f"forbidden tool '{forbidden}' was called")

    if s.expect_min_calls is not None and len(called) < s.expect_min_calls:
        failures.append(
            f"tool-call count {len(called)} below minimum {s.expect_min_calls}"
        )
    if s.expect_max_calls is not None and len(called) > s.expect_max_calls:
        failures.append(
            f"tool-call count {len(called)} above maximum {s.expect_max_calls}"
        )

    if s.expect_no_errors:
        errs = [c for c in audit.tool_calls if c.status == "error"]
        if errs:
            failures.append(
                f"{len(errs)} tool call(s) errored: "
                + ", ".join(f"{e.name} ({e.error})" for e in errs[:3])
            )

    if s.expect_fingerprint:
        actual = audit.fingerprint()
        if actual != s.expect_fingerprint:
            failures.append(
                f"fingerprint mismatch: expected {s.expect_fingerprint[:16]}…, "
                f"got {actual[:16]}…"
            )

    return ScenarioResult(
        name=s.name, passed=not failures, audit=audit, failures=failures
    )
