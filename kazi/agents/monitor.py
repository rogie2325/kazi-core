"""
Performance monitor — tracks call success/failure and fires bad actors.

Two independent thresholds can trigger a "fired" verdict:
  consecutive_threshold   N straight failures → fired immediately
  failure_rate_threshold  >X% failures over the last `window_size` calls → fired

Either threshold can be disabled by setting it to None.  Both fire only once
per component per lifetime; call reset(name) to rehabilitate after a fix.

Works for tools (ToolRegistry) and agents (Supervisor) — any string-named
component that produces successes or failures maps onto the same tracker.

Usage::

    from kazi.agents.monitor import PerformanceMonitor

    def alert(name: str, reason: str) -> None:
        print(f"FIRED: {name} — {reason}")

    monitor = PerformanceMonitor(consecutive_threshold=3, on_fired=alert)

    monitor.record("my_tool", success=False)
    monitor.record("my_tool", success=False)
    monitor.record("my_tool", success=False)
    # → prints "FIRED: my_tool — 3 consecutive failures"

    print(monitor.health("my_tool"))
"""
from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ComponentHealth:
    """Point-in-time health snapshot for one tool or agent."""

    name: str
    total_calls: int
    window_calls: int
    failures_in_window: int
    consecutive_failures: int
    failure_rate: float  # within the rolling window, 0.0–1.0
    fired: bool
    fired_reason: str = ""

    @property
    def success_rate(self) -> float:
        return 1.0 - self.failure_rate

    def __str__(self) -> str:
        status = f"FIRED ({self.fired_reason})" if self.fired else "active"
        return (
            f"{self.name}: {status} | "
            f"calls={self.total_calls} "
            f"fail_rate={self.failure_rate:.0%} "
            f"consecutive_failures={self.consecutive_failures}"
        )


class PerformanceMonitor:
    """
    Rolling-window failure tracker that fires components exceeding thresholds.

    Parameters
    ----------
    window_size
        Number of recent calls to consider for the failure-rate threshold.
    consecutive_threshold
        Fire after this many back-to-back failures.  None = disabled.
    failure_rate_threshold
        Fire when failure_rate exceeds this fraction over the window.
        Only evaluated once the window is full.  None = disabled.
    min_calls
        Minimum number of calls before the rate threshold is evaluated.
        Prevents false positives on brand-new components.
    on_fired
        Optional callback called once when a component is fired:
        ``on_fired(name: str, reason: str) -> None``
    """

    def __init__(
        self,
        window_size: int = 20,
        consecutive_threshold: int | None = 5,
        failure_rate_threshold: float | None = 0.5,
        min_calls: int = 5,
        on_fired: Callable[[str, str], None] | None = None,
    ) -> None:
        self._window = window_size
        self._consec_threshold = consecutive_threshold
        self._rate_threshold = failure_rate_threshold
        self._min_calls = min_calls
        self._on_fired = on_fired

        self._history: dict[str, deque[bool]] = {}  # True = success
        self._consecutive: dict[str, int] = {}
        self._total: dict[str, int] = {}
        self._fired: dict[str, str] = {}  # name → reason

    # ── Public API ────────────────────────────────────────────────────────

    def record(self, name: str, *, success: bool) -> bool:
        """
        Record one call result for `name`.

        Returns True the first time this component crosses a firing threshold
        (and also invokes on_fired).  Returns False on all subsequent calls.
        Already-fired components are not re-evaluated.
        """
        if name in self._fired:
            return False

        if name not in self._history:
            self._history[name] = deque(maxlen=self._window)
            self._consecutive[name] = 0
            self._total[name] = 0

        self._history[name].append(success)
        self._total[name] += 1

        if success:
            self._consecutive[name] = 0
            return False

        self._consecutive[name] += 1
        reason = self._check_thresholds(name)
        if not reason:
            return False

        self._fired[name] = reason
        logger.warning(
            "PerformanceMonitor: FIRED %r — %s "
            "(total_calls=%d, consecutive_failures=%d)",
            name, reason, self._total[name], self._consecutive[name],
        )
        if self._on_fired:
            try:
                self._on_fired(name, reason)
            except Exception as exc:
                logger.error("on_fired callback raised for %r: %s", name, exc)
        return True

    def is_fired(self, name: str) -> bool:
        """True if this component has been fired."""
        return name in self._fired

    def health(self, name: str) -> ComponentHealth:
        """Return a point-in-time health snapshot for `name`."""
        history = list(self._history.get(name, []))
        window_calls = len(history)
        failures = history.count(False)
        rate = failures / window_calls if window_calls else 0.0
        return ComponentHealth(
            name=name,
            total_calls=self._total.get(name, 0),
            window_calls=window_calls,
            failures_in_window=failures,
            consecutive_failures=self._consecutive.get(name, 0),
            failure_rate=rate,
            fired=name in self._fired,
            fired_reason=self._fired.get(name, ""),
        )

    def summary(self) -> list[ComponentHealth]:
        """Health snapshot for every tracked component, sorted by name."""
        all_names = set(self._history) | set(self._fired)
        return [self.health(n) for n in sorted(all_names)]

    def fired_names(self) -> list[str]:
        """Return names of all fired components."""
        return list(self._fired)

    def reset(self, name: str) -> None:
        """
        Clear history and un-fire a component (e.g. after a patch or restart).
        The component will be evaluated fresh from the next call onward.
        """
        self._history.pop(name, None)
        self._consecutive.pop(name, None)
        self._total.pop(name, None)
        self._fired.pop(name, None)
        logger.info("PerformanceMonitor: reset %r — back on probation", name)

    # ── Internal ──────────────────────────────────────────────────────────

    def _check_thresholds(self, name: str) -> str:
        consecutive = self._consecutive[name]
        total = self._total[name]

        if self._consec_threshold is not None and consecutive >= self._consec_threshold:
            return f"{consecutive} consecutive failures"

        if self._rate_threshold is not None and total >= self._min_calls:
            history = self._history[name]
            if len(history) >= self._window:
                rate = history.count(False) / len(history)
                if rate > self._rate_threshold:
                    pct = f"{rate:.0%}"
                    thr = f"{self._rate_threshold:.0%}"
                    return (
                        f"failure rate {pct} exceeds {thr} "
                        f"over last {self._window} calls"
                    )
        return ""
