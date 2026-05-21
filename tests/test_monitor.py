"""
Tests for PerformanceMonitor, ToolRegistry auto-firing, and Supervisor agent firing.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kazi.agents.monitor import ComponentHealth, PerformanceMonitor
from kazi.core.exceptions import ToolExecutionError
from kazi.core.registry import ToolDefinition, ToolRegistry, ToolSource

# ── PerformanceMonitor ────────────────────────────────────────────────────────

class TestPerformanceMonitor:
    def _make(self, **kwargs) -> PerformanceMonitor:
        return PerformanceMonitor(
            window_size=kwargs.pop("window_size", 10),
            consecutive_threshold=kwargs.pop("consecutive_threshold", 3),
            failure_rate_threshold=kwargs.pop("failure_rate_threshold", None),
            min_calls=kwargs.pop("min_calls", 5),
            **kwargs,
        )

    def test_no_fire_on_successes(self):
        m = self._make()
        for _ in range(10):
            assert m.record("tool", success=True) is False
        assert not m.is_fired("tool")

    def test_consecutive_threshold_fires(self):
        m = self._make(consecutive_threshold=3)
        m.record("tool", success=False)
        m.record("tool", success=False)
        fired = m.record("tool", success=False)
        assert fired is True
        assert m.is_fired("tool")

    def test_success_resets_consecutive_counter(self):
        m = self._make(consecutive_threshold=3)
        m.record("tool", success=False)
        m.record("tool", success=False)
        m.record("tool", success=True)  # resets counter
        m.record("tool", success=False)
        assert not m.is_fired("tool")  # only 1 consecutive failure after reset

    def test_already_fired_does_not_re_fire(self):
        fired_calls: list[str] = []
        m = self._make(consecutive_threshold=3, on_fired=lambda n, r: fired_calls.append(n))
        for _ in range(3):
            m.record("tool", success=False)
        for _ in range(5):
            result = m.record("tool", success=False)
            assert result is False
        assert fired_calls.count("tool") == 1

    def test_failure_rate_threshold(self):
        m = PerformanceMonitor(
            window_size=10,
            consecutive_threshold=None,
            failure_rate_threshold=0.5,
            min_calls=5,
        )
        # 5 successes, 5 failures = 50% exactly — not above threshold
        for _ in range(5):
            m.record("tool", success=True)
        for i in range(5):
            fired = m.record("tool", success=False)
            if i < 4:
                assert not fired
        # Now add one more failure to push above 50% in window
        fired = m.record("tool", success=False)
        assert fired is True

    def test_rate_threshold_requires_min_calls(self):
        m = PerformanceMonitor(
            window_size=10,
            consecutive_threshold=None,
            failure_rate_threshold=0.1,  # very low
            min_calls=5,
        )
        # Only 4 calls (below min_calls) — should not fire despite 100% failure rate
        for _ in range(4):
            assert m.record("tool", success=False) is False
        assert not m.is_fired("tool")

    def test_on_fired_callback_called(self):
        fired_args: list[tuple] = []
        m = self._make(consecutive_threshold=2, on_fired=lambda n, r: fired_args.append((n, r)))
        m.record("agent_x", success=False)
        m.record("agent_x", success=False)
        assert len(fired_args) == 1
        assert fired_args[0][0] == "agent_x"
        assert "consecutive" in fired_args[0][1]

    def test_health_snapshot(self):
        m = self._make(consecutive_threshold=5)
        m.record("tool", success=True)
        m.record("tool", success=False)
        m.record("tool", success=False)
        h = m.health("tool")
        assert isinstance(h, ComponentHealth)
        assert h.total_calls == 3
        assert h.consecutive_failures == 2
        assert h.failures_in_window == 2
        assert not h.fired

    def test_health_unknown_component(self):
        m = self._make()
        h = m.health("ghost")
        assert h.total_calls == 0
        assert h.fired is False

    def test_summary_includes_all_tracked(self):
        m = self._make(consecutive_threshold=2)
        m.record("alpha", success=True)
        m.record("beta", success=False)
        m.record("beta", success=False)  # fires beta
        names = {h.name for h in m.summary()}
        assert "alpha" in names
        assert "beta" in names

    def test_reset_clears_history_and_fired(self):
        m = self._make(consecutive_threshold=2)
        m.record("tool", success=False)
        m.record("tool", success=False)
        assert m.is_fired("tool")
        m.reset("tool")
        assert not m.is_fired("tool")
        # Can accumulate failures fresh after reset
        m.record("tool", success=False)
        assert not m.is_fired("tool")  # only 1 failure, threshold is 2

    def test_fired_names(self):
        m = self._make(consecutive_threshold=1)
        m.record("a", success=False)
        m.record("b", success=True)
        assert "a" in m.fired_names()
        assert "b" not in m.fired_names()


# ── ToolRegistry auto-firing ──────────────────────────────────────────────────

def _make_tool(name: str, handler) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description="test tool",
        parameters=[],
        source=ToolSource.NATIVE,
        handler=handler,
    )


class TestToolRegistryMonitor:
    @pytest.mark.asyncio
    async def test_success_recorded(self):
        monitor = PerformanceMonitor(consecutive_threshold=3)
        registry = ToolRegistry(monitor=monitor)
        registry.register(_make_tool("ok_tool", AsyncMock(return_value="done")))

        await registry.execute("ok_tool")
        h = monitor.health("ok_tool")
        assert h.total_calls == 1
        assert h.failures_in_window == 0

    @pytest.mark.asyncio
    async def test_failure_recorded_on_exception(self):
        monitor = PerformanceMonitor(consecutive_threshold=5)
        registry = ToolRegistry(monitor=monitor)
        registry.register(_make_tool("bad_tool", AsyncMock(side_effect=RuntimeError("boom"))))

        with pytest.raises(ToolExecutionError):
            await registry.execute("bad_tool")

        h = monitor.health("bad_tool")
        assert h.failures_in_window == 1
        assert h.consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_tool_auto_removed_when_fired(self):
        monitor = PerformanceMonitor(consecutive_threshold=3)
        registry = ToolRegistry(monitor=monitor)
        registry.register(_make_tool("flaky", AsyncMock(side_effect=RuntimeError("err"))))

        for _ in range(3):
            with pytest.raises(ToolExecutionError):
                await registry.execute("flaky")

        assert "flaky" not in registry

    @pytest.mark.asyncio
    async def test_tool_not_removed_below_threshold(self):
        monitor = PerformanceMonitor(consecutive_threshold=5)
        registry = ToolRegistry(monitor=monitor)
        registry.register(_make_tool("flaky", AsyncMock(side_effect=RuntimeError("err"))))

        for _ in range(4):
            with pytest.raises(ToolExecutionError):
                await registry.execute("flaky")

        assert "flaky" in registry


# ── Supervisor agent firing ───────────────────────────────────────────────────

def _make_agent(name: str, role: str = "tester") -> MagicMock:
    agent = MagicMock()
    agent.name = name
    agent.role = role
    agent.run = AsyncMock(return_value="ok")
    return agent


class TestSupervisorMonitor:
    def _make_kazi(self):
        from kazi.brain.graph_builder import GraphBrain
        kazi = MagicMock()
        kazi._brain = MagicMock(spec=GraphBrain)
        return kazi

    @pytest.mark.asyncio
    async def test_success_recorded(self):
        from kazi.agents.supervisor import Supervisor
        monitor = PerformanceMonitor(consecutive_threshold=3)
        riley = _make_agent("Riley")
        crew = Supervisor([riley], kazi=self._make_kazi(), monitor=monitor)

        with patch.object(crew, "_route", new=AsyncMock(return_value=riley)):
            await crew.run("do stuff")

        h = monitor.health("Riley")
        assert h.total_calls == 1
        assert not h.fired

    @pytest.mark.asyncio
    async def test_agent_fired_after_consecutive_failures(self):
        from kazi.agents.supervisor import Supervisor
        monitor = PerformanceMonitor(consecutive_threshold=3)
        riley = _make_agent("Riley")
        riley.run = AsyncMock(side_effect=RuntimeError("agent crashed"))
        jordan = _make_agent("Jordan")
        crew = Supervisor([riley, jordan], kazi=self._make_kazi(), monitor=monitor)

        for _ in range(3):
            with patch.object(crew, "_route", new=AsyncMock(return_value=riley)):
                with pytest.raises(RuntimeError):
                    await crew.run("do stuff")

        assert "Riley" not in {a.name for a in crew.agents}
        assert monitor.is_fired("Riley")

    @pytest.mark.asyncio
    async def test_default_reassigned_when_default_fired(self):
        from kazi.agents.supervisor import Supervisor
        monitor = PerformanceMonitor(consecutive_threshold=2)
        riley = _make_agent("Riley")
        riley.run = AsyncMock(side_effect=RuntimeError("crashed"))
        jordan = _make_agent("Jordan")
        crew = Supervisor([riley, jordan], kazi=self._make_kazi(), monitor=monitor)
        assert crew._default == "Riley"

        for _ in range(2):
            with patch.object(crew, "_route", new=AsyncMock(return_value=riley)):
                with pytest.raises(RuntimeError):
                    await crew.run("task")

        assert crew._default == "Jordan"

    @pytest.mark.asyncio
    async def test_crew_health_and_fired_agents(self):
        from kazi.agents.supervisor import Supervisor
        monitor = PerformanceMonitor(consecutive_threshold=2)
        riley = _make_agent("Riley")
        riley.run = AsyncMock(side_effect=RuntimeError("bad"))
        jordan = _make_agent("Jordan")
        crew = Supervisor([riley, jordan], kazi=self._make_kazi(), monitor=monitor)

        for _ in range(2):
            with patch.object(crew, "_route", new=AsyncMock(return_value=riley)):
                with pytest.raises(RuntimeError):
                    await crew.run("task")

        assert "Riley" in crew.fired_agents()
        assert "Jordan" not in crew.fired_agents()
        health_names = {h.name for h in crew.crew_health()}
        assert "Riley" in health_names

    @pytest.mark.asyncio
    async def test_reinstate_and_add_agent(self):
        from kazi.agents.supervisor import Supervisor
        monitor = PerformanceMonitor(consecutive_threshold=2)
        riley = _make_agent("Riley")
        riley.run = AsyncMock(side_effect=RuntimeError("bad"))
        jordan = _make_agent("Jordan")
        crew = Supervisor([riley, jordan], kazi=self._make_kazi(), monitor=monitor)

        for _ in range(2):
            with patch.object(crew, "_route", new=AsyncMock(return_value=riley)):
                with pytest.raises(RuntimeError):
                    await crew.run("task")

        assert "Riley" not in {a.name for a in crew.agents}

        # Fix Riley and re-add
        riley.run = AsyncMock(return_value="fixed")
        crew.reinstate("Riley")
        crew.add_agent(riley)
        assert "Riley" in {a.name for a in crew.agents}
        assert not monitor.is_fired("Riley")

    def test_no_monitor_returns_empty_health(self):
        from kazi.agents.supervisor import Supervisor
        riley = _make_agent("Riley")
        crew = Supervisor([riley], kazi=self._make_kazi())
        assert crew.crew_health() == []
        assert crew.fired_agents() == []
        assert crew.agent_health("Riley") is None
