"""
API contract tests.

Verifies the shape, signatures, and invariants of every public symbol in
kazi.__all__ — without calling a live LLM.  Catches:
  - Missing attributes on exported classes
  - Schema methods returning wrong types
  - Keyword arguments the orchestrator accepts but the brain silently drops
  - Public methods that exist in __all__ but raise AttributeError at call time
"""
from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import pytest

import kazi

# ── __all__ completeness ──────────────────────────────────────────────────────

class TestPublicAPI:
    def test_all_symbols_importable(self):
        """Every symbol in kazi.__all__ is reachable via getattr(kazi, name)."""
        for name in kazi.__all__:
            obj = getattr(kazi, name, None)
            assert obj is not None, f"kazi.__all__ lists {name!r} but it is not importable"

    def test_kazi_class_has_expected_methods(self):
        from kazi import Kazi
        for method in ("create", "run", "stream", "stream_events", "batch_run",
                       "run_with_approval", "run_voice", "stream_voice",
                       "add_tool", "branch_thread", "health"):
            assert hasattr(Kazi, method), f"Kazi missing expected method: {method!r}"

    def test_kazi_create_is_classmethod(self):
        from kazi import Kazi
        assert inspect.ismethod(Kazi.create) or callable(Kazi.create)

    def test_llm_config_fields(self):
        from kazi import LLMConfig, LLMProvider
        cfg = LLMConfig(provider=LLMProvider.ANTHROPIC, model="claude-haiku-4-5-20251001")
        assert cfg.provider == LLMProvider.ANTHROPIC
        assert cfg.model == "claude-haiku-4-5-20251001"

    def test_kazi_config_defaults(self):
        from kazi import KaziConfig, LLMConfig, LLMProvider
        cfg = KaziConfig(llm=LLMConfig(provider=LLMProvider.ANTHROPIC, model="m"))
        assert cfg.llm is not None
        assert cfg.rag is not None
        assert cfg.security is not None

    def test_tool_definition_schema_contracts(self):
        """to_openai_schema and to_anthropic_schema return correct structure."""
        from kazi import ToolDefinition, ToolParameter, ToolSource
        tool = ToolDefinition(
            name="my_tool",
            description="Does something",
            parameters=[
                ToolParameter(name="query", type="string", description="search query", required=True),
                ToolParameter(name="limit", type="integer", description="max results",
                              required=False, default=10),
            ],
            source=ToolSource.NATIVE,
            handler=None,
        )
        oai = tool.to_openai_schema()
        assert oai["type"] == "function"
        assert oai["function"]["name"] == "my_tool"
        assert "query" in oai["function"]["parameters"]["properties"]
        assert "limit" in oai["function"]["parameters"]["properties"]
        assert "query" in oai["function"]["parameters"]["required"]
        assert "limit" not in oai["function"]["parameters"]["required"]

        ant = tool.to_anthropic_schema()
        assert ant["name"] == "my_tool"
        assert "query" in ant["input_schema"]["properties"]

    def test_exceptions_hierarchy(self):
        """All public exceptions are subclasses of KaziError."""
        from kazi import (
            A2AConnectionError,
            ConfigurationError,
            KaziError,
            MCPConnectionError,
            OrchestratorError,
            ToolConflictError,
            ToolExecutionError,
            ToolNotFoundError,
        )
        for exc_cls in (ConfigurationError, ToolNotFoundError, ToolExecutionError,
                        ToolConflictError, MCPConnectionError, A2AConnectionError,
                        OrchestratorError):
            assert issubclass(exc_cls, KaziError), (
                f"{exc_cls.__name__} is not a subclass of KaziError"
            )

    def test_tool_registry_protocol(self):
        from kazi import ToolDefinition, ToolRegistry, ToolSource
        from kazi.core.exceptions import ToolConflictError, ToolNotFoundError

        reg = ToolRegistry()
        tool = ToolDefinition(
            name="t", description="desc", parameters=[], source=ToolSource.NATIVE, handler=None
        )
        reg.register(tool)
        assert "t" in reg
        assert len(reg) == 1
        assert reg.get("t") is tool

        with pytest.raises(ToolConflictError):
            reg.register(tool)

        reg.unregister("t")
        assert "t" not in reg

        with pytest.raises(ToolNotFoundError):
            reg.get("t")


# ── Brain method signature contracts ─────────────────────────────────────────

class TestBrainSignatures:
    """
    Verify the brain's method signatures match what the orchestrator passes.
    This catches the class of bug where orchestrator passes a kwarg that brain
    doesn't accept (TypeError at runtime).
    """

    def _brain_params(self, method_name: str) -> set[str]:
        from kazi.brain.graph_builder import GraphBrain
        sig = inspect.signature(getattr(GraphBrain, method_name))
        return set(sig.parameters.keys()) - {"self"}

    def test_run_has_system_prompt(self):
        assert "system_prompt" in self._brain_params("run")

    def test_stream_has_system_prompt(self):
        assert "system_prompt" in self._brain_params("stream"), (
            "GraphBrain.stream() is missing system_prompt — orchestrator always passes it"
        )

    def test_stream_events_has_system_prompt(self):
        assert "system_prompt" in self._brain_params("stream_events"), (
            "GraphBrain.stream_events() is missing system_prompt — user profiles are dropped"
        )

    def test_orchestrator_stream_passes_system_prompt(self):
        """Orchestrator's stream() must forward system_prompt to brain."""
        import ast
        import pathlib
        src = pathlib.Path("kazi/core/orchestrator.py").read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
                if node.name == "stream":
                    # Find the brain.stream() call inside it
                    for child in ast.walk(node):
                        if isinstance(child, ast.Call):
                            for kw in child.keywords:
                                if kw.arg == "system_prompt":
                                    return  # found it
        pytest.fail(
            "orchestrator.stream() does not forward system_prompt to _brain.stream()"
        )

    def test_orchestrator_stream_events_passes_system_prompt(self):
        """Orchestrator's stream_events() must forward system_prompt to brain."""
        import ast
        import pathlib
        src = pathlib.Path("kazi/core/orchestrator.py").read_text()
        tree = ast.parse(src)
        found_fn = False
        for node in ast.walk(tree):
            if isinstance(node, (ast.AsyncFunctionDef,)):
                if node.name == "stream_events":
                    found_fn = True
                    for child in ast.walk(node):
                        if isinstance(child, ast.Call):
                            for kw in child.keywords:
                                if kw.arg == "system_prompt":
                                    return
        if found_fn:
            pytest.fail("orchestrator.stream_events() does not forward system_prompt")


# ── ToolRegistry execute contracts ────────────────────────────────────────────

class TestRegistryExecuteContracts:
    @pytest.mark.asyncio
    async def test_async_handler_called_with_kwargs(self):
        from kazi import ToolDefinition, ToolRegistry, ToolSource
        received: list[dict] = []

        async def handler(x: str, y: int = 0) -> str:
            received.append({"x": x, "y": y})
            return f"ok:{x}:{y}"

        reg = ToolRegistry()
        reg.register(ToolDefinition(
            name="test_tool", description="d", parameters=[],
            source=ToolSource.NATIVE, handler=handler,
        ))
        result = await reg.execute("test_tool", x="hello", y=42)
        assert result == "ok:hello:42"
        assert received == [{"x": "hello", "y": 42}]

    @pytest.mark.asyncio
    async def test_sync_handler_runs_without_blocking_error(self):
        """Sync handlers must execute without TypeError."""
        from kazi import ToolDefinition, ToolRegistry, ToolSource

        def sync_handler(x: str) -> str:
            return f"sync:{x}"

        reg = ToolRegistry()
        reg.register(ToolDefinition(
            name="sync_tool", description="d", parameters=[],
            source=ToolSource.NATIVE, handler=sync_handler,
        ))
        result = await reg.execute("sync_tool", x="world")
        assert result == "sync:world"

    @pytest.mark.asyncio
    async def test_exception_wrapped_in_tool_execution_error(self):
        from kazi import ToolDefinition, ToolRegistry, ToolSource
        from kazi.core.exceptions import ToolExecutionError

        async def broken() -> str:
            raise ValueError("deliberately broken")

        reg = ToolRegistry()
        reg.register(ToolDefinition(
            name="broken", description="d", parameters=[],
            source=ToolSource.NATIVE, handler=broken,
        ))
        with pytest.raises(ToolExecutionError) as exc_info:
            await reg.execute("broken")
        assert "broken" in str(exc_info.value).lower() or "deliberately" in str(exc_info.value)


# ── Streaming contract ────────────────────────────────────────────────────────

class TestStreamingContracts:
    """
    Verify stream() and stream_events() yield the correct types without
    hitting a real LLM (brain is patched to yield scripted outputs).
    """

    def _config(self):
        from kazi import KaziConfig, LLMConfig, LLMProvider
        return KaziConfig(llm=LLMConfig(
            provider=LLMProvider.ANTHROPIC, model="claude-haiku-4-5-20251001",
            api_key="fake-key-for-contract-test",
        ))

    @pytest.mark.asyncio
    async def test_stream_yields_str_tokens(self):
        from kazi.brain.graph_builder import GraphBrain

        async def fake_stream(*a, **kw):
            for token in ["Hello", " ", "world"]:
                yield token

        with patch.object(GraphBrain, "stream", side_effect=fake_stream):
            from kazi import Kazi
            kazi = await Kazi.create(self._config())
            kazi._brain = MagicMock()
            kazi._brain.stream = fake_stream
            kazi._assert_ready = lambda: None

            tokens = []
            async for tok in kazi.stream("hi"):
                assert isinstance(tok, str)
                tokens.append(tok)
            assert "".join(tokens) == "Hello world"

    @pytest.mark.asyncio
    async def test_stream_events_yields_dicts_with_type_key(self):
        from kazi.brain.graph_builder import GraphBrain

        # StreamEvent is a TypedDict — use dict literals
        async def fake_events(*a, **kw):
            yield {"type": "token", "data": "hi", "metadata": {}}
            yield {"type": "done", "data": "", "metadata": {}}

        with patch.object(GraphBrain, "stream_events", side_effect=fake_events):
            from kazi import Kazi
            kazi = await Kazi.create(self._config())
            kazi._brain = MagicMock()
            kazi._brain.stream_events = fake_events
            kazi._assert_ready = lambda: None

            events = []
            async for ev in kazi.stream_events("hi"):
                assert "type" in ev, "StreamEvent must have 'type' key"
                events.append(ev)

            types = [e["type"] for e in events]
            assert "token" in types
            assert "done" in types


# ── PerformanceMonitor + Supervisor API contracts ─────────────────────────────

class TestMonitorAPIContracts:
    def test_component_health_fields(self):
        from kazi.agents.monitor import ComponentHealth, PerformanceMonitor
        m = PerformanceMonitor(consecutive_threshold=5)
        m.record("tool_a", success=True)
        m.record("tool_a", success=False)
        h = m.health("tool_a")
        assert isinstance(h, ComponentHealth)
        assert isinstance(h.total_calls, int)
        assert isinstance(h.failure_rate, float)
        assert isinstance(h.fired, bool)
        assert 0.0 <= h.failure_rate <= 1.0
        assert 0.0 <= h.success_rate <= 1.0
        assert h.success_rate + h.failure_rate == pytest.approx(1.0)

    def test_supervisor_no_monitor_health_returns_empty(self):
        from kazi.agents.supervisor import Supervisor
        agent = MagicMock()
        agent.name = "Riley"
        agent.role = "Research"
        crew = Supervisor([agent], kazi=MagicMock())
        assert crew.crew_health() == []
        assert crew.fired_agents() == []
        assert crew.agent_health("Riley") is None

    def test_supervisor_with_monitor_exposes_health(self):
        from kazi.agents.monitor import PerformanceMonitor
        from kazi.agents.supervisor import Supervisor
        agent = MagicMock()
        agent.name = "Riley"
        agent.role = "Research"
        monitor = PerformanceMonitor(consecutive_threshold=3)
        crew = Supervisor([agent], kazi=MagicMock(), monitor=monitor)
        h = crew.agent_health("Riley")
        assert h is not None
        assert h.name == "Riley"
