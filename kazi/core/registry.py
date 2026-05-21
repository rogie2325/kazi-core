from __future__ import annotations

import inspect
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from inspect import iscoroutinefunction
from typing import Any

from kazi.core.exceptions import ToolConflictError, ToolExecutionError, ToolNotFoundError

# Imported lazily to avoid a circular import at module load time.
# PerformanceMonitor is optional; ToolRegistry works fine without it.
_PerformanceMonitor = None


class ToolSource(Enum):
    NATIVE = "native"
    MCP = "mcp"
    RAG = "rag"
    A2A = "a2a"


@dataclass
class ToolParameter:
    name: str
    type: str  # JSON Schema type string
    description: str
    required: bool = True
    default: Any = None
    enum: list | None = None


@dataclass
class ToolDefinition:
    """Universal tool definition normalised across all sources."""

    name: str
    description: str
    parameters: list[ToolParameter]
    source: ToolSource
    handler: Callable | None = None
    mcp_server: str | None = None
    a2a_agent_url: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_openai_schema(self) -> dict:
        properties: dict[str, Any] = {}
        required: list[str] = []
        for p in self.parameters:
            prop: dict[str, Any] = {"type": p.type, "description": p.description}
            if p.enum:
                prop["enum"] = p.enum
            properties[p.name] = prop
            if p.required:
                required.append(p.name)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def to_anthropic_schema(self) -> dict:
        properties: dict[str, Any] = {}
        required: list[str] = []
        for p in self.parameters:
            prop: dict[str, Any] = {"type": p.type, "description": p.description}
            if p.enum:
                prop["enum"] = p.enum
            properties[p.name] = prop
            if p.required:
                required.append(p.name)
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


_PYTHON_TO_JSON: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


class ToolRegistry:
    """
    Central registry normalising tools from every source (native, MCP, RAG, A2A)
    into a single searchable catalogue.

    Optionally accepts a PerformanceMonitor to automatically remove tools that
    exceed failure thresholds::

        from kazi.agents.monitor import PerformanceMonitor
        monitor = PerformanceMonitor(consecutive_threshold=5)
        registry = ToolRegistry(monitor=monitor)
    """

    def __init__(self, monitor=None) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._categories: dict[str, list[str]] = {}
        self._monitor = monitor

    # ── registration ──────────────────────────────────────────────────────

    def register(self, tool: ToolDefinition, category: str = "general") -> None:
        if tool.name in self._tools:
            raise ToolConflictError(
                f"Tool '{tool.name}' already registered "
                f"(source: {self._tools[tool.name].source.value})"
            )
        self._tools[tool.name] = tool
        self._categories.setdefault(category, []).append(tool.name)

    def register_function(
        self,
        func: Callable,
        name: str | None = None,
        description: str | None = None,
        category: str = "general",
    ) -> ToolDefinition:
        """Register a plain Python function as a NATIVE tool."""
        tool_name = name or func.__name__
        tool_desc = description or (inspect.getdoc(func) or f"Executes {tool_name}")

        sig = inspect.signature(func)
        hints = func.__annotations__
        params: list[ToolParameter] = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue
            raw_type = hints.get(param_name, str)
            # Handle Optional[X] / Union[X, None] / X | None → X
            # Note: PEP 604 unions (`int | None`) are types.UnionType, which has
            # no __origin__ before Python 3.14, so detect them explicitly.
            origin = getattr(raw_type, "__origin__", None)
            args = getattr(raw_type, "__args__", ())
            if args and (origin is not None or isinstance(raw_type, types.UnionType)):
                raw_type = next((a for a in args if a is not type(None)), str)

            json_type = _PYTHON_TO_JSON.get(raw_type, "string")
            has_default = param.default is not inspect.Parameter.empty
            params.append(
                ToolParameter(
                    name=param_name,
                    type=json_type,
                    description=f"Parameter: {param_name}",
                    required=not has_default,
                    default=param.default if has_default else None,
                )
            )

        tool = ToolDefinition(
            name=tool_name,
            description=tool_desc,
            parameters=params,
            source=ToolSource.NATIVE,
            handler=func,
        )
        self.register(tool, category)
        return tool

    def unregister(self, name: str) -> None:
        if name not in self._tools:
            raise ToolNotFoundError(f"Tool '{name}' not found")
        del self._tools[name]
        for cat_tools in self._categories.values():
            if name in cat_tools:
                cat_tools.remove(name)

    # ── lookup ────────────────────────────────────────────────────────────

    def get(self, name: str) -> ToolDefinition:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolNotFoundError(f"Tool '{name}' not found in registry")

    def list_tools(self, category: str | None = None) -> list[ToolDefinition]:
        if category:
            return [self._tools[n] for n in self._categories.get(category, [])]
        return list(self._tools.values())

    def list_categories(self) -> list[str]:
        return list(self._categories.keys())

    def search(self, query: str) -> list[ToolDefinition]:
        q = query.lower()
        return [
            t for t in self._tools.values()
            if q in t.name.lower() or q in t.description.lower()
        ]

    # ── schema export ─────────────────────────────────────────────────────

    def get_schemas(self, fmt: str = "openai", names: set | None = None) -> list[dict]:
        tools = self._tools.values() if names is None else (
            t for t in self._tools.values() if t.name in names
        )
        if fmt == "openai":
            return [t.to_openai_schema() for t in tools]
        if fmt == "anthropic":
            return [t.to_anthropic_schema() for t in tools]
        raise ValueError(f"Unknown schema format: {fmt!r}")

    # ── execution ─────────────────────────────────────────────────────────

    async def execute(self, tool_name: str, **kwargs: Any) -> Any:
        """
        Execute any registered tool regardless of source.

        If a PerformanceMonitor is attached, every call outcome is recorded.
        When a tool is fired the monitor auto-unregisters it so the LLM stops
        receiving it in subsequent schema exports.

        Audit + shadow
        --------------
        When an ``AuditRecorder`` is bound via ``kazi.core.audit.run_context``
        (the orchestrator does this automatically for ``audit=True`` runs),
        every call is recorded with its name, args, result, duration, and status.

        When the run is in shadow mode (``shadow=True``), the real handler is
        bypassed and a deterministic stub result is returned to the agent.
        The stub still flows through the LLM as a tool result, so the agent's
        downstream reasoning is preserved — but no side effects occur.
        """
        import time as _time

        from kazi.core.audit import get_recorder, is_shadow
        from kazi.utils.metrics import record_tool_call
        from kazi.utils.telemetry import span

        tool = self.get(tool_name)
        if tool.handler is None:
            raise ToolExecutionError(tool_name, RuntimeError("No handler attached"))

        recorder = get_recorder()
        shadow = is_shadow()
        start = _time.monotonic()

        with span("kazi.tool.execute", {"tool.name": tool_name, "tool.source": tool.source.value}):
            # ── Shadow path: skip real execution, return deterministic stub ──
            if shadow:
                stub = (
                    f"[SHADOW] {tool_name} would have been called with "
                    f"{dict(kwargs)!r}. No side effects executed."
                )
                duration_ms = (_time.monotonic() - start) * 1000
                if recorder is not None:
                    recorder.record_tool_call(
                        name=tool_name,
                        args=kwargs,
                        result=stub,
                        duration_ms=duration_ms,
                        status="shadow",
                    )
                record_tool_call(tool_name, success=True)
                return stub

            # ── Live path ────────────────────────────────────────────────────
            try:
                if iscoroutinefunction(tool.handler):
                    result = await tool.handler(**kwargs)
                else:
                    result = tool.handler(**kwargs)
                duration_ms = (_time.monotonic() - start) * 1000
                if self._monitor is not None:
                    self._monitor.record(tool_name, success=True)
                if recorder is not None:
                    recorder.record_tool_call(
                        name=tool_name,
                        args=kwargs,
                        result=str(result)[:1000],
                        duration_ms=duration_ms,
                        status="ok",
                    )
                record_tool_call(tool_name, success=True)
                return result
            except Exception as exc:
                duration_ms = (_time.monotonic() - start) * 1000
                if recorder is not None:
                    recorder.record_tool_call(
                        name=tool_name,
                        args=kwargs,
                        result=None,
                        duration_ms=duration_ms,
                        status="error",
                        error=str(exc)[:500],
                    )
                record_tool_call(tool_name, success=False)
                if self._monitor is not None:
                    fired = self._monitor.record(tool_name, success=False)
                    if fired and tool_name in self._tools:
                        import logging as _logging
                        _logging.getLogger(__name__).warning(
                            "ToolRegistry: auto-removing fired tool %r", tool_name
                        )
                        self.unregister(tool_name)
                raise ToolExecutionError(tool_name, exc) from exc

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
