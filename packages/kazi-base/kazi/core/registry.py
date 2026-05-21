from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from inspect import iscoroutinefunction
from enum import Enum
from typing import Any, Callable, Optional

from kazi.core.exceptions import ToolConflictError, ToolNotFoundError, ToolExecutionError


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
    enum: Optional[list] = None


@dataclass
class ToolDefinition:
    """Universal tool definition normalised across all sources."""

    name: str
    description: str
    parameters: list[ToolParameter]
    source: ToolSource
    handler: Optional[Callable] = None
    mcp_server: Optional[str] = None
    a2a_agent_url: Optional[str] = None
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
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._categories: dict[str, list[str]] = {}

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
        name: Optional[str] = None,
        description: Optional[str] = None,
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
            # Handle Optional[X] → X
            origin = getattr(raw_type, "__origin__", None)
            if origin is type(None):
                raw_type = str
            args = getattr(raw_type, "__args__", ())
            if origin is not None and args:
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

    def list_tools(self, category: Optional[str] = None) -> list[ToolDefinition]:
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

    def get_schemas(self, fmt: str = "openai") -> list[dict]:
        if fmt == "openai":
            return [t.to_openai_schema() for t in self._tools.values()]
        if fmt == "anthropic":
            return [t.to_anthropic_schema() for t in self._tools.values()]
        raise ValueError(f"Unknown schema format: {fmt!r}")

    # ── execution ─────────────────────────────────────────────────────────

    async def execute(self, tool_name: str, **kwargs: Any) -> Any:
        """Execute any registered tool regardless of source."""
        tool = self.get(tool_name)
        if tool.handler is None:
            raise ToolExecutionError(tool_name, RuntimeError("No handler attached"))
        try:
            if iscoroutinefunction(tool.handler):
                return await tool.handler(**kwargs)
            return tool.handler(**kwargs)
        except Exception as exc:
            raise ToolExecutionError(tool_name, exc) from exc

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
