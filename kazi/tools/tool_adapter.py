"""
Adapters that convert tool schemas from other frameworks into
Kazi ToolDefinitions so they slot into the unified registry.
"""
from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from kazi.core.registry import ToolDefinition, ToolParameter, ToolSource


def from_openai_schema(schema: dict, handler: Callable) -> ToolDefinition:
    """
    Convert an OpenAI function-calling schema dict into a ToolDefinition.

    schema format: {"type": "function", "function": {"name": ..., ...}}
    """
    fn = schema.get("function", schema)
    props = fn.get("parameters", {}).get("properties", {})
    required = set(fn.get("parameters", {}).get("required", []))
    params = [
        ToolParameter(
            name=k,
            type=v.get("type", "string"),
            description=v.get("description", ""),
            required=k in required,
            enum=v.get("enum"),
        )
        for k, v in props.items()
    ]
    return ToolDefinition(
        name=fn["name"],
        description=fn.get("description", ""),
        parameters=params,
        source=ToolSource.NATIVE,
        handler=handler,
    )


def from_anthropic_schema(schema: dict, handler: Callable) -> ToolDefinition:
    """Convert an Anthropic tool-use schema dict into a ToolDefinition."""
    props = schema.get("input_schema", {}).get("properties", {})
    required = set(schema.get("input_schema", {}).get("required", []))
    params = [
        ToolParameter(
            name=k,
            type=v.get("type", "string"),
            description=v.get("description", ""),
            required=k in required,
            enum=v.get("enum"),
        )
        for k, v in props.items()
    ]
    return ToolDefinition(
        name=schema["name"],
        description=schema.get("description", ""),
        parameters=params,
        source=ToolSource.NATIVE,
        handler=handler,
    )


def from_langchain_tool(lc_tool: Any) -> ToolDefinition:
    """
    Wrap a LangChain BaseTool as a Kazi ToolDefinition.
    Requires langchain-core installed.
    """
    import asyncio

    async def handler(**kwargs):
        if inspect.iscoroutinefunction(lc_tool.arun):
            return await lc_tool.arun(**kwargs)
        return await asyncio.to_thread(lc_tool.run, **kwargs)

    # LangChain tools expose args_schema (a Pydantic model)
    params: list[ToolParameter] = []
    schema = getattr(lc_tool, "args_schema", None)
    if schema:
        json_schema = schema.model_json_schema() if hasattr(schema, "model_json_schema") else {}
        props = json_schema.get("properties", {})
        required = set(json_schema.get("required", []))
        for k, v in props.items():
            params.append(ToolParameter(
                name=k,
                type=v.get("type", "string"),
                description=v.get("description", ""),
                required=k in required,
            ))

    return ToolDefinition(
        name=lc_tool.name,
        description=lc_tool.description or "",
        parameters=params,
        source=ToolSource.NATIVE,
        handler=handler,
        metadata={"source_framework": "langchain"},
    )
