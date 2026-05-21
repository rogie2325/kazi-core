"""Tests for tool_adapter — OpenAI, Anthropic, and LangChain schema conversion."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from kazi.core.registry import ToolSource
from kazi.tools.tool_adapter import from_anthropic_schema, from_langchain_tool, from_openai_schema

# ── from_openai_schema ────────────────────────────────────────────────────────

def test_openai_name_and_description():
    schema = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
    tool = from_openai_schema(schema, handler=lambda: None)
    assert tool.name == "get_weather"
    assert tool.description == "Get weather for a city"
    assert tool.source == ToolSource.NATIVE


def test_openai_required_and_optional_params():
    schema = {
        "type": "function",
        "function": {
            "name": "search",
            "description": "",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Max results"},
                },
                "required": ["query"],
            },
        },
    }
    tool = from_openai_schema(schema, handler=lambda q, limit=10: None)
    params = {p.name: p for p in tool.parameters}

    assert params["query"].required is True
    assert params["query"].type == "string"
    assert params["limit"].required is False
    assert params["limit"].type == "integer"


def test_openai_enum_param():
    schema = {
        "type": "function",
        "function": {
            "name": "set_mode",
            "description": "",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "description": "Operating mode",
                        "enum": ["fast", "accurate", "balanced"],
                    }
                },
                "required": ["mode"],
            },
        },
    }
    tool = from_openai_schema(schema, handler=lambda mode: None)
    mode_param = tool.parameters[0]
    assert mode_param.enum == ["fast", "accurate", "balanced"]


def test_openai_flat_schema_without_function_wrapper():
    """Accepts schema where the 'function' key is missing — treats dict itself as the spec."""
    schema = {
        "name": "simple_tool",
        "description": "A simple tool",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }
    tool = from_openai_schema(schema, handler=lambda: None)
    assert tool.name == "simple_tool"


def test_openai_handler_is_attached_and_callable():
    async def my_handler(query: str) -> str:
        return f"result: {query}"

    schema = {
        "function": {
            "name": "search",
            "description": "",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }
    }
    tool = from_openai_schema(schema, handler=my_handler)
    assert tool.handler is my_handler


def test_openai_no_properties_produces_empty_params():
    schema = {
        "function": {
            "name": "ping",
            "description": "No args",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }
    }
    tool = from_openai_schema(schema, handler=lambda: None)
    assert tool.parameters == []


# ── from_anthropic_schema ─────────────────────────────────────────────────────

def test_anthropic_name_and_description():
    schema = {
        "name": "summarise",
        "description": "Summarise a document",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    }
    tool = from_anthropic_schema(schema, handler=lambda: None)
    assert tool.name == "summarise"
    assert tool.description == "Summarise a document"
    assert tool.source == ToolSource.NATIVE


def test_anthropic_required_and_optional_params():
    schema = {
        "name": "translate",
        "description": "",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to translate"},
                "target_lang": {"type": "string", "description": "Target language"},
            },
            "required": ["text"],
        },
    }
    tool = from_anthropic_schema(schema, handler=lambda text, target_lang="en": None)
    params = {p.name: p for p in tool.parameters}

    assert params["text"].required is True
    assert params["target_lang"].required is False


def test_anthropic_enum_param():
    schema = {
        "name": "classify",
        "description": "",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Category",
                    "enum": ["A", "B", "C"],
                }
            },
            "required": ["category"],
        },
    }
    tool = from_anthropic_schema(schema, handler=lambda category: None)
    assert tool.parameters[0].enum == ["A", "B", "C"]


def test_anthropic_missing_input_schema_gives_no_params():
    schema = {"name": "no_schema", "description": ""}
    tool = from_anthropic_schema(schema, handler=lambda: None)
    assert tool.parameters == []


def test_anthropic_handler_attached():
    handler = AsyncMock(return_value="ok")
    schema = {
        "name": "tool",
        "description": "",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    }
    tool = from_anthropic_schema(schema, handler=handler)
    assert tool.handler is handler


# ── from_langchain_tool ───────────────────────────────────────────────────────

def _make_lc_tool(name: str, description: str, schema: dict | None = None, has_arun: bool = True):
    """Build a minimal duck-typed LangChain tool mock."""
    mock = MagicMock()
    mock.name = name
    mock.description = description

    if schema:
        args_schema = MagicMock()
        args_schema.model_json_schema.return_value = schema
        mock.args_schema = args_schema
    else:
        mock.args_schema = None

    if has_arun:
        mock.arun = AsyncMock(return_value="async result")
    else:
        del mock.arun  # simulate tool without arun
        mock.run = MagicMock(return_value="sync result")

    return mock


def test_langchain_name_and_description():
    lc = _make_lc_tool("web_search", "Search the web")
    tool = from_langchain_tool(lc)
    assert tool.name == "web_search"
    assert tool.description == "Search the web"
    assert tool.source == ToolSource.NATIVE
    assert tool.metadata["source_framework"] == "langchain"


def test_langchain_params_extracted_from_args_schema():
    json_schema = {
        "properties": {
            "query": {"type": "string", "description": "The search query"},
            "max_results": {"type": "integer", "description": "Limit"},
        },
        "required": ["query"],
    }
    lc = _make_lc_tool("search", "Search", schema=json_schema)
    tool = from_langchain_tool(lc)
    params = {p.name: p for p in tool.parameters}

    assert "query" in params
    assert params["query"].required is True
    assert params["query"].type == "string"
    assert params["max_results"].required is False


def test_langchain_no_args_schema_gives_empty_params():
    lc = _make_lc_tool("ping", "Ping", schema=None)
    tool = from_langchain_tool(lc)
    assert tool.parameters == []


@pytest.mark.asyncio
async def test_langchain_handler_calls_arun():
    lc = _make_lc_tool("tool", "desc", has_arun=True)
    tool = from_langchain_tool(lc)
    result = await tool.handler(query="test")
    lc.arun.assert_called_once_with(query="test")
    assert result == "async result"


@pytest.mark.asyncio
async def test_langchain_handler_falls_back_to_run_when_arun_not_async():
    """When arun is not a coroutine function, the adapter calls run() instead."""
    lc = MagicMock()
    lc.name = "sync_tool"
    lc.description = "Sync only"
    lc.args_schema = None
    # arun is a plain lambda — iscoroutinefunction returns False → adapter calls run()
    lc.arun = lambda **kw: "should not be returned"
    lc.run = MagicMock(return_value="sync run result")

    tool = from_langchain_tool(lc)
    result = await tool.handler(query="hello")
    lc.run.assert_called_once_with(query="hello")
    assert result == "sync run result"


def test_openai_roundtrip_through_registry():
    """A ToolDefinition built from an OpenAI schema can be registered and retrieved."""
    from kazi.core.registry import ToolRegistry

    schema = {
        "function": {
            "name": "do_thing",
            "description": "Does a thing",
            "parameters": {
                "type": "object",
                "properties": {"x": {"type": "string", "description": "Input"}},
                "required": ["x"],
            },
        }
    }
    registry = ToolRegistry()
    tool = from_openai_schema(schema, handler=lambda x: x)
    registry.register(tool)
    retrieved = registry.get("do_thing")
    assert retrieved.name == "do_thing"
    assert retrieved.parameters[0].name == "x"


def test_anthropic_roundtrip_through_registry():
    from kazi.core.registry import ToolRegistry

    schema = {
        "name": "anthropic_tool",
        "description": "Anthropic flavoured",
        "input_schema": {
            "type": "object",
            "properties": {"prompt": {"type": "string", "description": "Prompt"}},
            "required": ["prompt"],
        },
    }
    registry = ToolRegistry()
    tool = from_anthropic_schema(schema, handler=lambda prompt: prompt)
    registry.register(tool)
    assert registry.get("anthropic_tool").parameters[0].name == "prompt"
