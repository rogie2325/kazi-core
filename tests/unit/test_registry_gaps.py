"""
Covers the uncovered lines in kazi.core.registry:
  - enum params in schema export
  - Optional[X] type inference in register_function
  - unregister unknown tool
  - list_tools by category
  - get_schemas unknown format
  - sync and async execute paths
  - ToolExecutionError on handler failure
  - __len__ and __contains__
"""
from unittest.mock import AsyncMock

import pytest

from kazi.core.exceptions import (
    ToolConflictError,
    ToolExecutionError,
    ToolNotFoundError,
)
from kazi.core.registry import (
    ToolDefinition,
    ToolParameter,
    ToolRegistry,
    ToolSource,
)


def _simple_tool(name: str) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description="A test tool",
        parameters=[],
        source=ToolSource.NATIVE,
        handler=AsyncMock(return_value="ok"),
    )


# ── Enum param in schema export ───────────────────────────────────────────────

def test_openai_schema_includes_enum():
    tool = ToolDefinition(
        name="classify",
        description="Classify input",
        parameters=[
            ToolParameter(
                name="category",
                type="string",
                description="Category",
                required=True,
                enum=["A", "B", "C"],
            )
        ],
        source=ToolSource.NATIVE,
    )
    schema = tool.to_openai_schema()
    props = schema["function"]["parameters"]["properties"]
    assert props["category"]["enum"] == ["A", "B", "C"]


def test_anthropic_schema_includes_enum():
    tool = ToolDefinition(
        name="classify",
        description="Classify input",
        parameters=[
            ToolParameter(
                name="category",
                type="string",
                description="Category",
                required=True,
                enum=["X", "Y"],
            )
        ],
        source=ToolSource.NATIVE,
    )
    schema = tool.to_anthropic_schema()
    props = schema["input_schema"]["properties"]
    assert props["category"]["enum"] == ["X", "Y"]


def test_schema_export_omits_enum_when_none():
    """Parameters without enum must not get an 'enum' key in the schema."""
    tool = ToolDefinition(
        name="search",
        description="",
        parameters=[ToolParameter(name="q", type="string", description="", required=True)],
        source=ToolSource.NATIVE,
    )
    schema = tool.to_openai_schema()
    props = schema["function"]["parameters"]["properties"]
    assert "enum" not in props["q"]


# ── register_function — type inference ───────────────────────────────────────

def test_register_function_infers_string_type():
    registry = ToolRegistry()

    def my_tool(text: str) -> str:
        return text

    td = registry.register_function(my_tool)
    assert td.parameters[0].type == "string"


def test_register_function_infers_integer_type():
    registry = ToolRegistry()

    def counter(n: int) -> int:
        return n

    td = registry.register_function(counter)
    assert td.parameters[0].type == "integer"


def test_register_function_infers_boolean_type():
    registry = ToolRegistry()

    def toggle(enabled: bool) -> None:
        pass

    td = registry.register_function(toggle)
    assert td.parameters[0].type == "boolean"


def test_register_function_optional_param_is_not_required():
    registry = ToolRegistry()

    def with_optional(text: str, limit: int | None = None) -> str:
        return text

    td = registry.register_function(with_optional)
    params = {p.name: p for p in td.parameters}
    assert params["text"].required is True
    assert params["limit"].required is False


def test_register_function_optional_extracts_inner_type():
    """Optional[int] should produce type='integer', not type='string'."""
    registry = ToolRegistry()

    def func(n: int | None = None) -> None:
        pass

    td = registry.register_function(func)
    assert td.parameters[0].type == "integer"


def test_register_function_unannotated_param_defaults_to_string():
    registry = ToolRegistry()

    def no_hints(x) -> None:
        pass

    td = registry.register_function(no_hints)
    assert td.parameters[0].type == "string"


def test_register_function_uses_docstring_as_description():
    registry = ToolRegistry()

    def documented(x: str) -> str:
        """Return x uppercased."""
        return x.upper()

    td = registry.register_function(documented)
    assert "uppercased" in td.description


def test_register_function_uses_explicit_description():
    registry = ToolRegistry()

    def undoc(x: str) -> str:
        pass

    td = registry.register_function(undoc, description="Custom description")
    assert td.description == "Custom description"


def test_register_function_uses_explicit_name():
    registry = ToolRegistry()

    def internal_name(x: str) -> str:
        pass

    td = registry.register_function(internal_name, name="public_name")
    assert td.name == "public_name"
    assert "public_name" in registry


# ── unregister ────────────────────────────────────────────────────────────────

def test_unregister_removes_tool():
    registry = ToolRegistry()
    registry.register(_simple_tool("to_remove"))
    registry.unregister("to_remove")
    assert "to_remove" not in registry


def test_unregister_raises_for_unknown_tool():
    registry = ToolRegistry()
    with pytest.raises(ToolNotFoundError, match="not found"):
        registry.unregister("nonexistent")


def test_unregister_removes_from_category():
    registry = ToolRegistry()
    registry.register(_simple_tool("t"), category="mycat")
    registry.unregister("t")
    assert registry.list_tools(category="mycat") == []


# ── list_tools by category ────────────────────────────────────────────────────

def test_list_tools_by_category_returns_only_that_category():
    registry = ToolRegistry()
    registry.register(_simple_tool("a"), category="cat1")
    registry.register(_simple_tool("b"), category="cat2")
    registry.register(_simple_tool("c"), category="cat1")

    cat1 = registry.list_tools(category="cat1")
    assert len(cat1) == 2
    names = {t.name for t in cat1}
    assert names == {"a", "c"}


def test_list_tools_unknown_category_returns_empty():
    registry = ToolRegistry()
    registry.register(_simple_tool("x"), category="real")
    assert registry.list_tools(category="nonexistent") == []


def test_list_tools_no_category_returns_all():
    registry = ToolRegistry()
    registry.register(_simple_tool("a"), category="cat1")
    registry.register(_simple_tool("b"), category="cat2")
    all_tools = registry.list_tools()
    assert len(all_tools) == 2


# ── get_schemas ───────────────────────────────────────────────────────────────

def test_get_schemas_openai_format():
    registry = ToolRegistry()
    registry.register(_simple_tool("t"))
    schemas = registry.get_schemas(fmt="openai")
    assert len(schemas) == 1
    assert schemas[0]["type"] == "function"


def test_get_schemas_anthropic_format():
    registry = ToolRegistry()
    registry.register(_simple_tool("t"))
    schemas = registry.get_schemas(fmt="anthropic")
    assert len(schemas) == 1
    assert "input_schema" in schemas[0]


def test_get_schemas_unknown_format_raises():
    registry = ToolRegistry()
    with pytest.raises(ValueError, match="Unknown schema format"):
        registry.get_schemas(fmt="unknown_format")


# ── execute ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_execute_async_handler():
    registry = ToolRegistry()
    handler = AsyncMock(return_value="async result")
    tool = ToolDefinition(
        name="async_tool",
        description="",
        parameters=[],
        source=ToolSource.NATIVE,
        handler=handler,
    )
    registry.register(tool)
    result = await registry.execute("async_tool")
    assert result == "async result"
    handler.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_sync_handler():
    registry = ToolRegistry()

    def sync_handler(x: int) -> int:
        return x * 2

    tool = ToolDefinition(
        name="sync_tool",
        description="",
        parameters=[ToolParameter(name="x", type="integer", description="", required=True)],
        source=ToolSource.NATIVE,
        handler=sync_handler,
    )
    registry.register(tool)
    result = await registry.execute("sync_tool", x=5)
    assert result == 10


@pytest.mark.asyncio
async def test_execute_raises_tool_not_found():
    registry = ToolRegistry()
    with pytest.raises(ToolNotFoundError):
        await registry.execute("ghost_tool")


@pytest.mark.asyncio
async def test_execute_wraps_handler_exception_in_tool_execution_error():
    registry = ToolRegistry()

    async def broken(**kwargs):
        raise ValueError("handler exploded")

    tool = ToolDefinition(
        name="broken",
        description="",
        parameters=[],
        source=ToolSource.NATIVE,
        handler=broken,
    )
    registry.register(tool)

    with pytest.raises(ToolExecutionError) as exc_info:
        await registry.execute("broken")

    assert exc_info.value.tool_name == "broken"
    assert "handler exploded" in str(exc_info.value)


@pytest.mark.asyncio
async def test_execute_raises_when_no_handler():
    registry = ToolRegistry()
    tool = ToolDefinition(
        name="no_handler",
        description="",
        parameters=[],
        source=ToolSource.NATIVE,
        handler=None,
    )
    registry.register(tool)
    with pytest.raises(ToolExecutionError, match="No handler"):
        await registry.execute("no_handler")


# ── __len__ and __contains__ ──────────────────────────────────────────────────

def test_len_empty_registry():
    assert len(ToolRegistry()) == 0


def test_len_after_registration():
    registry = ToolRegistry()
    registry.register(_simple_tool("a"))
    registry.register(_simple_tool("b"))
    assert len(registry) == 2


def test_len_after_unregister():
    registry = ToolRegistry()
    registry.register(_simple_tool("a"))
    registry.unregister("a")
    assert len(registry) == 0


def test_contains_true():
    registry = ToolRegistry()
    registry.register(_simple_tool("present"))
    assert "present" in registry


def test_contains_false():
    registry = ToolRegistry()
    assert "absent" not in registry


# ── search ────────────────────────────────────────────────────────────────────

def test_search_finds_by_name():
    registry = ToolRegistry()
    registry.register(_simple_tool("database_query"))
    results = registry.search("database")
    assert any(t.name == "database_query" for t in results)


def test_search_finds_by_description():
    registry = ToolRegistry()
    tool = ToolDefinition(
        name="t", description="Searches DuckDuckGo for results",
        parameters=[], source=ToolSource.NATIVE,
    )
    registry.register(tool)
    results = registry.search("duckduckgo")
    assert len(results) == 1


def test_search_is_case_insensitive():
    registry = ToolRegistry()
    registry.register(_simple_tool("WebSearch"))
    assert len(registry.search("websearch")) == 1
    assert len(registry.search("WEBSEARCH")) == 1


def test_search_returns_empty_when_no_match():
    registry = ToolRegistry()
    registry.register(_simple_tool("calculator"))
    assert registry.search("zzz_no_match_zzz") == []


# ── conflict handling ─────────────────────────────────────────────────────────

def test_register_duplicate_name_raises():
    registry = ToolRegistry()
    registry.register(_simple_tool("dup"))
    with pytest.raises(ToolConflictError, match="already registered"):
        registry.register(_simple_tool("dup"))


def test_register_same_name_different_source_still_raises():
    registry = ToolRegistry()
    registry.register(_simple_tool("conflict"))
    mcp_tool = ToolDefinition(
        name="conflict", description="", parameters=[],
        source=ToolSource.MCP, handler=None,
    )
    with pytest.raises(ToolConflictError):
        registry.register(mcp_tool)
