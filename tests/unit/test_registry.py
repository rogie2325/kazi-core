"""Unit tests for the ToolRegistry."""
import pytest

from kazi.core.exceptions import ToolConflictError, ToolExecutionError, ToolNotFoundError
from kazi.core.registry import ToolDefinition, ToolParameter, ToolRegistry, ToolSource


def make_tool(name: str = "my_tool") -> ToolDefinition:
    async def handler(x: str) -> str:
        return f"result: {x}"

    return ToolDefinition(
        name=name,
        description="A test tool",
        parameters=[ToolParameter(name="x", type="string", description="input", required=True)],
        source=ToolSource.NATIVE,
        handler=handler,
    )


def test_register_and_get():
    reg = ToolRegistry()
    tool = make_tool("tool_a")
    reg.register(tool)
    assert reg.get("tool_a") is tool


def test_register_conflict_raises():
    reg = ToolRegistry()
    reg.register(make_tool("dup"))
    with pytest.raises(ToolConflictError):
        reg.register(make_tool("dup"))


def test_get_missing_raises():
    reg = ToolRegistry()
    with pytest.raises(ToolNotFoundError):
        reg.get("nope")


def test_contains():
    reg = ToolRegistry()
    reg.register(make_tool("t"))
    assert "t" in reg
    assert "missing" not in reg


def test_list_tools_by_category():
    reg = ToolRegistry()
    reg.register(make_tool("a"), category="cat_a")
    reg.register(make_tool("b"), category="cat_b")
    assert len(reg.list_tools("cat_a")) == 1
    assert len(reg.list_tools()) == 2


def test_search():
    reg = ToolRegistry()
    reg.register(make_tool("weather_tool"))
    reg.register(make_tool("stock_tool"))
    results = reg.search("weather")
    assert len(results) == 1 and results[0].name == "weather_tool"


def test_register_function():
    reg = ToolRegistry()

    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    tool = reg.register_function(add)
    assert tool.name == "add"
    assert len(tool.parameters) == 2


@pytest.mark.asyncio
async def test_execute():
    reg = ToolRegistry()
    reg.register(make_tool("exec_tool"))
    result = await reg.execute("exec_tool", x="hello")
    assert result == "result: hello"


@pytest.mark.asyncio
async def test_execute_error_wraps():
    reg = ToolRegistry()

    async def bad_handler(**kwargs):
        raise ValueError("boom")

    tool = ToolDefinition(
        name="bad",
        description="fails",
        parameters=[],
        source=ToolSource.NATIVE,
        handler=bad_handler,
    )
    reg.register(tool)
    with pytest.raises(ToolExecutionError):
        await reg.execute("bad")


def test_openai_schema():
    reg = ToolRegistry()
    reg.register(make_tool("schema_tool"))
    schemas = reg.get_schemas(fmt="openai")
    assert len(schemas) == 1
    assert schemas[0]["function"]["name"] == "schema_tool"


def test_anthropic_schema():
    reg = ToolRegistry()
    reg.register(make_tool("anthro_tool"))
    schemas = reg.get_schemas(fmt="anthropic")
    assert schemas[0]["name"] == "anthro_tool"
    assert "input_schema" in schemas[0]


def test_unregister():
    reg = ToolRegistry()
    reg.register(make_tool("rm_me"))
    reg.unregister("rm_me")
    assert "rm_me" not in reg
