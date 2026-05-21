"""
Real MCP integration tests — spawns an actual stdio MCP server subprocess.

Uses tests/fixtures/mcp_echo_server.py as the server, which exposes
two tools (echo, add). Tests cover full connection setup, tool discovery,
execution, security policy filtering, and retry behaviour.
"""
import sys
from pathlib import Path

import pytest

from kazi.core.config import MCPConfig
from kazi.core.registry import ToolRegistry
from kazi.core.security import MCPSecurityPolicy
from kazi.tools.mcp_client import MCPBridge

FIXTURE_SERVER = str(Path(__file__).parent.parent / "fixtures" / "mcp_echo_server.py")
SERVER_CMD = f"{sys.executable} {FIXTURE_SERVER}"


async def _connected_bridge(security=None, **config_kwargs) -> MCPBridge:
    config = MCPConfig(servers={"echo": SERVER_CMD}, **config_kwargs)
    registry = ToolRegistry()
    bridge = MCPBridge(config, registry, security=security)
    await bridge.connect_all()
    return bridge


# ── Connection and tool discovery ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_connect_registers_tools():
    bridge = await _connected_bridge()
    try:
        assert "echo__echo" in bridge.registry
        assert "echo__add" in bridge.registry
    finally:
        await bridge.disconnect_all()


@pytest.mark.asyncio
async def test_tool_parameters_correctly_parsed():
    bridge = await _connected_bridge()
    try:
        echo_tool = bridge.registry.get("echo__echo")
        assert echo_tool.parameters[0].name == "text"
        assert echo_tool.parameters[0].type == "string"
        assert echo_tool.parameters[0].required is True

        add_tool = bridge.registry.get("echo__add")
        param_names = {p.name for p in add_tool.parameters}
        assert param_names == {"a", "b"}
    finally:
        await bridge.disconnect_all()


@pytest.mark.asyncio
async def test_tool_source_is_mcp():
    from kazi.core.registry import ToolSource
    bridge = await _connected_bridge()
    try:
        assert bridge.registry.get("echo__echo").source == ToolSource.MCP
    finally:
        await bridge.disconnect_all()


@pytest.mark.asyncio
async def test_tools_for_server_lists_correctly():
    bridge = await _connected_bridge()
    try:
        tools = bridge.tools_for_server("echo")
        assert "echo__echo" in tools
        assert "echo__add" in tools
    finally:
        await bridge.disconnect_all()


# ── Tool execution ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_execute_echo_tool():
    bridge = await _connected_bridge()
    try:
        result = await bridge.execute_tool("echo", "echo", {"text": "hello from kazi"})
        assert result == "hello from kazi"
    finally:
        await bridge.disconnect_all()


@pytest.mark.asyncio
async def test_execute_add_tool():
    bridge = await _connected_bridge()
    try:
        result = await bridge.execute_tool("echo", "add", {"a": 17, "b": 25})
        assert result.strip() == "42"
    finally:
        await bridge.disconnect_all()


@pytest.mark.asyncio
async def test_execute_via_registered_handler():
    """Tool handler registered in the registry must route back through the bridge."""
    bridge = await _connected_bridge()
    try:
        result = await bridge.registry.execute("echo__echo", text="via handler")
        assert result == "via handler"
    finally:
        await bridge.disconnect_all()


@pytest.mark.asyncio
async def test_execute_add_via_registry():
    bridge = await _connected_bridge()
    try:
        result = await bridge.registry.execute("echo__add", a=100, b=23)
        assert result.strip() == "123"
    finally:
        await bridge.disconnect_all()


@pytest.mark.asyncio
async def test_echo_preserves_special_characters():
    bridge = await _connected_bridge()
    try:
        payload = "Hello\nWorld\t<xml>& more"
        result = await bridge.execute_tool("echo", "echo", {"text": payload})
        assert result == payload
    finally:
        await bridge.disconnect_all()


@pytest.mark.asyncio
async def test_echo_handles_unicode():
    bridge = await _connected_bridge()
    try:
        payload = "日本語テスト 🚀"
        result = await bridge.execute_tool("echo", "echo", {"text": payload})
        assert result == payload
    finally:
        await bridge.disconnect_all()


@pytest.mark.asyncio
async def test_echo_handles_empty_string():
    bridge = await _connected_bridge()
    try:
        result = await bridge.execute_tool("echo", "echo", {"text": ""})
        assert isinstance(result, str)
    finally:
        await bridge.disconnect_all()


# ── Security policy ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_denylist_blocks_tool_at_registration():
    policy = MCPSecurityPolicy(denylist=["echo__echo"])
    bridge = await _connected_bridge(security=policy)
    try:
        assert "echo__echo" not in bridge.registry
        assert "echo__add" in bridge.registry  # not denied
    finally:
        await bridge.disconnect_all()


@pytest.mark.asyncio
async def test_allowlist_permits_only_listed_tools():
    policy = MCPSecurityPolicy(allowlist=["echo__echo"])
    bridge = await _connected_bridge(security=policy)
    try:
        assert "echo__echo" in bridge.registry
        assert "echo__add" not in bridge.registry
    finally:
        await bridge.disconnect_all()


@pytest.mark.asyncio
async def test_allowlist_glob_pattern():
    policy = MCPSecurityPolicy(allowlist=["echo__*"])
    bridge = await _connected_bridge(security=policy)
    try:
        assert "echo__echo" in bridge.registry
        assert "echo__add" in bridge.registry
    finally:
        await bridge.disconnect_all()


@pytest.mark.asyncio
async def test_denylist_wildcard_blocks_all_tools():
    policy = MCPSecurityPolicy(denylist=["echo__*"])
    bridge = await _connected_bridge(security=policy)
    try:
        assert len(bridge.registry.list_tools(category="mcp_echo")) == 0
    finally:
        await bridge.disconnect_all()


# ── Multiple executions / session reuse ───────────────────────────────────────

@pytest.mark.asyncio
async def test_execute_multiple_calls_on_same_session():
    """MCP sessions are reused across calls — verify no session state leaks."""
    bridge = await _connected_bridge()
    try:
        r1 = await bridge.execute_tool("echo", "echo", {"text": "first"})
        r2 = await bridge.execute_tool("echo", "echo", {"text": "second"})
        r3 = await bridge.execute_tool("echo", "echo", {"text": "third"})
        assert r1 == "first"
        assert r2 == "second"
        assert r3 == "third"
    finally:
        await bridge.disconnect_all()


@pytest.mark.asyncio
async def test_alternating_tool_calls():
    bridge = await _connected_bridge()
    try:
        echo_result = await bridge.execute_tool("echo", "echo", {"text": "ping"})
        add_result = await bridge.execute_tool("echo", "add", {"a": 3, "b": 4})
        assert echo_result == "ping"
        assert add_result.strip() == "7"
    finally:
        await bridge.disconnect_all()


# ── Disconnect ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_disconnect_all_clears_sessions():
    bridge = await _connected_bridge()
    await bridge.disconnect_all()
    assert bridge.list_servers() == []


@pytest.mark.asyncio
async def test_disconnect_is_idempotent():
    """Calling disconnect_all twice must not raise."""
    bridge = await _connected_bridge()
    await bridge.disconnect_all()
    await bridge.disconnect_all()
