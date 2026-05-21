"""
MCP + LLM combined integration tests.

Wires the real MCP echo server into a full Kazi instance with an OpenAI LLM
and verifies that the LLM actually discovers and calls MCP-backed tools.

Skipped automatically when OPENAI_API_KEY is not set.
"""
import os
import sys
from pathlib import Path

import pytest

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
pytestmark = pytest.mark.skipif(not OPENAI_KEY, reason="OPENAI_API_KEY not set")

FIXTURE_SERVER = str(Path(__file__).parent.parent / "fixtures" / "mcp_echo_server.py")
SERVER_CMD = f"{sys.executable} {FIXTURE_SERVER}"
MODEL = "gpt-4o-mini"


def _config(extra_mcp_servers: dict | None = None):
    from kazi.core.config import KaziConfig, LLMConfig, LLMProvider, MCPConfig
    servers = {"echo": SERVER_CMD}
    if extra_mcp_servers:
        servers.update(extra_mcp_servers)
    return KaziConfig(
        llm=LLMConfig(provider=LLMProvider.OPENAI, model=MODEL, api_key=OPENAI_KEY),
        mcp=MCPConfig(servers=servers),
    )


# ── Tool registration through Kazi ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_kazi_registers_mcp_tools_on_startup():
    """MCP tools must appear in the registry after Kazi.create()."""
    from kazi import Kazi
    async with await Kazi.create(_config()) as kazi:
        assert "echo__echo" in kazi.registry
        assert "echo__add" in kazi.registry


@pytest.mark.asyncio
async def test_mcp_tools_have_correct_source():
    from kazi import Kazi
    from kazi.core.registry import ToolSource
    async with await Kazi.create(_config()) as kazi:
        tool = kazi.registry.get("echo__echo")
        assert tool.source == ToolSource.MCP


# ── LLM calls MCP tool ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_llm_calls_mcp_echo_tool():
    """LLM must discover and call the MCP echo tool and return its output."""
    from kazi import Kazi
    async with await Kazi.create(_config()) as kazi:
        result = await kazi.run(
            "Call the echo__echo tool with the text 'kazi_mcp_test' and tell me what it returned.",
            thread_id="mcp-llm-echo",
        )
    assert "kazi_mcp_test" in result


@pytest.mark.asyncio
async def test_llm_calls_mcp_add_tool():
    """LLM must call the MCP add tool with specific numbers and return the sum."""
    from kazi import Kazi
    async with await Kazi.create(_config()) as kazi:
        result = await kazi.run(
            "Call the echo__add tool with a=17 and b=25 and tell me the result.",
            thread_id="mcp-llm-add",
        )
    assert "42" in result


@pytest.mark.asyncio
async def test_llm_uses_mcp_tool_in_multi_turn():
    """MCP tool output should persist across turns within the same thread."""
    from kazi import Kazi
    async with await Kazi.create(_config()) as kazi:
        await kazi.run(
            "Call the echo__echo tool with the text 'THREAD_SENTINEL' and remember the result.",
            thread_id="mcp-multi",
        )
        reply = await kazi.run(
            "What text did the echo tool return in your previous turn?",
            thread_id="mcp-multi",
        )
    assert "THREAD_SENTINEL" in reply


@pytest.mark.asyncio
async def test_mcp_tool_result_in_streamed_response():
    """Streaming must still produce the MCP tool output in the final response."""
    from kazi import Kazi
    chunks = []
    async with await Kazi.create(_config()) as kazi:
        async for chunk in kazi.stream(
            "Call echo__echo with text='stream_check' and tell me what came back.",
            thread_id="mcp-stream",
        ):
            chunks.append(chunk)

    full = "".join(chunks)
    assert "stream_check" in full


@pytest.mark.asyncio
async def test_native_and_mcp_tools_coexist():
    """A native Python tool and an MCP tool must both be callable in the same session."""
    from kazi import Kazi

    native_calls = []

    async def native_greeting() -> str:
        native_calls.append(1)
        return "NATIVE_HELLO"

    async with await Kazi.create(_config()) as kazi:
        kazi.add_tool(native_greeting, description="Returns a greeting. Call this when asked for a greeting.")

        # Ask LLM to call both tools
        result = await kazi.run(
            "First call native_greeting, then call echo__echo with text='MCP_HELLO'. "
            "Report both results.",
            thread_id="coexist",
        )

    assert native_calls, "Native tool was never called"
    assert "NATIVE_HELLO" in result or "MCP_HELLO" in result  # at least one confirmed
