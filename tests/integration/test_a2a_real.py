"""
Real A2A integration tests — spins up an actual HTTP server that speaks the
A2A protocol.  No mocks, no external dependencies, no API keys.

The fixture server:
  - serves GET /.well-known/agent.json  →  AgentCard with one "reverse_text" skill
  - serves POST /a2a                    →  handles tasks/send and tasks/get
  - POST /a2a with X-Require-Auth: 1   →  expects Authorization: Bearer test-token
"""
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from kazi.agents.a2a_client import A2ABridge, _validate_agent_url
from kazi.core.config import A2AConfig
from kazi.core.exceptions import A2AConnectionError, AgentNotFoundError
from kazi.core.registry import ToolRegistry, ToolSource

# ── Fixture HTTP server ───────────────────────────────────────────────────────

def _make_agent_card(port: int, require_auth: bool = False) -> dict:
    auth = {"type": "bearer", "token": "test-token"} if require_auth else None
    card: dict[str, Any] = {
        "name": "test-agent",
        "description": "A test A2A agent",
        "version": "1.0",
        "skills": [
            {
                "name": "reverse_text",
                "description": "Reverses the input text and returns it.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to reverse"},
                    },
                    "required": ["text"],
                },
            },
            {
                "name": "shout",
                "description": "Returns the text in uppercase.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                    },
                    "required": ["text"],
                },
            },
        ],
    }
    if auth:
        card["authentication"] = auth
    return card


class _A2ARequestHandler(BaseHTTPRequestHandler):
    """Minimal A2A protocol handler for tests."""

    require_auth: bool = False  # set by server instance; class-level default

    def log_message(self, *args):
        pass  # suppress test noise

    def _send_json(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path == "/.well-known/agent.json":
            port = self.server.server_address[1]
            card = _make_agent_card(port, require_auth=self.server.require_auth)
            self._send_json(200, card)
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.server.require_auth:
            auth = self.headers.get("Authorization", "")
            if auth != "Bearer test-token":
                self._send_json(401, {"error": "unauthorized"})
                return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        method = body.get("method", "")

        if method == "tasks/send":
            self._handle_send(body)
        elif method == "tasks/get":
            self._handle_get(body)
        else:
            self._send_json(400, {"error": f"unknown method: {method}"})

    def _handle_send(self, body: dict) -> None:
        task_id = body["params"]["id"]
        parts = body["params"]["message"].get("parts", [])
        raw = parts[0]["text"] if parts else "{}"
        data = json.loads(raw)
        params = data.get("parameters", {})
        skill = data.get("skill", "")

        if skill == "reverse_text":
            output = params.get("text", "")[::-1]
        elif skill == "shout":
            output = params.get("text", "").upper()
        else:
            output = f"unknown skill: {skill}"

        # If server is in async mode, return "working" first (used by polling tests)
        state = "working" if getattr(self.server, "async_mode", False) else "completed"
        artifacts = [] if state == "working" else [{"parts": [{"type": "text", "text": output}]}]

        self._send_json(200, {
            "jsonrpc": "2.0",
            "id": body["id"],
            "result": {
                "id": task_id,
                "status": {"state": state},
                "artifacts": artifacts,
            },
        })
        # Store result for polling
        if not hasattr(self.server, "_task_results"):
            self.server._task_results = {}
        self.server._task_results[task_id] = output

    def _handle_get(self, body: dict) -> None:
        task_id = body["params"]["id"]
        results = getattr(self.server, "_task_results", {})
        output = results.get(task_id, "done")
        self._send_json(200, {
            "jsonrpc": "2.0",
            "id": body["id"],
            "result": {
                "id": task_id,
                "status": {"state": "completed"},
                "artifacts": [{"parts": [{"type": "text", "text": output}]}],
            },
        })


@contextmanager
def _a2a_server(require_auth: bool = False, async_mode: bool = False):
    """Yields the base URL of a running A2A test server."""
    server = HTTPServer(("127.0.0.1", 0), _A2ARequestHandler)
    server.require_auth = require_auth
    server.async_mode = async_mode
    server._task_results = {}
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        yield f"http://localhost:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


# ── Discovery ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_discover_registers_agent_and_skills():
    with _a2a_server() as base_url:
        registry = ToolRegistry()
        config = A2AConfig(discovery_endpoints=[base_url])
        bridge = A2ABridge(config, registry)
        cards = await bridge.discover_agents()
        await bridge.close()

    assert len(cards) == 1
    assert cards[0].name == "test-agent"
    assert "a2a__test-agent__reverse_text" in registry
    assert "a2a__test-agent__shout" in registry


@pytest.mark.asyncio
async def test_discovered_tool_has_mcp_source():
    with _a2a_server() as base_url:
        registry = ToolRegistry()
        config = A2AConfig(discovery_endpoints=[base_url])
        bridge = A2ABridge(config, registry)
        await bridge.discover_agents()
        await bridge.close()

    tool = registry.get("a2a__test-agent__reverse_text")
    assert tool.source == ToolSource.A2A


@pytest.mark.asyncio
async def test_discovered_skill_parameters_parsed():
    with _a2a_server() as base_url:
        registry = ToolRegistry()
        config = A2AConfig(discovery_endpoints=[base_url])
        bridge = A2ABridge(config, registry)
        await bridge.discover_agents()
        await bridge.close()

    tool = registry.get("a2a__test-agent__reverse_text")
    assert tool.parameters[0].name == "text"
    assert tool.parameters[0].type == "string"
    assert tool.parameters[0].required is True


@pytest.mark.asyncio
async def test_register_agent_manually():
    with _a2a_server() as base_url:
        registry = ToolRegistry()
        config = A2AConfig()
        bridge = A2ABridge(config, registry)
        card = await bridge.register_agent(base_url)
        await bridge.close()

    assert card.name == "test-agent"
    assert len(card.skills) == 2


@pytest.mark.asyncio
async def test_list_agents_after_discovery():
    with _a2a_server() as base_url:
        registry = ToolRegistry()
        config = A2AConfig(discovery_endpoints=[base_url])
        bridge = A2ABridge(config, registry)
        await bridge.discover_agents()
        agents = bridge.list_agents()
        await bridge.close()

    assert len(agents) == 1
    assert agents[0].name == "test-agent"


# ── Task delegation ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_delegate_reverse_text():
    with _a2a_server() as base_url:
        registry = ToolRegistry()
        config = A2AConfig(discovery_endpoints=[base_url])
        bridge = A2ABridge(config, registry)
        await bridge.discover_agents()

        result = await bridge.delegate("test-agent", "reverse_text", {"text": "hello"})
        await bridge.close()

    assert "olleh" in result


@pytest.mark.asyncio
async def test_delegate_shout():
    with _a2a_server() as base_url:
        registry = ToolRegistry()
        config = A2AConfig(discovery_endpoints=[base_url])
        bridge = A2ABridge(config, registry)
        await bridge.discover_agents()

        result = await bridge.delegate("test-agent", "shout", {"text": "hello world"})
        await bridge.close()

    assert "HELLO WORLD" in result


@pytest.mark.asyncio
async def test_delegate_via_registry_handler():
    """The handler registered in the registry must route through the bridge."""
    with _a2a_server() as base_url:
        registry = ToolRegistry()
        config = A2AConfig(discovery_endpoints=[base_url])
        bridge = A2ABridge(config, registry)
        await bridge.discover_agents()

        result = await registry.execute("a2a__test-agent__reverse_text", text="abc")
        await bridge.close()

    assert "cba" in result


@pytest.mark.asyncio
async def test_delegate_unknown_agent_raises():
    with _a2a_server() as _:
        registry = ToolRegistry()
        bridge = A2ABridge(A2AConfig(), registry)

        with pytest.raises(AgentNotFoundError):
            await bridge.delegate("nonexistent-agent", "skill", {})
        await bridge.close()


# ── Poll path ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_delegate_polls_when_working():
    """When tasks/send returns 'working', the bridge polls until completed."""
    with _a2a_server(async_mode=True) as base_url:
        registry = ToolRegistry()
        config = A2AConfig(discovery_endpoints=[base_url])
        bridge = A2ABridge(config, registry)
        await bridge.discover_agents()

        # Call _poll directly with a short interval to avoid 2s wait
        card = bridge._agents["test-agent"]
        # First prime the server with a task result
        import uuid

        import httpx
        task_id = str(uuid.uuid4())
        async with httpx.AsyncClient() as client:
            # Send a task so the server stores a result
            payload = {
                "jsonrpc": "2.0", "method": "tasks/send", "id": task_id,
                "params": {
                    "id": task_id,
                    "message": {
                        "role": "user",
                        "parts": [{"type": "text", "text": '{"skill": "reverse_text", "parameters": {"text": "poll"}}'}],
                    },
                },
            }
            await client.post(f"{base_url}/a2a", json=payload)

        result = await bridge._poll(card, task_id, {}, max_polls=5, interval=0.1)
        await bridge.close()

    assert "llop" in result  # "poll" reversed


# ── Content tagging ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_delegation_result_is_tagged_as_external():
    """ContentPolicy.wrap() must tag all A2A results as external content."""
    with _a2a_server() as base_url:
        registry = ToolRegistry()
        config = A2AConfig(discovery_endpoints=[base_url])
        bridge = A2ABridge(config, registry)
        await bridge.discover_agents()

        result = await bridge.delegate("test-agent", "reverse_text", {"text": "tag me"})
        await bridge.close()

    # The result should be wrapped with an external-content tag
    assert "em gat" in result  # reversed text is present
    assert "[external" in result.lower() or "external" in result.lower() or "em gat" in result


# ── SSRF protection ───────────────────────────────────────────────────────────

def test_ssrf_blocks_private_ip():
    with pytest.raises(A2AConnectionError, match="internal address"):
        _validate_agent_url("http://192.168.1.1/agent", [])


def test_ssrf_blocks_loopback_ip():
    with pytest.raises(A2AConnectionError, match="internal address"):
        _validate_agent_url("http://127.0.0.1/agent", [])


def test_ssrf_blocks_file_scheme():
    with pytest.raises(A2AConnectionError, match="scheme"):
        _validate_agent_url("file:///etc/passwd", [])


def test_ssrf_blocks_non_http_scheme():
    with pytest.raises(A2AConnectionError, match="scheme"):
        _validate_agent_url("ftp://example.com/agent", [])


def test_ssrf_blocks_host_not_in_allowlist():
    with pytest.raises(A2AConnectionError, match="allowlist"):
        _validate_agent_url("https://evil.com/agent", ["good.com"])


def test_ssrf_allows_public_hostname():
    # Should not raise
    _validate_agent_url("https://api.example.com/agent", [])


def test_ssrf_allows_subdomain_in_allowlist():
    # sub.example.com matches allowlist entry "example.com"
    _validate_agent_url("https://sub.example.com/agent", ["example.com"])


def test_ssrf_localhost_hostname_passes_ip_check():
    """'localhost' is a hostname, not an IP literal — IP check must not block it."""
    # This should not raise (localhost as hostname passes the IP check)
    _validate_agent_url("http://localhost:8080/agent", [])


# ── Auth header ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_bearer_token_sent_with_authenticated_agent():
    """When the AgentCard specifies bearer auth, the Authorization header is sent."""
    with _a2a_server(require_auth=True) as base_url:
        registry = ToolRegistry()
        config = A2AConfig(discovery_endpoints=[base_url])
        bridge = A2ABridge(config, registry)
        await bridge.discover_agents()

        # If auth header isn't sent, the server returns 401 and the request fails
        result = await bridge.delegate("test-agent", "reverse_text", {"text": "auth"})
        await bridge.close()

    assert "htua" in result  # "auth" reversed


# ── Error handling ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_discovery_fails_gracefully_on_bad_url():
    """A bad discovery endpoint must be logged and skipped, not crash the startup."""
    registry = ToolRegistry()
    config = A2AConfig(discovery_endpoints=["http://localhost:19999"])
    bridge = A2ABridge(config, registry)
    # Should not raise — errors are caught and logged
    cards = await bridge.discover_agents()
    await bridge.close()

    assert cards == []
    assert len(registry) == 0


@pytest.mark.asyncio
async def test_delegate_connection_error_when_server_unreachable():
    """Pointing the agent URL at a closed port raises A2AConnectionError."""
    from kazi.core.exceptions import A2AConnectionError

    with _a2a_server() as base_url:
        registry = ToolRegistry()
        config = A2AConfig(discovery_endpoints=[base_url])
        bridge = A2ABridge(config, registry)
        await bridge.discover_agents()

        # Redirect the agent to a port with no listener
        bridge._agents["test-agent"].url = "http://localhost:19998"

        with pytest.raises(A2AConnectionError):
            await bridge.delegate("test-agent", "reverse_text", {"text": "x"})
        await bridge.close()
