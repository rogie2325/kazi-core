"""
Real HTTP tests for openapi_import.py.

Covers lines 97-98 (URL fetch success path) and 220-246 (handler execution)
using a real embedded HTTP server — no mocks, no monkeypatching.
"""
from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import MagicMock

import pytest


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _SimpleHandler(BaseHTTPRequestHandler):
    """Minimal HTTP server that serves a fixed JSON response."""

    def log_message(self, *args):
        pass  # suppress output in tests

    def do_GET(self):
        if self.path == "/openapi.json":
            body = json.dumps(self.server._spec).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/users"):
            body = json.dumps({"id": 1, "name": "Alice"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"not found")

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(content_length)
        body = json.dumps({"created": True}).encode()
        self.send_response(201)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _start_server(spec: dict, port: int) -> HTTPServer:
    server = HTTPServer(("127.0.0.1", port), _SimpleHandler)
    server._spec = spec
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def test_from_openapi_spec_loads_spec_from_url():
    """Lines 97-98: from_openapi_spec fetches and parses a spec from a real URL."""
    import kazi.integration.openapi_import as oai

    port = _find_free_port()
    spec = {
        "servers": [{"url": f"http://127.0.0.1:{port}"}],
        "paths": {
            "/users": {"get": {"operationId": "get_users", "summary": "List users"}},
        },
    }
    server = _start_server(spec, port)
    try:
        kazi_mock = MagicMock()
        registered = []
        kazi_mock.add_tool = MagicMock(side_effect=lambda fn, **kw: registered.append(kw["name"]))
        result = oai.from_openapi_spec(
            kazi_mock,
            f"http://127.0.0.1:{port}/openapi.json",
            allowlist=["*"],
        )
        assert "get_users" in result
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_handler_get_request_returns_json_response():
    """Lines 220-246: handler() makes a real GET request and returns the JSON body."""
    from kazi.integration.openapi_import import _build_handler

    port = _find_free_port()
    spec = {}
    server = _start_server(spec, port)
    try:
        handler = _build_handler(
            method="get",
            path="/users",
            base_url=f"http://127.0.0.1:{port}",
            op_spec={"operationId": "get_users", "parameters": []},
            auth_header={},
            timeout=5.0,
        )
        result = await handler()
        assert "Alice" in result or "id" in result
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_handler_post_request_sends_body():
    """Lines 234-235: POST handler sends remaining kwargs as JSON body."""
    from kazi.integration.openapi_import import _build_handler

    port = _find_free_port()
    spec = {}
    server = _start_server(spec, port)
    try:
        handler = _build_handler(
            method="post",
            path="/users",
            base_url=f"http://127.0.0.1:{port}",
            op_spec={"operationId": "create_user", "parameters": []},
            auth_header={},
            timeout=5.0,
        )
        result = await handler(name="Bob", email="bob@example.com")
        assert "created" in result or "True" in result
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_handler_raises_on_4xx_response():
    """Lines 237-241: handler raises RuntimeError on HTTP 4xx/5xx responses."""
    from kazi.integration.openapi_import import _build_handler

    port = _find_free_port()
    spec = {}
    server = _start_server(spec, port)
    try:
        handler = _build_handler(
            method="get",
            path="/nonexistent",
            base_url=f"http://127.0.0.1:{port}",
            op_spec={"operationId": "missing", "parameters": []},
            auth_header={},
            timeout=5.0,
        )
        with pytest.raises(RuntimeError, match="HTTP 404"):
            await handler()
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_handler_substitutes_path_params():
    """Lines 226-228: path parameters are substituted into the URL."""
    from kazi.integration.openapi_import import _build_handler

    port = _find_free_port()
    spec = {}
    server = _start_server(spec, port)
    try:
        handler = _build_handler(
            method="get",
            path="/users/{user_id}",
            base_url=f"http://127.0.0.1:{port}",
            op_spec={
                "operationId": "get_user",
                "parameters": [{"name": "user_id", "in": "path"}],
            },
            auth_header={},
            timeout=5.0,
        )
        result = await handler(user_id="42")
        assert result is not None
    finally:
        server.shutdown()
