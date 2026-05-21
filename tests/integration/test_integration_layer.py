"""
Real integration tests for kazi/integration/: scanner, openapi_import,
fastapi_mount, and the package __init__.py re-exports.

No LLM API calls are made in this file.  A minimal stub is used for the
`kazi` parameter in scanner / openapi tests because those helpers only
call kazi.add_tool() — they don't touch the LLM at all.  FastAPI mount
tests use a real FastAPI app + ASGI transport.
"""
from __future__ import annotations

import json
import threading
import types
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

# ── Minimal kazi stub (not mocking the LLM — just the add_tool interface) ───

class _KaziStub:
    """Records add_tool calls without starting any LLM infrastructure."""

    def __init__(self):
        self._tools: dict[str, dict] = {}

    def add_tool(
        self,
        fn: Any,
        *,
        name: str | None = None,
        description: str | None = None,
        category: str = "custom",
    ) -> None:
        key = name or fn.__name__
        self._tools[key] = {"fn": fn, "description": description, "category": category}

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())


# ═══════════════════════════════════════════════════════════════════════════════
# kazi.integration package __init__
# ═══════════════════════════════════════════════════════════════════════════════

def test_integration_init_exports_all_symbols():
    from kazi.integration import (
        expose_to_agent,
        from_openapi_spec,
        mount_to,
        register_module,
    )
    assert callable(expose_to_agent)
    assert callable(register_module)
    assert callable(from_openapi_spec)
    assert callable(mount_to)


# ═══════════════════════════════════════════════════════════════════════════════
# kazi.integration.scanner — expose_to_agent + register_module
# ═══════════════════════════════════════════════════════════════════════════════

def test_expose_to_agent_bare_decorator_marks_function():
    from kazi.integration.scanner import expose_to_agent, is_exposed

    @expose_to_agent
    def my_tool(x: int) -> int:
        return x * 2

    assert is_exposed(my_tool)
    assert my_tool(3) == 6  # function still works normally


def test_expose_to_agent_parameterised_form():
    from kazi.integration.scanner import expose_to_agent, is_exposed

    @expose_to_agent(name="lookup_user", description="Find a user by email", category="crm")
    def find_user(email: str) -> dict:
        return {"email": email}

    assert is_exposed(find_user)
    assert find_user("a@b.com") == {"email": "a@b.com"}


def test_expose_to_agent_stores_metadata():
    from kazi.integration.scanner import _MARKER_ATTR, expose_to_agent

    @expose_to_agent(name="custom_name", description="does something", category="ops")
    def fn():
        pass

    meta = getattr(fn, _MARKER_ATTR)
    assert meta["name"] == "custom_name"
    assert meta["description"] == "does something"
    assert meta["category"] == "ops"


def test_is_exposed_false_for_plain_function():
    from kazi.integration.scanner import is_exposed

    def ordinary():
        pass

    assert is_exposed(ordinary) is False


def test_is_exposed_false_for_non_callable():
    from kazi.integration.scanner import is_exposed

    assert is_exposed(42) is False
    assert is_exposed("string") is False


def test_register_module_registers_decorated_functions():
    from kazi.integration.scanner import expose_to_agent, register_module

    mod = types.ModuleType("fake_services")

    @expose_to_agent
    def get_invoice(id: int) -> dict:
        return {}

    @expose_to_agent(name="send_email", description="Send an email")
    def send_email_fn(to: str) -> None:
        pass

    def internal_fn():
        pass

    mod.get_invoice = get_invoice
    mod.send_email_fn = send_email_fn
    mod.internal_fn = internal_fn
    mod.__name__ = "fake_services"

    kazi = _KaziStub()
    registered = register_module(kazi, mod)

    assert "get_invoice" in registered
    assert "send_email" in registered
    assert "internal_fn" not in registered


def test_register_module_only_allowlist():
    from kazi.integration.scanner import register_module

    mod = types.ModuleType("svc")

    def get_order(id: int) -> dict:
        return {}

    def delete_order(id: int) -> None:
        pass

    mod.get_order = get_order
    mod.delete_order = delete_order
    mod.__name__ = "svc"

    kazi = _KaziStub()
    registered = register_module(kazi, mod, only=["get_order"])

    assert "get_order" in registered
    assert "delete_order" not in registered


def test_register_module_exclude_list():
    from kazi.integration.scanner import expose_to_agent, register_module

    mod = types.ModuleType("svc")

    @expose_to_agent
    def get_users():
        pass

    @expose_to_agent
    def delete_users():
        pass

    mod.get_users = get_users
    mod.delete_users = delete_users
    mod.__name__ = "svc"

    kazi = _KaziStub()
    registered = register_module(kazi, mod, exclude=["delete_users"])

    assert "get_users" in registered
    assert "delete_users" not in registered


def test_register_module_include_undecorated():
    from kazi.integration.scanner import register_module

    mod = types.ModuleType("svc")

    def plain_fn():
        pass

    mod.plain_fn = plain_fn
    mod.__name__ = "svc"

    kazi = _KaziStub()
    registered = register_module(kazi, mod, include_undecorated=True)

    assert "plain_fn" in registered


def test_register_module_skips_private_names():
    from kazi.integration.scanner import register_module

    mod = types.ModuleType("svc")

    def _private():
        pass

    mod._private = _private
    mod.__name__ = "svc"

    kazi = _KaziStub()
    registered = register_module(kazi, mod, include_undecorated=True)

    assert "_private" not in registered


def test_register_module_skips_non_callables():
    from kazi.integration.scanner import register_module

    mod = types.ModuleType("svc")
    mod.MY_CONST = 42
    mod.__name__ = "svc"

    kazi = _KaziStub()
    registered = register_module(kazi, mod, include_undecorated=True)

    assert "MY_CONST" not in registered


def test_register_module_skips_classes():
    from kazi.integration.scanner import register_module

    mod = types.ModuleType("svc")

    class MyClass:
        def method(self):
            pass

    mod.MyClass = MyClass
    mod.__name__ = "svc"

    kazi = _KaziStub()
    registered = register_module(kazi, mod, include_undecorated=True)

    assert "MyClass" not in registered


def test_register_module_handles_add_tool_error_gracefully():
    """If add_tool raises, register_module logs and continues."""
    from kazi.integration.scanner import expose_to_agent, register_module

    mod = types.ModuleType("svc")

    @expose_to_agent
    def good_tool():
        pass

    @expose_to_agent
    def bad_tool():
        pass

    mod.good_tool = good_tool
    mod.bad_tool = bad_tool
    mod.__name__ = "svc"

    class _FaultyKazi:
        _tools: dict = {}

        def add_tool(self, fn, *, name=None, description=None, category="custom"):
            if name == "bad_tool" or fn.__name__ == "bad_tool":
                raise ValueError("registration failed")
            self._tools[name or fn.__name__] = fn

    kazi = _FaultyKazi()
    registered = register_module(kazi, mod)

    assert "good_tool" in registered
    assert "bad_tool" not in registered


def test_register_module_returns_empty_for_empty_module():
    from kazi.integration.scanner import register_module

    mod = types.ModuleType("empty")
    mod.__name__ = "empty"

    kazi = _KaziStub()
    registered = register_module(kazi, mod)
    assert registered == []


# ═══════════════════════════════════════════════════════════════════════════════
# kazi.integration.openapi_import — pure helpers
# ═══════════════════════════════════════════════════════════════════════════════

def test_slug_replaces_non_identifier_chars():
    from kazi.integration.openapi_import import _slug

    assert _slug("get-users") == "get_users"
    assert _slug("list users") == "list_users"
    assert _slug("GET/users/{id}") == "GET_users__id"


def test_slug_strips_leading_trailing_underscores():
    from kazi.integration.openapi_import import _slug

    assert _slug("-prefix") == "prefix"


def test_derive_name_uses_operation_id():
    from kazi.integration.openapi_import import _derive_name

    op = {"operationId": "getUserById"}
    assert _derive_name(op, "get", "/users/{id}", "operationId") == "getUserById"


def test_derive_name_falls_back_to_method_path():
    from kazi.integration.openapi_import import _derive_name

    op = {}
    name = _derive_name(op, "get", "/users/profile", "operationId")
    assert name == "get_users_profile"


def test_derive_name_skips_path_param_segments():
    from kazi.integration.openapi_import import _derive_name

    op = {}
    name = _derive_name(op, "get", "/users/{id}/orders", "operationId")
    assert name == "get_users_orders"


def test_from_openapi_spec_registers_get_endpoints():
    from kazi.integration.openapi_import import from_openapi_spec

    spec = {
        "servers": [{"url": "http://api.example.com"}],
        "paths": {
            "/users": {
                "get": {
                    "operationId": "list_users",
                    "summary": "List all users",
                }
            }
        },
    }

    kazi = _KaziStub()
    registered = from_openapi_spec(kazi, spec)

    assert "list_users" in registered
    assert "list_users" in kazi.tool_names


def test_from_openapi_spec_uses_base_url_from_servers():
    from kazi.integration.openapi_import import from_openapi_spec

    spec = {
        "servers": [{"url": "https://api.example.com/v1"}],
        "paths": {
            "/items": {
                "get": {"operationId": "list_items", "summary": "List items"}
            }
        },
    }

    kazi = _KaziStub()
    registered = from_openapi_spec(kazi, spec)
    assert "list_items" in registered


def test_from_openapi_spec_base_url_parameter_overrides_servers():
    from kazi.integration.openapi_import import from_openapi_spec

    spec = {
        "servers": [{"url": "http://wrong.example.com"}],
        "paths": {
            "/ping": {"get": {"operationId": "get_ping", "summary": "Ping"}}
        },
    }

    kazi = _KaziStub()
    registered = from_openapi_spec(kazi, spec, base_url="http://right.example.com")
    assert "get_ping" in registered


def test_from_openapi_spec_allowlist_filters_post_by_default():
    from kazi.integration.openapi_import import from_openapi_spec

    spec = {
        "servers": [{"url": "http://api.example.com"}],
        "paths": {
            "/users": {
                "get": {"operationId": "list_users", "summary": "List"},
                "post": {"operationId": "create_user", "summary": "Create"},
            }
        },
    }

    kazi = _KaziStub()
    registered = from_openapi_spec(kazi, spec)

    assert "list_users" in registered
    assert "create_user" not in registered


def test_from_openapi_spec_custom_allowlist_all():
    from kazi.integration.openapi_import import from_openapi_spec

    spec = {
        "servers": [{"url": "http://api.example.com"}],
        "paths": {
            "/users": {
                "post": {"operationId": "create_user", "summary": "Create"},
            }
        },
    }

    kazi = _KaziStub()
    registered = from_openapi_spec(kazi, spec, allowlist=["*"])
    assert "create_user" in registered


def test_from_openapi_spec_denylist_excludes_matching_tools():
    from kazi.integration.openapi_import import from_openapi_spec

    spec = {
        "servers": [{"url": "http://api.example.com"}],
        "paths": {
            "/users": {
                "get": {"operationId": "list_users", "summary": "List"},
            },
            "/admin": {
                "get": {"operationId": "get_admin", "summary": "Admin"},
            },
        },
    }

    kazi = _KaziStub()
    registered = from_openapi_spec(kazi, spec, allowlist=["*"], denylist=["get_admin"])
    assert "list_users" in registered
    assert "get_admin" not in registered


def test_from_openapi_spec_no_base_url_returns_empty():
    from kazi.integration.openapi_import import from_openapi_spec

    spec = {"paths": {"/users": {"get": {"operationId": "list_users"}}}}

    kazi = _KaziStub()
    registered = from_openapi_spec(kazi, spec)
    assert registered == []


def test_from_openapi_spec_non_dict_spec_returns_empty():
    from kazi.integration.openapi_import import from_openapi_spec

    kazi = _KaziStub()
    registered = from_openapi_spec(kazi, "not-a-url-that-will-resolve://bad")
    assert registered == []


def test_from_openapi_spec_skips_non_http_methods():
    from kazi.integration.openapi_import import from_openapi_spec

    spec = {
        "servers": [{"url": "http://api.example.com"}],
        "paths": {
            "/users": {
                "head": {"operationId": "head_users"},
                "options": {"operationId": "options_users"},
                "get": {"operationId": "list_users", "summary": "List"},
            }
        },
    }

    kazi = _KaziStub()
    registered = from_openapi_spec(kazi, spec)
    assert "list_users" in registered
    assert "head_users" not in registered
    assert "options_users" not in registered


# ── Handler execution against a real local HTTP server ────────────────────────

class _SimpleAPIHandler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        if self.path.startswith("/users/"):
            user_id = self.path.split("/")[-1]
            body = json.dumps({"id": user_id, "name": "Alice"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/users"):
            body = json.dumps([{"id": "1"}]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        body = json.dumps({"id": "new", "created": True}).encode()
        self.send_response(201)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@contextmanager
def _local_server():
    server = HTTPServer(("127.0.0.1", 0), _SimpleAPIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        yield f"http://localhost:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


@pytest.mark.asyncio
async def test_openapi_handler_executes_get_request():
    from kazi.integration.openapi_import import from_openapi_spec

    with _local_server() as base_url:
        spec = {
            "servers": [{"url": base_url}],
            "paths": {
                "/users": {
                    "get": {"operationId": "list_users", "summary": "List users"},
                }
            },
        }
        kazi = _KaziStub()
        registered = from_openapi_spec(kazi, spec)
        assert "list_users" in registered

        handler = kazi._tools["list_users"]["fn"]
        result = await handler()
        assert "id" in result or "1" in result


@pytest.mark.asyncio
async def test_openapi_handler_substitutes_path_params():
    from kazi.integration.openapi_import import from_openapi_spec

    with _local_server() as base_url:
        spec = {
            "servers": [{"url": base_url}],
            "paths": {
                "/users/{user_id}": {
                    "get": {
                        "operationId": "get_user",
                        "summary": "Get user",
                        "parameters": [
                            {"name": "user_id", "in": "path", "required": True,
                             "schema": {"type": "string"}},
                        ],
                    }
                }
            },
        }
        kazi = _KaziStub()
        from_openapi_spec(kazi, spec)

        handler = kazi._tools["get_user"]["fn"]
        result = await handler(user_id="42")
        assert "42" in result


@pytest.mark.asyncio
async def test_openapi_handler_sends_query_params():
    from kazi.integration.openapi_import import from_openapi_spec

    with _local_server() as base_url:
        spec = {
            "servers": [{"url": base_url}],
            "paths": {
                "/users": {
                    "get": {
                        "operationId": "list_users",
                        "summary": "List users",
                        "parameters": [
                            {"name": "limit", "in": "query", "schema": {"type": "integer"}},
                        ],
                    }
                }
            },
        }
        kazi = _KaziStub()
        from_openapi_spec(kazi, spec)

        handler = kazi._tools["list_users"]["fn"]
        result = await handler(limit=10)
        assert result  # server responded


@pytest.mark.asyncio
async def test_openapi_handler_raises_on_4xx():
    from kazi.integration.openapi_import import from_openapi_spec

    with _local_server() as base_url:
        spec = {
            "servers": [{"url": base_url}],
            "paths": {
                "/notfound": {
                    "get": {"operationId": "get_notfound", "summary": "Not found"},
                }
            },
        }
        kazi = _KaziStub()
        from_openapi_spec(kazi, spec)

        handler = kazi._tools["get_notfound"]["fn"]
        with pytest.raises(RuntimeError, match="HTTP 404"):
            await handler()


@pytest.mark.asyncio
async def test_openapi_handler_sends_auth_header():
    """auth_header dict is forwarded on every request."""
    received_headers: list[str] = []

    class _AuthCapture(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            received_headers.append(self.headers.get("Authorization", ""))
            body = b'{"ok": true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(("127.0.0.1", 0), _AuthCapture)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]

    try:
        from kazi.integration.openapi_import import from_openapi_spec

        spec = {
            "servers": [{"url": f"http://localhost:{port}"}],
            "paths": {
                "/ping": {"get": {"operationId": "get_ping", "summary": "Ping"}},
            },
        }
        kazi = _KaziStub()
        from_openapi_spec(kazi, spec, auth_header={"Authorization": "Bearer test-token"})
        handler = kazi._tools["get_ping"]["fn"]
        await handler()
        assert received_headers[-1] == "Bearer test-token"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


# ═══════════════════════════════════════════════════════════════════════════════
# kazi.integration.fastapi_mount
# ═══════════════════════════════════════════════════════════════════════════════

def test_mount_to_raises_import_error_without_fastapi(monkeypatch):
    """Importing works without FastAPI; calling mount_to without it raises ImportError."""
    import builtins
    real_import = builtins.__import__

    def _block_fastapi(name, *args, **kwargs):
        if name == "fastapi":
            raise ImportError("no module named fastapi")
        return real_import(name, *args, **kwargs)

    from kazi.integration.fastapi_mount import mount_to
    monkeypatch.setattr(builtins, "__import__", _block_fastapi)

    with pytest.raises(ImportError, match="fastapi is required"):
        mount_to(object(), None)


def test_mount_to_raises_type_error_for_non_fastapi_app():
    from kazi.integration.fastapi_mount import mount_to

    with pytest.raises(TypeError, match="FastAPI"):
        mount_to("not-a-fastapi-app", None)


@pytest.mark.asyncio
async def test_mount_to_adds_routes_at_prefix():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from kazi.integration.fastapi_mount import mount_to

    # Build a minimal kazi stub that provides as_app()
    inner_app = FastAPI()

    @inner_app.get("/health")
    def health():
        return {"status": "ok"}

    class _KaziWithApp:
        def as_app(self, **kwargs):
            return inner_app

    host_app = FastAPI()
    host_app.get("/existing")(lambda: {"host": True})

    result = mount_to(host_app, _KaziWithApp(), prefix="/ai")
    assert result is host_app

    client = TestClient(host_app)
    resp = client.get("/ai/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}

    resp = client.get("/existing")
    assert resp.status_code == 200


def test_mount_to_returns_the_same_app_for_chaining():
    from fastapi import FastAPI

    from kazi.integration.fastapi_mount import mount_to

    inner_app = FastAPI()

    @inner_app.get("/health")
    def health():
        return {"ok": True}

    class _KaziStubApp:
        def as_app(self, **kwargs):
            return inner_app

    host_app = FastAPI()
    returned = mount_to(host_app, _KaziStubApp(), prefix="/kazi")
    assert returned is host_app


# ── OpenAPI URL-fetch paths (lines 97-98, 106-107) ────────────────────────────

def test_from_openapi_spec_fetches_spec_from_url():
    """from_openapi_spec accepts a URL, fetches it with httpx, and registers tools."""
    from kazi.integration.openapi_import import from_openapi_spec

    spec_dict = {
        "servers": [{"url": "http://api.example.com"}],
        "paths": {
            "/items": {"get": {"operationId": "list_items", "summary": "List items"}},
        },
    }

    class _SpecHandler(BaseHTTPRequestHandler):
        def log_message(self, *_): pass
        def do_GET(self):
            body = json.dumps(spec_dict).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(("127.0.0.1", 0), _SpecHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        kazi = _KaziStub()
        registered = from_openapi_spec(
            kazi, f"http://127.0.0.1:{port}/openapi.json", base_url="http://api.example.com"
        )
        assert "list_items" in registered
    finally:
        server.shutdown()


def test_from_openapi_spec_url_returns_non_dict_returns_empty():
    """When URL returns a non-dict body (e.g. a list), registers nothing."""
    from kazi.integration.openapi_import import from_openapi_spec

    class _ListHandler(BaseHTTPRequestHandler):
        def log_message(self, *_): pass
        def do_GET(self):
            body = b'["not", "a", "dict"]'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(("127.0.0.1", 0), _ListHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        kazi = _KaziStub()
        registered = from_openapi_spec(kazi, f"http://127.0.0.1:{port}/openapi.json",
                                       base_url="http://api.example.com")
        assert registered == []
    finally:
        server.shutdown()


# ── openapi_import — structural edge cases ────────────────────────────────────

def test_from_openapi_spec_skips_non_dict_methods_entry():
    """A paths entry where the methods value is not a dict is silently skipped."""
    from kazi.integration.openapi_import import from_openapi_spec

    spec = {
        "servers": [{"url": "http://api.example.com"}],
        "paths": {
            "/bad": "not-a-dict",          # line 128: skip
            "/good": {"get": {"operationId": "get_good", "summary": "OK"}},
        },
    }
    kazi = _KaziStub()
    registered = from_openapi_spec(kazi, spec)
    assert "get_good" in registered


def test_from_openapi_spec_skips_non_dict_op_entry():
    """A methods entry where the op value is not a dict is silently skipped."""
    from kazi.integration.openapi_import import from_openapi_spec

    spec = {
        "servers": [{"url": "http://api.example.com"}],
        "paths": {
            "/items": {
                "get": "not-a-dict",       # line 133: skip
                "post": {"operationId": "create_item", "summary": "Create"},
            },
        },
    }
    kazi = _KaziStub()
    registered = from_openapi_spec(kazi, spec, allowlist=["*"])
    assert "create_item" in registered


def test_from_openapi_spec_logs_warning_when_add_tool_raises(caplog):
    """When add_tool raises, the error is logged and the rest still registers."""
    import logging

    from kazi.integration.openapi_import import from_openapi_spec

    class _FaultyStub(_KaziStub):
        def add_tool(self, fn, *, name=None, **kw):
            if name == "list_items":
                raise RuntimeError("registration failed")
            super().add_tool(fn, name=name, **kw)

    spec = {
        "servers": [{"url": "http://api.example.com"}],
        "paths": {
            "/items": {"get": {"operationId": "list_items", "summary": "List"}},
            "/other": {"get": {"operationId": "get_other", "summary": "Other"}},
        },
    }
    kazi = _FaultyStub()
    with caplog.at_level(logging.WARNING, logger="kazi.integration.openapi_import"):
        registered = from_openapi_spec(kazi, spec)
    assert "list_items" not in registered
    assert "get_other" in registered
    assert "failed to register" in caplog.text


@pytest.mark.asyncio
async def test_openapi_handler_sends_post_body():
    """POST requests with body params include the body as JSON."""
    received_bodies: list[bytes] = []

    class _BodyCapture(BaseHTTPRequestHandler):
        def log_message(self, *_): pass
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            received_bodies.append(self.rfile.read(length))
            body = b'{"created": true}'
            self.send_response(201)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(("127.0.0.1", 0), _BodyCapture)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        from kazi.integration.openapi_import import from_openapi_spec

        spec = {
            "servers": [{"url": f"http://127.0.0.1:{port}"}],
            "paths": {
                "/items": {
                    "post": {"operationId": "create_item", "summary": "Create item"},
                }
            },
        }
        kazi = _KaziStub()
        from_openapi_spec(kazi, spec, allowlist=["*"])
        handler = kazi._tools["create_item"]["fn"]
        result = await handler(name="Widget", price=9.99)
        body = json.loads(received_bodies[0])
        assert "name" in body or "price" in body
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_openapi_handler_returns_text_for_non_json_response():
    """When response has no JSON content-type, handler returns raw text."""
    class _TextHandler(BaseHTTPRequestHandler):
        def log_message(self, *_): pass
        def do_GET(self):
            body = b"plain text response"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(("127.0.0.1", 0), _TextHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        from kazi.integration.openapi_import import from_openapi_spec

        spec = {
            "servers": [{"url": f"http://127.0.0.1:{port}"}],
            "paths": {
                "/text": {"get": {"operationId": "get_text", "summary": "Text endpoint"}},
            },
        }
        kazi = _KaziStub()
        from_openapi_spec(kazi, spec)
        handler = kazi._tools["get_text"]["fn"]
        result = await handler()
        assert "plain text response" in result
    finally:
        server.shutdown()
