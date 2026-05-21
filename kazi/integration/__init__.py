"""
kazi.integration — adoption primitives for wrapping existing codebases.

These let an AI coding agent (or a human) bolt kazi onto a working
SaaS / microservice / monolith **without rewriting any of the host
application's code**.

Three primary entry points:

  expose_to_agent — decorator that marks a function as agent-callable.
  register_module — scan a Python module, register every decorated function.
  from_openapi_spec — generate tools from an OpenAPI 3 spec URL or dict.
  mount_to — mount kazi HTTP routes onto an existing FastAPI app.

Typical contractor workflow::

    # client_app/services.py — existing code, ZERO modifications needed
    def get_invoice(id: int) -> dict: ...
    def send_email(to: str, body: str) -> None: ...

    # client_app/ai_layer.py — the only file you write
    from kazi import Kazi, KaziConfig
    from kazi.integration import expose_to_agent, register_module, mount_to
    from client_app import services

    # Option 1: decorate (preferred when you can touch the file)
    services.get_invoice = expose_to_agent(services.get_invoice)

    # Option 2: scan a module — no touching their code
    kazi = await Kazi.create(KaziConfig())
    register_module(kazi, services, only=["get_invoice", "send_email"])

    # Mount AI routes onto their existing FastAPI app
    mount_to(client_app.fastapi_app, kazi, prefix="/ai")
"""
from __future__ import annotations

from kazi.integration.fastapi_mount import mount_to
from kazi.integration.openapi_import import from_openapi_spec
from kazi.integration.scanner import expose_to_agent, register_module

__all__ = [
    "expose_to_agent",
    "register_module",
    "from_openapi_spec",
    "mount_to",
]
