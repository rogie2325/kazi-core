"""
Python-module scanner: turn an existing codebase into an agent toolbox
without rewriting any of its code.

Two registration styles supported:

1. **Opt-in via decorator** (preferred when you control the source)::

       @expose_to_agent
       def get_invoice(id: int) -> dict:
           return billing.fetch(id)

       @expose_to_agent(name="lookup_user", description="Find a user by email")
       def find_user_by_email(email: str) -> dict:
           ...

   Then::

       register_module(kazi, my_app.services)   # registers only @expose_to_agent

2. **Bulk import by name** (when you can't touch the host code at all)::

       register_module(
           kazi,
           my_app.services,
           only=["get_invoice", "send_email"],   # explicit allowlist
           category="billing",
       )

The scanner walks the module's public callables, filters by the rules above,
and calls ``kazi.add_tool(fn)`` for each match.  Parameter schemas are
inferred from type hints exactly like ``kazi.add_tool(plain_function)``.
"""
from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kazi.core.orchestrator import Kazi

logger = logging.getLogger(__name__)

_MARKER_ATTR = "__kazi_expose__"


def expose_to_agent(
    func: Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    category: str = "custom",
) -> Callable:
    """
    Mark a function as agent-callable.

    Works as a bare decorator or with keyword overrides::

        @expose_to_agent
        def foo(x: int) -> int: ...

        @expose_to_agent(name="lookup", description="Find a user")
        def find_user(email: str) -> dict: ...

    The wrapped function is otherwise unchanged — call sites in the host
    application keep working exactly as before.  The decorator merely sets
    an attribute that ``register_module()`` looks for.
    """
    def _decorate(fn: Callable) -> Callable:
        setattr(fn, _MARKER_ATTR, {
            "name": name or fn.__name__,
            "description": description,
            "category": category,
        })
        return fn

    if func is not None and callable(func):
        # Bare-decorator form: @expose_to_agent
        return _decorate(func)
    # Parameterised form: @expose_to_agent(name=...)
    return _decorate


def is_exposed(fn: Any) -> bool:
    """Return True if ``fn`` was decorated with @expose_to_agent."""
    return callable(fn) and hasattr(fn, _MARKER_ATTR)


def register_module(
    kazi: Kazi,
    module: Any,
    *,
    only: list[str] | None = None,
    exclude: list[str] | None = None,
    category: str | None = None,
    include_undecorated: bool | None = None,
) -> list[str]:
    """
    Walk ``module`` and register every agent-callable function on ``kazi``.

    Selection rules (applied in this order):
      1. If ``only`` is given, the function name MUST appear in it.
         When ``only`` is set, ``include_undecorated`` defaults to True so
         the explicit allowlist works against undecorated code.
      2. If ``exclude`` is given, names in it are skipped.
      3. Otherwise, only functions decorated with @expose_to_agent are
         registered.  Pass ``include_undecorated=True`` to register every
         public callable (rare; use with care).

    Parameters
    ----------
    kazi               The Kazi instance to register tools on.
    module              A Python module object (or anything with ``dir(...)``).
    only                Explicit allowlist of function names.  When set,
                        bypasses the @expose_to_agent requirement.
    exclude             Names to skip.
    category            Override the category for every tool from this module.
                        Defaults to the decorator's category or "custom".
    include_undecorated Register undecorated public callables.  Default False
                        (or True when ``only`` is given).

    Returns the list of registered tool names.
    """
    if include_undecorated is None:
        include_undecorated = only is not None

    only_set = set(only or [])
    exclude_set = set(exclude or [])

    registered: list[str] = []
    for attr in dir(module):
        if attr.startswith("_"):
            continue
        fn = getattr(module, attr)
        if not callable(fn):
            continue
        # Skip classes / class instances unless they're plain functions
        if not (inspect.isfunction(fn) or inspect.iscoroutinefunction(fn) or inspect.ismethod(fn)):
            continue

        if only_set and attr not in only_set:
            continue
        if attr in exclude_set:
            continue

        marker = getattr(fn, _MARKER_ATTR, None)
        if marker is None and not include_undecorated:
            continue

        tool_name = (marker or {}).get("name") or attr
        tool_desc = (marker or {}).get("description")
        tool_cat = category or (marker or {}).get("category") or "custom"

        try:
            kazi.add_tool(
                fn,
                name=tool_name,
                description=tool_desc,
                category=tool_cat,
            )
            registered.append(tool_name)
            logger.debug("register_module: registered %s (category=%s)", tool_name, tool_cat)
        except Exception as exc:
            logger.warning(
                "register_module: failed to register %s from %s: %s",
                attr, getattr(module, "__name__", "?"), exc,
            )

    logger.info(
        "register_module: %d tool(s) registered from %s",
        len(registered), getattr(module, "__name__", "<module>"),
    )
    return registered
