"""
Generate Kazi tools from an OpenAPI 3 specification.

Lets a contracted AI engineer point at a client's existing REST API and
expose every documented endpoint as an agent-callable tool — without writing
or modifying any of the host code::

    kazi = await Kazi.create(KaziConfig())
    from_openapi_spec(
        kazi,
        "https://api.client.com/openapi.json",
        base_url="https://api.client.com",
        auth_header={"Authorization": "Bearer ..."},
        allowlist=["get_*", "list_*"],   # read-only by default
        category="client_api",
    )

Each operation becomes a tool whose handler issues the HTTP request, returns
the response body (with sensible truncation), and surfaces non-2xx as an
exception so the agent's audit captures the failure.

Limitations / safe defaults
---------------------------
- Only ``application/json`` request and response bodies are handled.
- File uploads, multipart, server-sent events: NOT supported.
- ``allowlist`` defaults to read-only verb prefixes — explicit opt-in is
  required to expose mutating endpoints.
- Path / query / body parameter names with non-identifier chars are skipped.
"""
from __future__ import annotations

import fnmatch
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kazi.core.orchestrator import Kazi

logger = logging.getLogger(__name__)

# Default allowlist: only read-only verbs.  Override explicitly to expose mutations.
_DEFAULT_READ_ONLY = ["get_*", "list_*", "search_*", "find_*", "fetch_*"]


def from_openapi_spec(
    kazi: Kazi,
    spec: str | dict,
    *,
    base_url: str | None = None,
    auth_header: dict[str, str] | None = None,
    allowlist: list[str] | None = None,
    denylist: list[str] | None = None,
    category: str = "openapi",
    timeout: float = 30.0,
    name_from: str = "operationId",
) -> list[str]:
    """
    Import every operation from an OpenAPI 3 spec as a Kazi tool.

    Parameters
    ----------
    kazi       The Kazi instance to register tools on.
    spec        Either a URL to fetch the JSON spec from, or a dict that has
                already been parsed.  YAML specs must be parsed by the caller.
    base_url    Server base URL.  Falls back to ``spec["servers"][0]["url"]``.
    auth_header Headers added to every outgoing request.  Use this to attach
                a bearer token or API key.  NEVER log this dict — store the
                token in a secret manager and inject at startup.
    allowlist   fnmatch patterns the tool name must match.  Default is
                ``["get_*", "list_*", "search_*", "find_*", "fetch_*"]`` —
                only read endpoints.  Pass ``["*"]`` to expose everything.
    denylist    fnmatch patterns the tool name must NOT match.  Applied after
                allowlist.
    category    Tool category (groups tools in the registry view).
    timeout     Per-request HTTP timeout in seconds.
    name_from   Where to derive the tool name from.  ``operationId`` (default)
                falls back to ``<method>_<path>`` when missing.

    Returns the list of registered tool names.

    Requires httpx for the runtime client.  Falls back gracefully if the spec
    is unreachable — registers nothing and logs the error.
    """
    try:
        import httpx
    except ImportError as exc:
        raise ImportError(
            "httpx is required for OpenAPI import. "
            "Install: pip install httpx"
        ) from exc

    # Load spec dict
    if isinstance(spec, str):
        try:
            resp = httpx.get(spec, timeout=timeout)
            resp.raise_for_status()
            spec_dict = resp.json()
        except Exception as exc:
            logger.error("from_openapi_spec: failed to fetch %s: %s", spec, exc)
            return []
    else:
        spec_dict = spec

    if not isinstance(spec_dict, dict):
        logger.error("from_openapi_spec: spec must be a URL or dict")
        return []

    # Resolve base URL
    if base_url is None:
        servers = spec_dict.get("servers") or []
        if servers and isinstance(servers, list):
            base_url = servers[0].get("url", "")
    if not base_url:
        logger.error("from_openapi_spec: no base_url provided and spec has no servers[]")
        return []

    base_url = base_url.rstrip("/")

    allow = allowlist or list(_DEFAULT_READ_ONLY)
    deny = denylist or []

    paths = spec_dict.get("paths") or {}
    registered: list[str] = []

    for path, methods in paths.items():
        if not isinstance(methods, dict):
            continue
        for method, op in methods.items():
            if method.lower() not in {"get", "post", "put", "patch", "delete"}:
                continue
            if not isinstance(op, dict):
                continue

            tool_name = _derive_name(op, method, path, name_from)
            if not tool_name:
                continue

            if not any(fnmatch.fnmatch(tool_name, pat) for pat in allow):
                logger.debug("from_openapi_spec: %s filtered out by allowlist", tool_name)
                continue
            if any(fnmatch.fnmatch(tool_name, pat) for pat in deny):
                logger.debug("from_openapi_spec: %s filtered out by denylist", tool_name)
                continue

            handler = _build_handler(
                method=method.lower(),
                path=path,
                base_url=base_url,
                op_spec=op,
                auth_header=auth_header or {},
                timeout=timeout,
            )

            description = op.get("summary") or op.get("description") or f"{method.upper()} {path}"

            try:
                kazi.add_tool(
                    handler,
                    name=tool_name,
                    description=description,
                    category=category,
                )
                registered.append(tool_name)
            except Exception as exc:
                logger.warning("from_openapi_spec: failed to register %s: %s", tool_name, exc)

    logger.info(
        "from_openapi_spec: %d tool(s) registered (base_url=%s)",
        len(registered), base_url,
    )
    return registered


def _derive_name(op: dict, method: str, path: str, name_from: str) -> str:
    """Pick a tool name for an operation."""
    if name_from == "operationId":
        op_id = op.get("operationId")
        if op_id:
            return _slug(op_id)
    # Fallback: method + path → method_segment_segment
    segments = [s for s in path.split("/") if s and not s.startswith("{")]
    return _slug(f"{method}_{'_'.join(segments) or 'root'}")


def _slug(s: str) -> str:
    """Normalise to a registry-safe identifier."""
    return "".join(c if c.isalnum() or c == "_" else "_" for c in s).strip("_")


def _build_handler(
    *,
    method: str,
    path: str,
    base_url: str,
    op_spec: dict,
    auth_header: dict[str, str],
    timeout: float,
) -> Callable:
    """
    Build an async tool handler that issues the HTTP request for this operation.

    Parameter extraction:
      - Path params (in: path) are substituted into the URL template.
      - Query params (in: query) become URL query string.
      - JSON body params are sent as the request body when present.

    The handler is intentionally simple — it does not bind to a specific
    parameter schema.  The agent passes a plain dict of kwargs; we route
    each to the right place based on the op spec.
    """
    import httpx

    params_spec = op_spec.get("parameters") or []
    path_param_names = {p["name"] for p in params_spec if p.get("in") == "path"}
    query_param_names = {p["name"] for p in params_spec if p.get("in") == "query"}

    async def handler(**kwargs: Any) -> str:
        # Split kwargs into path / query / body buckets
        path_args = {k: kwargs.pop(k) for k in list(kwargs) if k in path_param_names}
        query_args = {k: kwargs.pop(k) for k in list(kwargs) if k in query_param_names}
        body = kwargs  # whatever's left becomes the JSON body

        # Substitute path params
        url_path = path
        for name, value in path_args.items():
            url_path = url_path.replace("{" + name + "}", str(value))
        url = base_url + url_path

        async with httpx.AsyncClient(timeout=timeout) as client:
            req_kwargs: dict[str, Any] = {"headers": dict(auth_header)}
            if query_args:
                req_kwargs["params"] = query_args
            if body and method in {"post", "put", "patch"}:
                req_kwargs["json"] = body
            resp = await client.request(method, url, **req_kwargs)
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"HTTP {resp.status_code} from {method.upper()} {url}: "
                    f"{resp.text[:200]}"
                )
            # Return the response body — truncated by ContentPolicy downstream
            ctype = resp.headers.get("content-type", "")
            if "json" in ctype:
                return str(resp.json())
            return resp.text

    # Give the handler a useful __name__ so registry inference picks it up
    handler.__name__ = _slug(op_spec.get("operationId") or f"{method}_{path}")
    return handler
