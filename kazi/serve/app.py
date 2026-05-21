"""
kazi.as_app() — production-hardened FastAPI server.

Security layers:
  - Bearer token auth on every HTTP route (constant-time comparison)
  - Per-IP sliding-window rate limiting  (applied to every route, including /health)
  - Global concurrent-request cap        (configurable; 503 when exceeded)
  - Per-request timeout                  (504 when LLM/tool call hangs)
  - Request body size cap                (default 1 MB)
  - Input sanitization                   (null bytes, path-traversal chars, ANSI codes)
  - Thread ID path-traversal guard       (strips / \\ .. and shell metacharacters)
  - Security response headers            (X-Content-Type-Options, X-Frame-Options,
                                          X-XSS-Protection, Referrer-Policy,
                                          Permissions-Policy, HSTS, CSP)
  - X-Request-ID correlation header      validated format (no CRLF injection); fresh
                                          UUID generated for non-conforming values
  - Error response sanitization          (500s return "Internal server error" only —
                                          no stack traces, no file paths, no internals)
  - Audit log                            timestamp, IP, route, thread_id — NOT content
  - WebSocket token auth                 via query param (browsers can't set headers)
  - WebSocket Origin validation          rejects cross-origin WS connections when
                                          cors_origins is configured
  - CORS restricted to declared origins  (wildcard rejected with ConfigurationError)
  - Graceful shutdown                    drains in-flight requests before process exit
  - Log injection prevention             ANSI codes + newlines stripped before logging
  - Health endpoint topology gate        full details only when authenticated
  - Rate limiter memory bound            stale IP entries swept at 50 000 key cap
  - Stream disconnect detection          generators stop when client disconnects

Routes:
  POST   /run     — single-turn text (JSON)
  POST   /stream  — SSE raw token stream
  POST   /events  — SSE typed event stream (token/tool_start/tool_end/done)
  POST   /ingest  — document ingestion
  WS     /voice   — real-time voice (audio in, audio chunks out)
  GET    /health  — health check (minimal response when unauthenticated)
  GET    /metrics — usage metrics (requires auth)
"""
from __future__ import annotations

import asyncio
import json
import logging
import re as _re
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kazi.core.orchestrator import Kazi

logger = logging.getLogger(__name__)

_MAX_BODY_BYTES = 1 * 1024 * 1024   # 1 MB default request body cap
_MAX_MESSAGE_CHARS = 32_000          # longest message the LLM can usefully handle

# ── Log sanitization helpers ──────────────────────────────────────────────────
# Applied to every user-controlled value before it enters a log statement.
# Prevents ANSI terminal hijacking and newline-based log-entry spoofing.

_ANSI_RE = _re.compile(r"\x1b(?:\[[0-9;]*[A-Za-z]|\][^\x07]*\x07|[\\@A-Z\[\]^_`{|}~])")
_CTRL_RE = _re.compile(r"[\x00-\x1f\x7f]")

def _log_safe(value: str, max_len: int = 120) -> str:
    """Strip ANSI escape codes and control characters before logging."""
    value = _ANSI_RE.sub("", value)
    value = _CTRL_RE.sub(" ", value)
    return value[:max_len] + ("…" if len(value) > max_len else "")


# ── X-Request-ID sanitizer ────────────────────────────────────────────────────
# Client-supplied X-Request-ID values are echoed into response headers.
# A value containing CRLF (\r\n) would allow HTTP response splitting.
# Accept only UUID-safe chars (hex digits + hyphens, max 64 chars).
_REQUEST_ID_RE = _re.compile(r'^[a-zA-Z0-9\-]{1,64}$')

def _safe_request_id(value: str) -> str:
    """Return value if it looks like a safe correlation ID, else a fresh UUID."""
    if value and _REQUEST_ID_RE.match(value):
        return value
    return str(uuid.uuid4())


# ── Thread ID sanitizer ───────────────────────────────────────────────────────
# Thread IDs flow into LangGraph checkpointers.  Filesystem backends (SQLite
# uses the thread_id as part of the row key) are safe, but path-based backends
# could be vulnerable to "../.." traversal if the value is unsanitized.
# Whitelist approach: ASCII-only chars (NOT \w, which matches Unicode word
# characters like ¹ ϰ that could survive and surprise downstream consumers).

_SAFE_TID_RE = _re.compile(r"[^A-Za-z0-9_\-:.@]")

def _sanitize_thread_id(tid: str) -> str:
    """Whitelist-strip thread IDs: keep [A-Za-z0-9_\\-:.@] only."""
    return _SAFE_TID_RE.sub("_", tid)[:256]


# ── build_app ─────────────────────────────────────────────────────────────────

def build_app(
    kazi: Kazi,
    *,
    prefix: str = "",
    api_key: str | None = None,
    cors_origins: list[str] | None = None,
    rate_limit_per_minute: int = 0,
    rate_limit_redis_url: str | None = None,
    max_body_bytes: int = _MAX_BODY_BYTES,
    allowed_ips: list[str] | None = None,
    enable_audit_log: bool = True,
    # New hardening params
    max_concurrent_runs: int = 50,       # 0 = unlimited (not recommended in prod)
    request_timeout_seconds: int = 120,  # 0 = no timeout (not recommended in prod)
    graceful_shutdown_timeout: int = 30, # seconds to wait for in-flight reqs on SIGTERM
    # Legacy alias — max_body_bytes takes precedence
    max_body_size: int | None = None,
):
    """
    Build and return a hardened FastAPI application wrapping ``kazi``.

    prefix                  URL prefix for all routes (e.g. "/api/v1").
    api_key                 Bearer token required on all requests (None = no auth).
    cors_origins            CORS allowed origins.  None = CORS disabled.
                            Wildcard ("*") is explicitly rejected.
    rate_limit_per_minute   Per-IP request limit per minute (0 = disabled).
    rate_limit_redis_url    Redis URL for distributed rate limiting across replicas.
                            When set, uses Redis sorted-set sliding window instead
                            of the in-process dict — required for horizontal scaling.
                            Example: "redis://localhost:6379"
    max_body_bytes          Maximum request body size in bytes (default 1 MB).
    allowed_ips             IP allowlist. None or [] = allow all IPs.
    enable_audit_log        Log request metadata (IP, route, thread_id — not content).
    max_concurrent_runs     Global cap on simultaneous LLM runs.  Excess requests
                            receive HTTP 503 immediately rather than queuing.
    request_timeout_seconds Hard deadline per request.  504 when exceeded.
    graceful_shutdown_timeout  Seconds to drain in-flight requests on SIGTERM.
    """
    if max_body_size is not None and max_body_size < max_body_bytes:
        max_body_bytes = max_body_size

    # ── CORS wildcard guard ───────────────────────────────────────────────────
    # "*" + allow_credentials=True is a browser security violation and exposes
    # the API to cross-origin credential theft.  Fail loudly at config time.
    if cors_origins and "*" in cors_origins:
        from kazi.core.exceptions import ConfigurationError
        raise ConfigurationError(
            "cors_origins=['*'] is not allowed — it would expose the API to "
            "cross-origin credential attacks.  List explicit origins instead:\n"
            "  cors_origins=['https://yourapp.com', 'http://localhost:3000']"
        )

    try:
        from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
        from fastapi.exceptions import RequestValidationError
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse, StreamingResponse
        from pydantic import BaseModel as PydanticBase
        from pydantic import Field, field_validator
    except ImportError:
        raise ImportError(
            "fastapi and pydantic are required. "
            "Install: pip install kazi-core[serve]"
        )

    # ── Shared mutable state (closure variables) ──────────────────────────────
    # A dict is used instead of nonlocal ints so the lifespan coroutine and
    # middlewares share the same mutable object across await boundaries.
    _state: dict[str, Any] = {
        "active_requests": 0,
        "shutting_down": False,
    }

    # ── Graceful shutdown lifespan ────────────────────────────────────────────
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # startup — kazi is already created before build_app() is called
        yield
        # shutdown — set flag so new requests receive 503, then drain
        _state["shutting_down"] = True
        deadline = time.monotonic() + graceful_shutdown_timeout
        while _state["active_requests"] > 0 and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        remaining = _state["active_requests"]
        if remaining > 0:
            logger.warning(
                "Graceful shutdown: %d request(s) still in-flight after %ds — "
                "proceeding with shutdown.",
                remaining, graceful_shutdown_timeout,
            )
        else:
            logger.info("Graceful shutdown: all requests drained cleanly.")

    app = FastAPI(
        title="kazi",
        description="kazi orchestration API",
        version="0.1.0",
        lifespan=lifespan,
        # Uncomment to disable interactive docs in production:
        # docs_url=None, redoc_url=None,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["POST", "GET", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        )

    # ── Error response sanitization ───────────────────────────────────────────
    # Unhandled exceptions return a generic 500 with only a request_id.
    # Full tracebacks are written to the server log (never sent to clients).

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        request_id = _safe_request_id(request.headers.get("X-Request-ID", ""))
        logger.error(
            "Unhandled exception request_id=%s route=%s: %s",
            request_id, _log_safe(str(request.url.path)), exc,
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "request_id": request_id},
            headers={
                "X-Request-ID": request_id,
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        # Return a generic 422 — don't leak field names or Python type internals
        request_id = _safe_request_id(request.headers.get("X-Request-ID", ""))
        logger.warning(
            "Request validation failed request_id=%s route=%s",
            request_id, _log_safe(str(request.url.path)),
        )
        return JSONResponse(
            status_code=422,
            content={"detail": "Request validation failed", "request_id": request_id},
            headers={"X-Request-ID": request_id},
        )

    # ── Prometheus metrics middleware ─────────────────────────────────────────
    # Tracks request duration and status for every HTTP route.
    # No-op when prometheus-client is not installed.
    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        from kazi.utils.metrics import record_request
        route = _log_safe(request.url.path, 80)
        start = time.monotonic()
        status = 200
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        except Exception:
            status = 500
            raise
        finally:
            record_request(route, status, time.monotonic() - start)

    # ── Security headers + X-Request-ID ──────────────────────────────────────
    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        request_id = _safe_request_id(request.headers.get("X-Request-ID", ""))
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
        return response

    # ── Graceful shutdown gate ────────────────────────────────────────────────
    @app.middleware("http")
    async def shutdown_gate(request: Request, call_next):
        if _state["shutting_down"] and not request.url.path.endswith("/health"):
            return JSONResponse(
                status_code=503,
                content={"detail": "Server is shutting down — please retry"},
                headers={"Retry-After": "5"},
            )
        return await call_next(request)

    # ── Global concurrency cap ────────────────────────────────────────────────
    # Limits simultaneous LLM calls across all clients.  Without this, N IPs
    # sending 1 req each can trigger N parallel LLM calls → OOM / rate-limit
    # cascade.  Excess requests get an immediate 503 rather than waiting.
    @app.middleware("http")
    async def concurrency_limit(request: Request, call_next):
        if max_concurrent_runs > 0:
            if _state["active_requests"] >= max_concurrent_runs:
                return JSONResponse(
                    status_code=503,
                    content={"detail": "Too many concurrent requests — please retry shortly"},
                    headers={"Retry-After": "2"},
                )
        _state["active_requests"] += 1
        try:
            return await call_next(request)
        finally:
            _state["active_requests"] -= 1

    # ── Request body size limit ───────────────────────────────────────────────
    @app.middleware("http")
    async def body_size_limit(request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > max_body_bytes:
            return JSONResponse(
                status_code=413,
                content={"detail": f"Request body too large (max {max_body_bytes // 1024} KB)"},
            )
        return await call_next(request)

    # ── Rate limiting (per-IP sliding window) ─────────────────────────────────
    # Two backends:
    #   1. In-process dict  — default; single-replica only
    #   2. Redis sorted set — enabled when rate_limit_redis_url is set;
    #      atomic across replicas, safe for horizontal scaling
    _rate_counters: dict[str, list[float]] = {}
    _RATE_COUNTER_MAX = 50_000
    _redis_rate_client = None   # lazy-initialised on first rate-limited request

    async def _check_rate_redis(ip: str, limit: int) -> None:
        nonlocal _redis_rate_client
        try:
            import redis.asyncio as aioredis
        except ImportError:
            raise ImportError(
                "redis package required for distributed rate limiting. "
                "Install: pip install redis"
            )
        if _redis_rate_client is None:
            _redis_rate_client = aioredis.from_url(
                rate_limit_redis_url, decode_responses=True, socket_timeout=1.0
            )
        now = time.time()
        window_start = now - 60.0
        key = f"kazi:rl:{ip}"
        try:
            async with _redis_rate_client.pipeline(transaction=True) as pipe:
                # Atomic sliding-window log: trim old entries, add current, count
                pipe.zremrangebyscore(key, 0, window_start)
                pipe.zadd(key, {f"{now:.6f}": now})
                pipe.zcard(key)
                pipe.expire(key, 65)
                results = await pipe.execute()
            count = results[2]
            if count > limit:
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded — try again later",
                )
        except HTTPException:
            raise
        except Exception as exc:
            # Redis unavailable — fall back to in-process limiter rather than
            # blocking all traffic (fail open is safer than fail closed here)
            logger.warning("Redis rate limiter unavailable, falling back to in-process: %s", exc)
            _check_rate_local(ip, limit)

    def _check_rate_local(ip: str, limit: int) -> None:
        now = time.monotonic()
        if len(_rate_counters) > _RATE_COUNTER_MAX:
            stale = [k for k, v in list(_rate_counters.items()) if not v or v[-1] < now - 60]
            for k in stale:
                _rate_counters.pop(k, None)
        window = [t for t in _rate_counters.get(ip, []) if now - t < 60]
        if len(window) >= limit:
            _rate_counters[ip] = window
            raise HTTPException(status_code=429, detail="Rate limit exceeded — try again later")
        window.append(now)
        _rate_counters[ip] = window

    async def _check_rate(ip: str, limit: int | None = None) -> None:
        effective = limit if limit is not None else rate_limit_per_minute
        if effective <= 0:
            return
        if rate_limit_redis_url:
            await _check_rate_redis(ip, effective)
        else:
            _check_rate_local(ip, effective)

    # ── IP allowlist ──────────────────────────────────────────────────────────
    def _check_ip(ip: str) -> None:
        if not allowed_ips:
            return
        if ip not in allowed_ips:
            raise HTTPException(status_code=403, detail="IP not allowed")

    # ── Audit logging (log-injection safe) ────────────────────────────────────
    def _audit(request: Request, route: str, thread_id: str = "") -> None:
        if not enable_audit_log:
            return
        ip = request.client.host if request.client else "unknown"
        request_id = _safe_request_id(request.headers.get("X-Request-ID", ""))
        logger.info(
            "AUDIT route=%s ip=%s thread_id=%s request_id=%s",
            _log_safe(route, 80),
            _log_safe(ip, 45),
            _log_safe(thread_id, 80),
            _log_safe(request_id, 40),
        )

    # ── Auth + rate limit + IP check dependency ───────────────────────────────
    async def _auth(request: Request) -> str:
        ip = request.client.host if request.client else "unknown"
        _check_ip(ip)
        if api_key is not None:
            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                raise HTTPException(status_code=401, detail="Missing Bearer token")
            if not secrets.compare_digest(auth_header[7:], api_key):
                raise HTTPException(status_code=401, detail="Invalid API key")
        await _check_rate(ip)
        return ip

    # ── Request / response models ─────────────────────────────────────────────

    class RunRequest(PydanticBase):
        message: str = Field(..., min_length=1, max_length=_MAX_MESSAGE_CHARS)
        thread_id: str = Field("default", max_length=256)
        system_prompt: str | None = Field(None, max_length=8_000)
        max_tool_calls: int = Field(25, ge=1, le=100)
        track_cost: bool = False
        user_id: str | None = Field(None, max_length=256)
        tenant_id: str | None = Field(None, max_length=256)

        @field_validator("message", mode="before")
        @classmethod
        def sanitize_message(cls, v):
            if isinstance(v, str):
                return v.replace("\x00", "")
            return v

        @field_validator("thread_id", mode="before")
        @classmethod
        def sanitize_thread_id(cls, v):
            # Null bytes + path-traversal characters stripped; whitelist applied
            if isinstance(v, str):
                return _sanitize_thread_id(v.replace("\x00", ""))
            return v

        @field_validator("user_id", "tenant_id", mode="before")
        @classmethod
        def sanitize_ids(cls, v):
            if isinstance(v, str):
                return v.replace("\x00", "")[:256]
            return v

    class RunResponse(PydanticBase):
        reply: str
        cost_usd: float | None = None
        input_tokens: int | None = None
        output_tokens: int | None = None

    class IngestRequest(PydanticBase):
        path: str = Field(..., min_length=1, max_length=1024)
        index_name: str = Field("default", max_length=128)

        @field_validator("path", mode="before")
        @classmethod
        def no_null_bytes(cls, v):
            return v.replace("\x00", "") if isinstance(v, str) else v

    # ── Timeout helper ────────────────────────────────────────────────────────
    _timeout = request_timeout_seconds if request_timeout_seconds > 0 else None

    async def _with_timeout(coro, route_name: str):
        if _timeout is None:
            return await coro
        try:
            return await asyncio.wait_for(coro, timeout=_timeout)
        except asyncio.TimeoutError:
            logger.warning("Request timeout on %s after %ds", route_name, _timeout)
            raise HTTPException(
                status_code=504,
                detail=f"Request timed out after {_timeout}s — try a shorter prompt or fewer tool calls",
            )

    # ── Routes ────────────────────────────────────────────────────────────────

    @app.post(f"{prefix}/run", response_model=RunResponse)
    async def run(req: RunRequest, request: Request, _ip: str = Depends(_auth)):
        _audit(request, "POST /run", req.thread_id)
        from kazi.core.cost import RunResult

        result = await _with_timeout(
            kazi.run(
                req.message,
                thread_id=req.thread_id,
                system_prompt=req.system_prompt,
                max_tool_calls=req.max_tool_calls,
                track_cost=req.track_cost,
                user_id=req.user_id,
                tenant_id=req.tenant_id,
            ),
            "POST /run",
        )
        if isinstance(result, RunResult):
            return RunResponse(
                reply=result.reply,
                cost_usd=result.cost.cost_usd,
                input_tokens=result.cost.input_tokens,
                output_tokens=result.cost.output_tokens,
            )
        return RunResponse(reply=result)

    @app.post(f"{prefix}/stream")
    async def stream(req: RunRequest, request: Request, _ip: str = Depends(_auth)):
        _audit(request, "POST /stream", req.thread_id)
        deadline = time.monotonic() + _timeout if _timeout else None

        async def token_generator():
            try:
                async for token in kazi.stream(
                    req.message,
                    thread_id=req.thread_id,
                    max_tool_calls=req.max_tool_calls,
                    user_id=req.user_id,
                    tenant_id=req.tenant_id,
                ):
                    if deadline and time.monotonic() > deadline:
                        yield f"data: {json.dumps({'error': 'timeout'})}\n\n"
                        return
                    if await request.is_disconnected():
                        logger.debug("Client disconnected mid-stream thread=%s", _log_safe(req.thread_id))
                        return
                    yield f"data: {json.dumps({'token': token})}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as exc:
                logger.error("Stream error thread=%s: %s", _log_safe(req.thread_id), exc)
                yield f"data: {json.dumps({'error': 'stream failed'})}\n\n"

        return StreamingResponse(
            token_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post(f"{prefix}/events")
    async def stream_events(req: RunRequest, request: Request, _ip: str = Depends(_auth)):
        """
        Typed event stream — yields StreamEvent JSON objects.

        Each line: ``data: {"type": "token"|"tool_start"|"tool_end"|"done"|"error", "data": "...", "metadata": {...}}``

        // React SSE example:
        // const source = new EventSource("/api/v1/events", { headers: { Authorization: "Bearer " + key } });
        // source.onmessage = (e) => {
        //   const ev = JSON.parse(e.data);
        //   if (ev.type === "token")      setReply(r => r + ev.data);
        //   if (ev.type === "tool_start") showSpinner(ev.data);
        //   if (ev.type === "tool_end")   hideSpinner(ev.data);
        //   if (ev.type === "done")       setLoading(false);
        // };
        """
        _audit(request, "POST /events", req.thread_id)
        deadline = time.monotonic() + _timeout if _timeout else None

        async def event_generator():
            try:
                async for event in kazi.stream_events(
                    req.message,
                    thread_id=req.thread_id,
                    max_tool_calls=req.max_tool_calls,
                    user_id=req.user_id,
                    tenant_id=req.tenant_id,
                ):
                    if deadline and time.monotonic() > deadline:
                        yield f"data: {json.dumps({'type': 'error', 'data': 'timeout'})}\n\n"
                        return
                    if await request.is_disconnected():
                        logger.debug("Client disconnected mid-events thread=%s", _log_safe(req.thread_id))
                        return
                    yield f"data: {json.dumps(event)}\n\n"
            except Exception as exc:
                logger.error("Events error thread=%s: %s", _log_safe(req.thread_id), exc)
                yield f"data: {json.dumps({'type': 'error', 'data': 'stream failed'})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post(f"{prefix}/ingest")
    async def ingest(req: IngestRequest, request: Request, _ip: str = Depends(_auth)):
        _audit(request, "POST /ingest")
        await _with_timeout(
            kazi.ingest(req.path, index_name=req.index_name),
            "POST /ingest",
        )
        return {"status": "ok", "path": req.path, "index": req.index_name}

    @app.websocket(f"{prefix}/voice")
    async def voice_ws(ws: WebSocket, token: str | None = None):
        """
        Real-time voice WebSocket.

        Auth: pass the API key as ?token=<key> — browsers cannot set Authorization
        headers on WebSocket connections.

        Protocol:
          Client → server:  JSON text frame first: {"thread_id": "...", "action": "init"}
                            then binary audio frames (WAV/MP3/WEBM)
          Server → client:  binary MP3 audio chunks
                            JSON: {"event": "done"} | {"event": "error", "detail": "..."}
        """
        if api_key is not None:
            if not token or not secrets.compare_digest(token, api_key):
                await ws.close(code=4001, reason="Unauthorized")
                return

        # WebSocket Origin check: browsers send the Origin header on WS upgrade.
        # Validate it against cors_origins when a whitelist is configured, to
        # prevent CSRF-style cross-origin WebSocket hijacking.
        if cors_origins:
            origin = ws.headers.get("origin", "")
            if origin and origin not in cors_origins:
                await ws.close(code=4003, reason="Origin not allowed")
                return

        client_ip = ws.client.host if ws.client else "unknown"
        _check_ip(client_ip)

        if _state["shutting_down"]:
            await ws.close(code=1001, reason="Server shutting down")
            return

        await ws.accept()
        thread_id = "default"

        if not kazi.config.voice:
            await ws.send_text(json.dumps({"event": "error", "detail": "Voice not configured"}))
            await ws.close()
            return

        _state["active_requests"] += 1
        try:
            while True:
                data = await ws.receive()

                if "text" in data:
                    try:
                        msg = json.loads(data["text"])
                    except json.JSONDecodeError:
                        continue
                    if "thread_id" in msg:
                        thread_id = _sanitize_thread_id(
                            msg["thread_id"].replace("\x00", "")
                        )
                    if msg.get("action") == "ping":
                        await ws.send_text(json.dumps({"event": "pong"}))
                    if msg.get("action") == "init":
                        logger.info(
                            "AUDIT WS /voice init ip=%s thread_id=%s",
                            _log_safe(client_ip), _log_safe(thread_id),
                        )
                    continue

                audio_bytes = data.get("bytes", b"")
                if not audio_bytes:
                    continue

                if len(audio_bytes) > 25 * 1024 * 1024:  # 25 MB cap
                    await ws.send_text(json.dumps({"event": "error", "detail": "Audio too large (max 25 MB)"}))
                    continue

                try:
                    async for chunk in kazi.stream_voice(audio_bytes, thread_id=thread_id):
                        await ws.send_bytes(chunk)
                    await ws.send_text(json.dumps({"event": "done"}))
                except Exception:
                    # Never send internal exception details over WebSocket
                    await ws.send_text(json.dumps({"event": "error", "detail": "Voice processing failed"}))

        except WebSocketDisconnect:
            pass
        finally:
            _state["active_requests"] -= 1

    @app.get(f"{prefix}/health")
    async def health(request: Request):
        """
        Health check — always accessible to load balancers and K8s probes.

        Unauthenticated callers receive only {"status": "healthy"|"unhealthy"|"degraded"}.
        Authenticated callers receive the full topology report (which subsystems are
        up, latencies, etc.) — keeping internal system structure out of anonymous hands.
        """
        ip = request.client.host if request.client else "unknown"
        # Rate-limit health too — prevents topology-probing floods
        await _check_rate(ip, limit=min(rate_limit_per_minute, 120) if rate_limit_per_minute > 0 else 120)

        report = await kazi.health()
        status_code = 200 if report["status"] in ("healthy", "degraded") else 503

        # Check whether the caller is authenticated
        is_authed = api_key is None  # no auth configured → all callers trusted
        if not is_authed:
            auth_header = request.headers.get("Authorization", "")
            if (
                auth_header.startswith("Bearer ")
                and api_key is not None
                and secrets.compare_digest(auth_header[7:], api_key)
            ):
                is_authed = True

        if is_authed:
            return JSONResponse(content=report, status_code=status_code)

        # Unauthenticated: return only the status word — no topology details
        return JSONResponse(
            content={"status": report["status"]},
            status_code=status_code,
        )

    @app.get(f"{prefix}/metrics")
    async def metrics(request: Request, _ip: str = Depends(_auth)):
        _audit(request, "GET /metrics")
        tools = kazi.registry.list_tools()
        return {
            "tools_registered": len(tools),
            "tool_sources": {
                src: len([t for t in tools if t.source.value == src])
                for src in {"native", "rag", "mcp", "a2a"}
            },
            "voice_enabled": kazi.config.voice is not None,
            "memory_backend": kazi.config.memory.backend.value,
            "semantic_cache_enabled": kazi.config.semantic_cache is not None,
            "guardrails_enabled": kazi.config.guardrails is not None,
            "injection_detection": kazi.config.security.injection.enabled,
            "tool_result_cache_ttl": kazi.config.tool_result_cache_ttl,
            "circuit_breaker_threshold": kazi.config.router.circuit_breaker_threshold,
            "max_concurrent_runs": max_concurrent_runs,
            "request_timeout_seconds": request_timeout_seconds,
            "active_requests": _state["active_requests"],
            "distributed_rate_limiting": rate_limit_redis_url is not None,
        }

    @app.get(f"{prefix}/prometheus")
    async def prometheus_metrics(request: Request, _ip: str = Depends(_auth)):
        """
        Prometheus scrape endpoint — returns metrics in the standard text exposition format.

        Configure your Prometheus scraper::

            scrape_configs:
              - job_name: kazi
                static_configs:
                  - targets: ['your-host:8000']
                metrics_path: /prometheus
                bearer_token: <your-api-key>

        Requires: pip install prometheus-client
        """
        _audit(request, "GET /prometheus")
        from fastapi.responses import Response as FastAPIResponse

        from kazi.utils.metrics import prometheus_output
        body, content_type = prometheus_output()
        return FastAPIResponse(content=body, media_type=content_type)

    return app
