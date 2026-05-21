"""
Example 11: Strongest production setup with OpenRouter

OpenRouter gives you access to 100+ models (Claude, GPT, Llama, Mistral,
DeepSeek, Qwen, Gemini, etc.) through a single API key with an
OpenAI-compatible endpoint.

This example shows the full production stack:

  Primary model    → Claude Sonnet 4.6 via OpenRouter
  Tool-call turns  → Mistral Small (fast, cheap for mechanical turns)
  Fallback model   → Llama 3.3 70B via OpenRouter (open-source, no outage correlation)
  Summarizer       → DeepSeek R1 (cheap, accurate for compression)

  Circuit breaker  → trips after 5 failures, cools down for 60s
  Retry attempts   → 3 per model, Retry-After respected on 429s
  Memory           → Redis (persistent, cluster-ready)
  Security         → Bearer auth, rate limiting, security headers, audit log
  Observability    → OpenTelemetry spans + MLflow metrics

The key insight: OpenRouter + kazi routing gives you TWO layers of resilience:
  1. kazi retries within a model (handles transient errors)
  2. kazi switches to fallback (handles sustained provider outages)
  3. OpenRouter can also do its own routing underneath (optional)

Install::

    pip install kazi-core[openai,serve,redis]
    pip install mlflow  # optional

Env vars::

    OPENROUTER_API_KEY=sk-or-v1-...
    REDIS_URL=redis://localhost:6379

Run::

    python examples/11_openrouter_strongest_setup.py
"""
from __future__ import annotations

import asyncio
import os
import time

from kazi import (
    Kazi,
    KaziConfig,
    LLMConfig,
    LLMProvider,
    MemoryConfig,
    MemoryBackend,
    UserProfile,
    ExperimentTracker,
    web_search_tool,
    python_sandbox_tool,
    configure_logging,
)
from kazi.core.router import ModelRoute, RouterConfig
from kazi.core.secrets import SecretRef

# ── Constants ──────────────────────────────────────────────────────────────────

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_KEY = SecretRef.from_env("OPENROUTER_API_KEY")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# ── Config ─────────────────────────────────────────────────────────────────────

def make_config() -> KaziConfig:
    return KaziConfig(
        # ── Primary: Claude Sonnet 4.6 via OpenRouter ─────────────────────
        # OpenRouter uses the OpenAI-compatible API format.
        # Set provider=OPENAI and point base_url at OpenRouter.
        llm=LLMConfig(
            provider=LLMProvider.OPENAI,
            model="anthropic/claude-sonnet-4-6",
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_KEY,
            temperature=0.1,
            max_tokens=4096,
        ),

        # ── Model routing ─────────────────────────────────────────────────
        router=RouterConfig(
            # Fallback: Llama 3.3 70B — open source, zero outage correlation with Claude
            # If Anthropic is down, Llama 3.3 via OpenRouter is almost certainly up.
            fallback=ModelRoute(
                model="meta-llama/llama-3.3-70b-instruct",
                provider="openai",
                base_url=OPENROUTER_BASE_URL,
                api_key=OPENROUTER_KEY,
            ),
            # Tool-call turns: use Mistral Small — 5-10× cheaper, ~same quality for
            # mechanical "which tool should I call next" decisions.
            tool_call=ModelRoute(
                model="mistralai/mistral-small-3.1-24b-instruct",
                provider="openai",
                base_url=OPENROUTER_BASE_URL,
                api_key=OPENROUTER_KEY,
            ),
            # Summarizer: DeepSeek R1 — very cheap, strong at compression tasks.
            summarizer=ModelRoute(
                model="deepseek/deepseek-r1",
                provider="openai",
                base_url=OPENROUTER_BASE_URL,
                api_key=OPENROUTER_KEY,
            ),
            # Retry config
            max_retry_attempts=3,
            retry_base_delay=1.0,
            # Circuit breaker: trip after 5 consecutive failures, cool for 60s.
            # When open, requests skip straight to fallback without wasting time.
            circuit_breaker_threshold=5,
            circuit_breaker_cooldown=60.0,
        ),

        # ── Persistent memory: Redis ───────────────────────────────────────
        # Redis survives restarts and works across multiple server instances.
        # Switch to MemoryBackend.POSTGRES for SQL-based checkpointing.
        memory=MemoryConfig(
            backend=MemoryBackend.REDIS,
            connection_string=REDIS_URL,
            max_conversation_turns=100,
        ),

        # ── Per-tenant tool isolation ─────────────────────────────────────
        tenant_tools={
            "tenant:free": {"web_search"},
            "tenant:pro": {"web_search", "execute_python"},
        },
    )


# ── Deploy as FastAPI server ───────────────────────────────────────────────────

async def run_server():
    """
    Spin up the production API server with all security layers enabled.
    Point your React frontend at http://localhost:8000.
    """
    import uvicorn
    configure_logging(level="INFO")

    config = make_config()
    kazi = await Kazi.create(config)

    kazi.registry.register(web_search_tool(max_results=10), category="search")
    kazi.registry.register(python_sandbox_tool(timeout=15), category="compute")

    app = kazi.as_app(
        prefix="/api/v1",
        api_key=os.environ["KAZI_API_KEY"],          # required in prod — set the env var
        cors_origins=[
            "http://localhost:3000",                  # React dev
            "https://yourapp.com",                    # production frontend
        ],
        rate_limit_per_minute=60,                     # 60 req/min per IP
        max_body_bytes=512 * 1024,                    # 512 KB max body
        enable_audit_log=True,                        # logs IP + thread_id, never message content
        # allowed_ips=["10.0.0.0/8"],                # uncomment to allowlist internal IPs only
    )

    print("kazi API live — http://0.0.0.0:8000/api/v1")
    print("Routes: POST /api/v1/run | POST /api/v1/stream | WS /api/v1/voice | GET /api/v1/health")

    server_config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        # In production: run behind nginx/Caddy for TLS termination
        # ssl_keyfile="./key.pem",
        # ssl_certfile="./cert.pem",
        workers=1,  # kazi uses async; single worker is fine; use gunicorn for multiprocess
        lifespan="on",
        access_log=True,
    )
    server = uvicorn.Server(server_config)
    await server.serve()


# ── Chat example: all resilience layers active ────────────────────────────────

async def chat_example():
    """
    Show the full retry → circuit breaker → fallback → per-tenant flow
    in a simple chat interaction.
    """
    configure_logging(level="INFO")

    config = make_config()
    tracker = ExperimentTracker(backend="mlflow", project="openrouter-prod")
    profile = UserProfile()

    # Store user preferences once (at onboarding time in a real app)
    profile.update("alice", {
        "role": "senior engineer",
        "prefers": "concise technical answers",
        "timezone": "UTC",
    })

    async with await Kazi.create(config) as kazi:
        kazi.registry.register(web_search_tool(), category="search")
        kazi.registry.register(python_sandbox_tool(timeout=15), category="compute")

        prompts = [
            ("What are the performance tradeoffs between Llama 3.3 and Claude Sonnet?", "tenant:pro"),
            ("Write Python code to benchmark two sorting algorithms", "tenant:pro"),
            ("Search for the latest OpenRouter pricing", "tenant:free"),   # free tier: search only
        ]

        for msg, tenant in prompts:
            t0 = time.monotonic()
            result = await kazi.run(
                msg,
                user_id="alice",          # injects profile preamble automatically
                tenant_id=tenant,         # restricts visible tools by plan tier
                thread_id="alice:demo",
                track_cost=True,
            )
            elapsed_ms = (time.monotonic() - t0) * 1000

            tracker.log_run_result(
                message=msg,
                result=result,
                model=config.llm.model,
                thread_id="alice:demo",
                latency_ms=elapsed_ms,
                extra={"tenant": tenant},
            )

            print(f"\nQ [{tenant}]: {msg}")
            print(f"A: {result.reply[:200]}…")
            print(f"   ${result.cost.cost_usd:.5f}  {elapsed_ms:.0f}ms  "
                  f"{result.cost.input_tokens}in/{result.cost.output_tokens}out")

    tracker.finish()


# ── React fetch snippet ────────────────────────────────────────────────────────
#
# // POST /api/v1/run from React
# const res = await fetch("http://localhost:8000/api/v1/run", {
#     method: "POST",
#     headers: {
#         "Content-Type": "application/json",
#         "Authorization": "Bearer " + process.env.REACT_APP_KAZI_KEY,
#     },
#     body: JSON.stringify({
#         message: userInput,
#         thread_id: `user:${userId}:${sessionId}`,
#         tenant_id: userPlan,   // "tenant:free" | "tenant:pro"
#         user_id: userId,
#     }),
# });
# const { reply, cost_usd } = await res.json();
#
# // SSE stream
# const source = new EventSource(
#     `http://localhost:8000/api/v1/stream?` + new URLSearchParams({
#         message: userInput,
#         thread_id: `user:${userId}`,
#     }),
#     { headers: { Authorization: "Bearer " + key } }   // requires EventSource polyfill for headers
# );
# source.onmessage = (e) => {
#     const { token } = JSON.parse(e.data);
#     if (token) setReply(prev => prev + token);
# };
#
# // WebSocket voice
# const ws = new WebSocket(`ws://localhost:8000/api/v1/voice?token=${apiKey}`);
# ws.onopen = () => ws.send(JSON.stringify({ action: "init", thread_id: `user:${userId}` }));
# ws.onmessage = (e) => {
#     if (typeof e.data === "string") {
#         const { event } = JSON.parse(e.data);
#         if (event === "done") setListening(false);
#     } else {
#         playAudioChunk(e.data);   // pipe binary MP3 to Web Audio API
#     }
# };


if __name__ == "__main__":
    # asyncio.run(run_server())   # start the API server
    asyncio.run(chat_example())   # run the chat demo
