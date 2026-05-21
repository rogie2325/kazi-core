"""
Example 8: Production FastAPI server

Deploy kazi as a REST API + WebSocket voice server for your React app.

Routes:
  POST /run           — single-turn chat (JSON in, JSON out)
  POST /stream        — SSE token stream for real-time UI
  POST /ingest        — background document ingestion
  WS   /voice         — WebSocket real-time voice
  GET  /health        — liveness/readiness probe
  GET  /metrics       — usage counters

Install::

    pip install kazi-core[anthropic,serve]

Run::

    python examples/08_serve_api.py
    # or in production:
    uvicorn examples.08_serve_api:app --host 0.0.0.0 --port 8000 --workers 4

Calling from React (fetch example)::

    const res = await fetch("http://localhost:8000/run", {
        method: "POST",
        headers: {"Content-Type": "application/json", "Authorization": "Bearer your-key"},
        body: JSON.stringify({message: "Hello!", thread_id: "user:alice:1"}),
    });
    const {reply} = await res.json();

SSE streaming in React::

    const source = new EventSource("/stream?message=Hello&thread_id=user:alice:1", {
        headers: {"Authorization": "Bearer your-key"},
    });
    source.onmessage = (e) => setReply(prev => prev + e.data);
"""
import asyncio
import os

from kazi import Kazi, KaziConfig, LLMConfig, LLMProvider, MemoryConfig, MemoryBackend
from kazi import web_search_tool, python_sandbox_tool


async def build_kazi():
    config = KaziConfig(
        llm=LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            model="claude-sonnet-4-6",
        ),
        memory=MemoryConfig(
            backend=MemoryBackend.SQLITE,
            connection_string="sqlite:///api_memory.db",
        ),
    )
    kazi = await Kazi.create(config)
    kazi.registry.register(web_search_tool(), category="search")
    kazi.registry.register(python_sandbox_tool(timeout=10), category="compute")
    return kazi


async def main():
    import uvicorn

    kazi = await build_kazi()

    app = kazi.as_app(
        api_key=os.getenv("KAZI_API_KEY", "change-me-in-production"),
        cors_origins=[
            "http://localhost:3000",   # React dev server
            "https://yourapp.com",     # production
        ],
        rate_limit_per_minute=60,      # per-IP rate limiting
    )

    print("Starting kazi API server on http://0.0.0.0:8000")
    print("  POST /run      — chat")
    print("  POST /stream   — SSE stream")
    print("  POST /ingest   — ingest documents")
    print("  WS   /voice    — real-time voice")
    print("  GET  /health   — health check")

    config = uvicorn.Config(app, host="0.0.0.0", port=8000, lifespan="on")
    server = uvicorn.Server(config)
    await server.serve()


# For deployment with `uvicorn examples.08_serve_api:app --host 0.0.0.0 --port 8000`
app = None  # populated at module load when using uvicorn directly


async def _init_app():
    global app
    kazi = await build_kazi()
    app = kazi.as_app(
        api_key=os.getenv("KAZI_API_KEY", "change-me-in-production"),
        cors_origins=["http://localhost:3000", "https://yourapp.com"],
        rate_limit_per_minute=60,
    )


if __name__ == "__main__":
    asyncio.run(main())
