# Kazi Orchestrate

[![CI](https://github.com/erose2502/kazi-core/actions/workflows/ci.yml/badge.svg)](https://github.com/erose2502/kazi-core/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/erose2502/kazi-core/branch/main/graph/badge.svg)](https://codecov.io/gh/erose2502/kazi-core)
[![PyPI](https://img.shields.io/pypi/v/kazi-core)](https://pypi.org/project/kazi-core/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Production-grade AI orchestration in 20 lines. Wires together LangGraph (stateful execution), LlamaIndex (RAG), MCP (tool protocols), and A2A (agent delegation) behind a single clean API.

```python
import asyncio
from kazi import Kazi, KaziConfig, LLMConfig, LLMProvider

config = KaziConfig(llm=LLMConfig(provider=LLMProvider.ANTHROPIC, model="claude-sonnet-4-6"))

async def main():
    async with await Kazi.create(config) as kazi:
        await kazi.ingest("./docs")
        print(await kazi.run("Summarise the key points from the docs"))

asyncio.run(main())
```

---

## What it does

Kazi Orchestrate composes four layers that otherwise require weeks of wiring:

| Layer | Library | What it gives you |
|---|---|---|
| Stateful execution | LangGraph | Persistent conversation memory, tool loops, streaming |
| Knowledge retrieval | LlamaIndex | Document ingestion, vector search, hybrid RAG |
| Tool protocols | MCP | Connect any MCP server; tools appear automatically |
| Agent delegation | A2A | Discover and delegate to specialist agents over HTTP |

Every layer is optional — use only what you need.

---

## Install

```bash
# Core + one LLM provider
pip install kazi-core[anthropic]
pip install kazi-core[openai]
pip install kazi-core[google]

# Individual layers
pip install kazi-base      # registry, config, security
pip install kazi-rag       # LlamaIndex RAG layer
pip install kazi-mcp       # MCP client bridge
pip install kazi-a2a       # A2A agent bridge
pip install kazi-voice     # Real-time voice (STT + TTS)

# Production add-ons
pip install kazi-core[serve]        # FastAPI server
pip install kazi-core[arq]          # ARQ async job queue
pip install kazi-core[celery]       # Celery job queue
pip install kazi-core[data]         # pandas dataframe tools
pip install kazi-core[data-polars]  # Polars dataframe tools
pip install kazi-core[redis]        # Redis memory backend
pip install kazi-core[postgres]     # Postgres memory backend
```

---

## Quick start

### Chat with your documents

```python
import asyncio
from kazi import Kazi, KaziConfig, LLMConfig, LLMProvider, RAGConfig, MemoryConfig, MemoryBackend

async def main():
    config = KaziConfig(
        llm=LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4o"),
        memory=MemoryConfig(backend=MemoryBackend.SQLITE, connection_string="sqlite:///memory.db"),
    )
    async with await Kazi.create(config) as kazi:
        await kazi.ingest("./company_docs", index_name="company")
        r1 = await kazi.run("What are our Q3 OKRs?", thread_id="alice:session-1")
        r2 = await kazi.run("Which team owns the top one?", thread_id="alice:session-1")

asyncio.run(main())
```

### Streaming

```python
async with await Kazi.create(config) as kazi:
    async for token in kazi.stream("Explain quantum entanglement simply."):
        print(token, end="", flush=True)
```

### Multi-modal (images + text)

```python
# Pass local paths, URLs, or raw bytes — kazi handles encoding automatically
reply = await kazi.run(
    "What trends do you see in this chart?",
    images=["./q3_revenue.png"],           # local file
    # images=["https://example.com/chart.png"],  # URL
    # images=[raw_bytes],                         # bytes
)
```

Vision-capable models: Claude Sonnet 4.6+, GPT-4o.

### Custom tools

```python
async def get_weather(city: str) -> str:
    """Return current weather for a city."""
    ...

async with await Kazi.create(config) as kazi:
    kazi.add_tool(get_weather, description="Get live weather for any city")
    result = await kazi.run("What's the weather in Tokyo right now?")
```

### MCP server

```python
from kazi import KaziConfig, MCPConfig

config = KaziConfig(
    mcp=MCPConfig(servers={
        "filesystem": "npx -y @modelcontextprotocol/server-filesystem /tmp",
        "github": "npx -y @modelcontextprotocol/server-github",
    })
)

async with await Kazi.create(config) as kazi:
    result = await kazi.run("List the files in /tmp and summarise what you find.")
```

### A2A agent delegation

```python
from kazi import KaziConfig, A2AConfig

config = KaziConfig(
    a2a=A2AConfig(
        discovery_endpoints=["https://agents.internal/summarizer"],
        allowed_hosts=["agents.internal"],
    )
)

async with await Kazi.create(config) as kazi:
    result = await kazi.run("Summarise this 50-page report: ...")
```

---

## Voice agents

Real-time voice with cross-modal memory. Voice and chat sessions share the same
LangGraph state — the agent remembers what was said across both modalities.

```bash
pip install kazi-voice[openai]          # Whisper STT + OpenAI TTS
pip install kazi-voice[deepgram]        # Deepgram STT
pip install kazi-voice[elevenlabs]      # ElevenLabs TTS
```

```python
from kazi import KaziConfig, VoiceConfig, STTProvider, TTSProvider

config = KaziConfig(
    llm=LLMConfig(provider=LLMProvider.ANTHROPIC, model="claude-sonnet-4-6"),
    voice=VoiceConfig(
        stt_provider=STTProvider.OPENAI,    # OPENAI | DEEPGRAM
        tts_provider=TTSProvider.OPENAI,    # OPENAI | ELEVENLABS
        tts_voice="nova",
    ),
)

async with await Kazi.create(config) as kazi:
    # Push-to-talk: audio bytes in → MP3 bytes out
    reply_audio = await kazi.run_voice(audio_bytes, thread_id="user:alice")

    # Streaming: first audio chunk arrives before LLM finishes (~500ms latency)
    async for chunk in kazi.stream_voice(audio_bytes, thread_id="user:alice"):
        await websocket.send_bytes(chunk)

    # Text turn — agent remembers the voice conversation
    reply = await kazi.run("What did I just ask you?", thread_id="user:alice")
```

---

## Sub-agents and Supervisor

Build multi-agent systems where specialized agents handle different domains.
Each sub-agent has its own personality, tool set, and memory namespace.

```python
from kazi.agents.subagent import SubAgent, SubAgentConfig
from kazi.agents.supervisor import Supervisor

async with await Kazi.create(config) as kazi:
    research_agent = SubAgent(kazi, SubAgentConfig(
        name="research",
        role="Research Specialist",
        system_prompt="You find accurate information using web search. Always cite sources.",
        tools=["web_search"],               # restricted tool set
    ))

    analyst_agent = SubAgent(kazi, SubAgentConfig(
        name="analyst",
        role="Data Analyst",
        system_prompt="You write and run Python code to process data and generate insights.",
        tools=["python_sandbox"],
        llm_override=ModelRoute(model="gpt-4o", provider="openai"),  # cross-model routing
    ))

    # Manual routing
    result = await research_agent.run("What are the top AI frameworks in 2025?")

    # Supervisor auto-routes based on query content
    supervisor = Supervisor(kazi, agents=[research_agent, analyst_agent])
    result = await supervisor.run("Calculate compound interest on $10,000 at 8% for 10 years")
```

Cross-model routing: each agent can use a different LLM provider while maintaining
consistent personality via system_prompt injection on every turn.

---

## Long-term user memory

Store cross-session user preferences that are automatically injected into every system prompt.

```python
from kazi import UserProfile

profile = UserProfile(storage_dir=".kazi_profiles")
profile.update("alice", {
    "role": "senior data scientist",
    "prefers": "concise bullet-point answers",
    "expertise": "Python, SQL, machine learning",
})

async with await Kazi.create(config) as kazi:
    # Agent automatically knows Alice's background on every turn
    reply = await kazi.run(
        "Recommend a feature engineering approach for tabular data.",
        user_id="alice",
        thread_id="alice:session-42",
    )
```

---

## Human-in-the-loop

Pause execution before tool calls for human review, modification, or rejection.

```python
async def my_approval(tool_calls):
    for call in tool_calls:
        print(f"Agent wants: {call['name']}({call['args']})")
    answer = input("Approve? [y/n]: ")
    return tool_calls if answer == "y" else None   # None = reject

reply = await kazi.run_with_approval(
    "Delete all staging resources",
    approval_callback=my_approval,
    thread_id="ops-session",
)
```

---

## Structured output

Parse the agent's reply into a Pydantic model with a single parameter.

```python
from pydantic import BaseModel

class Analysis(BaseModel):
    summary: str
    sentiment: str
    confidence: float

result: Analysis = await kazi.run(
    "Analyse this customer review: 'Great product but slow shipping'",
    response_schema=Analysis,
)
print(result.sentiment)  # "mixed"
```

---

## Cost tracking

```python
result = await kazi.run("Summarise the Q3 report", track_cost=True)
# result is RunResult(reply="...", cost=RunCost(cost_usd=0.0023, input_tokens=1200, output_tokens=340))
print(f"${result.cost.cost_usd:.4f}  {result.cost.input_tokens} in / {result.cost.output_tokens} out")
```

Pricing is built-in for claude-opus-4-7, claude-sonnet-4-6, claude-haiku-4-5, gpt-4o, gpt-4o-mini, and more.

---

## Experiment tracking

Log token counts, latency, and cost to MLflow or Weights & Biases.

```bash
pip install mlflow    # or: pip install wandb
```

```python
from kazi import ExperimentTracker
import time

tracker = ExperimentTracker(backend="mlflow", project="my-agent")

t0 = time.monotonic()
result = await kazi.run("Analyse Q3 results", track_cost=True)
tracker.log_run_result(
    message="Analyse Q3 results",
    result=result,
    model=config.llm.model,
    latency_ms=(time.monotonic() - t0) * 1000,
)
```

---

## Deploy as a REST API

```bash
pip install kazi-core[serve]
```

```python
import uvicorn

async with await Kazi.create(config) as kazi:
    app = kazi.as_app(
        api_key="your-secret",
        cors_origins=["https://yourapp.com"],
        rate_limit_per_minute=60,
    )
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Routes: `POST /run`, `POST /stream` (SSE), `POST /ingest`, `WS /voice`, `GET /health`, `GET /metrics`.

Call from React:
```js
const res = await fetch("/run", {
    method: "POST",
    headers: {"Content-Type": "application/json", "Authorization": "Bearer your-secret"},
    body: JSON.stringify({message: "Hello!", thread_id: "user:alice:1"}),
});
const {reply} = await res.json();
```

---

## Background job queues

For long-running agent tasks that shouldn't block your API response.

```bash
pip install kazi-core[arq]   # async-native (recommended)
pip install kazi-core[celery]  # Django/Flask compatible
```

**ARQ:**
```python
from kazi.queue.arq_worker import KaziQueue, build_worker_settings

WorkerSettings = build_worker_settings(config, redis_url="redis://localhost:6379")
# Start worker: python -m arq worker.WorkerSettings

queue = KaziQueue(redis_url="redis://localhost:6379")
job_id = await queue.enqueue("Analyse Q3 expenses", thread_id="user:1")
result = await queue.get_result(job_id, timeout=120)
print(result["reply"])
```

**Celery:**
```python
from kazi.queue.celery_worker import build_celery_app

app = build_celery_app(config, broker="redis://localhost:6379")
# Start worker: celery -A tasks worker --loglevel=info

job = app.send_task("kazi.run", args=["Summarise the report"], kwargs={"thread_id": "user:1"})
print(job.get(timeout=120)["reply"])
```

---

## Per-tenant tool isolation

For SaaS products where different customers should see different tool sets.

```python
config = KaziConfig(
    tenant_tools={
        "tenant:free":       {"web_search"},
        "tenant:pro":        {"web_search", "python_sandbox"},
        "tenant:enterprise": {"web_search", "python_sandbox", "query_db"},
    },
)

async with await Kazi.create(config) as kazi:
    # Free user — only web_search is visible
    reply = await kazi.run("Search for Python tutorials", tenant_id="tenant:free")

    # Enterprise user — all tools visible
    reply = await kazi.run("Query the database for Q3 data", tenant_id="tenant:enterprise")
```

---

## Validate your config

```bash
python -m kazi validate kazi.yaml
```

Checks config syntax, lists provider/model, and does a live connectivity check on all subsystems.

---

## Configuration reference

All configuration is done through a single `KaziConfig` dataclass. Every field has a safe default.

### LLM

```python
from kazi import LLMConfig, LLMProvider

LLMConfig(
    provider=LLMProvider.ANTHROPIC,   # OPENAI | ANTHROPIC | GOOGLE | LOCAL
    model="claude-sonnet-4-6",
    temperature=0.1,
    max_tokens=4096,
    api_key="sk-...",          # plain string or SecretRef — never logged
    base_url=None,             # set for local / proxied endpoints
)
```

`api_key` is automatically wrapped in a `SecretRef` so it never appears in logs or `repr()`. At runtime, kazi also checks `KAZI_API_KEY`, `OPENAI_API_KEY`, and `ANTHROPIC_API_KEY` in that order.

**Load from env or YAML:**

```python
config = KaziConfig.from_env()         # reads KAZI_LLM_PROVIDER, KAZI_API_KEY, etc.
config = KaziConfig.from_yaml("kazi.yaml")
```

### RAG

```python
from kazi import RAGConfig, VectorStoreBackend

RAGConfig(
    vector_store=VectorStoreBackend.IN_MEMORY,   # CHROMA | PINECONE | QDRANT | WEAVIATE
    embedding_model="text-embedding-3-small",
    chunk_size=1024,
    chunk_overlap=128,
    similarity_top_k=5,
    persist_dir="./kazi_index",
)
```

### Memory

```python
from kazi import MemoryConfig, MemoryBackend

MemoryConfig(
    backend=MemoryBackend.SQLITE,        # IN_MEMORY | SQLITE | REDIS | POSTGRES
    connection_string="sqlite:///kazi_memory.db",
    max_conversation_turns=50,
)
```

### Model routing

Route different turn types to different models. All routes inherit unset fields from the primary LLM.

```python
from kazi.core.router import ModelRoute, RouterConfig

config = KaziConfig(
    llm=LLMConfig(provider=LLMProvider.ANTHROPIC, model="claude-sonnet-4-6"),
    router=RouterConfig(
        # Fall back to cheaper model when primary fails (after 3 retries with backoff)
        fallback=ModelRoute(model="claude-haiku-4-5", provider="anthropic"),
        # Use a faster model for tool-call reasoning turns
        tool_call=ModelRoute(model="claude-haiku-4-5", provider="anthropic"),
        # Use a small cheap model for conversation summarisation
        summarizer=ModelRoute(model="gpt-4o-mini", provider="openai"),
    ),
)
```

Retry behaviour: before switching to the fallback, kazi retries the primary model up to 3 times
with exponential backoff (1s → 2s → 4s + jitter).

### Token budget

```python
from kazi.core.token_budget import TokenBudgetConfig

TokenBudgetConfig(
    max_tokens_per_run=100_000,        # hard stop per kazi.run() call; None = unlimited
    warn_at_fraction=0.8,              # log a warning at 80% consumed
    summarize_after_turns=20,          # compress history after N turns
    max_chars_per_tool_result=50_000,  # also enforced by ContentPolicy
    max_tool_description_chars=120,    # truncate tool descriptions in system prompt
    max_tools_per_prompt=20,           # inject only top-N most relevant tools per turn
)
```

---

## Built-in tools

```python
from kazi import (
    web_search_tool, read_file_tool, write_file_tool, list_directory_tool,
    sql_query_tool, python_sandbox_tool, data_query_tool, data_summary_tool,
)

kazi.registry.register(web_search_tool(max_results=10))
kazi.registry.register(read_file_tool(root_dir="/safe/dir"))
kazi.registry.register(write_file_tool(root_dir="/safe/dir"))
kazi.registry.register(list_directory_tool(root_dir="/safe/dir"))
kazi.registry.register(sql_query_tool("postgresql://user:pass@db/prod", read_only=True))
kazi.registry.register(python_sandbox_tool(timeout=15))
kazi.registry.register(data_query_tool(root_dir="./data"))    # pandas / Polars CSV/Excel/Parquet
kazi.registry.register(data_summary_tool(root_dir="./data"))  # schema + stats
```

| Tool | What it does | Key safety note |
|---|---|---|
| `web_search_tool` | DuckDuckGo search | Results capped at 20 |
| `read_file_tool` | Read a local file | `root_dir` blocks path traversal |
| `write_file_tool` | Write a local file | `root_dir` blocks path traversal |
| `list_directory_tool` | List directory contents | `root_dir` blocks path traversal |
| `sql_query_tool` | Run SQL queries | `read_only=True` blocks non-SELECT + injection |
| `python_sandbox_tool` | Run Python in a subprocess | Subprocess timeout + resource limits |
| `data_query_tool` | Filter/query CSV, Excel, Parquet | `root_dir` sandbox; MAX_ROWS=5000 |
| `data_summary_tool` | Schema + stats for data files | Same sandbox; no row limit |

---

## Security

kazi is built with a layered security model. Every layer is configurable but secure by default.

### Content policy

All tool results are tagged as external content before they enter the LLM context. This creates a clear trust boundary that makes prompt injection from external sources much harder to exploit.

```python
from kazi.core.security import SecurityConfig, ContentPolicy

config = KaziConfig(
    security=SecurityConfig(
        content=ContentPolicy(
            tag_external_content=True,   # default: True — wraps results in <external_content>
            max_result_chars=50_000,
            on_tool_call=lambda name, args: args if name != "delete_file" else None,
            on_tool_result=lambda name, result: result,
        )
    )
)
```

### MCP allowlist / denylist

```python
from kazi.core.security import MCPSecurityPolicy

MCPSecurityPolicy(allowlist=["filesystem__read_*", "github__list_*"])
MCPSecurityPolicy(denylist=["filesystem__delete_*", "shell__*"])
```

### Thread authentication

```python
import jwt
from kazi.core.security import ThreadPolicy

config = KaziConfig(
    security=SecurityConfig(
        threads=ThreadPolicy(
            require_auth=True,
            validator=lambda thread_id, token: (
                thread_id.startswith(jwt.decode(token, PUBLIC_KEY, ["RS256"])["sub"] + ":")
            )
        )
    )
)
await kazi.run("...", thread_id="user:123:session:abc", user_token=jwt_token)
```

### SSRF protection

Agent URLs are validated before any HTTP request is made. Private IP ranges, loopback addresses, and non-HTTP schemes are blocked by default.

### SQL injection

`sql_query_tool(read_only=True)` (the default) strips comments, checks for SELECT-only statements, and blocks multi-statement injection.

### Path traversal

All file tools resolve paths with `Path.resolve()` before checking against `root_dir` — symlink chains and `../../` escapes are caught.

---

## Observability

kazi emits OpenTelemetry spans when `opentelemetry-sdk` is installed. Silent no-ops otherwise.

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(provider)

async with await Kazi.create(config) as kazi:
    result = await kazi.run("...")  # emits kazi.run, kazi.tool.execute spans
```

---

## Examples

Ten runnable examples in [`examples/`](examples/):

| File | What it shows |
|---|---|
| [`01_simple_rag_agent.py`](examples/01_simple_rag_agent.py) | Ingest documents and run multi-turn Q&A |
| [`02_multi_tool_agent.py`](examples/02_multi_tool_agent.py) | Custom tools + built-in tools + MCP server |
| [`03_supervisor_delegation.py`](examples/03_supervisor_delegation.py) | A2A supervisor with remote specialist agents |
| [`04_full_orchestration.py`](examples/04_full_orchestration.py) | All four layers, persistent memory, streaming |
| [`05_custom_providers.py`](examples/05_custom_providers.py) | Bedrock, Vertex, Azure, HuggingFace, Cohere |
| [`06_voice_agent.py`](examples/06_voice_agent.py) | Push-to-talk, streaming voice, WebSocket server |
| [`07_subagents_and_supervisor.py`](examples/07_subagents_and_supervisor.py) | Local sub-agents with tool isolation + auto-routing |
| [`08_serve_api.py`](examples/08_serve_api.py) | FastAPI server for React frontends |
| [`09_background_jobs.py`](examples/09_background_jobs.py) | ARQ + Celery async job queues |
| [`10_multimodal_and_memory.py`](examples/10_multimodal_and_memory.py) | Vision input, user profiles, experiment tracking, tenant isolation |

---

## Development

```bash
git clone https://github.com/erose2502/kazi
cd kazi
pip install -e ".[dev]"

pytest                          # run tests
ruff check kazi tests          # lint
mypy kazi                      # type-check
bandit -r kazi -ll             # security scan
pip-audit                       # dependency vulnerability scan
python -m kazi validate kazi.yaml  # validate your config
```

Tests live in `tests/unit/` (fast, no external deps) and `tests/integration/` (mock transports). Coverage gate: 55%.

---

## Packages

kazi is a monorepo. The workspace root is not published; individual packages are:

| Package | Contents |
|---|---|
| `kazi-base` | Config, registry, exceptions, security, secrets |
| `kazi-rag` | LlamaIndex data layer (ingest, index, query) |
| `kazi-mcp` | MCP client bridge + built-in tools (file, database, web, sandbox) |
| `kazi-a2a` | A2A agent discovery and delegation |
| `kazi-voice` | Real-time STT/TTS pipeline (Whisper, Deepgram, ElevenLabs) |
| `kazi-core` | `Kazi` top-level class + LangGraph brain |
| `kazi` | Batteries-included meta-package (installs everything) |

---

## License

MIT
