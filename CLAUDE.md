# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install all dependencies for local development
pip install -e ".[dev]"
# Or with uv workspaces (preferred)
uv sync

# Run all tests
pytest

# Run only unit tests (fast, no external deps)
pytest tests/unit/

# Run a single test file
pytest tests/unit/test_registry.py

# Run a single test by name
pytest tests/unit/test_registry.py::test_register_tool

# Run integration tests
pytest tests/integration/

# Lint
ruff check kazi/ tests/

# Auto-fix lint issues
ruff check --fix kazi/ tests/

# Type check
mypy kazi/ --ignore-missing-imports

# Security static analysis
bandit -r kazi/ -ll --skip B603,B607

# Dependency vulnerability scan
pip-audit --requirement <(pip freeze) --skip-editable

# Validate a config file
python -m kazi validate kazi.yaml
```

Coverage gate is 55% (enforced in CI over combined unit + integration runs). `pyproject.toml` lists modules excluded from coverage (voice, MCP client, sandbox, etc.) because they require live external APIs.

## Architecture

Kazi Orchestrate is a **monorepo** of six published packages plus a root `kazi/` source tree used for development:

```
kazi/            ← authoritative source (editable install via root pyproject.toml)
packages/
  kazi-base/          ← config, registry, exceptions, security, secrets
  kazi-rag/           ← LlamaIndex data layer (ingest, index, query)
  kazi-mcp/           ← MCP client bridge + built-in tools
  kazi-a2a/           ← A2A agent discovery and delegation
  kazi-voice/         ← STT/TTS pipeline
  kazi-core/  ← Kazi top-level class + LangGraph brain
  kazi/               ← meta-package (installs everything)
tests/
  unit/           ← fast, no external deps
  integration/    ← mock transports
  e2e/
examples/         ← 10 runnable end-to-end demos
```

The `packages/` directories each re-export from the root `kazi/` source tree (see `packages/kazi-core/kazi/__init__.py`). The root `kazi/` is the single source of truth for all logic.

### Core layers and their modules

| Layer | Module | Key files |
|---|---|---|
| Orchestrator | `kazi/core/orchestrator.py` | `Kazi` class — factory, `run()`, `stream()`, `ingest()`, `run_voice()` |
| LangGraph brain | `kazi/brain/` | `graph_builder.py` (graph wiring + circuit breaker), `nodes.py` (summariser, reflection), `state.py`, `memory.py` |
| Config | `kazi/core/config.py` | All `*Config` dataclasses; `KaziConfig.from_env()`, `.from_yaml()` |
| Tool registry | `kazi/core/registry.py` | `ToolRegistry`, `ToolDefinition` — single registry for all tool sources |
| LLM adapters | `kazi/llm/` | `openai.py`, `anthropic.py`, `google.py`, `local.py` — all wrap `BaseChatModel` |
| Model routing | `kazi/core/router.py` | `RouterConfig` with `fallback`, `tool_call`, `summarizer` routes |
| RAG | `kazi/data/` | `index_manager.py` (LlamaIndex), `ingest.py`, `query_engine.py` |
| MCP | `kazi/tools/mcp_client.py` | `MCPBridge` — spawns stdio MCP servers, exposes their tools to the registry |
| A2A | `kazi/agents/` | `a2a_client.py`, `discovery.py`, `delegation.py`, `subagent.py`, `supervisor.py` |
| Memory | `kazi/memory/profile.py` | `UserProfile` — cross-session user preferences injected into system prompt |
| Security | `kazi/core/security.py` | `ContentPolicy`, `MCPSecurityPolicy`, `ThreadPolicy`, `InjectionDetectionConfig` |
| Token budget | `kazi/core/token_budget.py` | `TokenBudgetConfig`, hard stop + auto-summarise |
| Secrets | `kazi/core/secrets.py` | `SecretRef` — wraps API keys so they never appear in `repr()` or logs |
| Cost tracking | `kazi/core/cost.py` | `RunCost`, `RunResult`, `CostAccumulator` |
| Built-in tools | `kazi/tools/builtin/` | `web_search`, `file_system`, `database`, `dataframe` |
| Serve | `kazi/serve/app.py` | FastAPI app via `kazi.as_app()` |
| Queue | `kazi/queue/` | `arq_worker.py`, `celery_worker.py`, `webhook.py` |
| Semantic cache | `kazi/cache/semantic.py` | Vector-similarity cache for repeated queries |
| Voice | `kazi/voice/` | `pipeline.py`, `stt.py`, `tts.py` |

### Startup flow

`Kazi.create(config)` calls `_startup()` which initialises layers in order:
1. `UserProfile` store (always-on, zero overhead)
2. `SemanticCache` (if configured)
3. `VoicePipeline` (if `config.voice` is set)
4. `IndexManager` (LlamaIndex — no I/O yet)
5. `MCPBridge` (spawns MCP server subprocesses)
6. `A2AClient` (registers remote agents into registry)
7. LangGraph graph (built from `graph_builder.py`) with checkpointer (SQLite default; Redis/Postgres optional)

### LangGraph graph structure

`graph_builder.py` builds a `StateGraph[AgentState]` with:
- `agent` node — calls the LLM, handles tool routing, model fallback with circuit breaker, exponential backoff (1s→2s→4s + jitter), token budget enforcement
- `tools` node — executes tool calls from the registry (MCP, A2A, built-in, custom)
- Conditional edge: after `agent`, routes to `tools` if tool calls are present, else `END`
- `tools` always returns to `agent`

The circuit breaker (`_CircuitBreaker`) is per-provider, CLOSED→OPEN after 5 consecutive retryable failures, HALF_OPEN after 60s cooldown.

### Key design patterns

**Tool registration**: all tools — built-in, MCP, A2A, custom — flow through `ToolRegistry`. The `kazi.add_tool()` convenience method wraps a Python callable into a `ToolDefinition`. The registry is the single source of truth the LangGraph `tools` node uses.

**Security boundary**: all tool results are tagged as `<external_content>` before entering LLM context (`ContentPolicy.tag_external_content=True` by default). This is the primary prompt-injection defense.

**API key handling**: `LLMConfig.api_key` is always coerced to `SecretRef` in `__post_init__`. Call `.resolved_api_key()` at execution time; never access `.api_key` directly in LLM adapters.

**Thread IDs**: sanitized with a whitelist regex (`[a-zA-Z0-9_\-:.@]`) in `orchestrator.py` before reaching LangGraph checkpointer. Recommended format: `user:<id>:session:<id>`.

**Cross-modal memory**: voice and chat share the same LangGraph checkpointer state via `thread_id`. `VoicePipeline` calls `kazi.run()` internally after transcription.

### Package dependency graph

```
kazi
  └── kazi-core[openai,anthropic,...]
        ├── kazi-base     (config, registry, security, secrets)
        ├── kazi-rag      (LlamaIndex data layer)
        ├── kazi-mcp      (MCP client + built-in tools)
        ├── kazi-a2a      (A2A delegation)
        └── langgraph, langchain-core, aiosqlite
```

`kazi-voice` is independent — install alongside orchestrator for voice support.

### CI

CI runs on Python 3.10–3.13. Jobs: `test` (lint → mypy → unit tests → integration tests → codecov) and `security` (bandit + pip-audit). Bandit skips B603/B607 (subprocess use in `sandbox.py` is intentional).
