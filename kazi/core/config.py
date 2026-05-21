from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from kazi.core.router import ModelRoute, RouterConfig
from kazi.core.secrets import SecretRef
from kazi.core.security import SecurityConfig
from kazi.core.token_budget import TokenBudgetConfig


class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LOCAL = "local"


class VectorStoreBackend(Enum):
    CHROMA = "chroma"
    PINECONE = "pinecone"
    QDRANT = "qdrant"
    WEAVIATE = "weaviate"
    IN_MEMORY = "in_memory"


class MemoryBackend(Enum):
    SQLITE = "sqlite"
    REDIS = "redis"
    POSTGRES = "postgres"
    IN_MEMORY = "in_memory"


class STTProvider(Enum):
    OPENAI = "openai"
    DEEPGRAM = "deepgram"


class TTSProvider(Enum):
    OPENAI = "openai"
    ELEVENLABS = "elevenlabs"


@dataclass
class VoiceConfig:
    """
    Configuration for real-time voice I/O.

    STT (speech-to-text) and TTS (text-to-speech) providers are independent —
    mix and match freely (e.g. Deepgram STT + ElevenLabs TTS).

    API keys default to None and fall back to environment variables:
      OPENAI_API_KEY, DEEPGRAM_API_KEY, ELEVENLABS_API_KEY.

    Cross-modal memory: voice and chat sessions share memory automatically
    when the same thread_id is used. The LangGraph checkpointer is the single
    source of truth — modality is irrelevant to the stored state.
    """
    stt_provider: STTProvider = STTProvider.OPENAI
    stt_model: str = "whisper-1"
    stt_api_key: str | None = None

    tts_provider: TTSProvider = TTSProvider.OPENAI
    tts_model: str = "tts-1"
    tts_voice: str = "nova"       # OpenAI: alloy | echo | fable | onyx | nova | shimmer
    tts_speed: float = 1.0
    tts_api_key: str | None = None

    language: str | None = None  # None = auto-detect

    # ElevenLabs-specific
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"  # Rachel
    elevenlabs_model: str = "eleven_turbo_v2"


@dataclass
class LLMConfig:
    provider: LLMProvider = LLMProvider.OPENAI
    model: str = "gpt-4o"
    temperature: float = 0.1
    max_tokens: int = 4096
    # api_key accepts a plain string (convenient) or a SecretRef (recommended).
    # SecretRef values are never logged; plain strings will appear in repr().
    api_key: str | SecretRef | None = None
    base_url: str | None = None
    # Reproducibility seed.  Passed through to providers that support it
    # (OpenAI, Anthropic).  Combine with temperature=0 for deterministic
    # replay of validator-flagged runs.  None = no seed sent.
    seed: int | None = None
    # Pass any LangChain BaseChatModel here to bypass the built-in provider
    # lookup entirely — useful for Mistral, Cohere, or any custom model.
    custom_llm: Any | None = None

    def __post_init__(self) -> None:
        # Coerce plain strings into SecretRef so the value is never logged
        self.api_key = SecretRef.coerce(self.api_key)

    def resolved_api_key(self) -> str | None:
        """Return the resolved API key value (call at execution time, not at config time)."""
        if isinstance(self.api_key, SecretRef):
            return self.api_key.resolve()
        return self.api_key  # type: ignore[return-value]  # already None

    def deterministic(self, *, seed: int = 42) -> LLMConfig:
        """
        Return a copy of this LLMConfig configured for reproducible runs.

        Sets ``temperature=0`` and the given ``seed``.  Use when a validator
        needs to replay a specific run-time issue::

            cfg = KaziConfig(llm=LLMConfig(...).deterministic(seed=42))
            kazi = await Kazi.create(cfg)
            reply = await kazi.run("the failing prompt")
        """
        import copy
        clone = copy.copy(self)
        clone.temperature = 0.0
        clone.seed = seed
        return clone


@dataclass
class RAGConfig:
    vector_store: VectorStoreBackend = VectorStoreBackend.IN_MEMORY
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 1024
    chunk_overlap: int = 128
    similarity_top_k: int = 5
    hybrid_search: bool = False
    reranker: str | None = None
    persist_dir: str = "./kazi_index"
    # Connection details for external vector stores (Pinecone, Qdrant, Weaviate).
    # Not used for IN_MEMORY or CHROMA (which uses persist_dir).
    vector_store_url: str | None = None
    vector_store_api_key: str | None = None
    # Chroma collection name — defaults to the index_name passed at ingest time.
    chroma_collection: str | None = None
    # Pass any LlamaIndex BaseEmbedding to bypass the built-in OpenAI embedding
    # — useful for HuggingFace, Cohere, local models, or test fixtures.
    custom_embedding: Any | None = None
    # Pass any LlamaIndex LLM to override the synthesis model used for
    # RAG response generation (separate from the chat LLM used by the brain).
    custom_synthesis_llm: Any | None = None


@dataclass
class MCPConfig:
    # Map of server name → command/URL  e.g. {"fs": "npx @mcp/filesystem /tmp"}
    servers: dict[str, str] = field(default_factory=dict)
    timeout: int = 30
    max_retries: int = 3
    sandbox_enabled: bool = True


@dataclass
class A2AConfig:
    # URLs where remote Agent Cards live
    discovery_endpoints: list[str] = field(default_factory=list)
    publish_card: bool = False
    delegation_timeout: int = 120
    max_concurrent_delegations: int = 5
    # SSRF guard: only fetch agent cards from these hostnames (empty = allow all non-private hosts)
    allowed_hosts: list[str] = field(default_factory=list)
    max_retries: int = 3


@dataclass
class MemoryConfig:
    backend: MemoryBackend = MemoryBackend.IN_MEMORY
    connection_string: str = "sqlite:///kazi_memory.db"
    max_conversation_turns: int = 50
    # summarize_after is now owned by TokenBudgetConfig; kept here for
    # backwards-compat but TokenBudgetConfig.summarize_after_turns takes precedence
    summarize_after: int = 20


@dataclass
class KaziConfig:
    """Single unified configuration for the entire orchestration layer."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    a2a: A2AConfig = field(default_factory=A2AConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    budget: TokenBudgetConfig = field(default_factory=TokenBudgetConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    voice: VoiceConfig | None = None   # None = voice disabled
    verbose: bool = False
    telemetry_enabled: bool = False
    # Per-tenant tool isolation: maps tenant_id → set of allowed tool names.
    # When a run() call passes tenant_id=, only the listed tools are visible.
    # Empty dict = no isolation (all tools visible to all tenants).
    tenant_tools: dict[str, set[str]] = field(default_factory=dict)

    # ── Cost budget enforcement ───────────────────────────────────────────
    # 0.0 = disabled.  Checked before each run; raises BudgetExceededError.
    max_cost_per_run_usd: float = 0.0
    # Per-user daily spend cap (in-memory tracking; use Redis for multi-process).
    max_daily_cost_per_user_usd: float = 0.0

    # ── Semantic response cache ───────────────────────────────────────────
    # None = disabled.  When set, kazi checks the cache before every LLM call.
    semantic_cache: Any | None = None   # SemanticCacheConfig instance

    # ── Output guardrails ─────────────────────────────────────────────────
    # None = disabled.  When set, every LLM reply is validated before returning.
    guardrails: Any | None = None       # GuardrailConfig instance

    # ── Tool result cache ─────────────────────────────────────────────────
    # TTL in seconds for caching tool call results (0 = disabled).
    # Same tool name + same arguments within the TTL window returns the cached value.
    tool_result_cache_ttl: int = 0

    # ── Declarative tool imports ──────────────────────────────────────────
    # List of import directives executed at Kazi._startup() time.  Each entry
    # is one of:
    #   {"import": "my_app.services.get_invoice"}          → single function
    #   {"module": "my_app.services", "only": [...]}        → bulk scan
    #   {"openapi": "<url>", "base_url": "...", ...}        → REST API import
    # Lets a YAML config wire tools without any Python — the contractor workflow
    # for adopting an existing client codebase.
    tools_imports: list[dict] = field(default_factory=list)

    @classmethod
    def to_json_schema(cls) -> dict:
        """
        Return a JSON Schema (Draft 2020-12) describing the YAML config surface.

        Designed for two consumers:

        1. **Coding LLMs** generating ``kazi.yaml`` for a client — they can
           validate their output against this schema before submission so
           an entire round-trip of "write → run → error → fix" is avoided.

        2. **IDE tooling** — editors that consume JSON Schema (VSCode, IntelliJ,
           Cursor) get autocomplete and inline validation on ``kazi.yaml``.

        The schema is generated by introspecting the dataclass tree so it
        stays in sync with the code with zero manual maintenance.

        Use the CLI to dump it::

            python -m kazi config-schema > kazi.schema.json
        """
        from kazi.core.schema import kazi_config_schema
        return kazi_config_schema()

    @classmethod
    def from_yaml(cls, path: str) -> KaziConfig:
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)

        llm_data = data.get("llm", {})
        if "provider" in llm_data:
            llm_data["provider"] = LLMProvider(llm_data["provider"])
        # Promote plain api_key string to SecretRef via __post_init__

        rag_data = data.get("rag", {})
        if "vector_store" in rag_data:
            rag_data["vector_store"] = VectorStoreBackend(rag_data["vector_store"])

        mem_data = data.get("memory", {})
        if "backend" in mem_data:
            mem_data["backend"] = MemoryBackend(mem_data["backend"])

        from kazi.core.router import ModelRoute, RouterConfig
        from kazi.core.security import (
            ContentPolicy,
            MCPSecurityPolicy,
            SecurityConfig,
        )
        from kazi.core.token_budget import TokenBudgetConfig

        sec_data = data.get("security", {})
        security = SecurityConfig(
            content=ContentPolicy(**sec_data.get("content", {})),
            mcp=MCPSecurityPolicy(**sec_data.get("mcp", {})),
            verify_tls=sec_data.get("verify_tls", True),
        )

        budget = TokenBudgetConfig(**data.get("budget", {}))

        def _parse_route(d: dict) -> ModelRoute:
            return ModelRoute(**d)

        router_data = data.get("router", {})
        router = RouterConfig(
            fallback=_parse_route(router_data["fallback"]) if "fallback" in router_data else None,
            tool_call=_parse_route(router_data["tool_call"]) if "tool_call" in router_data else None,
            summarizer=_parse_route(router_data["summarizer"]) if "summarizer" in router_data else None,
        )

        voice = None
        if "voice" in data:
            v = data["voice"]
            if "stt_provider" in v:
                v["stt_provider"] = STTProvider(v["stt_provider"])
            if "tts_provider" in v:
                v["tts_provider"] = TTSProvider(v["tts_provider"])
            voice = VoiceConfig(**v)

        # Parse the optional `tools:` block — a list of declarative import
        # directives applied at Kazi._startup() time.  See kazi.integration.
        tools_imports: list[dict] = []
        for entry in data.get("tools", []) or []:
            if isinstance(entry, dict):
                tools_imports.append(entry)
            elif isinstance(entry, str):
                # Shorthand: a bare string is treated as {"import": "..."}
                tools_imports.append({"import": entry})

        return cls(
            llm=LLMConfig(**llm_data),
            rag=RAGConfig(**rag_data),
            mcp=MCPConfig(**data.get("mcp", {})),
            a2a=A2AConfig(**data.get("a2a", {})),
            memory=MemoryConfig(**mem_data),
            security=security,
            budget=budget,
            router=router,
            voice=voice,
            verbose=data.get("verbose", False),
            telemetry_enabled=data.get("telemetry_enabled", False),
            tools_imports=tools_imports,
        )

    @classmethod
    def from_env(cls) -> KaziConfig:
        provider_str = os.getenv("KAZI_LLM_PROVIDER", "openai")
        # api_key from env — use SecretRef so the value never appears in repr
        api_key: SecretRef | None = None
        for var in ("KAZI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            if os.environ.get(var):
                api_key = SecretRef.from_env(var)
                break

        return cls(
            llm=LLMConfig(
                provider=LLMProvider(provider_str),
                model=os.getenv("KAZI_LLM_MODEL", "gpt-4o"),
                api_key=api_key,
                base_url=os.getenv("KAZI_BASE_URL"),
            ),
            verbose=os.getenv("KAZI_VERBOSE", "").lower() in ("1", "true"),
        )
