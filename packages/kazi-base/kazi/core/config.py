from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union

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


@dataclass
class LLMConfig:
    provider: LLMProvider = LLMProvider.OPENAI
    model: str = "gpt-4o"
    temperature: float = 0.1
    max_tokens: int = 4096
    # api_key accepts a plain string (convenient) or a SecretRef (recommended).
    # SecretRef values are never logged; plain strings will appear in repr().
    api_key: Union[str, SecretRef, None] = None
    base_url: Optional[str] = None

    def __post_init__(self) -> None:
        # Coerce plain strings into SecretRef so the value is never logged
        self.api_key = SecretRef.coerce(self.api_key)

    def resolved_api_key(self) -> Optional[str]:
        """Return the resolved API key value (call at execution time, not at config time)."""
        if isinstance(self.api_key, SecretRef):
            return self.api_key.resolve()
        return self.api_key  # type: ignore[return-value]  # already None


@dataclass
class RAGConfig:
    vector_store: VectorStoreBackend = VectorStoreBackend.IN_MEMORY
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 1024
    chunk_overlap: int = 128
    similarity_top_k: int = 5
    hybrid_search: bool = False
    reranker: Optional[str] = None
    persist_dir: str = "./kazi_index"


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
    verbose: bool = False
    telemetry_enabled: bool = False

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

        return cls(
            llm=LLMConfig(**llm_data),
            rag=RAGConfig(**rag_data),
            mcp=MCPConfig(**data.get("mcp", {})),
            a2a=A2AConfig(**data.get("a2a", {})),
            memory=MemoryConfig(**mem_data),
            verbose=data.get("verbose", False),
            telemetry_enabled=data.get("telemetry_enabled", False),
        )

    @classmethod
    def from_env(cls) -> KaziConfig:
        provider_str = os.getenv("KAZI_LLM_PROVIDER", "openai")
        # api_key from env — use SecretRef so the value never appears in repr
        api_key: Union[SecretRef, None] = None
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
