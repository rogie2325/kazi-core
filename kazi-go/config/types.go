package config

import "github.com/rogie2325/kazi/secrets"

type LLMProvider string

type VectorStoreBackend string

type MemoryBackend string

type STTProvider string

type TTSProvider string

const (
    LLMProviderOpenAI    LLMProvider = "openai"
    LLMProviderAnthropic LLMProvider = "anthropic"
    LLMProviderGoogle    LLMProvider = "google"
    LLMProviderLocal     LLMProvider = "local"
)

const (
    VectorStoreChroma   VectorStoreBackend = "chroma"
    VectorStorePinecone VectorStoreBackend = "pinecone"
    VectorStoreQdrant   VectorStoreBackend = "qdrant"
    VectorStoreWeaviate VectorStoreBackend = "weaviate"
    VectorStoreInMemory VectorStoreBackend = "in_memory"
)

const (
    MemoryBackendSQLite  MemoryBackend = "sqlite"
    MemoryBackendRedis   MemoryBackend = "redis"
    MemoryBackendPostgres MemoryBackend = "postgres"
    MemoryBackendInMemory MemoryBackend = "in_memory"
)

const (
    STTProviderOpenAI  STTProvider = "openai"
    STTProviderDeepgram STTProvider = "deepgram"
)

const (
    TTSProviderOpenAI     TTSProvider = "openai"
    TTSProviderElevenLabs TTSProvider = "elevenlabs"
)

type VoiceConfig struct {
    STTProvider STTProvider `json:"stt_provider" yaml:"stt_provider"`
    STTModel    string      `json:"stt_model" yaml:"stt_model"`
    STTAPIKey   *secrets.SecretRef `json:"stt_api_key,omitempty" yaml:"stt_api_key,omitempty"`

    TTSProvider TTSProvider `json:"tts_provider" yaml:"tts_provider"`
    TTSModel    string      `json:"tts_model" yaml:"tts_model"`
    TTSVoice    string      `json:"tts_voice" yaml:"tts_voice"`
    TTSSpeed    float64     `json:"tts_speed" yaml:"tts_speed"`
    TTSAPIKey   *secrets.SecretRef `json:"tts_api_key,omitempty" yaml:"tts_api_key,omitempty"`

    Language string `json:"language,omitempty" yaml:"language,omitempty"`

    ElevenLabsVoiceID string `json:"elevenlabs_voice_id" yaml:"elevenlabs_voice_id"`
    ElevenLabsModel   string `json:"elevenlabs_model" yaml:"elevenlabs_model"`
}

type LLMConfig struct {
    Provider    LLMProvider       `json:"provider" yaml:"provider"`
    Model       string            `json:"model" yaml:"model"`
    Temperature float64           `json:"temperature" yaml:"temperature"`
    MaxTokens   int               `json:"max_tokens" yaml:"max_tokens"`
    APIKey      *secrets.SecretRef `json:"api_key,omitempty" yaml:"api_key,omitempty"`
    BaseURL     string            `json:"base_url,omitempty" yaml:"base_url,omitempty"`
    Seed        *int              `json:"seed,omitempty" yaml:"seed,omitempty"`
    CustomLLM   any               `json:"-" yaml:"-"`
}

func (c LLMConfig) ResolvedAPIKey() (string, bool) {
    if c.APIKey == nil {
        return "", false
    }
    return c.APIKey.Resolve()
}

func (c LLMConfig) Deterministic(seed int) LLMConfig {
    c.Temperature = 0.0
    c.Seed = &seed
    return c
}

type RAGConfig struct {
    VectorStore     VectorStoreBackend `json:"vector_store" yaml:"vector_store"`
    EmbeddingModel  string             `json:"embedding_model" yaml:"embedding_model"`
    ChunkSize       int                `json:"chunk_size" yaml:"chunk_size"`
    ChunkOverlap    int                `json:"chunk_overlap" yaml:"chunk_overlap"`
    SimilarityTopK  int                `json:"similarity_top_k" yaml:"similarity_top_k"`
    HybridSearch    bool               `json:"hybrid_search" yaml:"hybrid_search"`
    Reranker        string             `json:"reranker,omitempty" yaml:"reranker,omitempty"`
    PersistDir      string             `json:"persist_dir" yaml:"persist_dir"`
    VectorStoreURL  string             `json:"vector_store_url,omitempty" yaml:"vector_store_url,omitempty"`
    VectorStoreAPIKey *secrets.SecretRef `json:"vector_store_api_key,omitempty" yaml:"vector_store_api_key,omitempty"`
    ChromaCollection string            `json:"chroma_collection,omitempty" yaml:"chroma_collection,omitempty"`
    CustomEmbedding any               `json:"-" yaml:"-"`
    CustomSynthesisLLM any            `json:"-" yaml:"-"`
}

type MCPConfig struct {
    Servers        map[string]string `json:"servers" yaml:"servers"`
    TimeoutSeconds int               `json:"timeout" yaml:"timeout"`
    MaxRetries     int               `json:"max_retries" yaml:"max_retries"`
    SandboxEnabled bool              `json:"sandbox_enabled" yaml:"sandbox_enabled"`
}

type A2AConfig struct {
    DiscoveryEndpoints      []string `json:"discovery_endpoints" yaml:"discovery_endpoints"`
    PublishCard             bool     `json:"publish_card" yaml:"publish_card"`
    DelegationTimeoutSeconds int     `json:"delegation_timeout" yaml:"delegation_timeout"`
    MaxConcurrentDelegations int     `json:"max_concurrent_delegations" yaml:"max_concurrent_delegations"`
    AllowedHosts            []string `json:"allowed_hosts" yaml:"allowed_hosts"`
    MaxRetries              int      `json:"max_retries" yaml:"max_retries"`
}

type MemoryConfig struct {
    Backend              MemoryBackend `json:"backend" yaml:"backend"`
    ConnectionString     string        `json:"connection_string" yaml:"connection_string"`
    MaxConversationTurns int           `json:"max_conversation_turns" yaml:"max_conversation_turns"`
    SummarizeAfterTurns  int           `json:"summarize_after" yaml:"summarize_after"`
}

type MCPSecurityPolicy struct {
    Allowlist    []string `json:"allowlist" yaml:"allowlist"`
    Denylist     []string `json:"denylist" yaml:"denylist"`
    ValidateArgs bool     `json:"validate_args" yaml:"validate_args"`
}

type SecurityConfig struct {
    TagExternalContent bool             `json:"tag_external_content" yaml:"tag_external_content"`
    MCP                MCPSecurityPolicy `json:"mcp" yaml:"mcp"`
}

type TokenBudgetConfig struct {
    MaxTokens            int `json:"max_tokens" yaml:"max_tokens"`
    SummarizeAfterTurns  int `json:"summarize_after_turns" yaml:"summarize_after_turns"`
}

type ModelRoute struct {
    Provider LLMProvider `json:"provider" yaml:"provider"`
    Model    string      `json:"model" yaml:"model"`
}

type RouterConfig struct {
    Fallback   ModelRoute `json:"fallback" yaml:"fallback"`
    ToolCall   ModelRoute `json:"tool_call" yaml:"tool_call"`
    Summarizer ModelRoute `json:"summarizer" yaml:"summarizer"`
}

type Config struct {
    LLM      LLMConfig     `json:"llm" yaml:"llm"`
    RAG      RAGConfig     `json:"rag" yaml:"rag"`
    MCP      MCPConfig     `json:"mcp" yaml:"mcp"`
    A2A      A2AConfig     `json:"a2a" yaml:"a2a"`
    Memory   MemoryConfig  `json:"memory" yaml:"memory"`
    Security SecurityConfig `json:"security" yaml:"security"`
    Budget   TokenBudgetConfig `json:"budget" yaml:"budget"`
    Router   RouterConfig  `json:"router" yaml:"router"`
    Voice    *VoiceConfig  `json:"voice,omitempty" yaml:"voice,omitempty"`

    Verbose          bool `json:"verbose" yaml:"verbose"`
    TelemetryEnabled bool `json:"telemetry_enabled" yaml:"telemetry_enabled"`

    TenantTools map[string][]string `json:"tenant_tools" yaml:"tenant_tools"`

    MaxCostPerRunUSD      float64 `json:"max_cost_per_run_usd" yaml:"max_cost_per_run_usd"`
    MaxDailyCostPerUserUSD float64 `json:"max_daily_cost_per_user_usd" yaml:"max_daily_cost_per_user_usd"`

    SemanticCache any `json:"-" yaml:"-"`
    Guardrails    any `json:"-" yaml:"-"`

    ToolResultCacheTTLSeconds int `json:"tool_result_cache_ttl" yaml:"tool_result_cache_ttl"`
    ToolImports               []string `json:"tool_imports" yaml:"tool_imports"`
}

func DefaultConfig() Config {
    return Config{
        LLM: LLMConfig{
            Provider:    LLMProviderOpenAI,
            Model:       "gpt-4o",
            Temperature: 0.1,
            MaxTokens:   4096,
        },
        RAG: RAGConfig{
            VectorStore:     VectorStoreInMemory,
            EmbeddingModel:  "text-embedding-3-small",
            ChunkSize:       1024,
            ChunkOverlap:    128,
            SimilarityTopK:  5,
            HybridSearch:    false,
            PersistDir:      "./kazi_index",
        },
        MCP: MCPConfig{
            Servers:        map[string]string{},
            TimeoutSeconds: 30,
            MaxRetries:     3,
            SandboxEnabled: true,
        },
        A2A: A2AConfig{
            DiscoveryEndpoints:      []string{},
            PublishCard:             false,
            DelegationTimeoutSeconds: 120,
            MaxConcurrentDelegations: 5,
            AllowedHosts:            []string{},
            MaxRetries:              3,
        },
        Memory: MemoryConfig{
            Backend:              MemoryBackendInMemory,
            ConnectionString:     "sqlite:///kazi_memory.db",
            MaxConversationTurns: 50,
            SummarizeAfterTurns:  20,
        },
        Security: SecurityConfig{
            TagExternalContent: true,
            MCP: MCPSecurityPolicy{
                Allowlist:    []string{},
                Denylist:     []string{},
                ValidateArgs: true,
            },
        },
        Budget: TokenBudgetConfig{
            MaxTokens:           0,
            SummarizeAfterTurns: 0,
        },
        Router: RouterConfig{},
        Voice:  nil,
        Verbose: false,
        TelemetryEnabled: false,
        TenantTools: map[string][]string{},
        MaxCostPerRunUSD: 0,
        MaxDailyCostPerUserUSD: 0,
        ToolResultCacheTTLSeconds: 0,
        ToolImports: []string{},
    }
}
