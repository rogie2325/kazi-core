# kazi-base

Core configuration, registry, security primitives, and LLM abstractions for the kazi ecosystem.

This is the foundation package that all other kazi packages build on. Install it directly only if you are building a custom integration — most users should install `kazi-core` or `kazi` instead.

## Install

```bash
pip install kazi-base[anthropic]
pip install kazi-base[openai]
pip install kazi-base[google]
pip install kazi-base[local]
```

## What's included

- `KaziConfig` / `LLMConfig` / `RouterConfig` — dataclass-based configuration with YAML and env-var loading
- `SecretRef` — wraps API keys so they never appear in logs or `repr()`
- `ToolRegistry` — central registry for all tools available to the agent
- `SecurityConfig` / `ContentPolicy` / `MCPSecurityPolicy` / `ThreadPolicy` — layered security primitives
- `TokenBudgetConfig` — token usage tracking and conversation compression

## License

MIT
