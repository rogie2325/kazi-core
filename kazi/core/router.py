"""
Model routing configuration for kazi.

Three routing slots, each taking an optional ModelRoute:

  fallback   — retry here when the primary LLM call raises an exception
  tool_call  — use for turns where the LLM is reasoning over tool results
               (cheaper/faster model; these turns are mechanical)
  summarizer — use for background conversation summarisation
               (output is internal; a cheap model is fine)

All fields on ModelRoute are optional except `model`: unset fields inherit
from the primary LLMConfig, so you only need to specify what changes.

Example — cross-provider failover + cheap tool routing::

    from kazi.core.router import RouterConfig, ModelRoute
    from kazi.core.secrets import SecretRef

    RouterConfig(
        fallback=ModelRoute(
            model="claude-haiku-4-5-20251001",
            provider="anthropic",
            api_key=SecretRef.from_env("ANTHROPIC_API_KEY"),
        ),
        tool_call=ModelRoute(model="gpt-4o-mini"),
        summarizer=ModelRoute(model="gpt-4o-mini"),
    )
"""
from __future__ import annotations

from dataclasses import dataclass

from kazi.core.secrets import SecretRef


@dataclass
class ModelRoute:
    """
    Configuration for a single alternative model endpoint.

    Only `model` is required; everything else inherits from the primary
    LLMConfig when left as None.

    api_key accepts a plain string or SecretRef — plain strings are coerced
    to SecretRef automatically so they are never logged.
    """

    model: str
    provider: str | None = None
    api_key: str | SecretRef | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    base_url: str | None = None

    def __post_init__(self) -> None:
        self.api_key = SecretRef.coerce(self.api_key)

    def resolved_api_key(self) -> str | None:
        if isinstance(self.api_key, SecretRef):
            return self.api_key.resolve()
        return self.api_key  # type: ignore[return-value]


@dataclass
class RouterConfig:
    """
    Model routing configuration — wired into KaziConfig.

    fallback
        Retried once when the primary LLM call raises any exception
        (network error, rate limit, provider outage). Intended for
        cross-provider redundancy.

    tool_call
        Used when the agent is reasoning over tool results — i.e. the
        last message in context is a ToolMessage. These turns decide
        which tool to call next and do not require a frontier model.
        Setting this to a fast, cheap model significantly cuts cost on
        long tool chains without degrading final output quality.

    summarizer
        Used for background conversation summarisation triggered by
        TokenBudgetConfig.summarize_after_turns. The summary is internal
        state, never shown directly to the user.

    Note: if LLMConfig.custom_llm is set, it is used for primary turns
    regardless of routing. Routes always go through the built-in provider
    lookup (openai / anthropic / google / local).
    """

    fallback: ModelRoute | None = None
    tool_call: ModelRoute | None = None
    summarizer: ModelRoute | None = None
    # Retry knobs — applied to every LLM invocation
    max_retry_attempts: int = 3      # attempts before falling to fallback (min 1)
    retry_base_delay: float = 1.0    # seconds for first backoff interval
    # Circuit breaker — trips after this many consecutive retryable failures
    # and stays open for circuit_breaker_cooldown seconds before trying again.
    # Set to 0 to disable.
    circuit_breaker_threshold: int = 5
    circuit_breaker_cooldown: float = 60.0
