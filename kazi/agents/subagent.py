"""SubAgent — a Kazi instance with a fixed role, personality, and tool set."""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SubAgentConfig:
    """
    Defines the identity and behaviour of a sub-agent.

    name            Display name (e.g. "Riley").
    role            One-line role description used in routing ("Research & Intelligence").
    system_prompt   Full system prompt that defines personality, constraints, and tone.
                    This is injected on EVERY turn so the agent behaves consistently
                    regardless of which LLM model is active (cross-model routing).
    tools           Optional allowlist of tool names this agent may use.
                    Empty list = no restriction (inherits all registry tools).
                    Restriction is enforced via the system prompt — the LLM is
                    instructed to only call tools in this list.
    llm_override    Optional model name override for this specific agent.
                    Useful for giving a cheap model to a high-volume agent (e.g. Sam)
                    or a powerful model to a critical one (e.g. Cody).
                    Falls back to the parent KaziConfig LLM when None.
    """
    name: str
    role: str
    system_prompt: str
    tools: list[str] = field(default_factory=list)
    llm_override: str | None = None


class SubAgent:
    """
    A Kazi-backed agent with a fixed personality and role.

    The system_prompt is passed on every run/stream call so the agent's
    personality, tone, and constraints are consistent regardless of which
    underlying LLM serves the request.  Cross-model routing (fallback,
    tool_call, summarizer routes) still applies — the personality stays
    constant because it lives in the system prompt, not in the model.

    Memory is scoped by thread_id.  Using the same thread_id for both voice
    and chat gives the agent identical context across modalities.

    Tool restriction: if `tools` is set in SubAgentConfig, the system prompt
    instructs the agent to only use those tools. This is enforced at the LLM
    level, which is simpler and concurrency-safe vs. mutating the registry.
    """

    def __init__(self, config: SubAgentConfig, kazi) -> None:
        self.name = config.name
        self.role = config.role
        self._kazi = kazi
        self._system_prompt = self._build_system_prompt(config)

    def _build_system_prompt(self, config: SubAgentConfig) -> str:
        prompt = config.system_prompt.rstrip()
        if config.tools:
            tool_list = ", ".join(config.tools)
            prompt += (
                f"\n\nTool restriction: you may ONLY call the following tools: {tool_list}. "
                "Do not call any other tool even if it is available."
            )
        return prompt

    async def run(
        self,
        message: str,
        *,
        thread_id: str = "default",
        max_tool_calls: int = 25,
        user_token: str | None = None,
    ) -> str:
        """Run a single turn, returning the agent's text response."""
        return await self._kazi.run(
            message,
            thread_id=self._scoped_thread(thread_id),
            max_tool_calls=max_tool_calls,
            system_prompt=self._system_prompt,
            user_token=user_token,
        )

    async def stream(
        self,
        message: str,
        *,
        thread_id: str = "default",
        max_tool_calls: int = 25,
        user_token: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream tokens from the agent's response."""
        async for token in self._kazi.stream(
            message,
            thread_id=self._scoped_thread(thread_id),
            max_tool_calls=max_tool_calls,
            system_prompt=self._system_prompt,
            user_token=user_token,
        ):
            yield token

    async def run_voice(self, audio: bytes, *, thread_id: str = "default") -> bytes:
        """Transcribe audio → run → synthesize. Requires VoiceConfig on the parent Kazi."""
        self._kazi._assert_voice()
        from kazi.voice.stt import transcribe
        from kazi.voice.tts import synthesize
        cfg = self._kazi.config.voice
        text = await transcribe(audio, provider=cfg.stt_provider, api_key=cfg.stt_api_key,
                                model=cfg.stt_model, language=cfg.language)
        reply = await self._kazi.run(text, thread_id=self._scoped_thread(thread_id),
                                      system_prompt=self._system_prompt)
        return await synthesize(reply, provider=cfg.tts_provider, api_key=cfg.tts_api_key,
                                model=cfg.tts_model, voice=cfg.tts_voice, speed=cfg.tts_speed,
                                elevenlabs_voice_id=cfg.elevenlabs_voice_id,
                                elevenlabs_model=cfg.elevenlabs_model)

    async def stream_voice(self, audio: bytes, *, thread_id: str = "default") -> AsyncIterator[bytes]:
        """Low-latency streaming voice for this agent."""
        self._kazi._assert_voice()
        from kazi.voice.stt import transcribe
        from kazi.voice.tts import synthesize_stream
        cfg = self._kazi.config.voice
        text = await transcribe(audio, provider=cfg.stt_provider, api_key=cfg.stt_api_key,
                                model=cfg.stt_model, language=cfg.language)
        token_stream = self._kazi.stream(text, thread_id=self._scoped_thread(thread_id),
                                          system_prompt=self._system_prompt)
        async for chunk in synthesize_stream(
            token_stream, provider=cfg.tts_provider, api_key=cfg.tts_api_key,
            model=cfg.tts_model, voice=cfg.tts_voice, speed=cfg.tts_speed,
            elevenlabs_voice_id=cfg.elevenlabs_voice_id, elevenlabs_model=cfg.elevenlabs_model,
        ):
            yield chunk

    def _scoped_thread(self, thread_id: str) -> str:
        """
        Thread IDs are shared across voice and chat for the same agent+user pair.
        Prefixed with the agent name so Riley and Jordan don't share memory.
        """
        return f"{self.name.lower()}:{thread_id}"

    def __repr__(self) -> str:
        return f"SubAgent(name={self.name!r}, role={self.role!r})"
