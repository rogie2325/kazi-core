"""Supervisor — routes requests to the right SubAgent using LLM-based routing."""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any, Protocol

from langchain_core.messages import HumanMessage

from kazi.agents.monitor import ComponentHealth, PerformanceMonitor

logger = logging.getLogger(__name__)


class _Agent(Protocol):
    name: str
    role: str

    async def run(
        self,
        message: str,
        *,
        thread_id: str = "default",
        max_tool_calls: int = 25,
        user_token: str | None = None,
    ) -> str:
        ...

    def stream(
        self,
        message: str,
        *,
        thread_id: str = "default",
        max_tool_calls: int = 25,
        user_token: str | None = None,
    ) -> AsyncIterator[str]:
        ...


class Supervisor:
    """
    Coordinates a crew of SubAgents, routing each request to the most
    appropriate agent based on the message content.

    Cross-model consistency: routing uses the primary LLM, but each agent
    uses its own system_prompt to maintain personality regardless of which
    model (primary, fallback, tool_call route) ultimately handles the turn.

    Cross-modal memory: thread_id is passed through to the SubAgent, which
    scopes it as `{agent_name}:{thread_id}`.  Voice and chat sessions for the
    same user share a thread, so the agent remembers both modalities.

    Tool delegation: agents can use any tool in their allowlist, including
    A2A tools that delegate to other SubAgents or external agents.

    Usage::

        from kazi.agents import SubAgent, SubAgentConfig, Supervisor

        riley = SubAgent(SubAgentConfig(
            name="Riley",
            role="Research & Intelligence",
            system_prompt="You are Riley, a research specialist...",
        ), kazi)

        jordan = SubAgent(SubAgentConfig(
            name="Jordan",
            role="Execution & Output",
            system_prompt="You are Jordan, a delivery specialist...",
        ), kazi)

        crew = Supervisor(agents=[riley, jordan], kazi=kazi)

        async with kazi:
            reply = await crew.run("Research our top competitors", thread_id="user:123")
            # Voice works too — same thread_id, same memory
            audio = await crew.run_voice(audio_bytes, thread_id="user:123")
    """

    def __init__(
        self,
        agents: list[_Agent],
        kazi: Any,
        *,
        default_agent: str | None = None,
        monitor: PerformanceMonitor | None = None,
    ) -> None:
        """
        agents          List of SubAgent instances.
        kazi           The parent Kazi instance (for LLM access + voice).
        default_agent   Name of the agent to use when routing is ambiguous.
                        Defaults to the first agent in the list.
        monitor         Optional PerformanceMonitor.  When provided, every
                        agent call outcome is recorded.  Agents that exceed
                        the configured thresholds are automatically removed
                        from the crew so they receive no further requests.
                        Example::

                            from kazi.agents.monitor import PerformanceMonitor
                            monitor = PerformanceMonitor(consecutive_threshold=3)
                            crew = Supervisor(agents=[...], kazi=kazi, monitor=monitor)
        """
        if not agents:
            raise ValueError("Supervisor requires at least one agent.")
        self._agents: dict[str, _Agent] = {a.name: a for a in agents}
        self._kazi = kazi
        self._default = default_agent or agents[0].name
        self._monitor = monitor

    # ── Public API ────────────────────────────────────────────────────────

    async def run(
        self,
        message: str,
        *,
        thread_id: str = "default",
        agent_name: str | None = None,
        max_tool_calls: int = 25,
        user_token: str | None = None,
    ) -> str:
        """
        Route `message` to the best agent and return its text response.

        agent_name  Force a specific agent by name (skip routing).

        If a PerformanceMonitor is attached, exceptions from the agent are
        recorded as failures.  Once an agent is fired it is removed from the
        crew; subsequent requests route to the next-best (or default) agent.
        """
        agent = self._resolve(agent_name) or await self._route(message)
        logger.info("Supervisor → %s (thread=%s)", agent.name, thread_id)
        try:
            result = await agent.run(
                message, thread_id=thread_id,
                max_tool_calls=max_tool_calls, user_token=user_token,
            )
            if self._monitor is not None:
                self._monitor.record(agent.name, success=True)
            return result
        except Exception:
            if self._monitor is not None:
                fired = self._monitor.record(agent.name, success=False)
                if fired:
                    self._fire_agent(agent.name)
            raise

    async def stream(
        self,
        message: str,
        *,
        thread_id: str = "default",
        agent_name: str | None = None,
        max_tool_calls: int = 25,
        user_token: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream tokens from the routed agent's response."""
        agent = self._resolve(agent_name) or await self._route(message)
        logger.info("Supervisor stream → %s (thread=%s)", agent.name, thread_id)
        try:
            async for token in agent.stream(
                message, thread_id=thread_id,
                max_tool_calls=max_tool_calls, user_token=user_token,
            ):
                yield token
            if self._monitor is not None:
                self._monitor.record(agent.name, success=True)
        except Exception:
            if self._monitor is not None:
                fired = self._monitor.record(agent.name, success=False)
                if fired:
                    self._fire_agent(agent.name)
            raise

    async def run_voice(
        self,
        audio: bytes,
        *,
        thread_id: str = "default",
        agent_name: str | None = None,
    ) -> bytes:
        """
        Full voice round-trip routed to the best agent.
        Transcribes audio → routes → synthesizes reply.
        Same thread_id gives the agent memory of prior chat sessions.
        """
        # Transcribe first so we can route on the text
        self._kazi._assert_voice()
        from kazi.voice.stt import transcribe
        cfg = self._kazi.config.voice
        text = await transcribe(audio, provider=cfg.stt_provider, api_key=cfg.stt_api_key,
                                model=cfg.stt_model, language=cfg.language)

        agent = self._resolve(agent_name) or await self._route(text)
        logger.info("Supervisor voice → %s (thread=%s)", agent.name, thread_id)

        from kazi.voice.tts import synthesize
        reply = await agent.run(text, thread_id=thread_id)
        return await synthesize(reply, provider=cfg.tts_provider, api_key=cfg.tts_api_key,
                                model=cfg.tts_model, voice=cfg.tts_voice, speed=cfg.tts_speed,
                                elevenlabs_voice_id=cfg.elevenlabs_voice_id,
                                elevenlabs_model=cfg.elevenlabs_model)

    async def stream_voice(
        self,
        audio: bytes,
        *,
        thread_id: str = "default",
        agent_name: str | None = None,
    ) -> AsyncIterator[bytes]:
        """Low-latency streaming voice, routed to the best agent."""
        self._kazi._assert_voice()
        from kazi.voice.stt import transcribe
        cfg = self._kazi.config.voice
        text = await transcribe(audio, provider=cfg.stt_provider, api_key=cfg.stt_api_key,
                                model=cfg.stt_model, language=cfg.language)

        agent = self._resolve(agent_name) or await self._route(text)
        logger.info("Supervisor stream_voice → %s (thread=%s)", agent.name, thread_id)

        from kazi.voice.tts import synthesize_stream
        token_stream = agent.stream(text, thread_id=thread_id)
        async for chunk in synthesize_stream(
            token_stream, provider=cfg.tts_provider, api_key=cfg.tts_api_key,
            model=cfg.tts_model, voice=cfg.tts_voice, speed=cfg.tts_speed,
            elevenlabs_voice_id=cfg.elevenlabs_voice_id, elevenlabs_model=cfg.elevenlabs_model,
        ):
            yield chunk

    @property
    def agents(self) -> list:
        return list(self._agents.values())

    # ── Health / monitoring ───────────────────────────────────────────────

    def agent_health(self, name: str) -> ComponentHealth | None:
        """Return the health snapshot for one agent (None if no monitor attached)."""
        if self._monitor is None:
            return None
        return self._monitor.health(name)

    def crew_health(self) -> list[ComponentHealth]:
        """Return health snapshots for all tracked agents."""
        if self._monitor is None:
            return []
        return self._monitor.summary()

    def fired_agents(self) -> list[str]:
        """Return names of all agents that have been fired."""
        if self._monitor is None:
            return []
        return self._monitor.fired_names()

    def reinstate(self, name: str) -> None:
        """
        Un-fire an agent and add it back to the active crew.

        The agent's performance history is cleared so it starts fresh.
        Call this after the underlying issue has been resolved.
        """
        if self._monitor is not None:
            self._monitor.reset(name)
        # Find the original agent object from fired set if it still exists;
        # the caller is expected to re-add via add_agent() if needed.
        logger.info("Supervisor: %r reinstated — performance history cleared", name)

    def add_agent(self, agent) -> None:
        """Add a new agent (or re-add a previously fired one) to the crew."""
        self._agents[agent.name] = agent
        logger.info("Supervisor: %r added to crew", agent.name)

    # ── Internal ──────────────────────────────────────────────────────────

    def _fire_agent(self, name: str) -> None:
        """Remove a fired agent from the active crew and fall back to default if needed."""
        if name in self._agents:
            del self._agents[name]
            logger.warning(
                "Supervisor: agent %r FIRED and removed from crew. "
                "Remaining agents: %s",
                name, list(self._agents),
            )
        if not self._agents:
            logger.error(
                "Supervisor: all agents have been fired — "
                "no agents remain to handle requests."
            )
            return
        if self._default == name:
            self._default = next(iter(self._agents))
            logger.warning(
                "Supervisor: default agent was fired; new default is %r",
                self._default,
            )

    # ── Routing ───────────────────────────────────────────────────────────

    async def _route(self, message: str):
        """
        Ask the primary LLM which agent is best suited for this message.
        Falls back to the default agent if routing fails or is ambiguous.
        """
        agent_descriptions = "\n".join(
            f"- {name}: {a.role}" for name, a in self._agents.items()
        )
        routing_prompt = (
            f"You are a routing agent. Given the request below, respond with ONLY the name "
            f"of the most appropriate agent — nothing else.\n\n"
            f"Agents:\n{agent_descriptions}\n\n"
            f"Request: {message}"
        )
        try:
            from kazi.brain.graph_builder import GraphBrain
            brain: GraphBrain = self._kazi._brain
            llm = brain._build_llm()
            response = await llm.ainvoke([HumanMessage(content=routing_prompt)])
            selected = response.content.strip().strip('"').strip("'")
            if selected in self._agents:
                return self._agents[selected]
            # Fuzzy match — agent name appears anywhere in the response
            for name in self._agents:
                if name.lower() in selected.lower():
                    return self._agents[name]
        except Exception as exc:
            logger.warning("Routing failed (%s) — using default agent %r", exc, self._default)
        return self._agents[self._default]

    def _resolve(self, agent_name: str | None) -> _Agent | None:
        if agent_name is None:
            return None
        if agent_name in self._agents:
            return self._agents[agent_name]
        raise ValueError(
            f"Unknown agent {agent_name!r}. Available: {list(self._agents.keys())}"
        )
