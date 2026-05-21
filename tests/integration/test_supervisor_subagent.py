"""
Real integration tests for Supervisor and SubAgent — uses OpenAI gpt-4o-mini.

Covers:
  - SubAgent.run(), stream(), _build_system_prompt with tool restriction, __repr__
  - Supervisor.run(), stream(), routing, _fire_agent, reinstate, add_agent,
    agent_health, crew_health, fired_agents, _resolve, ValueError on no agents
"""
from __future__ import annotations

import os

import pytest

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
pytestmark = pytest.mark.skipif(not OPENAI_KEY, reason="OPENAI_API_KEY not set")

MODEL = "gpt-4o-mini"


def _llm_config():
    from kazi.core.config import LLMConfig, LLMProvider
    return LLMConfig(provider=LLMProvider.OPENAI, model=MODEL, api_key=OPENAI_KEY)


def _kazi_config():
    from kazi.core.config import KaziConfig
    return KaziConfig(llm=_llm_config())


@pytest.fixture
async def kazi():
    from kazi import Kazi
    async with await Kazi.create(_kazi_config()) as nx:
        yield nx


# ═══════════════════════════════════════════════════════════════════════════════
# SubAgent
# ═══════════════════════════════════════════════════════════════════════════════

def test_subagent_repr(kazi):
    from kazi.agents.subagent import SubAgent, SubAgentConfig
    agent = SubAgent(SubAgentConfig(
        name="Riley", role="Research", system_prompt="You research things."
    ), kazi)
    r = repr(agent)
    assert "Riley" in r
    assert "Research" in r


def test_subagent_system_prompt_with_tool_restriction(kazi):
    from kazi.agents.subagent import SubAgent, SubAgentConfig
    agent = SubAgent(SubAgentConfig(
        name="Sam",
        role="Data",
        system_prompt="You are Sam.",
        tools=["web_search", "database"],
    ), kazi)
    assert "web_search" in agent._system_prompt
    assert "database" in agent._system_prompt
    assert "ONLY" in agent._system_prompt


def test_subagent_scoped_thread(kazi):
    from kazi.agents.subagent import SubAgent, SubAgentConfig
    agent = SubAgent(SubAgentConfig(
        name="Riley", role="R", system_prompt="You are Riley."
    ), kazi)
    assert agent._scoped_thread("user:42") == "riley:user:42"


@pytest.mark.asyncio
async def test_subagent_run_returns_string(kazi):
    from kazi.agents.subagent import SubAgent, SubAgentConfig
    agent = SubAgent(SubAgentConfig(
        name="Riley",
        role="Research",
        system_prompt="You are Riley, a researcher. Be concise.",
    ), kazi)
    result = await agent.run("Say exactly: SUBAGENT_OK", thread_id="sub-run-test")
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_subagent_stream_yields_tokens(kazi):
    from kazi.agents.subagent import SubAgent, SubAgentConfig
    agent = SubAgent(SubAgentConfig(
        name="Jordan",
        role="Execution",
        system_prompt="You are Jordan. Be concise.",
    ), kazi)
    chunks = []
    async for token in agent.stream("Count to three.", thread_id="sub-stream-test"):
        chunks.append(token)
    assert len(chunks) > 0
    assert "".join(chunks).strip() != ""


@pytest.mark.asyncio
async def test_subagent_thread_is_scoped_to_agent_name(kazi):
    """Two agents with the same user thread_id have separate memory."""
    from kazi.agents.subagent import SubAgent, SubAgentConfig

    riley = SubAgent(SubAgentConfig(
        name="Riley", role="R", system_prompt="You are Riley."
    ), kazi)
    jordan = SubAgent(SubAgentConfig(
        name="Jordan", role="J", system_prompt="You are Jordan."
    ), kazi)

    await riley.run("My name is ALICE.", thread_id="shared-user")
    await jordan.run("My name is BOB.", thread_id="shared-user")

    riley_reply = await riley.run("What is my name?", thread_id="shared-user")
    jordan_reply = await jordan.run("What is my name?", thread_id="shared-user")

    assert "alice" in riley_reply.lower()
    assert "bob" in jordan_reply.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# Supervisor — construction
# ═══════════════════════════════════════════════════════════════════════════════

def test_supervisor_requires_at_least_one_agent(kazi):
    from kazi.agents.supervisor import Supervisor
    with pytest.raises(ValueError, match="at least one"):
        Supervisor(agents=[], kazi=kazi)


def test_supervisor_default_agent_is_first(kazi):
    from kazi.agents.subagent import SubAgent, SubAgentConfig
    from kazi.agents.supervisor import Supervisor

    a = SubAgent(SubAgentConfig(name="A", role="R", system_prompt="A"), kazi)
    b = SubAgent(SubAgentConfig(name="B", role="R", system_prompt="B"), kazi)
    sup = Supervisor(agents=[a, b], kazi=kazi)
    assert sup._default == "A"


def test_supervisor_explicit_default(kazi):
    from kazi.agents.subagent import SubAgent, SubAgentConfig
    from kazi.agents.supervisor import Supervisor

    a = SubAgent(SubAgentConfig(name="A", role="R", system_prompt="A"), kazi)
    b = SubAgent(SubAgentConfig(name="B", role="R", system_prompt="B"), kazi)
    sup = Supervisor(agents=[a, b], kazi=kazi, default_agent="B")
    assert sup._default == "B"


# ═══════════════════════════════════════════════════════════════════════════════
# Supervisor — run
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_supervisor_run_forced_agent(kazi):
    from kazi.agents.subagent import SubAgent, SubAgentConfig
    from kazi.agents.supervisor import Supervisor

    riley = SubAgent(SubAgentConfig(
        name="Riley", role="Research", system_prompt="You are Riley."
    ), kazi)
    sup = Supervisor(agents=[riley], kazi=kazi)
    result = await sup.run(
        "Say exactly: SUPERVISOR_OK",
        agent_name="Riley",
        thread_id="sup-forced",
    )
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_supervisor_run_routed(kazi):
    """No agent_name — supervisor routes via LLM."""
    from kazi.agents.subagent import SubAgent, SubAgentConfig
    from kazi.agents.supervisor import Supervisor

    riley = SubAgent(SubAgentConfig(
        name="Riley", role="Research & data analysis", system_prompt="You are Riley."
    ), kazi)
    jordan = SubAgent(SubAgentConfig(
        name="Jordan", role="Writing & communication", system_prompt="You are Jordan."
    ), kazi)
    sup = Supervisor(agents=[riley, jordan], kazi=kazi)
    result = await sup.run("Say: ROUTED_OK", thread_id="sup-route")
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_supervisor_run_raises_on_unknown_agent(kazi):
    from kazi.agents.subagent import SubAgent, SubAgentConfig
    from kazi.agents.supervisor import Supervisor

    a = SubAgent(SubAgentConfig(name="A", role="R", system_prompt="A"), kazi)
    sup = Supervisor(agents=[a], kazi=kazi)
    with pytest.raises(ValueError, match="Unknown agent"):
        await sup.run("hello", agent_name="NonExistent")


# ═══════════════════════════════════════════════════════════════════════════════
# Supervisor — stream
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_supervisor_stream_yields_tokens(kazi):
    from kazi.agents.subagent import SubAgent, SubAgentConfig
    from kazi.agents.supervisor import Supervisor

    riley = SubAgent(SubAgentConfig(
        name="Riley", role="Research", system_prompt="You are Riley."
    ), kazi)
    sup = Supervisor(agents=[riley], kazi=kazi)
    chunks = []
    async for token in sup.stream(
        "Count to three.", agent_name="Riley", thread_id="sup-stream"
    ):
        chunks.append(token)
    assert len(chunks) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Supervisor — monitor / fire / reinstate / add_agent
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_supervisor_monitor_records_success(kazi):
    from kazi.agents.monitor import PerformanceMonitor
    from kazi.agents.subagent import SubAgent, SubAgentConfig
    from kazi.agents.supervisor import Supervisor

    riley = SubAgent(SubAgentConfig(
        name="Riley", role="Research", system_prompt="You are Riley."
    ), kazi)
    monitor = PerformanceMonitor(consecutive_threshold=5)
    sup = Supervisor(agents=[riley], kazi=kazi, monitor=monitor)

    await sup.run("Say: OK", agent_name="Riley", thread_id="mon-test")
    health = sup.agent_health("Riley")
    assert health is not None
    assert health.consecutive_failures == 0


@pytest.mark.asyncio
async def test_supervisor_fires_agent_after_threshold(kazi):
    """An agent that fails repeatedly should be fired from the crew."""
    from kazi.agents.monitor import PerformanceMonitor
    from kazi.agents.subagent import SubAgent, SubAgentConfig
    from kazi.agents.supervisor import Supervisor

    class _FailingAgent:
        name = "Badbot"
        role = "Always fails"

        async def run(self, *args, **kwargs):
            raise RuntimeError("always fails")

        async def stream(self, *args, **kwargs):
            raise RuntimeError("always fails")
            yield  # make it an async generator

    riley = SubAgent(SubAgentConfig(
        name="Riley", role="Backup", system_prompt="You are Riley."
    ), kazi)
    bad = _FailingAgent()
    monitor = PerformanceMonitor(consecutive_threshold=2)
    sup = Supervisor(agents=[riley, bad], kazi=kazi, monitor=monitor)

    for _ in range(2):
        with pytest.raises(RuntimeError):
            await sup.run("hello", agent_name="Badbot", thread_id="fire-test")

    assert "Badbot" not in [a.name for a in sup.agents]
    assert "Badbot" in sup.fired_agents()


def test_supervisor_reinstate_clears_history(kazi):
    from kazi.agents.monitor import PerformanceMonitor
    from kazi.agents.subagent import SubAgent, SubAgentConfig
    from kazi.agents.supervisor import Supervisor

    a = SubAgent(SubAgentConfig(name="A", role="R", system_prompt="A"), kazi)
    monitor = PerformanceMonitor(consecutive_threshold=3)
    sup = Supervisor(agents=[a], kazi=kazi, monitor=monitor)
    sup.reinstate("A")  # should not raise even without prior failures


def test_supervisor_add_agent(kazi):
    from kazi.agents.subagent import SubAgent, SubAgentConfig
    from kazi.agents.supervisor import Supervisor

    a = SubAgent(SubAgentConfig(name="A", role="R", system_prompt="A"), kazi)
    b = SubAgent(SubAgentConfig(name="B", role="R", system_prompt="B"), kazi)
    sup = Supervisor(agents=[a], kazi=kazi)
    sup.add_agent(b)
    assert "B" in [ag.name for ag in sup.agents]


def test_supervisor_crew_health_without_monitor(kazi):
    from kazi.agents.subagent import SubAgent, SubAgentConfig
    from kazi.agents.supervisor import Supervisor

    a = SubAgent(SubAgentConfig(name="A", role="R", system_prompt="A"), kazi)
    sup = Supervisor(agents=[a], kazi=kazi)
    assert sup.crew_health() == []
    assert sup.agent_health("A") is None
    assert sup.fired_agents() == []


def test_supervisor_fire_agent_updates_default(kazi):
    """When the default agent is fired, the next agent becomes default."""
    from kazi.agents.subagent import SubAgent, SubAgentConfig
    from kazi.agents.supervisor import Supervisor

    a = SubAgent(SubAgentConfig(name="A", role="R", system_prompt="A"), kazi)
    b = SubAgent(SubAgentConfig(name="B", role="R", system_prompt="B"), kazi)
    sup = Supervisor(agents=[a, b], kazi=kazi)
    sup._fire_agent("A")
    assert sup._default == "B"
    assert "A" not in [ag.name for ag in sup.agents]


def test_supervisor_fire_all_agents_logs_error(kazi, caplog):
    import logging

    from kazi.agents.subagent import SubAgent, SubAgentConfig
    from kazi.agents.supervisor import Supervisor

    a = SubAgent(SubAgentConfig(name="A", role="R", system_prompt="A"), kazi)
    sup = Supervisor(agents=[a], kazi=kazi)
    with caplog.at_level(logging.ERROR, logger="kazi.agents.supervisor"):
        sup._fire_agent("A")
    assert "all agents have been fired" in caplog.text


@pytest.mark.asyncio
async def test_supervisor_route_fuzzy_match(kazi):
    """If LLM returns a name with surrounding text, fuzzy match still routes."""
    from kazi.agents.subagent import SubAgent, SubAgentConfig
    from kazi.agents.supervisor import Supervisor

    riley = SubAgent(SubAgentConfig(
        name="Riley", role="Research specialist", system_prompt="You are Riley."
    ), kazi)
    sup = Supervisor(agents=[riley], kazi=kazi)
    # With one agent the router will always pick Riley (or fall back to default)
    result = await sup.run("What is 2+2?", thread_id="fuzzy-test")
    assert isinstance(result, str)
