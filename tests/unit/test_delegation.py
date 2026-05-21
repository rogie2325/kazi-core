"""Tests for kazi.agents.delegation — delegate_to_best_agent, _pick_skill, fan_out."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from kazi.agents.agent_card import AgentCard, AgentSkill
from kazi.agents.delegation import _pick_skill, delegate_to_best_agent, fan_out


def _card(name: str, capabilities: list[str] = None, skills: list[AgentSkill] = None) -> AgentCard:
    return AgentCard(
        name=name,
        description=f"{name} agent",
        url=f"https://{name}.example.com",
        capabilities=capabilities or [],
        skills=skills or [],
    )


def _skill(name: str, description: str = "") -> AgentSkill:
    return AgentSkill(name=name, description=description or name)


def _bridge_with_agents(agents: list[AgentCard]) -> MagicMock:
    bridge = MagicMock()
    bridge.list_agents.return_value = agents
    bridge.delegate = AsyncMock(return_value="<external_content>delegated result</external_content>")
    return bridge


# ── delegate_to_best_agent ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_no_agents_returns_message():
    bridge = _bridge_with_agents([])
    result = await delegate_to_best_agent(bridge, "do something")
    assert "No remote agents" in result
    bridge.delegate.assert_not_called()


@pytest.mark.asyncio
async def test_picks_first_agent_when_no_hint():
    agents = [
        _card("alpha", skills=[_skill("run")]),
        _card("beta", skills=[_skill("run")]),
    ]
    bridge = _bridge_with_agents(agents)
    await delegate_to_best_agent(bridge, "do a task")
    # First agent used when no capability hint
    bridge.delegate.assert_called_once()
    assert bridge.delegate.call_args[0][0] == "alpha"


@pytest.mark.asyncio
async def test_picks_agent_matching_capability_hint():
    agents = [
        _card("general", capabilities=["text"], skills=[_skill("do")]),
        _card("coder", capabilities=["code-execution", "python"], skills=[_skill("run_code")]),
    ]
    bridge = _bridge_with_agents(agents)
    await delegate_to_best_agent(bridge, "run some code", capability_hint="python")
    bridge.delegate.assert_called_once()
    assert bridge.delegate.call_args[0][0] == "coder"


@pytest.mark.asyncio
async def test_falls_back_to_first_when_hint_unmatched():
    agents = [
        _card("alpha", capabilities=["text"], skills=[_skill("process")]),
        _card("beta", capabilities=["images"], skills=[_skill("analyse")]),
    ]
    bridge = _bridge_with_agents(agents)
    await delegate_to_best_agent(bridge, "do something", capability_hint="audio")
    # Neither matches "audio" → falls back to first
    assert bridge.delegate.call_args[0][0] == "alpha"


@pytest.mark.asyncio
async def test_returns_no_skills_message_when_chosen_agent_has_none():
    agents = [_card("empty", skills=[])]
    bridge = _bridge_with_agents(agents)
    result = await delegate_to_best_agent(bridge, "do something")
    assert "no skills" in result.lower()
    bridge.delegate.assert_not_called()


@pytest.mark.asyncio
async def test_delegates_with_task_description():
    skill = _skill("summarize", "Summarise text documents")
    agents = [_card("nlp", skills=[skill])]
    bridge = _bridge_with_agents(agents)
    await delegate_to_best_agent(bridge, "summarize this report")
    _, skill_name, params = bridge.delegate.call_args[0]
    assert "task_description" in params
    assert "summarize this report" in params["task_description"]


@pytest.mark.asyncio
async def test_returns_delegate_result():
    agents = [_card("worker", skills=[_skill("do")])]
    bridge = _bridge_with_agents(agents)
    result = await delegate_to_best_agent(bridge, "task")
    assert "delegated result" in result


# ── _pick_skill ───────────────────────────────────────────────────────────────

def test_pick_skill_returns_none_when_no_skills():
    card = _card("empty", skills=[])
    assert _pick_skill(card, "do something") is None


def test_pick_skill_returns_first_when_no_keyword_match():
    skills = [_skill("alpha"), _skill("beta"), _skill("gamma")]
    card = _card("multi", skills=skills)
    result = _pick_skill(card, "zzzzz no match here")
    assert result.name == "alpha"


def test_pick_skill_returns_matching_skill():
    skills = [
        _skill("translate", "Translate text between languages"),
        _skill("summarize", "Summarise documents"),
    ]
    card = _card("nlp", skills=skills)
    result = _pick_skill(card, "I need to summarize this long document")
    assert result.name == "summarize"


def test_pick_skill_matches_on_any_word():
    skills = [
        _skill("web_search", "Search the internet for information"),
        _skill("file_reader", "Read files from disk"),
    ]
    card = _card("tools", skills=skills)
    # "disk" is in "Read files from disk"
    result = _pick_skill(card, "read something from disk")
    assert result.name == "file_reader"


def test_pick_skill_returns_first_skill_for_single_skill_agent():
    card = _card("single", skills=[_skill("only_skill")])
    result = _pick_skill(card, "anything at all")
    assert result.name == "only_skill"


# ── fan_out ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fan_out_returns_results_in_order():
    results = {"task_a": "Result A", "task_b": "Result B", "task_c": "Result C"}

    async def mock_delegate(agent, skill, params):
        return results[params.get("id", agent)]

    bridge = MagicMock()
    bridge.delegate = mock_delegate

    tasks = [
        {"agent": "a", "skill": "do", "params": {"id": "task_a"}},
        {"agent": "b", "skill": "do", "params": {"id": "task_b"}},
        {"agent": "c", "skill": "do", "params": {"id": "task_c"}},
    ]
    output = await fan_out(bridge, tasks)

    assert output == ["Result A", "Result B", "Result C"]


@pytest.mark.asyncio
async def test_fan_out_empty_tasks_returns_empty_list():
    bridge = MagicMock()
    bridge.delegate = AsyncMock()
    output = await fan_out(bridge, [])
    assert output == []
    bridge.delegate.assert_not_called()


@pytest.mark.asyncio
async def test_fan_out_uses_empty_params_when_missing():
    """params key is optional in fan_out entries."""
    received_params = []

    async def mock_delegate(agent, skill, params):
        received_params.append(params)
        return "ok"

    bridge = MagicMock()
    bridge.delegate = mock_delegate

    await fan_out(bridge, [{"agent": "a", "skill": "s"}])
    assert received_params == [{}]


@pytest.mark.asyncio
async def test_fan_out_propagates_exception():
    bridge = MagicMock()
    bridge.delegate = AsyncMock(side_effect=RuntimeError("agent down"))

    with pytest.raises(RuntimeError, match="agent down"):
        await fan_out(bridge, [{"agent": "a", "skill": "s", "params": {}}])


@pytest.mark.asyncio
async def test_fan_out_all_tasks_are_dispatched():
    call_count = 0

    async def mock_delegate(agent, skill, params):
        nonlocal call_count
        call_count += 1
        return f"done {call_count}"

    bridge = MagicMock()
    bridge.delegate = mock_delegate

    tasks = [{"agent": f"a{i}", "skill": "s", "params": {}} for i in range(5)]
    results = await fan_out(bridge, tasks)

    assert call_count == 5
    assert len(results) == 5
