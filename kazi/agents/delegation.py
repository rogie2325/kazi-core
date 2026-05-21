"""
High-level delegation helpers for multi-agent workflows.

These sit above A2ABridge and let you describe tasks in natural language
rather than wiring skill names manually.
"""
from __future__ import annotations

import re

from kazi.agents.a2a_client import A2ABridge
from kazi.agents.agent_card import AgentCard, AgentSkill


async def delegate_to_best_agent(
    bridge: A2ABridge,
    task: str,
    capability_hint: str | None = None,
    *,
    _visited: frozenset[str] = frozenset(),
) -> str:
    """
    Pick the best available agent for a task and delegate to it.

    Selects by:
      1. ``capability_hint`` — substring match against agent capability tags
      2. Skill description scoring — picks the skill with the most task-word overlap
      3. Falls back to first available agent/skill when nothing matches

    Cycle detection
    ---------------
    Pass ``_visited`` when calling from inside an agent handler to prevent
    delegation loops.  If the chosen agent is already in the chain, the call
    is short-circuited and a descriptive error string is returned instead of
    making an infinite A2A call::

        result = await delegate_to_best_agent(
            bridge, "Analyse sales data",
            _visited=frozenset({"my-agent-name"}),
        )
    """
    agents = bridge.list_agents()
    if not agents:
        return "No remote agents available for delegation."

    chosen: AgentCard | None = None

    if capability_hint:
        hint_lower = capability_hint.lower()
        for agent in agents:
            if agent.name in _visited:
                continue
            if any(hint_lower in cap.lower() for cap in agent.capabilities):
                chosen = agent
                break

    if chosen is None:
        # Pick the agent (not already in the chain) whose skills best match the task
        best_score = -1
        for agent in agents:
            if agent.name in _visited:
                continue
            score = max((_score_skill(s, task) for s in agent.skills), default=0)
            if score > best_score:
                best_score = score
                chosen = agent

    if chosen is None:
        visited_names = ", ".join(sorted(_visited))
        return (
            f"Delegation cycle or no eligible agents: all available agents "
            f"({visited_names}) are already in the current delegation chain."
        )

    skill = _pick_best_skill(chosen, task)
    if skill is None:
        return f"Agent '{chosen.name}' has no skills defined."

    return await bridge.delegate(chosen.name, skill.name, {"task_description": task})


def _tokenize(text: str) -> set[str]:
    """Return a set of lowercase alphanum tokens from text."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _score_skill(skill: AgentSkill, task: str) -> int:
    """
    Score how well a skill description matches the task using word-overlap.

    Returns the count of unique task words found in the skill description.
    This is a simple but effective BM25-lite relevance signal that handles
    multi-word task descriptions much better than the previous any()-match.
    """
    task_tokens = _tokenize(task)
    skill_tokens = _tokenize(skill.description + " " + skill.name)
    # Ignore common stop words to avoid false positives on "the", "a", "in", etc.
    _STOP = {"a", "an", "the", "to", "of", "in", "for", "and", "or", "with", "is", "on"}
    meaningful = task_tokens - _STOP
    if not meaningful:
        return 0
    return len(meaningful & skill_tokens)


def _pick_best_skill(agent: AgentCard, task: str) -> AgentSkill | None:
    """Return the skill with the highest task-relevance score, or the first skill."""
    if not agent.skills:
        return None
    scored = [(s, _score_skill(s, task)) for s in agent.skills]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0]


# Backward-compatible alias used by tests and external callers
_pick_skill = _pick_best_skill


async def fan_out(
    bridge: A2ABridge,
    tasks: list[dict],
    *,
    _visited: frozenset[str] = frozenset(),
) -> list[str]:
    """
    Delegate multiple tasks in parallel, one per entry.

    Each entry: {"agent": str, "skill": str, "params": dict}
    Returns results in the same order.

    Pass ``_visited`` to propagate cycle-detection context from a parent agent::

        results = await fan_out(bridge, tasks, _visited=frozenset({"parent-agent"}))
    """
    import asyncio

    async def _one(entry: dict) -> str:
        agent_name = entry["agent"]
        if agent_name in _visited:
            return (
                f"Delegation cycle: '{agent_name}' is already in the current chain "
                f"{sorted(_visited)}."
            )
        return await bridge.delegate(agent_name, entry["skill"], entry.get("params", {}))

    return list(await asyncio.gather(*[_one(t) for t in tasks], return_exceptions=False))
