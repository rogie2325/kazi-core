"""
High-level delegation helpers for multi-agent workflows.

These sit above A2ABridge and let you describe tasks in natural language
rather than wiring skill names manually.
"""
from __future__ import annotations

from typing import Optional

from kazi.agents.a2a_client import A2ABridge
from kazi.agents.agent_card import AgentCard


async def delegate_to_best_agent(
    bridge: A2ABridge,
    task: str,
    capability_hint: Optional[str] = None,
) -> str:
    """
    Pick the best available agent for a task and delegate to it.

    Selects by matching `capability_hint` against agent capability tags,
    or falls back to the first available agent.
    """
    agents = bridge.list_agents()
    if not agents:
        return "No remote agents available for delegation."

    chosen: Optional[AgentCard] = None

    if capability_hint:
        for agent in agents:
            if any(capability_hint.lower() in cap.lower() for cap in agent.capabilities):
                chosen = agent
                break

    if chosen is None:
        chosen = agents[0]

    # Use the first skill that looks relevant, else the first skill
    skill = _pick_skill(chosen, task)
    if skill is None:
        return f"Agent '{chosen.name}' has no skills defined."

    return await bridge.delegate(chosen.name, skill.name, {"task_description": task})


def _pick_skill(agent: AgentCard, task: str):
    if not agent.skills:
        return None
    task_lower = task.lower()
    for skill in agent.skills:
        if any(word in skill.description.lower() for word in task_lower.split()):
            return skill
    return agent.skills[0]


async def fan_out(
    bridge: A2ABridge,
    tasks: list[dict],
) -> list[str]:
    """
    Delegate multiple tasks in parallel, one per entry.

    Each entry: {"agent": str, "skill": str, "params": dict}
    Returns results in the same order.
    """
    import asyncio

    async def _one(entry: dict) -> str:
        return await bridge.delegate(entry["agent"], entry["skill"], entry.get("params", {}))

    return list(await asyncio.gather(*[_one(t) for t in tasks], return_exceptions=False))
