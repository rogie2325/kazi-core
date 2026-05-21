"""
Pre-built LangGraph node factories for common patterns.

These are thin wrappers you can drop into custom graphs built
on top of the Kazi foundation.
"""
from __future__ import annotations

from collections.abc import Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from kazi.brain.state import AgentState


def make_summariser_node(llm, max_turns: int = 20) -> Callable:
    """
    Returns a node that summarises the conversation history when it
    exceeds `max_turns`, replacing old messages with a single summary.
    """

    async def summarise(state: AgentState) -> dict:
        msgs = list(state["messages"])
        if len(msgs) <= max_turns:
            return {}

        history_text = "\n".join(
            f"{m.__class__.__name__}: {m.content}" for m in msgs[:-max_turns]
        )
        prompt = (
            "Summarise the following conversation history in 3–5 sentences "
            "preserving all key facts and decisions:\n\n" + history_text
        )
        summary_msg = await llm.ainvoke([HumanMessage(content=prompt)])
        compressed = [SystemMessage(content=f"[Conversation summary]: {summary_msg.content}")]
        return {"messages": compressed + msgs[-max_turns:]}

    return summarise


def make_reflection_node(llm) -> Callable:
    """
    Returns a node that critiques the last AI response and appends
    a self-correction if it finds issues.
    """

    async def reflect(state: AgentState) -> dict:
        last = state["messages"][-1]
        if not isinstance(last, AIMessage):
            return {}

        critique_prompt = (
            "Review this response for accuracy, completeness, and helpfulness. "
            "If it has issues, provide a corrected version. "
            "If it is fine, just reply 'LGTM'.\n\n"
            f"Response to review:\n{last.content}"
        )
        critique = await llm.ainvoke([HumanMessage(content=critique_prompt)])
        if "LGTM" in critique.content.upper():
            return {}
        return {"messages": [AIMessage(content=critique.content)]}

    return reflect


def make_router_node(routes: dict[str, str]) -> Callable:
    """
    Returns a node that classifies intent and returns a routing key.

    `routes` maps intent labels to node names, e.g.
    {"research": "rag_agent", "action": "tool_agent"}.
    """

    async def route(state: AgentState) -> dict:
        last_human = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            None,
        )
        if last_human is None:
            return {"metadata": {**state.get("metadata", {}), "route": list(routes.values())[0]}}

        # Ensure content is a string before calling lower()
        content = last_human.content if isinstance(last_human.content, str) else str(last_human.content)
        intent = content.lower()
        chosen = list(routes.values())[0]
        for label, target in routes.items():
            if label.lower() in intent:
                chosen = target
                break
        return {"metadata": {**state.get("metadata", {}), "route": chosen}}

    return route
