"""Tests for pre-built LangGraph node factories in kazi.brain.nodes."""
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from kazi.brain.nodes import make_reflection_node, make_router_node, make_summariser_node


def _mock_llm(response_content: str) -> AsyncMock:
    llm = AsyncMock()
    response = MagicMock()
    response.content = response_content
    llm.ainvoke = AsyncMock(return_value=response)
    return llm


def _state(messages, metadata=None):
    return {
        "messages": messages,
        "metadata": metadata or {},
        "thread_id": "test",
        "current_step": "test",
        "tool_calls_made": 0,
        "max_tool_calls": 25,
        "system_prompt": None,
        "final_answer": None,
    }


# ── make_summariser_node ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_summariser_returns_empty_when_under_limit():
    """No summarisation when message count is within the limit."""
    llm = _mock_llm("should not be called")
    node = make_summariser_node(llm, max_turns=10)
    messages = [HumanMessage(content="hi"), AIMessage(content="hello")]
    result = await node(_state(messages))
    assert result == {}
    llm.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_summariser_returns_empty_when_exactly_at_limit():
    llm = _mock_llm("should not be called")
    node = make_summariser_node(llm, max_turns=3)
    messages = [HumanMessage(content=f"msg {i}") for i in range(3)]
    result = await node(_state(messages))
    assert result == {}
    llm.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_summariser_compresses_when_over_limit():
    """When history exceeds max_turns, old messages are replaced with a summary."""
    llm = _mock_llm("This is the compressed summary.")
    node = make_summariser_node(llm, max_turns=2)

    messages = [
        HumanMessage(content="msg 1"),
        AIMessage(content="reply 1"),
        HumanMessage(content="msg 2"),
        AIMessage(content="reply 2"),
        HumanMessage(content="msg 3"),
    ]
    result = await node(_state(messages))

    assert "messages" in result
    # First message should be the summary
    first = result["messages"][0]
    assert isinstance(first, SystemMessage)
    assert "compressed summary" in first.content


@pytest.mark.asyncio
async def test_summariser_retains_recent_messages():
    llm = _mock_llm("Summary here.")
    node = make_summariser_node(llm, max_turns=2)

    messages = [
        HumanMessage(content="old 1"),
        HumanMessage(content="old 2"),
        HumanMessage(content="recent 1"),
        HumanMessage(content="recent 2"),
    ]
    result = await node(_state(messages))

    # The last max_turns messages should still be present verbatim
    retained_contents = [m.content for m in result["messages"][1:]]
    assert "recent 1" in retained_contents
    assert "recent 2" in retained_contents


@pytest.mark.asyncio
async def test_summariser_calls_llm_with_history_text():
    llm = _mock_llm("Summary.")
    node = make_summariser_node(llm, max_turns=1)

    messages = [
        HumanMessage(content="tell me about kazi"),
        AIMessage(content="kazi is a framework"),
        HumanMessage(content="thanks"),
    ]
    await node(_state(messages))

    prompt_used = llm.ainvoke.call_args[0][0][0].content
    assert "kazi" in prompt_used.lower()


# ── make_reflection_node ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_reflection_skips_when_last_message_is_not_ai():
    llm = _mock_llm("should not be called")
    node = make_reflection_node(llm)
    messages = [HumanMessage(content="a question")]
    result = await node(_state(messages))
    assert result == {}
    llm.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_reflection_returns_empty_when_lgtm():
    llm = _mock_llm("LGTM — response is correct and complete.")
    node = make_reflection_node(llm)
    messages = [HumanMessage(content="q"), AIMessage(content="a")]
    result = await node(_state(messages))
    assert result == {}


@pytest.mark.asyncio
async def test_reflection_returns_empty_when_lgtm_lowercase():
    llm = _mock_llm("lgtm, looks great!")
    node = make_reflection_node(llm)
    messages = [AIMessage(content="original response")]
    result = await node(_state(messages))
    assert result == {}


@pytest.mark.asyncio
async def test_reflection_appends_correction_when_issues_found():
    llm = _mock_llm("The response contains an error. Corrected: The capital is Paris.")
    node = make_reflection_node(llm)
    messages = [HumanMessage(content="Capital of France?"), AIMessage(content="London")]
    result = await node(_state(messages))

    assert "messages" in result
    assert len(result["messages"]) == 1
    correction = result["messages"][0]
    assert isinstance(correction, AIMessage)
    assert "Paris" in correction.content


@pytest.mark.asyncio
async def test_reflection_passes_last_response_to_llm():
    llm = _mock_llm("LGTM")
    node = make_reflection_node(llm)
    messages = [HumanMessage(content="q"), AIMessage(content="this is the AI response")]
    await node(_state(messages))

    prompt = llm.ainvoke.call_args[0][0][0].content
    assert "this is the AI response" in prompt


# ── make_router_node ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_router_matches_intent_in_message():
    routes = {"research": "rag_agent", "action": "tool_agent"}
    node = make_router_node(routes)
    messages = [HumanMessage(content="I need to research the market trends")]
    result = await node(_state(messages))
    assert result["metadata"]["route"] == "rag_agent"


@pytest.mark.asyncio
async def test_router_falls_back_to_first_route_when_no_match():
    routes = {"research": "rag_agent", "action": "tool_agent"}
    node = make_router_node(routes)
    messages = [HumanMessage(content="tell me a joke")]
    result = await node(_state(messages))
    # No keyword match → first route
    assert result["metadata"]["route"] == "rag_agent"


@pytest.mark.asyncio
async def test_router_returns_first_route_when_no_human_messages():
    routes = {"research": "rag_agent", "action": "tool_agent"}
    node = make_router_node(routes)
    messages = [AIMessage(content="I can help with that")]
    result = await node(_state(messages))
    assert result["metadata"]["route"] == "rag_agent"


@pytest.mark.asyncio
async def test_router_preserves_existing_metadata():
    routes = {"action": "tool_agent"}
    node = make_router_node(routes)
    messages = [HumanMessage(content="take an action")]
    initial_metadata = {"user_id": "u123", "session": "s456"}
    result = await node(_state(messages, metadata=initial_metadata))
    assert result["metadata"]["user_id"] == "u123"
    assert result["metadata"]["session"] == "s456"
    assert result["metadata"]["route"] == "tool_agent"


@pytest.mark.asyncio
async def test_router_uses_last_human_message():
    """Only the most recent HumanMessage matters for routing."""
    routes = {"research": "rag_agent", "action": "tool_agent"}
    node = make_router_node(routes)
    messages = [
        HumanMessage(content="I want to do research"),   # old — says research
        AIMessage(content="Sure!"),
        HumanMessage(content="Actually take an action"),  # new — says action
    ]
    result = await node(_state(messages))
    assert result["metadata"]["route"] == "tool_agent"
