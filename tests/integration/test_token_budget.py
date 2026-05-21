"""
Integration tests for token budget — exercises the full LangGraph graph loop
with a mock LLM so no API keys are required.

These tests prove the budget state fix works: the budget object stored in
state["metadata"]["_budget"] is found and charged on every graph node.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage

from kazi.brain.graph_builder import GraphBrain
from kazi.core.config import KaziConfig
from kazi.core.exceptions import TokenBudgetExceeded
from kazi.core.registry import ToolRegistry
from kazi.core.token_budget import TokenBudgetConfig


def _mock_llm(response_text: str = "Hello!") -> MagicMock:
    """Mock LLM that always returns a fixed response with no tool calls."""
    llm = MagicMock()
    llm.bind_tools = lambda tools: llm
    llm.ainvoke = AsyncMock(return_value=AIMessage(content=response_text))
    return llm


def _brain_with_mock(response_text: str, budget: "TokenBudgetConfig") -> "GraphBrain":
    """Create a GraphBrain that uses a mock LLM via custom_llm."""
    from kazi.core.config import LLMConfig, LLMProvider
    llm = _mock_llm(response_text)
    cfg = KaziConfig(
        llm=LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            model="claude-haiku-4-5-20251001",
            custom_llm=llm,
        ),
        budget=budget,
    )
    return GraphBrain(cfg, ToolRegistry())


@pytest.mark.asyncio
async def test_budget_exceeded_raises():
    """Budget is charged and raises TokenBudgetExceeded when the limit is hit."""
    brain = _brain_with_mock("x" * 500, TokenBudgetConfig(max_tokens_per_run=5))
    with pytest.raises(TokenBudgetExceeded):
        await brain.run("tell me a story")


@pytest.mark.asyncio
async def test_budget_not_exceeded_within_limit():
    """A response well within budget completes normally and returns text."""
    brain = _brain_with_mock("Hi there!", TokenBudgetConfig(max_tokens_per_run=50_000))
    result = await brain.run("hello")
    assert result == "Hi there!"


@pytest.mark.asyncio
async def test_no_budget_limit_never_raises():
    """max_tokens_per_run=None means unlimited — even huge responses complete."""
    brain = _brain_with_mock("x" * 100_000, TokenBudgetConfig(max_tokens_per_run=None))
    result = await brain.run("hello")
    assert len(result) > 0


@pytest.mark.asyncio
async def test_budget_is_charged_not_zero():
    """Verify the budget object actually accumulates charges (not always zero)."""
    from kazi.core.token_budget import TokenBudget

    charged_values: list[int] = []
    original_check = TokenBudget._check

    def spy_check(self):
        charged_values.append(self._used)
        original_check(self)

    brain = _brain_with_mock("Hello, world!", TokenBudgetConfig(max_tokens_per_run=50_000))

    TokenBudget._check = spy_check
    try:
        await brain.run("hi")
    finally:
        TokenBudget._check = original_check

    # At least one charge must have been non-zero
    assert any(v > 0 for v in charged_values), (
        f"Budget was never charged — state access fix may have regressed. "
        f"Charged values seen: {charged_values}"
    )
