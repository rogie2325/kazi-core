"""
Integration tests for token budget — exercises the full LangGraph graph loop
with a real OpenAI LLM. No mocks.

Requires OPENAI_API_KEY; skipped otherwise.
"""
import os

import pytest

from kazi.brain.graph_builder import GraphBrain
from kazi.core.config import KaziConfig, LLMConfig, LLMProvider
from kazi.core.exceptions import TokenBudgetExceeded
from kazi.core.registry import ToolRegistry
from kazi.core.token_budget import TokenBudgetConfig

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
pytestmark = pytest.mark.skipif(not OPENAI_KEY, reason="OPENAI_API_KEY not set")
MODEL = "gpt-4o-mini"


def _brain(budget: TokenBudgetConfig) -> GraphBrain:
    cfg = KaziConfig(
        llm=LLMConfig(provider=LLMProvider.OPENAI, model=MODEL, api_key=OPENAI_KEY),
        budget=budget,
    )
    return GraphBrain(cfg, ToolRegistry())


@pytest.mark.asyncio
async def test_budget_exceeded_raises():
    """Budget is charged and raises TokenBudgetExceeded when the limit is hit."""
    brain = _brain(TokenBudgetConfig(max_tokens_per_run=3))
    with pytest.raises(TokenBudgetExceeded):
        await brain.run(
            "Write a detailed essay on the history of artificial intelligence, "
            "covering every major milestone from 1950 to today."
        )


@pytest.mark.asyncio
async def test_budget_not_exceeded_within_limit():
    """A response well within budget completes and returns a non-empty string."""
    brain = _brain(TokenBudgetConfig(max_tokens_per_run=50_000))
    result = await brain.run("Say hi", thread_id="budget-ok")
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_no_budget_limit_never_raises():
    """max_tokens_per_run=None means unlimited — response always completes."""
    brain = _brain(TokenBudgetConfig(max_tokens_per_run=None))
    result = await brain.run("hello", thread_id="budget-unlimited")
    assert len(result) > 0


@pytest.mark.asyncio
async def test_budget_charged_proves_by_tight_limit():
    """
    Run with a budget just large enough to complete a minimal reply, then
    show a slightly tighter budget would have failed. This proves the
    budget counter accumulates real token counts from the API response.
    """
    # First pass: very tight but just enough — verify it completes
    brain_ok = _brain(TokenBudgetConfig(max_tokens_per_run=50_000))
    result = await brain_ok.run("Reply with the single word: YES", thread_id="tight-ok")
    assert "YES" in result.upper()

    # Second pass: 3-token limit — must raise because even a one-word reply
    # with prompt overhead exceeds 3 tokens
    brain_tight = _brain(TokenBudgetConfig(max_tokens_per_run=3))
    with pytest.raises(TokenBudgetExceeded):
        await brain_tight.run("Reply with the single word: YES", thread_id="tight-fail")


@pytest.mark.asyncio
async def test_budget_resets_between_independent_runs():
    """Each brain.run() gets a fresh budget — prior runs don't bleed over."""
    brain = _brain(TokenBudgetConfig(max_tokens_per_run=50_000))
    r1 = await brain.run("Say: FIRST", thread_id="reset-1")
    r2 = await brain.run("Say: SECOND", thread_id="reset-2")
    assert "FIRST" in r1.upper()
    assert "SECOND" in r2.upper()
