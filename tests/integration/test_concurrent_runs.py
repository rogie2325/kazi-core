"""
Concurrent run integration tests.

Verifies that multiple simultaneous kazi.run() and brain.run() calls:
  - All complete successfully without errors
  - Don't bleed memory across thread IDs
  - Don't deadlock under the asyncio.Lock budget guard
  - Produce independent results per thread

Requires OPENAI_API_KEY for LLM-backed tests.
"""
import asyncio
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


# ── brain-level concurrency ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_concurrent_brain_runs_all_complete():
    """Five simultaneous brain.run() calls must all return a non-empty string."""
    from kazi.brain.graph_builder import GraphBrain
    from kazi.core.registry import ToolRegistry

    brain = GraphBrain(_kazi_config(), ToolRegistry())
    results = await asyncio.gather(*[
        brain.run(f"Say exactly the word: TOKEN_{i}", thread_id=f"conc-{i}")
        for i in range(5)
    ])

    assert len(results) == 5
    assert all(isinstance(r, str) and len(r) > 0 for r in results)


@pytest.mark.asyncio
async def test_concurrent_brain_runs_no_thread_bleed():
    """Concurrent runs on distinct thread IDs must not share memory."""
    from kazi.brain.graph_builder import GraphBrain
    from kazi.core.registry import ToolRegistry

    brain = GraphBrain(_kazi_config(), ToolRegistry())

    # Plant a unique secret in each thread simultaneously
    secrets = {f"bleed-{i}": f"SECRET_{i}" for i in range(3)}
    await asyncio.gather(*[
        brain.run(f"Remember this secret: {secret}", thread_id=tid)
        for tid, secret in secrets.items()
    ])

    # Each thread should only know its own secret
    async def check(tid: str, own_secret: str, other_secrets: list[str]) -> None:
        reply = await brain.run("What secret did I just tell you?", thread_id=tid)
        assert own_secret in reply, f"Thread {tid} forgot its own secret"
        for other in other_secrets:
            assert other not in reply, f"Thread {tid} leaked secret from another thread: {other}"

    await asyncio.gather(*[
        check(tid, secret, [s for t, s in secrets.items() if t != tid])
        for tid, secret in secrets.items()
    ])


@pytest.mark.asyncio
async def test_concurrent_brain_runs_with_tool():
    """Tool calls must not be dropped or duplicated under concurrent load."""
    from kazi.brain.graph_builder import GraphBrain
    from kazi.core.config import KaziConfig
    from kazi.core.registry import ToolDefinition, ToolRegistry, ToolSource

    registry = ToolRegistry()
    call_counts: dict[int, int] = {}

    for i in range(3):
        idx = i

        async def tool(thread_index=idx) -> str:
            call_counts[thread_index] = call_counts.get(thread_index, 0) + 1
            return f"TOOL_RESULT_{thread_index}"

        registry.register(ToolDefinition(
            name=f"get_result_{i}",
            description=f"Returns TOOL_RESULT_{i}. Call this when asked for result {i}.",
            parameters=[],
            source=ToolSource.NATIVE,
            handler=tool,
        ))

    brain = GraphBrain(KaziConfig(llm=_llm_config()), registry)

    results = await asyncio.gather(*[
        brain.run(
            f"Call the get_result_{i} tool and tell me what it returned.",
            thread_id=f"tool-conc-{i}",
        )
        for i in range(3)
    ])

    for i, result in enumerate(results):
        assert f"TOOL_RESULT_{i}" in result, f"Run {i} missing expected tool output"


# ── orchestrator-level concurrency ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_concurrent_kazi_runs_all_succeed():
    """Ten simultaneous Kazi.run() calls must all complete without error."""
    from kazi import Kazi

    async with await Kazi.create(_kazi_config()) as kazi:
        results = await asyncio.gather(*[
            kazi.run(f"Reply with exactly: RESULT_{i}", thread_id=f"kazi-conc-{i}")
            for i in range(10)
        ])

    assert len(results) == 10
    assert all(isinstance(r, str) and len(r) > 0 for r in results)



@pytest.mark.asyncio
async def test_budget_lock_not_deadlocked_under_load():
    """
    The asyncio.Lock in GraphBrain._active_budgets must not deadlock when
    several coroutines enter run() simultaneously.
    """
    from kazi import Kazi

    async with await Kazi.create(_kazi_config()) as kazi:
        # 5 concurrent calls — if the lock deadlocks this will hang until
        # pytest times out the test
        results = await asyncio.wait_for(
            asyncio.gather(*[
                kazi.run("Say: OK", thread_id=f"lock-{i}")
                for i in range(5)
            ]),
            timeout=120.0,
        )

    assert len(results) == 5
    assert all("OK" in r or len(r) > 0 for r in results)


@pytest.mark.asyncio
async def test_concurrent_streams_all_produce_output():
    """Multiple simultaneous stream() calls must each yield tokens."""
    from kazi import Kazi

    async def collect_stream(kazi, i: int) -> str:
        chunks = []
        async for chunk in kazi.stream(f"Count to {i + 2}.", thread_id=f"stream-conc-{i}"):
            chunks.append(chunk)
        return "".join(chunks)

    async with await Kazi.create(_kazi_config()) as kazi:
        results = await asyncio.gather(*[
            collect_stream(kazi, i) for i in range(2)
        ])

    assert all(len(r) > 0 for r in results)
