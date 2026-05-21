"""
Concurrency stress tests for audit / shadow / tenant isolation.

These prove the system maintains its invariants under parallel load.
Failures here are the kind unit tests miss but production triggers.

What's verified
===============

1. **Audit isolation under load** — 100 simultaneous runs each get their own
   recorder.  No record is observed in the wrong audit.

2. **Shadow mode under load** — 100 simultaneous shadow runs never invoke
   the real handler, even under contention.

3. **Mixed shadow / live runs** — interleaved runs cannot poison each other.

4. **Tool registry exec atomicity** — concurrent executes against the same
   tool produce per-call records with matching args, never mixed.

5. **Context cleanup** — after N parallel runs, the module-level contextvars
   return to their default (None / False).
"""
from __future__ import annotations

import asyncio

import pytest

from kazi.core.audit import get_recorder, is_shadow, run_context
from kazi.core.registry import ToolRegistry


@pytest.mark.asyncio
async def test_100_parallel_audited_runs_isolated():
    """100 parallel runs — each audit must contain only its own call."""
    registry = ToolRegistry()

    async def square(n: int) -> int:
        # Force a real await so scheduler interleaves these tasks
        await asyncio.sleep(0)
        return n * n

    registry.register_function(square, name="square")

    async def one_run(n: int):
        with run_context(audit=True, shadow=False) as ctx:
            result = await registry.execute("square", n=n)
            audit = ctx.recorder.finalize()
            return n, result, audit

    outcomes = await asyncio.gather(*[one_run(i) for i in range(100)])

    for n, result, audit in outcomes:
        assert result == n * n
        assert audit.tool_call_count == 1
        call = audit.tool_calls[0]
        assert call.name == "square"
        assert call.args == {"n": n}, (
            f"Audit for run {n} contains foreign args {call.args}"
        )


@pytest.mark.asyncio
async def test_100_parallel_shadow_runs_never_invoke_handler():
    """100 parallel shadow runs — handler never runs, even once."""
    registry = ToolRegistry()
    invocations: list[int] = []

    async def destructive(target: int) -> str:
        invocations.append(target)
        return f"deleted {target}"

    registry.register_function(destructive, name="destructive")

    async def one_shadow(target: int):
        with run_context(audit=True, shadow=True) as ctx:
            result = await registry.execute("destructive", target=target)
            audit = ctx.recorder.finalize()
            return target, result, audit

    outcomes = await asyncio.gather(*[one_shadow(i) for i in range(100)])

    # Handler must have been skipped 100/100 times
    assert invocations == [], f"Handler executed {len(invocations)} time(s) under shadow"

    for target, result, audit in outcomes:
        assert "[SHADOW]" in result
        assert audit.tool_call_count == 1
        assert audit.tool_calls[0].status == "shadow"
        assert audit.tool_calls[0].args == {"target": target}


@pytest.mark.asyncio
async def test_mixed_shadow_and_live_runs_do_not_poison_each_other():
    """
    Interleave 50 shadow runs and 50 live runs.

    Live runs must execute the handler.  Shadow runs must NOT.
    Their audits must reflect their respective modes.
    """
    registry = ToolRegistry()
    live_invocations: list[int] = []

    async def maybe_dangerous(n: int) -> str:
        live_invocations.append(n)
        return f"ran {n}"

    registry.register_function(maybe_dangerous, name="op")

    async def one(n: int, shadow: bool):
        with run_context(audit=True, shadow=shadow) as ctx:
            await registry.execute("op", n=n)
            audit = ctx.recorder.finalize()
            return n, shadow, audit

    # Build mixed task list — odd indices are shadow, even are live
    tasks = [one(i, shadow=(i % 2 == 1)) for i in range(100)]
    outcomes = await asyncio.gather(*tasks)

    expected_live = sorted(i for i in range(100) if i % 2 == 0)
    assert sorted(live_invocations) == expected_live, (
        f"Live invocations diverged.  Expected {len(expected_live)} ({expected_live[:5]}…), "
        f"got {len(live_invocations)}"
    )

    for n, shadow, audit in outcomes:
        assert audit.tool_call_count == 1
        call = audit.tool_calls[0]
        if shadow:
            assert call.status == "shadow"
        else:
            assert call.status == "ok"
        assert call.args == {"n": n}


@pytest.mark.asyncio
async def test_contextvars_reset_after_parallel_runs():
    """After many parallel runs, module-level state must be clean."""
    registry = ToolRegistry()

    async def noop() -> str:
        return "ok"

    registry.register_function(noop, name="noop")

    async def one():
        with run_context(audit=True, shadow=True):
            await registry.execute("noop")

    await asyncio.gather(*[one() for _ in range(50)])

    # The parent context (where this test runs) must see no leakage
    assert get_recorder() is None
    assert is_shadow() is False


@pytest.mark.asyncio
async def test_concurrent_executes_on_same_tool_record_distinct_args():
    """
    50 concurrent executes on the same tool with different args.
    Every recorder must end up with its own correct args (no cross-talk).
    """
    registry = ToolRegistry()

    async def slow_add(a: int, b: int) -> int:
        # Insert an await so the scheduler can interleave aggressively
        await asyncio.sleep(0.001)
        return a + b

    registry.register_function(slow_add, name="slow_add")

    async def one(pair: tuple[int, int]):
        a, b = pair
        with run_context(audit=True, shadow=False) as ctx:
            await registry.execute("slow_add", a=a, b=b)
            return pair, ctx.recorder.finalize()

    pairs = [(i, i * 2) for i in range(50)]
    outcomes = await asyncio.gather(*[one(p) for p in pairs])

    for pair, audit in outcomes:
        assert audit.tool_call_count == 1
        call = audit.tool_calls[0]
        assert (call.args["a"], call.args["b"]) == pair, (
            f"Audit cross-talk: expected args {pair}, got "
            f"({call.args['a']}, {call.args['b']})"
        )


@pytest.mark.asyncio
async def test_fingerprint_stable_across_parallel_identical_runs():
    """
    Same prompt + same tool registry + same args = same fingerprint,
    even across many parallel executions.  This proves determinism
    is preserved under load.
    """
    registry = ToolRegistry()

    async def constant() -> str:
        return "always-the-same"

    registry.register_function(constant, name="constant")

    async def one():
        with run_context(audit=True, shadow=False) as ctx:
            await registry.execute("constant")
            return ctx.recorder.finalize().fingerprint()

    fingerprints = await asyncio.gather(*[one() for _ in range(50)])
    unique = set(fingerprints)
    assert len(unique) == 1, (
        f"Expected one fingerprint across 50 identical runs, got {len(unique)}: {unique}"
    )
