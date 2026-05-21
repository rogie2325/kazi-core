"""
Tests for kazi/testing/replay.py and kazi/testing/eval.py.

Validates the record/replay harness and scenario evaluation harness — the
tooling that turns "tests pass" into "the integration behaves the same way
across commits, machines, and replays."
"""
from __future__ import annotations

import json

import pytest

from kazi.core.audit import RunAudit, ToolCallRecord
from kazi.testing.eval import EvalReport, Scenario, ScenarioResult
from kazi.testing.replay import (
    AuditFixture,
    FingerprintMismatch,
    assert_fingerprint,
    load_fixture,
    record_fixture,
)


def _make_audit(*calls: tuple[str, dict]) -> RunAudit:
    return RunAudit(
        tool_calls=[
            ToolCallRecord(name=name, args=dict(args), result="ok", duration_ms=1.0, status="ok")
            for name, args in calls
        ],
        shadow=False,
        thread_id="t",
    )


# ── AuditFixture round-trip ───────────────────────────────────────────────────

def test_fixture_to_dict_round_trip():
    audit = _make_audit(("get_invoice", {"id": 42}))
    fixture = AuditFixture(
        prompt="hello",
        fingerprint=audit.fingerprint(),
        tool_call_names=["get_invoice"],
        audit=audit,
        metadata={"model": "test"},
    )
    restored = AuditFixture.from_dict(fixture.to_dict())
    assert restored.prompt == "hello"
    assert restored.fingerprint == audit.fingerprint()
    assert restored.tool_call_names == ["get_invoice"]
    assert restored.audit.fingerprint() == audit.fingerprint()
    assert restored.metadata["model"] == "test"


def test_record_and_load_fixture(tmp_path):
    audit = _make_audit(("a", {}), ("b", {"x": 1}))
    path = str(tmp_path / "fix.json")
    record_fixture(path, "do a then b", audit, metadata={"git_sha": "abcdef"})

    on_disk = json.loads(open(path).read())
    assert on_disk["prompt"] == "do a then b"
    assert on_disk["fingerprint"] == audit.fingerprint()
    assert on_disk["tool_call_names"] == ["a", "b"]
    assert on_disk["metadata"]["git_sha"] == "abcdef"

    restored = load_fixture(path)
    assert restored.fingerprint == audit.fingerprint()


def test_record_fixture_creates_missing_parent_directory(tmp_path):
    nested = tmp_path / "deep" / "subdir" / "fix.json"
    record_fixture(str(nested), "p", _make_audit())
    assert nested.exists()


# ── assert_fingerprint ────────────────────────────────────────────────────────

def test_assert_fingerprint_passes_on_match():
    a = _make_audit(("x", {}))
    assert_fingerprint(a, a.fingerprint())  # raw string overload


def test_assert_fingerprint_passes_with_fixture():
    a = _make_audit(("x", {}))
    fixture = AuditFixture(
        prompt="", fingerprint=a.fingerprint(),
        tool_call_names=["x"], audit=a, metadata={},
    )
    assert_fingerprint(a, fixture)


def test_assert_fingerprint_raises_on_divergence():
    a = _make_audit(("x", {}))
    b = _make_audit(("y", {}))  # different tool name → different fingerprint
    with pytest.raises(FingerprintMismatch) as excinfo:
        assert_fingerprint(a, b.fingerprint())
    msg = str(excinfo.value)
    assert "fingerprint diverged" in msg


def test_assert_fingerprint_diff_is_human_readable():
    """The mismatch message should show tool-name diff for fast debugging."""
    expected = _make_audit(("get_invoice", {}), ("send_email", {}))
    actual = _make_audit(("get_invoice", {}), ("delete_invoice", {}))
    fixture = AuditFixture(
        prompt="", fingerprint=expected.fingerprint(),
        tool_call_names=["get_invoice", "send_email"],
        audit=expected, metadata={},
    )
    with pytest.raises(FingerprintMismatch) as excinfo:
        assert_fingerprint(actual, fixture)
    msg = str(excinfo.value)
    # The unified diff should mention at least one of the diverging tool names
    assert "send_email" in msg or "delete_invoice" in msg


# ── Scenario / EvalReport ─────────────────────────────────────────────────────

def test_scenario_result_str():
    r = ScenarioResult(name="t1", passed=True, audit=_make_audit(("a", {})))
    s = str(r)
    assert "[PASS]" in s
    assert "t1" in s
    assert "1 tool" in s


def test_eval_report_summary_and_failures():
    r1 = ScenarioResult(name="ok", passed=True, audit=None)
    r2 = ScenarioResult(name="bad", passed=False, audit=None, failures=["missing tool x"])
    report = EvalReport(results=[r1, r2])
    assert report.passed_count == 1
    assert report.total_count == 2
    assert report.all_passed is False
    assert "1/2" in report.summary()
    txt = report.failures_text()
    assert "bad" in txt
    assert "missing tool x" in txt


def test_eval_report_all_passed():
    report = EvalReport(results=[
        ScenarioResult(name=f"s{i}", passed=True, audit=None) for i in range(3)
    ])
    assert report.all_passed is True
    assert report.failures_text() == "(no failures)"


# ── Scenario runner via a fake Kazi ──────────────────────────────────────────

class _FakeAuditResult:
    def __init__(self, audit):
        self.audit = audit


class _FakeKazi:
    """Stub that returns canned audits — lets us test the eval runner offline."""

    def __init__(self, audits_by_prompt: dict[str, RunAudit]):
        self._map = audits_by_prompt

    async def run(self, prompt, *, thread_id, audit, shadow):
        if prompt not in self._map:
            raise KeyError(f"no canned audit for {prompt!r}")
        return _FakeAuditResult(self._map[prompt])


@pytest.mark.asyncio
async def test_scenario_runner_pass_when_required_tools_called():
    from kazi.testing.eval import run_scenarios
    audit = _make_audit(("get_invoice", {"id": 1}), ("send_email", {}))
    kazi = _FakeKazi({"do it": audit})
    report = await run_scenarios(kazi, [
        Scenario(name="needs_invoice", prompt="do it",
                 expect_tools=["get_invoice"]),
    ])
    assert report.all_passed
    assert report.passed_count == 1


@pytest.mark.asyncio
async def test_scenario_runner_fail_when_required_tool_missing():
    from kazi.testing.eval import run_scenarios
    audit = _make_audit(("send_email", {}))
    kazi = _FakeKazi({"x": audit})
    report = await run_scenarios(kazi, [
        Scenario(name="needs_invoice", prompt="x", expect_tools=["get_invoice"]),
    ])
    assert not report.all_passed
    assert "get_invoice" in report.failures_text()


@pytest.mark.asyncio
async def test_scenario_runner_fail_when_forbidden_tool_called():
    from kazi.testing.eval import run_scenarios
    audit = _make_audit(("delete_invoice", {"id": 1}))
    kazi = _FakeKazi({"clean up": audit})
    report = await run_scenarios(kazi, [
        Scenario(name="never_delete", prompt="clean up",
                 forbid_tools=["delete_invoice"]),
    ])
    assert not report.all_passed
    assert "delete_invoice" in report.failures_text()


@pytest.mark.asyncio
async def test_scenario_runner_exact_sequence_match():
    from kazi.testing.eval import run_scenarios
    audit = _make_audit(("a", {}), ("b", {}), ("c", {}))
    kazi = _FakeKazi({"seq": audit})
    report = await run_scenarios(kazi, [
        Scenario(name="exact_ok", prompt="seq", expect_exact=["a", "b", "c"]),
        Scenario(name="exact_bad", prompt="seq", expect_exact=["a", "c", "b"]),
    ])
    assert report.passed_count == 1


@pytest.mark.asyncio
async def test_scenario_runner_min_max_call_count():
    from kazi.testing.eval import run_scenarios
    audit = _make_audit(("x", {}), ("x", {}))
    kazi = _FakeKazi({"p": audit})
    report = await run_scenarios(kazi, [
        Scenario(name="too_few", prompt="p", expect_min_calls=3),
        Scenario(name="too_many", prompt="p", expect_max_calls=1),
        Scenario(name="just_right", prompt="p",
                 expect_min_calls=2, expect_max_calls=2),
    ])
    assert report.passed_count == 1


@pytest.mark.asyncio
async def test_scenario_runner_fingerprint_match():
    from kazi.testing.eval import run_scenarios
    audit = _make_audit(("only", {"v": 7}))
    kazi = _FakeKazi({"do": audit})
    expected_fp = audit.fingerprint()
    report = await run_scenarios(kazi, [
        Scenario(name="fp_match", prompt="do", expect_fingerprint=expected_fp),
        Scenario(name="fp_wrong", prompt="do", expect_fingerprint="0" * 64),
    ])
    assert report.passed_count == 1
    assert "fingerprint mismatch" in report.failures_text()


@pytest.mark.asyncio
async def test_scenario_runner_catches_run_exception():
    from kazi.testing.eval import run_scenarios

    class _BrokenKazi:
        async def run(self, *args, **kwargs):
            raise RuntimeError("brain offline")

    report = await run_scenarios(_BrokenKazi(), [
        Scenario(name="catches", prompt="x"),
    ])
    assert not report.all_passed
    assert "RuntimeError" in report.failures_text()


@pytest.mark.asyncio
async def test_scenario_runner_fails_when_audit_is_none():
    """When kazi.run() returns something with no audit attribute, fail gracefully."""
    from kazi.testing.eval import run_scenarios

    class _NoAuditKazi:
        async def run(self, *args, **kwargs):
            return "just a string, no audit"  # no .audit attribute

    report = await run_scenarios(_NoAuditKazi(), [
        Scenario(name="no_audit", prompt="p"),
    ])
    assert not report.all_passed
    assert "no audit" in report.failures_text()


@pytest.mark.asyncio
async def test_scenario_runner_fail_when_expect_no_errors_violated():
    """expect_no_errors=True fails when tool calls have status='error'."""
    from kazi.testing.eval import run_scenarios

    audit = RunAudit(
        tool_calls=[
            ToolCallRecord("get_data", {}, None, 1.0, "error", error="DB timeout"),
        ],
        shadow=False,
        thread_id="t",
    )
    kazi = _FakeKazi({"q": audit})
    report = await run_scenarios(kazi, [
        Scenario(name="no_errors_check", prompt="q", expect_no_errors=True),
    ])
    assert not report.all_passed
    assert "errored" in report.failures_text()


# ── replay_fixture ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_replay_fixture_succeeds_when_fingerprint_matches(tmp_path):
    """replay() completes without raising when the audit fingerprint matches."""
    from kazi.testing.replay import record_fixture, replay

    audit = _make_audit(("fetch_data", {"id": 1}))
    fixture_path = str(tmp_path / "replay_test.json")
    record_fixture(fixture_path, "fetch something", audit)

    class _ReplayKazi:
        async def run(self, prompt, *, thread_id, audit, shadow):
            return _FakeAuditResult(_make_audit(("fetch_data", {"id": 1})))

    result_audit = await replay(_ReplayKazi(), fixture_path)
    assert result_audit is not None
    assert result_audit.fingerprint() == audit.fingerprint()


@pytest.mark.asyncio
async def test_replay_fixture_raises_on_fingerprint_divergence(tmp_path):
    """replay() raises FingerprintMismatch when audit diverges from recording."""
    from kazi.testing.replay import FingerprintMismatch, record_fixture, replay

    original_audit = _make_audit(("step_a", {}), ("step_b", {}))
    fixture_path = str(tmp_path / "replay_diverge.json")
    record_fixture(fixture_path, "do steps", original_audit)

    class _DivergedKazi:
        async def run(self, prompt, *, thread_id, audit, shadow):
            return _FakeAuditResult(_make_audit(("step_a", {}), ("step_c", {})))

    with pytest.raises(FingerprintMismatch):
        await replay(_DivergedKazi(), fixture_path)
