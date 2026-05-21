"""
Record / replay harness for kazi audit traces.

The proof loop
==============

1. **Record**: run a prompt in shadow mode against your tool registry::

        result = await kazi.run(prompt, audit=True, shadow=True)
        record_fixture("fixtures/invoice_query.json", prompt, result.audit)

2. **Replay**: in CI or pre-commit, replay the same prompt and assert the
   audit fingerprint matches the recorded one::

        await replay(kazi, "fixtures/invoice_query.json")

If the agent's tool-call DAG changes — different tools, different arg shape,
different order — the fingerprint changes and the replay fails.  This turns
silent behavioural drift into a hard CI signal.

Combine with ``DeterministicConfig`` and ``shadow=True`` so the LLM output is
stable across runs.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from kazi.core.audit import RunAudit

if TYPE_CHECKING:
    from kazi.core.orchestrator import Kazi

logger = logging.getLogger(__name__)


class FingerprintMismatch(AssertionError):
    """Raised when a replayed audit fingerprint diverges from the fixture."""


@dataclass
class AuditFixture:
    """
    A recorded run, suitable for storing on disk and replaying later.

    Fields:
      prompt          The original user message.
      fingerprint     SHA-256 of the recorded audit's deterministic shape.
      tool_call_names Ordered list of tool names in the recorded run —
                      kept separately so a diff is human-readable.
      audit           The full RunAudit (for inspection / debugging).
      metadata        Free-form bag for the recording context (model, seed,
                      git SHA, timestamp, etc.).
    """
    prompt: str
    fingerprint: str
    tool_call_names: list[str]
    audit: RunAudit
    metadata: dict[str, Any]

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "fingerprint": self.fingerprint,
            "tool_call_names": list(self.tool_call_names),
            "audit": self.audit.to_dict(),
            "metadata": self.metadata,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: dict) -> AuditFixture:
        return cls(
            prompt=data["prompt"],
            fingerprint=data["fingerprint"],
            tool_call_names=list(data.get("tool_call_names") or []),
            audit=RunAudit.from_dict(data.get("audit") or {}),
            metadata=dict(data.get("metadata") or {}),
        )


# ── Record / load ─────────────────────────────────────────────────────────────


def record_fixture(
    path: str,
    prompt: str,
    audit: RunAudit,
    *,
    metadata: dict | None = None,
) -> AuditFixture:
    """
    Persist a fixture to disk.  Creates parent directories as needed.

    Pass any free-form metadata (model name, seed, git SHA, branch) so
    the validator can correlate fixture drift with code changes.
    """
    fixture = AuditFixture(
        prompt=prompt,
        fingerprint=audit.fingerprint(),
        tool_call_names=[c.name for c in audit.tool_calls],
        audit=audit,
        metadata=dict(metadata or {}),
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(fixture.to_json())
    logger.info(
        "Recorded fixture path=%s fingerprint=%s calls=%d",
        path, fixture.fingerprint[:12], len(fixture.tool_call_names),
    )
    return fixture


def load_fixture(path: str) -> AuditFixture:
    """Load a previously recorded fixture."""
    with open(path) as f:
        return AuditFixture.from_dict(json.load(f))


# ── Assertion helpers ─────────────────────────────────────────────────────────


def assert_fingerprint(audit: RunAudit, expected: str | AuditFixture) -> None:
    """
    Assert ``audit.fingerprint()`` matches the expected hash.

    The mismatch message includes a human-readable diff of tool-call names
    so a senior engineer can read the failure at a glance.
    """
    expected_fp = expected.fingerprint if isinstance(expected, AuditFixture) else expected
    actual_fp = audit.fingerprint()
    if actual_fp == expected_fp:
        return

    expected_names = (
        expected.tool_call_names
        if isinstance(expected, AuditFixture)
        else ["<no fixture>"]
    )
    actual_names = [c.name for c in audit.tool_calls]
    diff = _human_diff(expected_names, actual_names)
    raise FingerprintMismatch(
        f"Audit fingerprint diverged:\n"
        f"  expected: {expected_fp[:16]}…  ({len(expected_names)} calls)\n"
        f"  actual:   {actual_fp[:16]}…  ({len(actual_names)} calls)\n"
        f"  diff:\n{diff}"
    )


def _human_diff(expected: list[str], actual: list[str]) -> str:
    """Render an ordered side-by-side diff of tool-call names."""
    import difflib
    diff_lines = difflib.unified_diff(
        expected, actual, fromfile="expected", tofile="actual", lineterm=""
    )
    return "\n".join(f"    {line}" for line in diff_lines) or "    (no name diff — argument shape must differ)"


# ── Full replay ───────────────────────────────────────────────────────────────


async def replay(
    kazi: Kazi,
    fixture_path: str,
    *,
    thread_id: str | None = None,
    shadow: bool = True,
) -> RunAudit:
    """
    Re-run the prompt from ``fixture_path`` against ``kazi`` and assert the
    fingerprint matches.

    Returns the replayed RunAudit so the caller can inspect details on failure.

    By default the replay runs in shadow mode — no side effects — which is what
    you want in CI.  Set ``shadow=False`` only when intentionally exercising a
    live integration.

    Raises ``FingerprintMismatch`` when the agent's decisions diverged from
    the recording.
    """
    fixture = load_fixture(fixture_path)
    tid = thread_id or f"replay:{fixture.fingerprint[:12]}"

    result = await kazi.run(
        fixture.prompt,
        thread_id=tid,
        audit=True,
        shadow=shadow,
    )
    # kazi.run() with audit=True returns RunAuditResult
    audit: RunAudit = result.audit if hasattr(result, "audit") else result  # type: ignore[assignment]
    assert_fingerprint(audit, fixture)
    return audit
