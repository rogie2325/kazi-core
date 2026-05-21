"""
kazi.testing — verification harnesses that turn unit-test "passes"
into "this system behaves consistently."

Three primary tools:

  replay.py — record an audit DAG to JSON and replay it later to detect
              behavioural drift between commits.

  eval.py   — declarative scenarios: a prompt, expected tool-call shape, and
              an audit fingerprint.  Run with pytest -m eval for go/no-go
              decisions on a generated integration.

  invariants.py — property-based assertions that hold across all inputs
              (tenant isolation, audit cross-talk, registry contracts).

These are intended for CI gating, validator workflows, and rolling-out
AI-generated integrations to client codebases.
"""
from __future__ import annotations

from kazi.testing.replay import (
    AuditFixture,
    FingerprintMismatch,
    assert_fingerprint,
    load_fixture,
    record_fixture,
    replay,
)

__all__ = [
    "AuditFixture",
    "FingerprintMismatch",
    "assert_fingerprint",
    "load_fixture",
    "record_fixture",
    "replay",
]
