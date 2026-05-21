"""
Run audit + shadow mode for kazi.

Two features for AI-engineer / validator workflows:

1. ``RunAudit`` — a structured record of everything an agent did during a run:
   every tool call (name, args, result, duration, status), prompts, costs,
   and timings.  Returned alongside the reply when ``audit=True``.  This is the
   primary artefact a senior engineer reviews to validate generated code.

2. **Shadow mode** — when ``shadow=True``, tool calls are intercepted and
   replaced with a stub.  The agent still reasons over the (mock) tool result
   and produces a reply, but no side effects hit the real system.  Use this
   to dry-run an integration against a client's existing codebase before
   enabling live execution.

Both features hook in at ``ToolRegistry.execute()`` via context variables, so
they cover every tool source — native, MCP, A2A, RAG — transparently.

Usage::

    result = await kazi.run(
        "Process invoice #4521",
        audit=True,           # collect a RunAudit alongside the reply
        shadow=True,          # don't actually call tools — return mocks
    )
    print(result.reply)
    for call in result.audit.tool_calls:
        print(call)           # → [SHADOW ok] get_invoice(id=4521) -> ... (1.2 ms)
"""
from __future__ import annotations

import contextvars
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

# ── Per-run context variables ─────────────────────────────────────────────────
# Tool execution checks these on every call to decide whether to record / stub.

_current_recorder: contextvars.ContextVar = contextvars.ContextVar(
    "kazi_audit_recorder", default=None
)
_shadow_mode: contextvars.ContextVar = contextvars.ContextVar(
    "kazi_shadow_mode", default=False
)


def get_recorder() -> AuditRecorder | None:
    """Return the AuditRecorder bound to the current run, or None."""
    return _current_recorder.get()


def is_shadow() -> bool:
    """True when the current run is executing in shadow mode."""
    return _shadow_mode.get()


# ── Records ───────────────────────────────────────────────────────────────────


@dataclass
class ToolCallRecord:
    """One tool invocation captured during a run."""
    name: str
    args: dict
    result: str | None
    duration_ms: float
    status: str  # "ok" | "error" | "shadow"
    error: str | None = None

    def __str__(self) -> str:
        prefix = f"[{self.status.upper()}]"
        result_preview = (
            (self.result[:80] + "…") if self.result and len(self.result) > 80
            else (self.result or "")
        )
        return (
            f"{prefix} {self.name}({self._args_str()}) -> {result_preview} "
            f"({self.duration_ms:.1f} ms)"
            + (f" ERROR: {self.error}" if self.error else "")
        )

    def _args_str(self) -> str:
        return ", ".join(f"{k}={v!r}" for k, v in self.args.items())


@dataclass
class RunAudit:
    """
    Structured record of everything an agent did during one kazi.run() call.

    Use this for compliance review, validator dashboards, replay debugging, and
    cost attribution per tool.  Serialise with ``dataclasses.asdict()``.
    """
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    started_at: float = 0.0      # epoch seconds
    duration_ms: float = 0.0     # total wall-time
    shadow: bool = False         # was this a dry-run?
    thread_id: str = ""
    tenant_id: str = ""
    user_id: str = ""
    error: str | None = None     # set when the run itself raised

    @property
    def tool_call_count(self) -> int:
        return len(self.tool_calls)

    @property
    def error_count(self) -> int:
        return sum(1 for c in self.tool_calls if c.status == "error")

    @property
    def total_tool_ms(self) -> float:
        return sum(c.duration_ms for c in self.tool_calls)

    def summary(self) -> str:
        """One-line human summary, useful for log output."""
        mode = "SHADOW " if self.shadow else ""
        return (
            f"{mode}{self.tool_call_count} tool call(s), "
            f"{self.error_count} error(s), "
            f"{self.total_tool_ms:.0f} ms in tools, "
            f"{self.duration_ms:.0f} ms total"
        )

    # ── Fingerprint (deterministic) ───────────────────────────────────────────

    def fingerprint(self) -> str:
        """
        Stable SHA-256 hash of the agent's decisions during this run.

        The fingerprint covers ONLY the deterministic shape of the run:
        tool names, arg keys/values, call order, and final status.  Timing,
        wall-clock timestamps, and freeform result strings are excluded so
        the same logical run produces the same fingerprint across machines,
        versions, and replays.

        Use this for:
          - Regression detection: assert today's fingerprint matches yesterday's
            for a fixed prompt + seed + tool registry.
          - Deduplication: identical runs collapse to one entry.
          - Validator review: a senior engineer compares fingerprints across
            commits to catch silent behavioural drift.

        Combine with ``seed=`` on LLMConfig and ``shadow=True`` to make the
        whole run replayable.
        """
        canon = {
            "shadow": self.shadow,
            "tool_calls": [
                {
                    "name": c.name,
                    "args": _canonical(c.args),
                    "status": c.status,
                }
                for c in self.tool_calls
            ],
            "error_kind": _classify_error(self.error),
        }
        # sort_keys ensures dict-key ordering doesn't leak into the hash
        payload = json.dumps(canon, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def short_fingerprint(self) -> str:
        """First 12 chars of the SHA-256 fingerprint — readable in logs."""
        return self.fingerprint()[:12]

    # ── Serialization (record / replay) ──────────────────────────────────────

    def to_dict(self) -> dict:
        """Return a JSON-safe dict representation of the audit."""
        d = asdict(self)
        # asdict drops nothing of interest — tool_calls are dataclasses too
        return d

    def to_json(self, *, indent: int = 2) -> str:
        """Serialise the audit to a JSON string for on-disk storage."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: dict) -> RunAudit:
        """Reconstruct a RunAudit from a dict produced by ``to_dict()``."""
        raw_calls = data.get("tool_calls") or []
        calls = [
            ToolCallRecord(
                name=c["name"],
                args=dict(c.get("args") or {}),
                result=c.get("result"),
                duration_ms=float(c.get("duration_ms", 0)),
                status=c.get("status", "ok"),
                error=c.get("error"),
            )
            for c in raw_calls
        ]
        return cls(
            tool_calls=calls,
            started_at=float(data.get("started_at", 0.0)),
            duration_ms=float(data.get("duration_ms", 0.0)),
            shadow=bool(data.get("shadow", False)),
            thread_id=data.get("thread_id", ""),
            tenant_id=data.get("tenant_id", ""),
            user_id=data.get("user_id", ""),
            error=data.get("error"),
        )

    @classmethod
    def from_json(cls, payload: str) -> RunAudit:
        return cls.from_dict(json.loads(payload))


# ── Fingerprint helpers ───────────────────────────────────────────────────────


def _canonical(obj: Any) -> Any:
    """
    Recursively normalise a value so the same logical input produces the same
    hash regardless of dict ordering or insignificant type differences.
    """
    if isinstance(obj, dict):
        return {k: _canonical(obj[k]) for k in sorted(obj.keys(), key=str)}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, (str, int, bool, type(None))):
        return obj
    if isinstance(obj, float):
        # Round floats to 6 decimals so trivial precision noise doesn't change the hash
        return round(obj, 6)
    return str(obj)


def _classify_error(err: str | None) -> str:
    """
    Map a freeform error message to a coarse category for fingerprint stability.

    "ConnectionError: ..." → "ConnectionError"
    "TimeoutError(...)" → "TimeoutError"
    None → "none"
    """
    if not err:
        return "none"
    # Use the first token before ':' or '(' as the error class
    for sep in (":", "("):
        if sep in err:
            return err.split(sep, 1)[0].strip()
    return err.strip()[:50]


@dataclass
class RunAuditResult:
    """
    Returned by ``kazi.run()`` when ``audit=True``.

    Carries the reply, the optional cost record (if ``track_cost`` was also
    set), and the full ``RunAudit`` structure.
    """
    reply: str
    audit: RunAudit
    cost: Any | None = None  # RunCost — kept loose to avoid circular import


# ── Recorder ──────────────────────────────────────────────────────────────────


class AuditRecorder:
    """
    Collects ToolCallRecords during a single kazi.run() call.

    Created by the orchestrator when ``audit=True`` and bound via the
    ``_current_recorder`` contextvar so any tool execution path picks it up
    without needing to thread it through every call site.
    """

    def __init__(
        self,
        *,
        thread_id: str = "",
        tenant_id: str = "",
        user_id: str = "",
        shadow: bool = False,
    ) -> None:
        self._start = time.monotonic()
        self.audit = RunAudit(
            started_at=time.time(),
            shadow=shadow,
            thread_id=thread_id,
            tenant_id=tenant_id,
            user_id=user_id,
        )

    def record_tool_call(
        self,
        *,
        name: str,
        args: dict,
        result: str | None,
        duration_ms: float,
        status: str,
        error: str | None = None,
    ) -> None:
        self.audit.tool_calls.append(ToolCallRecord(
            name=name,
            args=dict(args),  # defensive copy
            result=result,
            duration_ms=duration_ms,
            status=status,
            error=error,
        ))

    def finalize(self, *, error: str | None = None) -> RunAudit:
        self.audit.duration_ms = (time.monotonic() - self._start) * 1000
        if error:
            self.audit.error = error
        return self.audit


# ── Context manager for one run ───────────────────────────────────────────────


class run_context:
    """
    Context manager that binds an AuditRecorder and shadow flag for one run.

    Used internally by Kazi.run(); rarely needed in user code.

    Usage::

        with run_context(audit=True, shadow=False, thread_id="t1") as ctx:
            reply = await brain.run(...)
            audit = ctx.recorder.finalize()
    """

    def __init__(
        self,
        *,
        audit: bool,
        shadow: bool,
        thread_id: str = "",
        tenant_id: str = "",
        user_id: str = "",
    ) -> None:
        self._audit = audit
        self._shadow = shadow
        self.recorder: AuditRecorder | None = None
        self._thread_id = thread_id
        self._tenant_id = tenant_id
        self._user_id = user_id
        self._tokens: list[Any] = []

    def __enter__(self) -> run_context:
        if self._audit:
            self.recorder = AuditRecorder(
                thread_id=self._thread_id,
                tenant_id=self._tenant_id,
                user_id=self._user_id,
                shadow=self._shadow,
            )
            self._tokens.append(_current_recorder.set(self.recorder))
        if self._shadow:
            self._tokens.append(_shadow_mode.set(True))
        return self

    def __exit__(self, *_exc) -> None:
        # Reset contextvars in reverse order
        for token in reversed(self._tokens):
            try:
                if token.var is _current_recorder:
                    _current_recorder.reset(token)
                elif token.var is _shadow_mode:
                    _shadow_mode.reset(token)
            except (ValueError, LookupError, RuntimeError):
                pass
