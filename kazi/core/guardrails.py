"""
Output guardrails: PII detection, blocklist filtering, length enforcement.

Applied to LLM responses before they are returned to the caller.

Three violation modes
---------------------
warn    Log the violation and return the original text unchanged (default).
block   Raise GuardrailViolationError — the caller must handle it.
redact  Replace matched text with [REDACTED:<LABEL>] and return cleaned text.

Usage::

    from kazi.core.guardrails import GuardrailConfig, check_output

    config = GuardrailConfig(
        pii_detection=True,
        blocklist_patterns=[r"confidential project X"],
        on_violation="redact",
    )
    result = check_output(llm_reply, config)
    print(result.text)          # cleaned text
    print(result.violations)    # list of violation labels
    print(result.clean)         # True when no violations found
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)

# ── Built-in PII patterns ──────────────────────────────────────────────────────

_PII_PATTERNS: list[tuple[str, str]] = [
    ("email",       r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    ("phone_us",    r"\b(?:\+1[\s.\-]?)?\(?[2-9]\d{2}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}\b"),
    ("ssn",         r"\b\d{3}-\d{2}-\d{4}\b"),
    ("credit_card", (
        r"\b(?:"
        r"4\d{12}(?:\d{3})?"              # Visa
        r"|5[1-5]\d{14}"                  # Mastercard
        r"|3[47]\d{13}"                   # Amex
        r"|6(?:011|5\d{2})\d{12}"         # Discover
        r"|3(?:0[0-5]|[68]\d)\d{11}"      # Diners
        r")\b"
    )),
    ("ipv4", r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"),
]

_REDACT_TMPL = "[REDACTED:{label}]"


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class GuardrailConfig:
    """
    Configuration for post-generation output validation.

    pii_detection       Scan for emails, phone numbers, SSNs, credit cards, IPv4s.
    blocklist_patterns  Regex patterns; any match triggers on_violation.
    max_output_chars    Hard cap on response length (0 = unlimited).
    on_violation        What to do when a rule fires: warn / block / redact.
    custom_pii_patterns Additional (label, regex) pairs checked alongside built-ins.
                        Example: [("internal_id", r"INT-\\d{6}")]
    """
    pii_detection: bool = False
    blocklist_patterns: list[str] = field(default_factory=list)
    max_output_chars: int = 0
    on_violation: Literal["warn", "block", "redact"] = "warn"
    custom_pii_patterns: list[tuple[str, str]] = field(default_factory=list)


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class GuardrailResult:
    """
    Result of a guardrail check.

    text        The (possibly redacted) output text.
    violations  List of violation labels found.
    clean       True when no violations were detected.
    """
    text: str
    violations: list[str]

    @property
    def clean(self) -> bool:
        return not self.violations


# ── Core check function ───────────────────────────────────────────────────────

def check_output(text: str, config: GuardrailConfig) -> GuardrailResult:
    """
    Apply guardrails to an LLM output string.

    Returns GuardrailResult.  Raises GuardrailViolationError when
    on_violation="block" and a rule fires.
    """
    from kazi.core.exceptions import GuardrailViolationError

    violations: list[str] = []
    working = text

    # 1. Max length
    if config.max_output_chars > 0 and len(working) > config.max_output_chars:
        label = f"output_too_long:{len(working)}>{config.max_output_chars}"
        violations.append(label)
        if config.on_violation == "block":
            raise GuardrailViolationError(
                f"Output too long: {len(working)} chars (max {config.max_output_chars})"
            )
        elif config.on_violation == "redact":
            working = working[: config.max_output_chars] + "… [truncated by guardrail]"
        else:
            logger.warning("Guardrail: %s", label)

    # 2. PII detection
    if config.pii_detection:
        patterns = _PII_PATTERNS + list(config.custom_pii_patterns)
        for label, pattern in patterns:
            matches = re.findall(pattern, working)
            if not matches:
                continue
            violation = f"pii:{label}:{len(matches)}_match(es)"
            violations.append(violation)
            if config.on_violation == "block":
                raise GuardrailViolationError(
                    f"PII detected in output ({label}): {len(matches)} match(es)"
                )
            elif config.on_violation == "redact":
                working = re.sub(
                    pattern,
                    _REDACT_TMPL.format(label=label.upper()),
                    working,
                )
            else:
                logger.warning("Guardrail: PII detected (%s) — %d match(es)", label, len(matches))

    # 3. Blocklist patterns
    for pattern in config.blocklist_patterns:
        try:
            m = re.search(pattern, working, re.IGNORECASE | re.DOTALL)
        except re.error as exc:
            logger.error("Guardrail: invalid regex '%s': %s", pattern[:60], exc)
            continue
        if not m:
            continue
        short = pattern[:40]
        violations.append(f"blocklist:{short}")
        if config.on_violation == "block":
            raise GuardrailViolationError(f"Blocklist pattern matched: {short}")
        elif config.on_violation == "redact":
            working = re.sub(pattern, "[REDACTED]", working, flags=re.IGNORECASE | re.DOTALL)
        else:
            logger.warning("Guardrail: blocklist pattern matched: %s", short)

    return GuardrailResult(text=working, violations=violations)
