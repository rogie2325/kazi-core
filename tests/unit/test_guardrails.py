"""
Unit tests for kazi/core/guardrails.py

Covers check_output() across all three violation modes (warn / block / redact)
for every rule type: max_output_chars, PII patterns, and blocklist patterns.
Pure Python — no external dependencies or API keys required.
"""
import pytest

from kazi.core.exceptions import GuardrailViolationError
from kazi.core.guardrails import GuardrailConfig, GuardrailResult, check_output

# ── GuardrailResult ───────────────────────────────────────────────────────────

def test_result_clean_when_no_violations():
    r = GuardrailResult(text="ok", violations=[])
    assert r.clean is True


def test_result_not_clean_when_violations_present():
    r = GuardrailResult(text="bad", violations=["pii:email:1_match(es)"])
    assert r.clean is False


# ── Clean text ────────────────────────────────────────────────────────────────

def test_clean_text_passes_through_unchanged():
    config = GuardrailConfig()
    result = check_output("Hello, world!", config)
    assert result.clean
    assert result.violations == []
    assert result.text == "Hello, world!"


def test_empty_string_is_clean():
    config = GuardrailConfig(pii_detection=True, blocklist_patterns=[r"secret"])
    result = check_output("", config)
    assert result.clean


# ── max_output_chars — warn ───────────────────────────────────────────────────

def test_max_chars_warn_records_violation_but_keeps_text():
    config = GuardrailConfig(max_output_chars=5, on_violation="warn")
    result = check_output("hello world", config)
    assert not result.clean
    assert any("output_too_long" in v for v in result.violations)
    assert result.text == "hello world"


def test_max_chars_exactly_at_limit_is_clean():
    config = GuardrailConfig(max_output_chars=5, on_violation="block")
    result = check_output("hello", config)
    assert result.clean


def test_max_chars_zero_means_unlimited():
    config = GuardrailConfig(max_output_chars=0)
    result = check_output("x" * 100_000, config)
    assert result.clean


# ── max_output_chars — block ──────────────────────────────────────────────────

def test_max_chars_block_raises_guardrail_error():
    config = GuardrailConfig(max_output_chars=5, on_violation="block")
    with pytest.raises(GuardrailViolationError, match="Output too long"):
        check_output("hello world", config)


# ── max_output_chars — redact ─────────────────────────────────────────────────

def test_max_chars_redact_truncates_text():
    config = GuardrailConfig(max_output_chars=5, on_violation="redact")
    result = check_output("hello world", config)
    assert not result.clean
    assert result.text.startswith("hello")
    assert "truncated" in result.text
    assert "world" not in result.text


# ── PII — email ───────────────────────────────────────────────────────────────

def test_pii_email_warn_records_violation_keeps_text():
    config = GuardrailConfig(pii_detection=True, on_violation="warn")
    result = check_output("Contact us at user@example.com for help.", config)
    assert not result.clean
    assert any("email" in v for v in result.violations)
    assert "user@example.com" in result.text


def test_pii_email_block_raises():
    config = GuardrailConfig(pii_detection=True, on_violation="block")
    with pytest.raises(GuardrailViolationError, match="PII detected"):
        check_output("Send to admin@kazi.ai", config)


def test_pii_email_redact_replaces_address():
    config = GuardrailConfig(pii_detection=True, on_violation="redact")
    result = check_output("Email: test@example.com here", config)
    assert not result.clean
    assert "test@example.com" not in result.text
    assert "[REDACTED:EMAIL]" in result.text


def test_pii_multiple_emails_counted():
    config = GuardrailConfig(pii_detection=True, on_violation="warn")
    result = check_output("a@a.com and b@b.com", config)
    assert any("2_match" in v for v in result.violations)


# ── PII — phone ───────────────────────────────────────────────────────────────

def test_pii_phone_us_detected_and_redacted():
    config = GuardrailConfig(pii_detection=True, on_violation="redact")
    result = check_output("Call 800-555-1234 now.", config)
    assert not result.clean
    assert any("phone_us" in v for v in result.violations)
    assert "[REDACTED:PHONE_US]" in result.text


# ── PII — SSN ─────────────────────────────────────────────────────────────────

def test_pii_ssn_detected_and_redacted():
    config = GuardrailConfig(pii_detection=True, on_violation="redact")
    result = check_output("SSN: 123-45-6789", config)
    assert not result.clean
    assert any("ssn" in v for v in result.violations)
    assert "[REDACTED:SSN]" in result.text


# ── PII — credit card ─────────────────────────────────────────────────────────

def test_pii_visa_card_detected():
    config = GuardrailConfig(pii_detection=True, on_violation="redact")
    result = check_output("Card number: 4111111111111111", config)
    assert not result.clean
    assert any("credit_card" in v for v in result.violations)


# ── PII — IPv4 ────────────────────────────────────────────────────────────────

def test_pii_ipv4_detected_and_redacted():
    config = GuardrailConfig(pii_detection=True, on_violation="redact")
    result = check_output("Server IP: 192.168.1.100", config)
    assert not result.clean
    assert any("ipv4" in v for v in result.violations)
    assert "[REDACTED:IPV4]" in result.text


# ── PII — disabled ────────────────────────────────────────────────────────────

def test_pii_disabled_ignores_patterns():
    config = GuardrailConfig(pii_detection=False)
    result = check_output("Email: user@example.com, SSN: 123-45-6789", config)
    assert result.clean


# ── Custom PII patterns ───────────────────────────────────────────────────────

def test_custom_pii_pattern_redacted():
    config = GuardrailConfig(
        pii_detection=True,
        on_violation="redact",
        custom_pii_patterns=[("internal_id", r"INT-\d{6}")],
    )
    result = check_output("Invoice INT-123456 processed", config)
    assert not result.clean
    assert any("internal_id" in v for v in result.violations)
    assert "[REDACTED:INTERNAL_ID]" in result.text


def test_custom_pii_no_match_is_clean():
    config = GuardrailConfig(
        pii_detection=True,
        on_violation="warn",
        custom_pii_patterns=[("internal_id", r"INT-\d{6}")],
    )
    result = check_output("No internal IDs here", config)
    assert result.clean


# ── Blocklist — warn ──────────────────────────────────────────────────────────

def test_blocklist_warn_records_violation_keeps_text():
    config = GuardrailConfig(blocklist_patterns=[r"secret project"], on_violation="warn")
    result = check_output("We are working on secret project Alpha.", config)
    assert not result.clean
    assert any("blocklist" in v for v in result.violations)
    assert "secret project" in result.text


def test_blocklist_case_insensitive_match():
    config = GuardrailConfig(blocklist_patterns=[r"forbidden"], on_violation="warn")
    result = check_output("This is FORBIDDEN content", config)
    assert not result.clean


def test_blocklist_no_match_stays_clean():
    config = GuardrailConfig(blocklist_patterns=[r"classified"])
    result = check_output("completely normal text", config)
    assert result.clean


# ── Blocklist — block ─────────────────────────────────────────────────────────

def test_blocklist_block_raises_on_match():
    config = GuardrailConfig(blocklist_patterns=[r"confidential"], on_violation="block")
    with pytest.raises(GuardrailViolationError, match="Blocklist"):
        check_output("This contains confidential data.", config)


# ── Blocklist — redact ────────────────────────────────────────────────────────

def test_blocklist_redact_removes_match():
    config = GuardrailConfig(blocklist_patterns=[r"classified"], on_violation="redact")
    result = check_output("The classified document is ready.", config)
    assert not result.clean
    assert "classified" not in result.text
    assert "[REDACTED]" in result.text


# ── Blocklist — invalid regex ─────────────────────────────────────────────────

def test_invalid_regex_in_blocklist_is_skipped_not_raised():
    config = GuardrailConfig(blocklist_patterns=[r"[invalid_regex"], on_violation="block")
    result = check_output("some normal text", config)
    assert result.clean


# ── Combined rules ────────────────────────────────────────────────────────────

def test_multiple_violation_types_all_accumulated():
    config = GuardrailConfig(
        pii_detection=True,
        blocklist_patterns=[r"forbidden"],
        on_violation="warn",
    )
    result = check_output("Email me@x.com about the forbidden topic", config)
    assert len(result.violations) >= 2
    assert any("email" in v for v in result.violations)
    assert any("blocklist" in v for v in result.violations)


def test_max_chars_then_pii_both_recorded_in_warn_mode():
    config = GuardrailConfig(
        max_output_chars=10,
        pii_detection=True,
        on_violation="warn",
    )
    result = check_output("user@example.com is long", config)
    assert len(result.violations) >= 2
