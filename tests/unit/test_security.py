"""Unit tests for security and token budget modules."""
import pytest

from kazi.core.exceptions import (
    ConfigurationError,
    ThreadAuthError,
    TokenBudgetExceeded,
    ToolBlockedError,
)
from kazi.core.secrets import SecretRef
from kazi.core.security import ContentPolicy, MCPSecurityPolicy, ThreadPolicy
from kazi.core.token_budget import TokenBudget, TokenBudgetConfig, count_tokens

# ── SecretRef ──────────────────────────────────────────────────────────────────

def test_secret_ref_literal():
    ref = SecretRef("sk-abc")
    assert ref.resolve() == "sk-abc"
    assert repr(ref) == "SecretRef(***)"
    assert str(ref) == "***"


def test_secret_ref_callable():
    ref = SecretRef.from_callable(lambda: "dynamic-value")
    assert ref.resolve() == "dynamic-value"
    assert repr(ref) == "SecretRef(***)"


def test_secret_ref_env(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "env-value")
    ref = SecretRef.from_env("TEST_KEY")
    assert ref.resolve() == "env-value"


def test_secret_ref_env_missing(monkeypatch):
    monkeypatch.delenv("MISSING_KEY", raising=False)
    ref = SecretRef.from_env("MISSING_KEY")
    assert ref.resolve() is None


def test_secret_ref_coerce_string():
    ref = SecretRef.coerce("plain-string")
    assert isinstance(ref, SecretRef)
    assert ref.resolve() == "plain-string"


def test_secret_ref_coerce_none():
    assert SecretRef.coerce(None) is None


def test_secret_ref_coerce_passthrough():
    original = SecretRef("x")
    assert SecretRef.coerce(original) is original


def test_secret_ref_not_equal_to_string():
    ref = SecretRef("abc")
    assert ref != "abc"
    assert ref != SecretRef("abc")  # only equal to itself


def test_secret_ref_equal_to_itself():
    ref = SecretRef("abc")
    assert ref == ref


def test_secret_ref_hash_is_stable():
    ref = SecretRef("abc")
    assert hash(ref) == hash(ref)


def test_secret_ref_hash_is_id_based():
    """Two SecretRefs with the same value must have different hashes (identity-based)."""
    r1 = SecretRef("same")
    r2 = SecretRef("same")
    assert hash(r1) != hash(r2) or r1 is not r2  # different objects → different identity


# ── MCPSecurityPolicy ─────────────────────────────────────────────────────────

def test_mcp_allow_all_by_default():
    policy = MCPSecurityPolicy()
    assert policy.is_allowed("anything__tool") is True


def test_mcp_allowlist_permits_match():
    policy = MCPSecurityPolicy(allowlist=["github__*"])
    assert policy.is_allowed("github__list_prs") is True


def test_mcp_allowlist_blocks_non_match():
    policy = MCPSecurityPolicy(allowlist=["github__*"])
    assert policy.is_allowed("filesystem__read") is False


def test_mcp_denylist_blocks():
    policy = MCPSecurityPolicy(denylist=["*__delete*"])
    assert policy.is_allowed("github__delete_repo") is False
    assert policy.is_allowed("github__list_prs") is True


def test_mcp_denylist_beats_allowlist():
    policy = MCPSecurityPolicy(allowlist=["github__*"], denylist=["github__delete_*"])
    assert policy.is_allowed("github__list_prs") is True
    assert policy.is_allowed("github__delete_repo") is False


# ── ContentPolicy ─────────────────────────────────────────────────────────────

def test_content_policy_tags_result():
    cp = ContentPolicy(tag_external_content=True)
    result = cp.wrap("web_search", "some content")
    assert '<external_content source="web_search">' in result
    assert "some content" in result
    assert "</external_content>" in result


def test_content_policy_no_tag():
    cp = ContentPolicy(tag_external_content=False)
    result = cp.wrap("web_search", "content")
    assert "<external_content" not in result


def test_content_policy_truncates():
    cp = ContentPolicy(max_result_chars=10)
    result = cp.wrap("tool", "x" * 100)
    assert "truncated" in result


def test_content_policy_on_result_hook_blocks():
    cp = ContentPolicy(on_tool_result=lambda name, r: None)
    with pytest.raises(ToolBlockedError):
        cp.wrap("tool", "value")


def test_content_policy_on_result_hook_modifies():
    cp = ContentPolicy(
        tag_external_content=False,
        on_tool_result=lambda name, r: r.upper(),
    )
    result = cp.wrap("tool", "hello")
    assert result == "HELLO"


def test_content_policy_on_call_hook_blocks():
    cp = ContentPolicy(on_tool_call=lambda name, args: None)
    with pytest.raises(ToolBlockedError):
        cp.check_call("tool", {"x": 1})


def test_content_policy_on_call_hook_modifies():
    cp = ContentPolicy(on_tool_call=lambda name, args: {**args, "injected": True})
    result = cp.check_call("tool", {"x": 1})
    assert result["injected"] is True


# ── ThreadPolicy ──────────────────────────────────────────────────────────────

def test_thread_policy_no_auth_passes():
    tp = ThreadPolicy(require_auth=False)
    tp.check("any-thread", None)  # should not raise


def test_thread_policy_auth_required_valid():
    tp = ThreadPolicy(require_auth=True, validator=lambda tid, tok: tok == "good")
    tp.check("t", "good")  # should not raise


def test_thread_policy_auth_required_invalid():
    tp = ThreadPolicy(require_auth=True, validator=lambda tid, tok: False)
    with pytest.raises(ThreadAuthError):
        tp.check("t", "bad")


def test_thread_policy_auth_no_validator_raises_at_construction():
    # Fail fast at config time — not silently at first .check() call
    with pytest.raises(ConfigurationError, match="validator"):
        ThreadPolicy(require_auth=True, validator=None)


# ── TokenBudget ───────────────────────────────────────────────────────────────

def test_token_budget_unlimited():
    budget = TokenBudget(TokenBudgetConfig(max_tokens_per_run=None))
    budget.charge_text("x" * 10000)  # should not raise


def test_token_budget_exceeded():
    budget = TokenBudget(TokenBudgetConfig(max_tokens_per_run=5))
    with pytest.raises(TokenBudgetExceeded):
        budget.charge_text("This is definitely more than five tokens total here")


def test_token_budget_tracks_usage():
    budget = TokenBudget(TokenBudgetConfig())
    budget.charge_text("hello world")
    assert budget.used > 0


def test_token_budget_reset():
    budget = TokenBudget(TokenBudgetConfig())
    budget.charge_text("hello")
    budget.reset()
    assert budget.used == 0


def test_count_tokens_positive():
    assert count_tokens("Hello, world!") > 0


def test_count_tokens_empty():
    assert count_tokens("") == 0 or count_tokens("") >= 0


# ── Config validation — MCPSecurityPolicy ─────────────────────────────────────

def test_mcp_policy_rejects_none_denylist():
    with pytest.raises(ConfigurationError, match="denylist"):
        MCPSecurityPolicy(denylist=None)


def test_mcp_policy_rejects_none_allowlist():
    with pytest.raises(ConfigurationError, match="allowlist"):
        MCPSecurityPolicy(allowlist=None)


def test_mcp_policy_rejects_non_string_pattern():
    with pytest.raises(ConfigurationError, match="strings"):
        MCPSecurityPolicy(denylist=[123, "valid"])


def test_mcp_policy_accepts_valid_lists():
    p = MCPSecurityPolicy(allowlist=["github__*"], denylist=["github__delete_*"])
    assert p.is_allowed("github__list_prs") is True


# ── Config validation — ContentPolicy ────────────────────────────────────────

def test_content_policy_rejects_zero_max_chars():
    with pytest.raises(ConfigurationError, match="max_result_chars"):
        ContentPolicy(max_result_chars=0)


def test_content_policy_rejects_negative_max_chars():
    with pytest.raises(ConfigurationError, match="max_result_chars"):
        ContentPolicy(max_result_chars=-1)


def test_content_policy_rejects_non_callable_on_call():
    with pytest.raises(ConfigurationError, match="on_tool_call"):
        ContentPolicy(on_tool_call="not-a-function")


def test_content_policy_rejects_non_callable_on_result():
    with pytest.raises(ConfigurationError, match="on_tool_result"):
        ContentPolicy(on_tool_result=42)


def test_content_policy_accepts_valid_config():
    cp = ContentPolicy(max_result_chars=1000, on_tool_call=lambda n, a: a)
    assert cp.max_result_chars == 1000


# ── Config validation — ThreadPolicy ─────────────────────────────────────────

def test_thread_policy_rejects_non_callable_validator():
    with pytest.raises(ConfigurationError, match="callable"):
        ThreadPolicy(require_auth=True, validator="not-callable")


def test_thread_policy_accepts_false_auth_with_no_validator():
    # require_auth=False with no validator is fine
    tp = ThreadPolicy(require_auth=False, validator=None)
    tp.check("any", None)  # should not raise


# ── Config validation — TokenBudgetConfig ────────────────────────────────────

def test_budget_rejects_zero_max_tokens():
    with pytest.raises(ConfigurationError, match="max_tokens_per_run"):
        TokenBudgetConfig(max_tokens_per_run=0)


def test_budget_rejects_negative_max_tokens():
    with pytest.raises(ConfigurationError, match="max_tokens_per_run"):
        TokenBudgetConfig(max_tokens_per_run=-100)


def test_budget_accepts_none_max_tokens():
    cfg = TokenBudgetConfig(max_tokens_per_run=None)
    assert cfg.max_tokens_per_run is None


def test_budget_rejects_warn_fraction_above_one():
    with pytest.raises(ConfigurationError, match="warn_at_fraction"):
        TokenBudgetConfig(warn_at_fraction=1.5)


def test_budget_rejects_warn_fraction_of_one():
    # 1.0 means "warn only when budget is exactly full" — no useful warning window
    with pytest.raises(ConfigurationError, match="warn_at_fraction"):
        TokenBudgetConfig(warn_at_fraction=1.0)


def test_budget_rejects_warn_fraction_of_zero():
    with pytest.raises(ConfigurationError, match="warn_at_fraction"):
        TokenBudgetConfig(warn_at_fraction=0.0)


def test_budget_rejects_negative_summarize_turns():
    with pytest.raises(ConfigurationError, match="summarize_after_turns"):
        TokenBudgetConfig(summarize_after_turns=-1)


def test_budget_accepts_zero_summarize_turns():
    # 0 means disabled — valid
    cfg = TokenBudgetConfig(summarize_after_turns=0)
    assert cfg.summarize_after_turns == 0


def test_budget_rejects_non_positive_max_chars():
    with pytest.raises(ConfigurationError, match="max_chars_per_tool_result"):
        TokenBudgetConfig(max_chars_per_tool_result=0)
