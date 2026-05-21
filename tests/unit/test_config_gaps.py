"""
Covers uncovered lines in kazi.core.config:
  - LLMConfig.__post_init__ coercion
  - KaziConfig.from_env
  - KaziConfig.from_yaml (including security and budget sections)
"""
import os
import tempfile

from kazi.core.config import (
    KaziConfig,
    LLMConfig,
    LLMProvider,
    MemoryBackend,
    VectorStoreBackend,
)
from kazi.core.secrets import SecretRef

# ── LLMConfig.__post_init__ ───────────────────────────────────────────────────

def test_llm_config_coerces_plain_string_to_secret_ref():
    cfg = LLMConfig(api_key="sk-abc-123")
    assert isinstance(cfg.api_key, SecretRef)
    assert cfg.api_key.resolve() == "sk-abc-123"


def test_llm_config_none_api_key_stays_none():
    cfg = LLMConfig(api_key=None)
    assert cfg.api_key is None


def test_llm_config_passes_through_existing_secret_ref():
    original = SecretRef.from_env("NONEXISTENT_KEY_XYZ")
    cfg = LLMConfig(api_key=original)
    assert cfg.api_key is original


def test_llm_config_resolved_api_key_returns_string():
    cfg = LLMConfig(api_key="my-key")
    assert cfg.resolved_api_key() == "my-key"


def test_llm_config_resolved_api_key_returns_none_when_unset():
    cfg = LLMConfig(api_key=None)
    assert cfg.resolved_api_key() is None


def test_llm_config_repr_does_not_expose_key():
    """The api_key must not appear in repr even after coercion."""
    cfg = LLMConfig(api_key="super-secret-key")
    assert "super-secret-key" not in repr(cfg)
    assert "super-secret-key" not in str(cfg)


# ── KaziConfig.from_env ──────────────────────────────────────────────────────

def test_from_env_defaults(monkeypatch):
    monkeypatch.delenv("KAZI_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("KAZI_LLM_MODEL", raising=False)
    monkeypatch.delenv("KAZI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("KAZI_VERBOSE", raising=False)
    monkeypatch.delenv("KAZI_BASE_URL", raising=False)

    cfg = KaziConfig.from_env()
    assert cfg.llm.provider == LLMProvider.OPENAI
    assert cfg.llm.model == "gpt-4o"
    assert cfg.llm.api_key is None
    assert cfg.verbose is False


def test_from_env_reads_provider(monkeypatch):
    monkeypatch.setenv("KAZI_LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("KAZI_LLM_MODEL", raising=False)
    monkeypatch.delenv("KAZI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    cfg = KaziConfig.from_env()
    assert cfg.llm.provider == LLMProvider.ANTHROPIC


def test_from_env_reads_model(monkeypatch):
    monkeypatch.setenv("KAZI_LLM_MODEL", "claude-opus-4-7")
    monkeypatch.delenv("KAZI_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("KAZI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    cfg = KaziConfig.from_env()
    assert cfg.llm.model == "claude-opus-4-7"


def test_from_env_uses_kazi_api_key_first(monkeypatch):
    monkeypatch.setenv("KAZI_API_KEY", "kazi-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.delenv("KAZI_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("KAZI_LLM_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    cfg = KaziConfig.from_env()
    assert cfg.llm.api_key is not None
    assert cfg.llm.resolved_api_key() == "kazi-key"


def test_from_env_falls_back_to_openai_key(monkeypatch):
    monkeypatch.delenv("KAZI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "oai-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("KAZI_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("KAZI_LLM_MODEL", raising=False)

    cfg = KaziConfig.from_env()
    assert cfg.llm.resolved_api_key() == "oai-key"


def test_from_env_verbose_flag(monkeypatch):
    monkeypatch.setenv("KAZI_VERBOSE", "true")
    monkeypatch.delenv("KAZI_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("KAZI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("KAZI_LLM_MODEL", raising=False)

    cfg = KaziConfig.from_env()
    assert cfg.verbose is True


def test_from_env_verbose_accepts_1(monkeypatch):
    monkeypatch.setenv("KAZI_VERBOSE", "1")
    monkeypatch.delenv("KAZI_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("KAZI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("KAZI_LLM_MODEL", raising=False)

    cfg = KaziConfig.from_env()
    assert cfg.verbose is True


# ── KaziConfig.from_yaml ─────────────────────────────────────────────────────

def _write_yaml(content: str) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    try:
        f.write(content)
        f.flush()
        return f.name
    finally:
        f.close()


def test_from_yaml_minimal():
    path = _write_yaml("llm:\n  provider: openai\n  model: gpt-4o\n")
    try:
        cfg = KaziConfig.from_yaml(path)
        assert cfg.llm.provider == LLMProvider.OPENAI
        assert cfg.llm.model == "gpt-4o"
    finally:
        os.unlink(path)


def test_from_yaml_anthropic_provider():
    path = _write_yaml("llm:\n  provider: anthropic\n  model: claude-sonnet-4-6\n")
    try:
        cfg = KaziConfig.from_yaml(path)
        assert cfg.llm.provider == LLMProvider.ANTHROPIC
    finally:
        os.unlink(path)


def test_from_yaml_rag_section():
    yaml_content = """\
llm:
  provider: openai
rag:
  vector_store: chroma
  chunk_size: 512
  similarity_top_k: 3
"""
    path = _write_yaml(yaml_content)
    try:
        cfg = KaziConfig.from_yaml(path)
        assert cfg.rag.vector_store == VectorStoreBackend.CHROMA
        assert cfg.rag.chunk_size == 512
        assert cfg.rag.similarity_top_k == 3
    finally:
        os.unlink(path)


def test_from_yaml_memory_section():
    yaml_content = """\
llm:
  provider: openai
memory:
  backend: sqlite
  connection_string: sqlite:///test.db
  max_conversation_turns: 100
"""
    path = _write_yaml(yaml_content)
    try:
        cfg = KaziConfig.from_yaml(path)
        assert cfg.memory.backend == MemoryBackend.SQLITE
        assert cfg.memory.connection_string == "sqlite:///test.db"
        assert cfg.memory.max_conversation_turns == 100
    finally:
        os.unlink(path)


def test_from_yaml_security_content_policy():
    """security.content section must be deserialised — not silently ignored."""
    yaml_content = """\
llm:
  provider: openai
security:
  content:
    tag_external_content: false
    max_result_chars: 10000
  verify_tls: false
"""
    path = _write_yaml(yaml_content)
    try:
        cfg = KaziConfig.from_yaml(path)
        assert cfg.security.content.tag_external_content is False
        assert cfg.security.content.max_result_chars == 10000
        assert cfg.security.verify_tls is False
    finally:
        os.unlink(path)


def test_from_yaml_security_mcp_policy():
    yaml_content = """\
llm:
  provider: openai
security:
  mcp:
    allowlist:
      - "github__*"
    denylist:
      - "github__delete_*"
"""
    path = _write_yaml(yaml_content)
    try:
        cfg = KaziConfig.from_yaml(path)
        assert "github__*" in cfg.security.mcp.allowlist
        assert "github__delete_*" in cfg.security.mcp.denylist
    finally:
        os.unlink(path)


def test_from_yaml_budget_section():
    """budget section must be deserialised — not silently ignored."""
    yaml_content = """\
llm:
  provider: openai
budget:
  max_tokens_per_run: 50000
  warn_at_fraction: 0.7
  summarize_after_turns: 10
  max_chars_per_tool_result: 20000
"""
    path = _write_yaml(yaml_content)
    try:
        cfg = KaziConfig.from_yaml(path)
        assert cfg.budget.max_tokens_per_run == 50000
        assert cfg.budget.warn_at_fraction == 0.7
        assert cfg.budget.summarize_after_turns == 10
        assert cfg.budget.max_chars_per_tool_result == 20000
    finally:
        os.unlink(path)


def test_from_yaml_defaults_when_sections_absent():
    """When security/budget sections are absent, safe defaults are used."""
    path = _write_yaml("llm:\n  provider: openai\n")
    try:
        cfg = KaziConfig.from_yaml(path)
        assert cfg.security.content.tag_external_content is True
        assert cfg.security.verify_tls is True
        assert cfg.budget.max_tokens_per_run is None
    finally:
        os.unlink(path)


def test_from_yaml_verbose_flag():
    path = _write_yaml("llm:\n  provider: openai\nverbose: true\n")
    try:
        cfg = KaziConfig.from_yaml(path)
        assert cfg.verbose is True
    finally:
        os.unlink(path)
