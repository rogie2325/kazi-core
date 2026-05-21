"""Unit tests for KaziConfig."""
from kazi.core.config import KaziConfig, LLMProvider, MemoryBackend


def test_defaults():
    cfg = KaziConfig()
    assert cfg.llm.provider == LLMProvider.OPENAI
    assert cfg.llm.model == "gpt-4o"
    assert cfg.memory.backend == MemoryBackend.IN_MEMORY


def test_from_env(monkeypatch):
    monkeypatch.setenv("KAZI_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("KAZI_LLM_MODEL", "claude-sonnet-4-6")
    monkeypatch.setenv("KAZI_API_KEY", "sk-test")

    cfg = KaziConfig.from_env()
    assert cfg.llm.provider == LLMProvider.ANTHROPIC
    assert cfg.llm.model == "claude-sonnet-4-6"
    assert cfg.llm.resolved_api_key() == "sk-test"
