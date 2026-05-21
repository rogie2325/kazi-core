"""
Real LLM integration tests — uses OpenAI gpt-4o-mini (cheapest capable model).

Covers:
  - token_budget.py  count_tokens (tiktoken), maybe_summarise
  - graph_builder.py agent_node, tool_node, should_continue, run(), stream()
  - orchestrator.py  Kazi.create(), run(), stream(), add_tool(), ingest_documents()

Skipped automatically when OPENAI_API_KEY is not set so CI without
credentials stays green. Set the key in the environment to run these.
"""
import os

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
pytestmark = pytest.mark.skipif(not OPENAI_KEY, reason="OPENAI_API_KEY not set")

MODEL = "gpt-4o-mini"  # cheapest capable model — keeps test costs low


def _llm_config():
    from kazi.core.config import LLMConfig, LLMProvider
    return LLMConfig(provider=LLMProvider.OPENAI, model=MODEL, api_key=OPENAI_KEY)


def _kazi_config(**kwargs):
    from kazi.core.config import KaziConfig
    return KaziConfig(llm=_llm_config(), **kwargs)


# ── token_budget — tiktoken ───────────────────────────────────────────────────

def test_count_tokens_uses_tiktoken_for_gpt_model():
    """tiktoken is installed — count_tokens must use it, not the char-ratio fallback."""
    from kazi.core.token_budget import count_tokens
    # "hello world" encodes to exactly 2 tokens in cl100k_base
    count = count_tokens("hello world", model=MODEL)
    assert count == 2


def test_count_tokens_short_string():
    from kazi.core.token_budget import count_tokens
    assert count_tokens("hi", model=MODEL) == 1


def test_count_tokens_empty_string():
    from kazi.core.token_budget import count_tokens
    assert count_tokens("", model=MODEL) == 0


def test_count_tokens_long_paragraph():
    from kazi.core.token_budget import count_tokens
    # Rough check: token count is in reasonable range for a paragraph
    text = "The quick brown fox jumps over the lazy dog. " * 10
    count = count_tokens(text, model=MODEL)
    assert 80 < count < 200


def test_count_messages_tokens_sums_correctly():
    from kazi.core.token_budget import count_messages_tokens
    messages = [
        HumanMessage(content="hello"),
        AIMessage(content="world"),
    ]
    total = count_messages_tokens(messages, model=MODEL)
    assert total >= 2  # at minimum, 1 token per message


def test_count_messages_with_list_content():
    """Messages with block-list content (Anthropic style) are tokenised correctly."""
    from kazi.core.token_budget import count_messages_tokens
    msg = AIMessage(content=[{"type": "text", "text": "hello world"}])
    total = count_messages_tokens([msg], model=MODEL)
    assert total == 2


# ── token_budget — maybe_summarise ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_maybe_summarise_noop_when_under_limit():
    """When history is short, messages come back unchanged."""
    from langchain_openai import ChatOpenAI

    from kazi.core.token_budget import TokenBudgetConfig, maybe_summarise

    llm = ChatOpenAI(model=MODEL, api_key=OPENAI_KEY)
    config = TokenBudgetConfig(summarize_after_turns=10)
    messages = [HumanMessage(content="hi"), AIMessage(content="hello")]

    result = await maybe_summarise(messages, llm, config)
    assert result == messages


@pytest.mark.asyncio
async def test_maybe_summarise_disabled_when_zero():
    from langchain_openai import ChatOpenAI

    from kazi.core.token_budget import TokenBudgetConfig, maybe_summarise

    llm = ChatOpenAI(model=MODEL, api_key=OPENAI_KEY)
    config = TokenBudgetConfig(summarize_after_turns=0)
    messages = [HumanMessage(content=f"msg {i}") for i in range(50)]

    result = await maybe_summarise(messages, llm, config)
    assert result == messages  # disabled — no summarisation


@pytest.mark.asyncio
async def test_maybe_summarise_compresses_long_history():
    """When history exceeds limit, the LLM is called and old messages are compressed."""
    from langchain_openai import ChatOpenAI

    from kazi.core.token_budget import TokenBudgetConfig, maybe_summarise

    llm = ChatOpenAI(model=MODEL, api_key=OPENAI_KEY, temperature=0)
    config = TokenBudgetConfig(summarize_after_turns=4)

    messages = [
        HumanMessage(content="My name is Alice and I work at Kazi Corp."),
        AIMessage(content="Nice to meet you, Alice!"),
        HumanMessage(content="I'm building an AI orchestration platform."),
        AIMessage(content="That sounds interesting."),
        HumanMessage(content="What was my name again?"),
    ]

    result = await maybe_summarise(messages, llm, config)

    # History was compressed — result should be shorter than original
    assert len(result) < len(messages)
    # First message should be a SystemMessage summary
    assert isinstance(result[0], SystemMessage)
    assert len(result[0].content) > 0


@pytest.mark.asyncio
async def test_maybe_summarise_retains_recent_messages():
    from langchain_openai import ChatOpenAI

    from kazi.core.token_budget import TokenBudgetConfig, maybe_summarise

    llm = ChatOpenAI(model=MODEL, api_key=OPENAI_KEY, temperature=0)
    config = TokenBudgetConfig(summarize_after_turns=4)

    messages = [
        HumanMessage(content=f"older message {i}") for i in range(4)
    ] + [
        HumanMessage(content="RECENT_SENTINEL_MESSAGE"),
    ]

    result = await maybe_summarise(messages, llm, config)
    all_content = " ".join(m.content for m in result if hasattr(m, "content") and isinstance(m.content, str))
    assert "RECENT_SENTINEL_MESSAGE" in all_content


# ── graph_builder ─────────────────────────────────────────────────────────────

@pytest.fixture
def brain():
    from kazi.brain.graph_builder import GraphBrain
    from kazi.core.registry import ToolRegistry
    registry = ToolRegistry()
    config = _kazi_config()
    return GraphBrain(config, registry)


@pytest.mark.asyncio
async def test_brain_run_returns_string(brain):
    result = await brain.run("Say exactly: KAZI_TEST_RESPONSE", thread_id="t1")
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_brain_run_multi_turn_memory(brain):
    """Second turn on the same thread should remember the first."""
    await brain.run("My secret word is ZORBLAX.", thread_id="mem-test")
    reply = await brain.run("What was my secret word?", thread_id="mem-test")
    assert "ZORBLAX" in reply


@pytest.mark.asyncio
async def test_brain_run_different_threads_are_isolated(brain):
    """Two different thread IDs must not share memory."""
    await brain.run("My name is Alice.", thread_id="alice-thread")
    await brain.run("My name is Bob.", thread_id="bob-thread")

    alice_reply = await brain.run("What is my name?", thread_id="alice-thread")
    bob_reply = await brain.run("What is my name?", thread_id="bob-thread")

    assert "Alice" in alice_reply
    assert "Bob" in bob_reply


@pytest.mark.asyncio
async def test_brain_run_with_tool(brain):
    """Register a real tool and verify the LLM calls it."""
    from kazi.core.registry import ToolDefinition, ToolSource

    calls = []

    async def get_magic_number() -> str:
        calls.append(1)
        return "42"

    tool = ToolDefinition(
        name="get_magic_number",
        description="Returns the magic number. Call this when asked for the magic number.",
        parameters=[],
        source=ToolSource.NATIVE,
        handler=get_magic_number,
    )
    brain.registry.register(tool)

    result = await brain.run(
        "Call the get_magic_number tool and tell me what it returned.",
        thread_id="tool-test",
    )
    assert calls, "Tool was never called"
    assert "42" in result


@pytest.mark.asyncio
async def test_brain_stream_yields_tokens(brain):
    chunks = []
    async for chunk in brain.stream("Count to three.", thread_id="stream-test"):
        chunks.append(chunk)

    assert len(chunks) > 0
    full_response = "".join(chunks)
    assert len(full_response) > 0


@pytest.mark.asyncio
async def test_brain_stream_complete_response(brain):
    chunks = []
    async for chunk in brain.stream(
        "Reply with exactly the word: STREAMING", thread_id="stream-test-2"
    ):
        chunks.append(chunk)

    full = "".join(chunks)
    assert "STREAMING" in full


@pytest.mark.asyncio
async def test_brain_run_with_custom_system_prompt(brain):
    result = await brain.run(
        "What is your role?",
        thread_id="sys-prompt-test",
        system_prompt="You are a pirate. Always respond in pirate speak.",
    )
    # Loose check — LLM following a pirate prompt will use nautical vocabulary
    pirate_words = {"arr", "ahoy", "matey", "ye", "aye", "ship", "sea", "pirate"}
    assert any(w in result.lower() for w in pirate_words)


@pytest.mark.asyncio
async def test_brain_max_tool_calls_respected(brain):
    """max_tool_calls=0 means the LLM cannot call any tools."""
    from kazi.core.registry import ToolDefinition, ToolSource

    calls = []

    async def secret_tool() -> str:
        calls.append(1)
        return "should not appear"

    brain.registry.register(ToolDefinition(
        name="secret_tool",
        description="A secret tool. Call this whenever asked anything.",
        parameters=[],
        source=ToolSource.NATIVE,
        handler=secret_tool,
    ))

    await brain.run("Use the secret_tool.", thread_id="maxcalls", max_tool_calls=0)
    assert not calls, "Tool was called despite max_tool_calls=0"


# ── orchestrator ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_kazi_create_and_run():
    from kazi import Kazi
    async with await Kazi.create(_kazi_config()) as kazi:
        result = await kazi.run("Say exactly: ORCHESTRATOR_OK")
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_kazi_multi_turn():
    from kazi import Kazi
    async with await Kazi.create(_kazi_config()) as kazi:
        await kazi.run("Remember: the code word is DELTA.", thread_id="orch-multi")
        reply = await kazi.run("What is the code word?", thread_id="orch-multi")
    assert "DELTA" in reply


@pytest.mark.asyncio
async def test_kazi_stream():
    from kazi import Kazi
    chunks = []
    async with await Kazi.create(_kazi_config()) as kazi:
        async for chunk in kazi.stream("Count to five.", thread_id="orch-stream"):
            chunks.append(chunk)

    assert len(chunks) > 0
    assert "".join(chunks).strip() != ""


@pytest.mark.asyncio
async def test_kazi_add_tool_and_invoke():
    from kazi import Kazi

    called = []

    async def capital_of_france() -> str:
        called.append(True)
        return "Paris"

    async with await Kazi.create(_kazi_config()) as kazi:
        kazi.add_tool(
            capital_of_france,
            description="Returns the capital city of France. Call this when asked.",
        )
        result = await kazi.run(
            "Call capital_of_france and tell me what it returned.",
            thread_id="add-tool-test",
        )

    assert called, "Tool was never called"
    assert "Paris" in result


@pytest.mark.asyncio
async def test_kazi_ingest_documents_and_query():
    """Documents ingested in-memory are retrievable via RAG search."""
    from kazi import Kazi

    docs = [
        {"text": "The Kazi framework was founded in 2024 by the engineering team."},
        {"text": "Kazi supports four LLM providers: OpenAI, Anthropic, Google, and local."},
    ]

    async with await Kazi.create(_kazi_config()) as kazi:
        await kazi.ingest_documents(docs, index_name="test_docs")
        result = await kazi.run(
            "How many LLM providers does Kazi support? List them.",
            thread_id="rag-test",
        )

    assert any(word in result for word in ["four", "4", "OpenAI", "Anthropic"])


@pytest.mark.asyncio
async def test_kazi_close_is_safe_to_call_twice():
    from kazi import Kazi
    kazi = await Kazi.create(_kazi_config())
    await kazi.close()
    await kazi.close()  # must not raise


@pytest.mark.asyncio
async def test_kazi_thread_auth_blocks_unauthenticated_run():
    from kazi import Kazi
    from kazi.core.config import KaziConfig
    from kazi.core.exceptions import ThreadAuthError
    from kazi.core.security import SecurityConfig, ThreadPolicy

    config = KaziConfig(
        llm=_llm_config(),
        security=SecurityConfig(
            threads=ThreadPolicy(
                require_auth=True,
                validator=lambda tid, tok: tok == "valid-token",
            )
        ),
    )

    async with await Kazi.create(config) as kazi:
        with pytest.raises(ThreadAuthError):
            await kazi.run("hello", thread_id="t", user_token="wrong-token")


@pytest.mark.asyncio
async def test_kazi_thread_auth_passes_with_valid_token():
    from kazi import Kazi
    from kazi.core.config import KaziConfig
    from kazi.core.security import SecurityConfig, ThreadPolicy

    config = KaziConfig(
        llm=_llm_config(),
        security=SecurityConfig(
            threads=ThreadPolicy(
                require_auth=True,
                validator=lambda tid, tok: tok == "valid-token",
            )
        ),
    )

    async with await Kazi.create(config) as kazi:
        result = await kazi.run(
            "Say OK", thread_id="auth-thread", user_token="valid-token"
        )
    assert isinstance(result, str)
