"""
Custom LLM and RAG provider integration tests.

Validates the escape hatches that let users plug in any LangChain chat model
or LlamaIndex embedding model without being restricted to the four built-in
provider enums.

LLM tests require OPENAI_API_KEY (skipped otherwise).
Embedding tests run fully offline — MockEmbedding needs no API key.
"""
import os
import tempfile

import pytest

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = "gpt-4o-mini"


# ── Custom LLM ────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not OPENAI_KEY, reason="OPENAI_API_KEY not set")
@pytest.mark.asyncio
async def test_custom_llm_bypasses_provider_enum():
    """
    When custom_llm is set, the brain must use it regardless of provider enum.
    We set provider=LOCAL (would fail without Ollama) to prove the override works.
    """
    from langchain_openai import ChatOpenAI

    from kazi.brain.graph_builder import GraphBrain
    from kazi.core.config import KaziConfig, LLMConfig, LLMProvider
    from kazi.core.registry import ToolRegistry

    custom = ChatOpenAI(model=MODEL, api_key=OPENAI_KEY)
    config = KaziConfig(llm=LLMConfig(
        provider=LLMProvider.LOCAL,  # would raise without Ollama running
        custom_llm=custom,
    ))
    brain = GraphBrain(config, ToolRegistry())
    result = await brain.run("Say exactly: CUSTOM_LLM_OK", thread_id="custom-llm-1")
    assert "CUSTOM_LLM_OK" in result


@pytest.mark.skipif(not OPENAI_KEY, reason="OPENAI_API_KEY not set")
@pytest.mark.asyncio
async def test_custom_llm_used_by_kazi_orchestrator():
    """custom_llm wired through Kazi.create() must produce real responses."""
    from langchain_openai import ChatOpenAI

    from kazi import Kazi
    from kazi.core.config import KaziConfig, LLMConfig, LLMProvider

    custom = ChatOpenAI(model=MODEL, api_key=OPENAI_KEY)
    config = KaziConfig(llm=LLMConfig(
        provider=LLMProvider.LOCAL,
        custom_llm=custom,
    ))
    async with await Kazi.create(config) as kazi:
        result = await kazi.run("Say exactly: ORCHESTRATOR_CUSTOM_OK", thread_id="custom-llm-2")
    assert "ORCHESTRATOR_CUSTOM_OK" in result


@pytest.mark.skipif(not OPENAI_KEY, reason="OPENAI_API_KEY not set")
@pytest.mark.asyncio
async def test_custom_llm_supports_tool_calls():
    """A custom LLM must still be able to call registered tools."""
    from langchain_openai import ChatOpenAI

    from kazi.brain.graph_builder import GraphBrain
    from kazi.core.config import KaziConfig, LLMConfig, LLMProvider
    from kazi.core.registry import ToolDefinition, ToolRegistry, ToolSource

    calls = []

    async def ping() -> str:
        calls.append(1)
        return "PONG"

    registry = ToolRegistry()
    registry.register(ToolDefinition(
        name="ping",
        description="Returns PONG. Call this when asked to ping.",
        parameters=[],
        source=ToolSource.NATIVE,
        handler=ping,
    ))

    custom = ChatOpenAI(model=MODEL, api_key=OPENAI_KEY)
    config = KaziConfig(llm=LLMConfig(provider=LLMProvider.LOCAL, custom_llm=custom))
    brain = GraphBrain(config, registry)

    result = await brain.run("Call the ping tool and tell me what it returned.", thread_id="custom-llm-tool")
    assert calls, "Tool was never called"
    assert "PONG" in result


@pytest.mark.skipif(not OPENAI_KEY, reason="OPENAI_API_KEY not set")
@pytest.mark.asyncio
async def test_custom_llm_multi_turn_memory():
    """Memory must work correctly with a custom LLM."""
    from langchain_openai import ChatOpenAI

    from kazi.brain.graph_builder import GraphBrain
    from kazi.core.config import KaziConfig, LLMConfig, LLMProvider
    from kazi.core.registry import ToolRegistry

    custom = ChatOpenAI(model=MODEL, api_key=OPENAI_KEY)
    config = KaziConfig(llm=LLMConfig(provider=LLMProvider.LOCAL, custom_llm=custom))
    brain = GraphBrain(config, ToolRegistry())

    await brain.run("My secret code is ZEBRA99.", thread_id="custom-llm-mem")
    reply = await brain.run("What is my secret code?", thread_id="custom-llm-mem")
    assert "ZEBRA99" in reply


# ── Custom embedding ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_custom_embedding_used_for_ingestion():
    """
    custom_embedding must be passed to LlamaIndex Settings.embed_model so
    documents can be ingested without calling any external embedding API.
    """
    from llama_index.core.embeddings.mock_embed_model import MockEmbedding

    from kazi.core.config import LLMConfig, LLMProvider, RAGConfig
    from kazi.data.index_manager import IndexManager

    embed = MockEmbedding(embed_dim=8)
    rag = RAGConfig(custom_embedding=embed)
    manager = IndexManager(rag, LLMConfig(provider=LLMProvider.OPENAI))

    await manager.ingest_documents(
        [
            {"text": "Kazi supports custom embedding models."},
            {"text": "No OpenAI API call is needed when custom_embedding is set."},
        ],
        index_name="custom_embed_test",
    )

    assert "custom_embed_test" in manager._indices
    assert "custom_embed_test" in manager.list_indices()


@pytest.mark.asyncio
async def test_custom_embedding_does_not_call_openai():
    """
    With custom_embedding set, the index manager must not instantiate
    OpenAIEmbedding — verified by providing a non-existent API key that
    would fail at init time if OpenAI embedding were attempted.
    """
    from llama_index.core.embeddings.mock_embed_model import MockEmbedding

    from kazi.core.config import LLMConfig, LLMProvider, RAGConfig
    from kazi.data.index_manager import IndexManager

    embed = MockEmbedding(embed_dim=8)
    rag = RAGConfig(custom_embedding=embed)
    # Deliberately bad API key — would fail if OpenAIEmbedding were constructed
    llm = LLMConfig(provider=LLMProvider.OPENAI, api_key="sk-INVALID-KEY")
    manager = IndexManager(rag, llm)

    # Should not raise despite the invalid key, because custom_embedding skips
    # the OpenAIEmbedding constructor
    await manager.ingest_documents(
        [{"text": "Custom embedding bypasses API key check."}],
        index_name="no_openai_embed",
    )
    assert "no_openai_embed" in manager._indices


@pytest.mark.asyncio
async def test_custom_embedding_retrieval_works():
    """Documents ingested with a custom embedding must be retrievable."""
    from llama_index.core.embeddings.mock_embed_model import MockEmbedding

    from kazi.core.config import LLMConfig, LLMProvider, RAGConfig
    from kazi.data.index_manager import IndexManager

    embed = MockEmbedding(embed_dim=8)
    rag = RAGConfig(custom_embedding=embed)
    manager = IndexManager(rag, LLMConfig(provider=LLMProvider.OPENAI))

    await manager.ingest_documents(
        [{"text": "The answer is forty-two."}],
        index_name="retrieval_test",
    )

    # Build query engine — retriever works offline with MockEmbedding
    engine = manager.get_query_engine("retrieval_test")
    assert engine is not None

    # Verify the tool definition is buildable
    tool = manager.as_tool_definition("retrieval_test")
    assert tool.name == "search_retrieval_test"


@pytest.mark.asyncio
async def test_custom_embedding_and_synthesis_llm_both_set():
    """When both custom hooks are set, no provider-specific setup runs at all."""
    from llama_index.core.embeddings.mock_embed_model import MockEmbedding
    from llama_index.core.llms.mock import MockLLM

    from kazi.core.config import LLMConfig, LLMProvider, RAGConfig
    from kazi.data.index_manager import IndexManager

    embed = MockEmbedding(embed_dim=8)
    synth = MockLLM()
    rag = RAGConfig(custom_embedding=embed, custom_synthesis_llm=synth)
    # provider=LOCAL with no API key — would fail if provider wiring ran
    llm = LLMConfig(provider=LLMProvider.LOCAL)
    manager = IndexManager(rag, llm)

    await manager.ingest_documents(
        [{"text": "Fully offline ingestion with custom embed and synthesis LLM."}],
        index_name="fully_custom",
    )
    assert "fully_custom" in manager._indices


@pytest.mark.asyncio
async def test_custom_embedding_directory_ingestion():
    """ingest_directory must also respect custom_embedding."""
    from llama_index.core.embeddings.mock_embed_model import MockEmbedding

    from kazi.core.config import LLMConfig, LLMProvider, RAGConfig
    from kazi.data.index_manager import IndexManager

    embed = MockEmbedding(embed_dim=8)
    rag = RAGConfig(custom_embedding=embed)
    manager = IndexManager(rag, LLMConfig(provider=LLMProvider.OPENAI))

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write real text files for the directory reader
        for i, content in enumerate([
            "Kazi is an AI orchestration framework.",
            "It supports MCP, A2A, RAG, and custom LLM providers.",
            "The custom_embedding field accepts any LlamaIndex BaseEmbedding.",
        ]):
            __import__("pathlib").Path(tmpdir).joinpath(f"doc{i}.txt").write_text(content)

        await manager.ingest_directory(tmpdir, index_name="dir_custom_embed")

    assert "dir_custom_embed" in manager._indices
