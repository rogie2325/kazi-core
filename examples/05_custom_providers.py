"""
Example 5: Custom LLM and embedding providers

By default kazi uses OpenAI, Anthropic, Google, or Ollama via the
LLMProvider enum. This example shows how to break out of that list and
use any LangChain chat model or LlamaIndex embedding model instead —
useful for AWS Bedrock, Google Vertex AI, Azure OpenAI, HuggingFace,
Cohere, Mistral, or any self-hosted model.

The escape hatches:
  LLMConfig(custom_llm=...)           — any LangChain BaseChatModel
  RAGConfig(custom_embedding=...)     — any LlamaIndex BaseEmbedding
  RAGConfig(custom_synthesis_llm=...) — override the RAG synthesis LLM
                                        independently of the chat LLM

When custom_llm is set, the provider= field is ignored entirely.
When both custom_embedding and custom_synthesis_llm are set, no
provider-specific setup runs at all — fully offline capable.
"""
import asyncio


# ── AWS Bedrock ────────────────────────────────────────────────────────────────
#
# Requires: pip install langchain-aws
# Auth:     AWS credentials in environment (AWS_ACCESS_KEY_ID, etc.)
#           or an IAM role on EC2/ECS/Lambda.

async def bedrock_example():
    from langchain_aws import ChatBedrock
    from kazi import Kazi, KaziConfig
    from kazi.core.config import LLMConfig, LLMProvider

    llm = ChatBedrock(
        model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
        region_name="us-east-1",
        # model_kwargs={"temperature": 0.1},
    )

    config = KaziConfig(
        llm=LLMConfig(
            provider=LLMProvider.OPENAI,  # ignored — custom_llm takes precedence
            custom_llm=llm,
        )
    )

    async with await Kazi.create(config) as kazi:
        result = await kazi.run("Summarise the key benefits of serverless architecture.")
        print(result)


# ── Google Vertex AI ───────────────────────────────────────────────────────────
#
# Requires: pip install langchain-google-vertexai
# Auth:     GOOGLE_APPLICATION_CREDENTIALS or gcloud auth application-default login

async def vertex_example():
    from langchain_google_vertexai import ChatVertexAI
    from kazi import Kazi, KaziConfig
    from kazi.core.config import LLMConfig, LLMProvider

    llm = ChatVertexAI(
        model="gemini-1.5-pro",
        project="my-gcp-project",
        location="us-central1",
        temperature=0.1,
    )

    config = KaziConfig(
        llm=LLMConfig(provider=LLMProvider.OPENAI, custom_llm=llm)
    )

    async with await Kazi.create(config) as kazi:
        result = await kazi.run("What is the capital of France?")
        print(result)


# ── Azure OpenAI ───────────────────────────────────────────────────────────────
#
# Requires: pip install langchain-openai
# Auth:     AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT environment variables

async def azure_example():
    import os
    from langchain_openai import AzureChatOpenAI
    from kazi import Kazi, KaziConfig
    from kazi.core.config import LLMConfig, LLMProvider

    llm = AzureChatOpenAI(
        azure_deployment="gpt-4o",
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version="2024-08-01-preview",
        temperature=0.1,
    )

    config = KaziConfig(
        llm=LLMConfig(provider=LLMProvider.OPENAI, custom_llm=llm)
    )

    async with await Kazi.create(config) as kazi:
        result = await kazi.run("Draft a one-paragraph product description for a B2B SaaS tool.")
        print(result)


# ── HuggingFace embeddings (offline RAG) ───────────────────────────────────────
#
# Requires: pip install llama-index-embeddings-huggingface sentence-transformers
# Auth:     None — model runs locally.

async def huggingface_embedding_example():
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from kazi import Kazi, KaziConfig
    from kazi.core.config import LLMConfig, LLMProvider, RAGConfig

    embed = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    config = KaziConfig(
        llm=LLMConfig(provider=LLMProvider.OPENAI),   # chat LLM uses OpenAI
        rag=RAGConfig(custom_embedding=embed),          # embeddings run locally
    )

    async with await Kazi.create(config) as kazi:
        await kazi.ingest_documents(
            [
                {"text": "Kazi is a production-grade AI orchestration library."},
                {"text": "It supports MCP, A2A, RAG, and custom LLM providers."},
            ],
            index_name="docs",
        )
        result = await kazi.run("What protocols does Kazi support?")
        print(result)


# ── Fully offline: custom embedding + custom synthesis LLM ────────────────────
#
# When both fields are set, no provider-specific setup runs — useful for
# air-gapped environments or CI pipelines that must never call an external API.
#
# Requires: pip install llama-index-embeddings-huggingface
#           pip install llama-index-llms-ollama  (or any local LlamaIndex LLM)

async def fully_offline_rag_example():
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.ollama import Ollama
    from langchain_ollama import ChatOllama
    from kazi import Kazi, KaziConfig
    from kazi.core.config import LLMConfig, LLMProvider, RAGConfig

    chat_llm = ChatOllama(model="llama3.2", base_url="http://localhost:11434")
    embed = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    synth_llm = Ollama(model="llama3.2", base_url="http://localhost:11434")

    config = KaziConfig(
        llm=LLMConfig(
            provider=LLMProvider.LOCAL,   # tells kazi to expect a local model
            custom_llm=chat_llm,          # but use this specific client
        ),
        rag=RAGConfig(
            custom_embedding=embed,       # embed locally
            custom_synthesis_llm=synth_llm,  # synthesise locally
        ),
    )

    async with await Kazi.create(config) as kazi:
        await kazi.ingest_documents(
            [{"text": "The answer to everything is forty-two."}],
            index_name="knowledge",
        )
        result = await kazi.run("What is the answer to everything?")
        print(result)


# ── Cohere ─────────────────────────────────────────────────────────────────────
#
# Requires: pip install langchain-cohere
# Auth:     COHERE_API_KEY environment variable

async def cohere_example():
    import os
    from langchain_cohere import ChatCohere
    from kazi import Kazi, KaziConfig
    from kazi.core.config import LLMConfig, LLMProvider

    llm = ChatCohere(
        model="command-r-plus",
        cohere_api_key=os.environ["COHERE_API_KEY"],
        temperature=0.1,
    )

    config = KaziConfig(
        llm=LLMConfig(provider=LLMProvider.OPENAI, custom_llm=llm)
    )

    async with await Kazi.create(config) as kazi:
        result = await kazi.run("List three advantages of retrieval-augmented generation.")
        print(result)


if __name__ == "__main__":
    # Run whichever example matches your credentials.
    # asyncio.run(bedrock_example())
    # asyncio.run(vertex_example())
    # asyncio.run(azure_example())
    # asyncio.run(huggingface_embedding_example())
    # asyncio.run(fully_offline_rag_example())
    # asyncio.run(cohere_example())
    print("Uncomment the example matching your provider and run again.")
