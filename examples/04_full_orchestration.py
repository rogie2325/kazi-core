"""
Example 4: Full orchestration — all four layers active

RAG + MCP tools + A2A agents + persistent memory + streaming output.
This is what a production deployment looks like.
"""
import asyncio
from kazi import (
    Kazi,
    KaziConfig,
    LLMConfig,
    RAGConfig,
    MCPConfig,
    A2AConfig,
    MemoryConfig,
    LLMProvider,
    MemoryBackend,
    read_file_tool,
    web_search_tool,
    python_sandbox_tool,
    configure_logging,
)


async def main():
    configure_logging(level="INFO")

    config = KaziConfig(
        llm=LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            model="claude-sonnet-4-6",
            temperature=0.1,
        ),
        rag=RAGConfig(
            persist_dir="./kazi_index",
            similarity_top_k=5,
            chunk_size=1024,
        ),
        mcp=MCPConfig(
            servers={
                # "github": "npx -y @modelcontextprotocol/server-github",
                # "postgres": "npx -y @modelcontextprotocol/server-postgres postgresql://...",
            },
            timeout=30,
        ),
        a2a=A2AConfig(
            discovery_endpoints=[
                # "http://localhost:8001",  # specialist agents
            ],
            delegation_timeout=120,
        ),
        memory=MemoryConfig(
            backend=MemoryBackend.SQLITE,
            connection_string="sqlite:///kazi_memory.db",
        ),
        verbose=True,
    )

    async with await Kazi.create(config) as kazi:
        # --- Ingest knowledge ---
        # await kazi.ingest("./company_docs", index_name="company")
        # await kazi.ingest("./product_docs", index_name="product")

        # --- Add tools ---
        kazi.registry.register(read_file_tool())
        kazi.registry.register(web_search_tool())
        kazi.registry.register(python_sandbox_tool(timeout=15))

        print(f"Registry: {len(kazi.registry)} tools across {kazi.registry.list_categories()}")

        # --- Streaming response ---
        print("\n--- Streaming response ---")
        async for token in kazi.stream(
            "Explain the difference between MCP and A2A protocols in 3 bullet points.",
            thread_id="session-prod-001",
        ):
            print(token, end="", flush=True)
        print()

        # --- Multi-turn with persistent memory ---
        r1 = await kazi.run(
            "My name is Alex and I'm building an AI orchestration platform.",
            thread_id="session-prod-001",
        )
        print(f"\nTurn 1: {r1}")

        r2 = await kazi.run(
            "What did I just tell you about myself?",
            thread_id="session-prod-001",
        )
        print(f"\nTurn 2 (tests memory): {r2}")


if __name__ == "__main__":
    asyncio.run(main())
