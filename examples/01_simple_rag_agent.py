"""
Example 1: Simple RAG agent

Ingest a directory of documents and chat with them.
20 lines of developer code, full production pipeline underneath.
"""
import asyncio
from kazi import Kazi, KaziConfig, LLMConfig, LLMProvider


async def main():
    config = KaziConfig(
        llm=LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4o",
        )
    )

    async with await Kazi.create(config) as kazi:
        # Ingest documents — automatically indexed and registered as a search tool
        await kazi.ingest("./docs", index_name="company_docs")

        # Ask questions — the LLM will automatically use the search tool
        answer = await kazi.run("What are the main topics covered in the documentation?")
        print(answer)

        # Multi-turn conversation on the same thread
        followup = await kazi.run(
            "Can you give me a 3-point summary?",
            thread_id="session-001",
        )
        print(followup)


if __name__ == "__main__":
    asyncio.run(main())
