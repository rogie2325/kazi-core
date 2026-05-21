"""
Example 3: Supervisor with A2A agent delegation

A supervisor agent delegates sub-tasks to specialised remote agents
via the A2A protocol. Each specialist agent exposes an Agent Card
at /.well-known/agent.json.
"""
import asyncio
from kazi import Kazi, KaziConfig, LLMConfig, A2AConfig, LLMProvider


async def main():
    config = KaziConfig(
        llm=LLMConfig(provider=LLMProvider.ANTHROPIC, model="claude-sonnet-4-6"),
        a2a=A2AConfig(
            discovery_endpoints=[
                # Remote specialist agents — each one exposes an Agent Card
                "http://localhost:8001",  # research-agent
                "http://localhost:8002",  # writing-agent
                "http://localhost:8003",  # data-analysis-agent
            ],
            delegation_timeout=60,
        ),
    )

    async with await Kazi.create(config) as kazi:
        # After create(), all discovered agent skills are in the registry
        # The supervisor LLM automatically knows about them

        print(f"Discovered {len(kazi.a2a.list_agents())} remote agents:")
        for agent in kazi.a2a.list_agents():
            print(f"  - {agent.name}: {len(agent.skills)} skills")

        # The supervisor decomposes this into sub-tasks and delegates
        result = await kazi.run(
            "Research the current state of AI agent frameworks, "
            "analyze which ones are growing fastest, "
            "and write a 500-word executive summary."
        )
        print("\n--- RESULT ---")
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
