"""
Example 7: Sub-agents and Supervisor pattern

Build a multi-agent system where specialized sub-agents handle different domains
and a Supervisor intelligently routes user requests between them.

Each sub-agent has:
  - Its own personality (system_prompt)
  - Its own tool set (restricted from the global registry)
  - Its own memory namespace (scoped thread_id)
  - The same underlying LLM (or a different one via llm_override)

Cross-model routing: agents can use different LLM providers while maintaining
consistent identity via their system_prompt injection on every turn.
"""
import asyncio

from kazi import (
    Kazi,
    KaziConfig,
    LLMConfig,
    LLMProvider,
    web_search_tool,
    python_sandbox_tool,
    sql_query_tool,
)
from kazi.agents.subagent import SubAgent, SubAgentConfig
from kazi.agents.supervisor import Supervisor


async def main():
    config = KaziConfig(
        llm=LLMConfig(provider=LLMProvider.ANTHROPIC, model="claude-sonnet-4-6"),
    )

    async with await Kazi.create(config) as kazi:
        # Register tools globally — sub-agents will each see a subset
        kazi.registry.register(web_search_tool(), category="search")
        kazi.registry.register(python_sandbox_tool(timeout=15), category="compute")
        # kazi.registry.register(sql_query_tool("postgresql://..."), category="data")

        # ── Define specialized sub-agents ──────────────────────────────────

        research_agent = SubAgent(
            kazi,
            SubAgentConfig(
                name="research",
                role="Research Specialist",
                system_prompt=(
                    "You are a focused research assistant. Your job is to find accurate, "
                    "up-to-date information using web search. Always cite your sources. "
                    "Be concise — bullet points preferred over paragraphs."
                ),
                tools=["web_search"],  # can only see web_search
            ),
        )

        analyst_agent = SubAgent(
            kazi,
            SubAgentConfig(
                name="analyst",
                role="Data Analyst",
                system_prompt=(
                    "You are a data analyst. You write and run Python code to process data, "
                    "compute statistics, and generate insights. Always show your working code."
                ),
                tools=["python_sandbox"],  # can only see python_sandbox
            ),
        )

        # Cross-model: analyst uses GPT-4o for code generation, primary uses Claude
        from kazi.core.router import ModelRoute
        analyst_agent_gpt = SubAgent(
            kazi,
            SubAgentConfig(
                name="analyst_gpt",
                role="Data Analyst (GPT-4o)",
                system_prompt="You are a precise data analyst. Write clean, commented Python code.",
                tools=["python_sandbox"],
                llm_override=ModelRoute(model="gpt-4o", provider="openai"),
            ),
        )

        # ── Option A: Route manually ───────────────────────────────────────
        print("=== Manual routing ===")

        research_result = await research_agent.run(
            "What are the top 3 AI agent frameworks in 2025?",
            thread_id="project-alpha",
        )
        print(f"Research: {research_result[:200]}…\n")

        analysis_result = await analyst_agent.run(
            "Calculate compound interest: $10,000 at 8% annually for 10 years",
            thread_id="project-alpha",  # shares memory with research_agent turn
        )
        print(f"Analysis: {analysis_result[:200]}…\n")

        # ── Option B: Supervisor auto-routes ──────────────────────────────
        print("=== Supervisor routing ===")

        supervisor = Supervisor(
            kazi,
            agents=[research_agent, analyst_agent],
        )

        # Supervisor reads the query and picks the right agent automatically
        tasks = [
            "Search for the latest news about LangGraph",
            "Write Python code to calculate the Fibonacci sequence up to 100",
            "What are the key differences between RAG and fine-tuning?",
        ]

        for task in tasks:
            result = await supervisor.run(task, thread_id="supervisor-session")
            print(f"Q: {task}")
            print(f"A: {result[:150]}…\n")


if __name__ == "__main__":
    asyncio.run(main())
