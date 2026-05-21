"""
kazi — Production-grade AI orchestration in 20 lines.

Wires together LangGraph (stateful execution), LlamaIndex (RAG),
MCP (tool protocols), and A2A (agent delegation) behind a single
clean API.

Quick start::

    import asyncio
    from kazi import Kazi, KaziConfig, LLMConfig, LLMProvider

    config = KaziConfig(llm=LLMConfig(provider=LLMProvider.ANTHROPIC, model="claude-sonnet-4-6"))

    async def main():
        async with await Kazi.create(config) as kazi:
            await kazi.ingest("./docs")
            print(await kazi.run("Summarise the key points from the docs"))

    asyncio.run(main())
"""

from kazi.core.orchestrator import Kazi
from kazi.core.config import (
    KaziConfig,
    LLMConfig,
    RAGConfig,
    MCPConfig,
    A2AConfig,
    MemoryConfig,
    LLMProvider,
    VectorStoreBackend,
    MemoryBackend,
)
from kazi.core.registry import ToolRegistry, ToolDefinition, ToolParameter, ToolSource
from kazi.core.exceptions import (
    KaziError,
    ConfigurationError,
    ToolNotFoundError,
    ToolExecutionError,
    ToolConflictError,
    MCPConnectionError,
    A2AConnectionError,
    OrchestratorError,
)
from kazi.tools.builtin import (
    web_search_tool,
    read_file_tool,
    write_file_tool,
    list_directory_tool,
    sql_query_tool,
)
from kazi.tools.sandbox import python_sandbox_tool
from kazi.tools.tool_adapter import from_openai_schema, from_anthropic_schema, from_langchain_tool
from kazi.agents.agent_card import AgentCard
from kazi.data.ingest import ingest_strings, ingest_web_pages
from kazi.utils.logging import configure_logging

__version__ = "0.1.1"

__all__ = [
    # Core
    "Kazi",
    "KaziConfig",
    "LLMConfig",
    "RAGConfig",
    "MCPConfig",
    "A2AConfig",
    "MemoryConfig",
    "LLMProvider",
    "VectorStoreBackend",
    "MemoryBackend",
    # Registry
    "ToolRegistry",
    "ToolDefinition",
    "ToolParameter",
    "ToolSource",
    # Exceptions
    "KaziError",
    "ConfigurationError",
    "ToolNotFoundError",
    "ToolExecutionError",
    "ToolConflictError",
    "MCPConnectionError",
    "A2AConnectionError",
    "OrchestratorError",
    # Built-in tools
    "web_search_tool",
    "read_file_tool",
    "write_file_tool",
    "list_directory_tool",
    "sql_query_tool",
    "python_sandbox_tool",
    # Adapters
    "from_openai_schema",
    "from_anthropic_schema",
    "from_langchain_tool",
    # Agents
    "AgentCard",
    # Data
    "ingest_strings",
    "ingest_web_pages",
    # Utils
    "configure_logging",
]
