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

from kazi.agents.agent_card import AgentCard
from kazi.agents.subagent import SubAgent, SubAgentConfig
from kazi.agents.supervisor import Supervisor
from kazi.cache.semantic import SemanticCache, SemanticCacheConfig
from kazi.core.audit import RunAudit, RunAuditResult, ToolCallRecord
from kazi.core.config import (
    A2AConfig,
    KaziConfig,
    LLMConfig,
    LLMProvider,
    MCPConfig,
    MemoryBackend,
    MemoryConfig,
    RAGConfig,
    STTProvider,
    TTSProvider,
    VectorStoreBackend,
    VoiceConfig,
)
from kazi.core.cost import CostReport, RunCost, RunResult, TenantCostLedger
from kazi.core.events import StreamEvent
from kazi.core.exceptions import (
    A2AConnectionError,
    BudgetExceededError,
    ConfigurationError,
    GuardrailViolationError,
    InjectionDetectedError,
    KaziError,
    MCPConnectionError,
    OrchestratorError,
    ToolConflictError,
    ToolExecutionError,
    ToolNotFoundError,
)
from kazi.core.guardrails import GuardrailConfig, GuardrailResult, check_output
from kazi.core.orchestrator import Kazi
from kazi.core.registry import ToolDefinition, ToolParameter, ToolRegistry, ToolSource
from kazi.core.router import ModelRoute, RouterConfig
from kazi.core.security import InjectionDetectionConfig
from kazi.data.ingest import ingest_strings, ingest_web_pages
from kazi.memory.profile import UserProfile
from kazi.queue.webhook import WebhookConfig, dispatch_webhook
from kazi.tools.builtin import (
    data_query_tool,
    data_summary_tool,
    list_directory_tool,
    read_file_tool,
    sql_query_tool,
    web_search_tool,
    write_file_tool,
)
from kazi.tools.sandbox import python_sandbox_tool
from kazi.tools.tool_adapter import from_anthropic_schema, from_langchain_tool, from_openai_schema
from kazi.utils.experiment import ExperimentTracker
from kazi.utils.logging import configure_logging

__version__ = "0.1.0"

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
    "ModelRoute",
    "RouterConfig",
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
    "data_query_tool",
    "data_summary_tool",
    "python_sandbox_tool",
    # Adapters
    "from_openai_schema",
    "from_anthropic_schema",
    "from_langchain_tool",
    # Agents
    "AgentCard",
    "SubAgent",
    "SubAgentConfig",
    "Supervisor",
    # Voice
    "VoiceConfig",
    "STTProvider",
    "TTSProvider",
    # Cost tracking
    "RunCost",
    "RunResult",
    "CostReport",
    "TenantCostLedger",
    # Audit / shadow
    "RunAudit",
    "RunAuditResult",
    "ToolCallRecord",
    # Data
    "ingest_strings",
    "ingest_web_pages",
    # Memory
    "UserProfile",
    # Utils
    "configure_logging",
    "ExperimentTracker",
    # Typed events
    "StreamEvent",
    # Guardrails
    "GuardrailConfig",
    "GuardrailResult",
    "check_output",
    # Semantic cache
    "SemanticCache",
    "SemanticCacheConfig",
    # Webhook
    "WebhookConfig",
    "dispatch_webhook",
    # Security extras
    "InjectionDetectionConfig",
    # New exceptions
    "GuardrailViolationError",
    "BudgetExceededError",
    "InjectionDetectedError",
]
