from __future__ import annotations


class KaziError(Exception):
    """Base exception for all Kazi errors."""


class ConfigurationError(KaziError):
    """Invalid or missing configuration."""


class ToolNotFoundError(KaziError):
    """Named tool does not exist in the registry."""


class ToolExecutionError(KaziError):
    """A tool raised an error during execution."""

    def __init__(self, tool_name: str, cause: Exception):
        self.tool_name = tool_name
        self.cause = cause
        super().__init__(f"Tool '{tool_name}' failed: {cause}")


class ToolConflictError(KaziError):
    """A tool with this name is already registered."""


class IndexNotFoundError(KaziError):
    """Named LlamaIndex index does not exist."""


class MCPConnectionError(KaziError):
    """Failed to connect to an MCP server."""


class MCPTimeoutError(KaziError):
    """MCP tool call timed out."""


class A2AConnectionError(KaziError):
    """Failed to reach a remote A2A agent."""


class A2ATimeoutError(KaziError):
    """A2A task delegation timed out."""


class AgentNotFoundError(KaziError):
    """Named remote agent has not been discovered."""


class OrchestratorError(KaziError):
    """Top-level orchestration failure."""


class MaxToolCallsExceeded(KaziError):
    """The graph hit its tool-call budget before finishing."""


# ── Security ──────────────────────────────────────────────────────────────────

class SecurityError(KaziError):
    """Base class for security-related failures."""


class ThreadAuthError(SecurityError):
    """Access to a thread was denied — missing or invalid user token."""


class ToolBlockedError(SecurityError):
    """A tool call or result was blocked by a ContentPolicy hook."""


class ToolArgValidationError(SecurityError):
    """Tool arguments supplied by the LLM failed schema validation."""


# ── Token budget ──────────────────────────────────────────────────────────────

class TokenBudgetExceeded(KaziError):
    """A run exceeded its configured token budget."""
