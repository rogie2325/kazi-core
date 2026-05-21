from __future__ import annotations

import logging
from typing import AsyncIterator, Optional

from kazi.core.config import KaziConfig
from kazi.core.registry import ToolRegistry

logger = logging.getLogger(__name__)


class Kazi:
    """
    Top-level orchestrator that wires all four layers together.

    Usage::

        kazi = await Kazi.create(config)
        response = await kazi.run("Summarise the Q3 report")
        await kazi.close()

    Or as an async context manager (recommended — guarantees cleanup)::

        async with await Kazi.create(config) as kazi:
            response = await kazi.run("...")

    Thread authentication (optional)::

        config = KaziConfig(
            security=SecurityConfig(
                threads=ThreadPolicy(
                    require_auth=True,
                    validator=lambda thread_id, token: verify_jwt(token, thread_id),
                )
            )
        )
        # now every run() call must pass user_token=
        await kazi.run("...", thread_id="user:123:session:abc", user_token=jwt)
    """

    def __init__(self, config: KaziConfig) -> None:
        self.config = config
        self.registry = ToolRegistry()
        self._brain = None
        self._data = None
        self._mcp = None
        self._a2a = None
        self._ready = False

    # ── factory ───────────────────────────────────────────────────────────

    @classmethod
    async def create(cls, config: Optional[KaziConfig] = None) -> "Kazi":
        inst = cls(config or KaziConfig())
        await inst._startup()
        return inst

    async def _startup(self) -> None:
        security = self.config.security

        # 1. Data layer — no I/O yet
        from kazi.data.index_manager import IndexManager
        self._data = IndexManager(self.config.rag, self.config.llm)

        # 2. MCP bridge — pass security policy so the bridge can filter tools
        if self.config.mcp.servers:
            from kazi.tools.mcp_client import MCPBridge
            self._mcp = MCPBridge(
                self.config.mcp,
                self.registry,
                security=security.mcp,
                validate_args=security.mcp.validate_args,
            )
            await self._mcp.connect_all()

        # 3. A2A bridge — pass full security config for TLS + content tagging
        if self.config.a2a.discovery_endpoints:
            from kazi.agents.a2a_client import A2ABridge
            self._a2a = A2ABridge(self.config.a2a, self.registry, security=security)
            await self._a2a.discover_agents()

        # 4. Brain — built last so it sees the full registry
        from kazi.brain.graph_builder import GraphBrain
        self._brain = GraphBrain(self.config, self.registry)

        self._ready = True
        logger.info(
            "Kazi ready — %d tools registered across categories: %s",
            len(self.registry),
            self.registry.list_categories(),
        )

    # ── document ingestion ────────────────────────────────────────────────

    async def ingest(
        self,
        path: str,
        index_name: str = "default",
        *,
        tool_name: Optional[str] = None,
        tool_description: Optional[str] = None,
    ) -> None:
        self._assert_ready()
        await self._data.ingest_directory(path, index_name=index_name)
        tool_def = self._data.as_tool_definition(
            index_name, tool_name=tool_name, description=tool_description,
        )
        if tool_def.name in self.registry:
            self.registry.unregister(tool_def.name)
        self.registry.register(tool_def, category="rag")

    async def ingest_documents(
        self,
        documents: list[dict],
        index_name: str = "default",
        *,
        tool_name: Optional[str] = None,
        tool_description: Optional[str] = None,
    ) -> None:
        self._assert_ready()
        await self._data.ingest_documents(documents, index_name=index_name)
        tool_def = self._data.as_tool_definition(
            index_name, tool_name=tool_name, description=tool_description,
        )
        if tool_def.name in self.registry:
            self.registry.unregister(tool_def.name)
        self.registry.register(tool_def, category="rag")

    # ── tool management ───────────────────────────────────────────────────

    def add_tool(self, func, *, name=None, description=None, category="custom") -> None:
        """Register a Python function as a tool available to the LLM."""
        self.registry.register_function(
            func, name=name, description=description, category=category
        )

    # ── inference ─────────────────────────────────────────────────────────

    async def run(
        self,
        message: str,
        *,
        thread_id: str = "default",
        max_tool_calls: int = 25,
        system_prompt: Optional[str] = None,
        user_token: Optional[str] = None,
    ) -> str:
        """
        Run a conversation turn.

        Parameters
        ──────────
        message        The user's input.
        thread_id      Conversation thread identifier. Calls with the same
                       thread_id share memory across runs.
        max_tool_calls Hard cap on tool iterations for this run.
        system_prompt  Override the auto-generated system prompt for this run.
        user_token     Required when SecurityConfig.threads.require_auth=True.
                       Passed to ThreadPolicy.validator(thread_id, user_token).
        """
        self._assert_ready()

        # Thread authentication gate
        self.config.security.threads.check(thread_id, user_token)

        return await self._brain.run(
            message,
            thread_id=thread_id,
            max_tool_calls=max_tool_calls,
            system_prompt=system_prompt,
        )

    async def stream(
        self,
        message: str,
        *,
        thread_id: str = "default",
        max_tool_calls: int = 25,
        user_token: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream tokens as they arrive from the LLM."""
        self._assert_ready()
        self.config.security.threads.check(thread_id, user_token)
        async for chunk in self._brain.stream(
            message, thread_id=thread_id, max_tool_calls=max_tool_calls
        ):
            yield chunk

    # ── lifecycle ─────────────────────────────────────────────────────────

    async def close(self) -> None:
        if self._mcp:
            await self._mcp.disconnect_all()
        if self._a2a:
            await self._a2a.close()
        self._ready = False

    async def __aenter__(self) -> "Kazi":
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()

    # ── layer access ─────────────────────────────────────────────────────

    @property
    def data(self):
        return self._data

    @property
    def mcp(self):
        return self._mcp

    @property
    def a2a(self):
        return self._a2a

    # ── helpers ───────────────────────────────────────────────────────────

    def _assert_ready(self) -> None:
        if not self._ready:
            raise RuntimeError(
                "Kazi is not initialised. Use `await Kazi.create(config)` "
                "instead of calling Kazi() directly."
            )
