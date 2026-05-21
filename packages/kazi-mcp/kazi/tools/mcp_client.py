from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from kazi.core.config import MCPConfig, KaziConfig
from kazi.core.exceptions import MCPConnectionError, MCPTimeoutError, ToolArgValidationError
from kazi.core.registry import ToolDefinition, ToolParameter, ToolSource, ToolRegistry
from kazi.core.security import MCPSecurityPolicy

logger = logging.getLogger(__name__)


def _validate_args(tool: ToolDefinition, args: dict) -> None:
    """
    Validate that LLM-supplied args conform to the declared parameter schema.
    Raises ToolArgValidationError with a clear message if they don't.
    """
    required = {p.name for p in tool.parameters if p.required}
    missing = required - set(args.keys())
    if missing:
        raise ToolArgValidationError(
            f"Tool '{tool.name}' missing required args: {sorted(missing)}"
        )
    declared = {p.name for p in tool.parameters}
    unknown = set(args.keys()) - declared
    if unknown:
        # Unknown args are a warning, not a hard error — LLMs sometimes hallucinate extras
        logger.warning("Tool '%s' received undeclared args: %s — passing through", tool.name, unknown)


class MCPBridge:
    """
    Connects to MCP servers, discovers tools, and registers them in the
    shared ToolRegistry — subject to the MCPSecurityPolicy allowlist/denylist.
    """

    def __init__(
        self,
        config: MCPConfig,
        registry: ToolRegistry,
        security: Optional[MCPSecurityPolicy] = None,
        validate_args: bool = True,
    ) -> None:
        self.config = config
        self.registry = registry
        self.security = security or MCPSecurityPolicy()
        self.validate_args = validate_args or self.security.validate_args
        self._sessions: dict[str, Any] = {}
        self._server_tools: dict[str, list[str]] = {}

    async def connect_all(self) -> None:
        for name, uri in self.config.servers.items():
            try:
                await self._connect_server(name, uri)
            except Exception as exc:
                logger.error("Failed to connect to MCP server '%s': %s", name, exc)
                raise MCPConnectionError(f"Cannot connect to MCP server '{name}'") from exc

    async def _connect_server(self, name: str, uri: str) -> None:
        from mcp import ClientSession

        if uri.startswith(("http://", "https://")):
            session = await self._connect_sse(uri)
        else:
            session = await self._connect_stdio(uri)

        self._sessions[name] = session
        tools_response = await session.list_tools()
        registered: list[str] = []
        skipped: list[str] = []

        for mcp_tool in tools_response.tools:
            candidate_name = f"{name}__{mcp_tool.name}"

            # Apply security policy before registration
            if not self.security.is_allowed(candidate_name):
                skipped.append(candidate_name)
                continue

            tool_def = self._convert(mcp_tool, name)
            self.registry.register(tool_def, category=f"mcp_{name}")
            registered.append(tool_def.name)

        self._server_tools[name] = registered
        if skipped:
            logger.info(
                "MCP server '%s': %d registered, %d blocked by policy: %s",
                name, len(registered), len(skipped), skipped,
            )
        else:
            logger.info(
                "MCP server '%s' registered %d tools: %s", name, len(registered), registered
            )

    async def _connect_stdio(self, command: str) -> Any:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        parts = command.split()
        params = StdioServerParameters(command=parts[0], args=parts[1:])
        cm = stdio_client(params)
        read, write = await cm.__aenter__()
        session = ClientSession(read, write)
        await session.__aenter__()
        await session.initialize()
        return session

    async def _connect_sse(self, url: str) -> Any:
        from mcp import ClientSession
        from mcp.client.sse import sse_client

        cm = sse_client(url)
        read, write = await cm.__aenter__()
        session = ClientSession(read, write)
        await session.__aenter__()
        await session.initialize()
        return session

    def _convert(self, mcp_tool: Any, server_name: str) -> ToolDefinition:
        params: list[ToolParameter] = []
        schema = getattr(mcp_tool, "inputSchema", None) or {}
        properties = schema.get("properties", {})
        required_set = set(schema.get("required", []))

        for pname, pschema in properties.items():
            params.append(ToolParameter(
                name=pname,
                type=pschema.get("type", "string"),
                description=pschema.get("description", f"Parameter: {pname}"),
                required=pname in required_set,
                default=pschema.get("default"),
                enum=pschema.get("enum"),
            ))

        orig_name = mcp_tool.name
        sname = server_name
        validate = self.validate_args

        async def handler(**kwargs):
            if validate:
                _validate_args(tool_def, kwargs)
            return await self.execute_tool(sname, orig_name, kwargs)

        tool_def = ToolDefinition(
            name=f"{server_name}__{orig_name}",
            description=mcp_tool.description or f"MCP tool: {orig_name}",
            parameters=params,
            source=ToolSource.MCP,
            handler=handler,
            mcp_server=server_name,
            metadata={"original_name": orig_name, "server": server_name},
        )
        return tool_def

    async def execute_tool(self, server_name: str, tool_name: str, arguments: dict) -> str:
        if server_name not in self._sessions:
            raise MCPConnectionError(f"Not connected to MCP server: {server_name}")

        session = self._sessions[server_name]
        try:
            result = await asyncio.wait_for(
                session.call_tool(tool_name, arguments=arguments),
                timeout=self.config.timeout,
            )
        except asyncio.TimeoutError:
            raise MCPTimeoutError(
                f"MCP tool '{tool_name}' on '{server_name}' timed out after {self.config.timeout}s"
            )

        if hasattr(result, "content"):
            texts = [b.text for b in result.content if hasattr(b, "text")]
            return "\n".join(texts) if texts else str(result)
        return str(result)

    async def disconnect_all(self) -> None:
        for name, session in self._sessions.items():
            try:
                await session.__aexit__(None, None, None)
            except Exception as exc:
                logger.warning("Error disconnecting from MCP server '%s': %s", name, exc)
        self._sessions.clear()

    def list_servers(self) -> list[str]:
        return list(self._sessions.keys())

    def tools_for_server(self, server_name: str) -> list[str]:
        return self._server_tools.get(server_name, [])
