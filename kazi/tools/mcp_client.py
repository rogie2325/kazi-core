from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

from kazi.core.config import MCPConfig
from kazi.core.exceptions import MCPConnectionError, MCPTimeoutError, ToolArgValidationError
from kazi.core.registry import ToolDefinition, ToolParameter, ToolRegistry, ToolSource
from kazi.core.security import MCPSecurityPolicy

logger = logging.getLogger(__name__)

_STOP = object()  # sentinel to break the call loop in _ServerHandle


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
        logger.warning("Tool '%s' received undeclared args: %s — passing through", tool.name, unknown)


class _ServerHandle:
    """
    Runs a single MCP server connection entirely inside one asyncio Task so that
    anyio cancel scopes are never crossed between tasks.  Callers communicate
    with the background task via an asyncio Queue.
    """

    def __init__(self, name: str, command: str, config: MCPConfig) -> None:
        self.name = name
        self.command = command
        self.config = config
        self._queue: asyncio.Queue = asyncio.Queue()
        self._ready_event: asyncio.Event = asyncio.Event()
        self._task: asyncio.Task | None = None
        self.tools: Any = None
        self._start_error: BaseException | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name=f"mcp-{self.name}")
        await self._ready_event.wait()
        if self._start_error is not None:
            raise self._start_error

    async def _run(self) -> None:
        try:
            if self.command.startswith(("http://", "https://")):
                await self._run_sse()
            else:
                await self._run_stdio()
        except Exception as exc:
            self._start_error = exc
            self._ready_event.set()
            self._drain_queue(exc)

    async def _run_stdio(self) -> None:
        import shlex

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        parts = shlex.split(self.command)
        params = StdioServerParameters(command=parts[0], args=parts[1:])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                self.tools = await session.list_tools()
                self._ready_event.set()
                await self._process_calls(session)

    async def _run_sse(self) -> None:
        from mcp import ClientSession
        from mcp.client.sse import sse_client

        async with sse_client(self.command) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                self.tools = await session.list_tools()
                self._ready_event.set()
                await self._process_calls(session)

    async def _process_calls(self, session: Any) -> None:
        while True:
            item = await self._queue.get()
            if item is _STOP:
                break
            future: asyncio.Future = item["future"]
            try:
                result = await session.call_tool(item["name"], arguments=item["args"])
                if not future.done():
                    future.set_result(result)
            except Exception as exc:
                if not future.done():
                    future.set_exception(exc)

    def _drain_queue(self, exc: BaseException) -> None:
        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item is not _STOP and not item["future"].done():
                item["future"].set_exception(MCPConnectionError("Server disconnected"))

    async def call(self, name: str, args: dict) -> Any:
        if self._task is None or self._task.done():
            raise MCPConnectionError(f"MCP server '{self.name}' is not connected")
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        await self._queue.put({"name": name, "args": args, "future": future})
        try:
            return await asyncio.wait_for(
                asyncio.shield(future), timeout=self.config.timeout
            )
        except asyncio.TimeoutError as exc:
            # Shield protects the future so the background task can still settle it
            raise MCPTimeoutError(
                f"MCP tool '{name}' on '{self.name}' timed out after {self.config.timeout}s"
            ) from exc

    async def stop(self) -> None:
        await self._queue.put(_STOP)
        if self._task and not self._task.done():
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except asyncio.TimeoutError:
                self._task.cancel()
                try:
                    await self._task
                except (asyncio.CancelledError, Exception):
                    pass


class MCPBridge:
    """
    Connects to MCP servers, discovers tools, and registers them in the
    shared ToolRegistry — subject to the MCPSecurityPolicy allowlist/denylist.
    """

    def __init__(
        self,
        config: MCPConfig,
        registry: ToolRegistry,
        security: MCPSecurityPolicy | None = None,
        validate_args: bool = True,
    ) -> None:
        self.config = config
        self.registry = registry
        self.security = security or MCPSecurityPolicy()
        self.validate_args = validate_args or self.security.validate_args
        self._handles: dict[str, _ServerHandle] = {}
        self._server_tools: dict[str, list[str]] = {}

    async def connect_all(self) -> None:
        for name, uri in self.config.servers.items():
            try:
                await self._connect_server(name, uri)
            except Exception as exc:
                logger.error("Failed to connect to MCP server '%s': %s", name, exc)
                raise MCPConnectionError(f"Cannot connect to MCP server '{name}'") from exc

    async def _connect_server(self, name: str, uri: str) -> None:
        handle = _ServerHandle(name, uri, self.config)
        await handle.start()

        self._handles[name] = handle
        registered: list[str] = []
        skipped: list[str] = []

        for mcp_tool in handle.tools.tools:
            candidate_name = f"{name}__{mcp_tool.name}"
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
        if server_name not in self._handles:
            raise MCPConnectionError(f"Not connected to MCP server: {server_name}")

        handle = self._handles[server_name]
        max_attempts = self.config.max_retries + 1
        last_exc: Exception = RuntimeError("unreachable")

        for attempt in range(max_attempts):
            try:
                result = await handle.call(tool_name, arguments)
                if hasattr(result, "content"):
                    texts = [b.text for b in result.content if hasattr(b, "text")]
                    return "\n".join(texts) if texts else str(result)
                return str(result)

            except MCPTimeoutError:
                raise  # timeouts are not retried
            except Exception as exc:
                last_exc = exc
                if attempt < max_attempts - 1:
                    delay = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(
                        "MCP tool '%s/%s' failed (attempt %d/%d): %s — retrying in %.1fs",
                        server_name, tool_name, attempt + 1, max_attempts, exc, delay,
                    )
                    await asyncio.sleep(delay)

        raise MCPConnectionError(
            f"MCP tool '{tool_name}' on '{server_name}' failed after {max_attempts} attempt(s)"
        ) from last_exc

    async def disconnect_all(self) -> None:
        for name, handle in list(self._handles.items()):
            try:
                await handle.stop()
            except Exception as exc:
                logger.warning("Error disconnecting from MCP server '%s': %s", name, exc)
        self._handles.clear()

    def list_servers(self) -> list[str]:
        return list(self._handles.keys())

    def tools_for_server(self, server_name: str) -> list[str]:
        return self._server_tools.get(server_name, [])
