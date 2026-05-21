#!/usr/bin/env python3
"""
Minimal stdio MCP server used as a real test fixture.

Exposes two tools:
  echo  — returns the input text unchanged
  add   — adds two integers and returns the result
"""
import asyncio

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

server = Server("kazi-test-echo-server")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="echo",
            description="Echo the input text back unchanged.",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to echo"},
                },
                "required": ["text"],
            },
        ),
        types.Tool(
            name="add",
            description="Add two integers and return the sum.",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "First operand"},
                    "b": {"type": "integer", "description": "Second operand"},
                },
                "required": ["a", "b"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "echo":
        return [types.TextContent(type="text", text=arguments["text"])]
    if name == "add":
        result = int(arguments["a"]) + int(arguments["b"])
        return [types.TextContent(type="text", text=str(result))]
    raise ValueError(f"Unknown tool: {name!r}")


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
