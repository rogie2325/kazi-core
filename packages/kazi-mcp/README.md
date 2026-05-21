# kazi-mcp

MCP (Model Context Protocol) client bridge and built-in tools for kazi. Connects any MCP server to the kazi tool registry automatically.

## Install

```bash
pip install kazi-mcp
pip install kazi-mcp[web-search]   # adds DuckDuckGo web search tool
pip install kazi-mcp[database]     # adds SQL query tool
```

## What's included

- MCP stdio and HTTP/SSE client — tools from connected servers appear in the registry automatically
- Built-in tools: `read_file`, `write_file`, `list_directory`, `web_search`, `sql_query`, `python_sandbox`
- MCP security policy — allowlist/denylist with glob patterns to control which MCP tools are registered

## Usage

```python
from kazi import KaziConfig, MCPConfig

config = KaziConfig(
    mcp=MCPConfig(servers={
        "filesystem": "npx -y @modelcontextprotocol/server-filesystem /tmp",
        "github": "npx -y @modelcontextprotocol/server-github",
    })
)
```

## License

MIT
