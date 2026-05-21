"""Built-in web search tool (DuckDuckGo, no API key required)."""
from __future__ import annotations

from kazi.core.registry import ToolDefinition, ToolParameter, ToolSource

_MAX_RESULTS = 20  # hard cap — prevents accidental server overload


async def _duckduckgo_search(query: str, max_results: int = 5) -> str:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return "duckduckgo-search package not installed. Run: pip install duckduckgo-search"

    max_results = max(1, min(max_results, _MAX_RESULTS))

    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(f"**{r['title']}**\n{r['href']}\n{r['body']}")

    return "\n\n".join(results) if results else "No results found."


def web_search_tool() -> ToolDefinition:
    """Return a ToolDefinition for DuckDuckGo web search."""
    return ToolDefinition(
        name="web_search",
        description="Search the web for current information using DuckDuckGo.",
        parameters=[
            ToolParameter(
                name="query",
                type="string",
                description="The search query",
                required=True,
            ),
            ToolParameter(
                name="max_results",
                type="integer",
                description=f"Max results to return (default 5, max {_MAX_RESULTS})",
                required=False,
                default=5,
            ),
        ],
        source=ToolSource.NATIVE,
        handler=_duckduckgo_search,
        metadata={"provider": "duckduckgo"},
    )
