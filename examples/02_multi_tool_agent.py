"""
Example 2: Multi-tool agent

Register custom Python tools, built-in tools, and an MCP server.
The LLM picks the right tool for each sub-task automatically.
"""
import asyncio
from kazi import (
    Kazi,
    KaziConfig,
    LLMConfig,
    MCPConfig,
    LLMProvider,
    web_search_tool,
    python_sandbox_tool,
)


# --- Custom tool (just a Python function) ---

async def get_stock_price(ticker: str) -> str:
    """Get the current stock price for a ticker symbol."""
    # In a real app this would call a financial API
    mock_prices = {"AAPL": 189.50, "GOOGL": 175.20, "MSFT": 415.80}
    price = mock_prices.get(ticker.upper(), "unknown")
    return f"{ticker.upper()}: ${price}" if price != "unknown" else f"Unknown ticker: {ticker}"


async def main():
    config = KaziConfig(
        llm=LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4o"),
        mcp=MCPConfig(
            servers={
                # Uncomment to connect a real MCP server:
                # "filesystem": "npx -y @modelcontextprotocol/server-filesystem /tmp",
            }
        ),
    )

    async with await Kazi.create(config) as kazi:
        # Register built-in tools
        kazi.registry.register(web_search_tool(), category="search")
        kazi.registry.register(python_sandbox_tool(timeout=15), category="compute")

        # Register the custom stock tool — auto-extracts parameters from type hints
        kazi.add_tool(get_stock_price, description="Get the current stock price for a ticker")

        # The LLM decides which tools to use
        result = await kazi.run(
            "What is the current stock price of Apple? "
            "Then write Python code to calculate what $10,000 invested would be worth "
            "if the stock grows 15% annually for 5 years."
        )
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
