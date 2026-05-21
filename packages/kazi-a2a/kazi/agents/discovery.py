"""
Agent discovery utilities — scan networks or registries for A2A agents.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

import httpx

from kazi.agents.agent_card import AgentCard

logger = logging.getLogger(__name__)


async def discover_from_urls(urls: list[str], timeout: int = 10) -> list[AgentCard]:
    """
    Attempt to fetch Agent Cards from a list of base URLs.
    Silently skips URLs that don't respond or don't have a valid card.
    """
    cards: list[AgentCard] = []

    async def _try(url: str) -> Optional[AgentCard]:
        card_url = url.rstrip("/") + "/.well-known/agent.json"
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(card_url)
                if resp.status_code == 200:
                    return AgentCard.from_dict(resp.json(), url)
        except Exception as exc:
            logger.debug("No agent card at %s: %s", card_url, exc)
        return None

    results = await asyncio.gather(*[_try(u) for u in urls], return_exceptions=False)
    cards = [r for r in results if r is not None]
    logger.info("Discovered %d agent(s) from %d URL(s)", len(cards), len(urls))
    return cards


async def scan_localhost(
    ports: Optional[list[int]] = None,
    timeout: float = 1.0,
) -> list[AgentCard]:
    """
    Scan localhost ports for running A2A agents. Useful during development.

    Default port range: 8000–8020.
    """
    scan_ports = ports or list(range(8000, 8021))
    urls = [f"http://localhost:{p}" for p in scan_ports]
    return await discover_from_urls(urls, timeout=int(timeout))
