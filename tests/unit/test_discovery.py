"""Tests for kazi.agents.discovery — URL-based and localhost scanning."""
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from kazi.agents.agent_card import AgentCard
from kazi.agents.discovery import discover_from_urls, scan_localhost

MOCK_CARD = {
    "name": "test-agent",
    "description": "A test agent",
    "version": "1.0",
    "skills": [
        {
            "name": "do_thing",
            "description": "Does a thing",
            "input_schema": {"properties": {"x": {"type": "string"}}, "required": ["x"]},
        }
    ],
}


def _make_mock_client(status: int = 200, json_data: dict | None = None):
    """Return a mock httpx.AsyncClient that responds with the given status and body."""
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = json_data or {}

    client = AsyncMock()
    client.get = AsyncMock(return_value=resp)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client


# ── discover_from_urls ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_discovers_single_valid_url():
    mock_client = _make_mock_client(status=200, json_data=MOCK_CARD)
    with patch("kazi.agents.discovery.httpx.AsyncClient", return_value=mock_client):
        cards = await discover_from_urls(["https://agents.example.com"])

    assert len(cards) == 1
    assert cards[0].name == "test-agent"
    assert len(cards[0].skills) == 1


@pytest.mark.asyncio
async def test_silently_skips_non_200_response():
    mock_client = _make_mock_client(status=404)
    with patch("kazi.agents.discovery.httpx.AsyncClient", return_value=mock_client):
        cards = await discover_from_urls(["https://no-agent.example.com"])

    assert cards == []


@pytest.mark.asyncio
async def test_silently_skips_url_that_raises():
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("kazi.agents.discovery.httpx.AsyncClient", return_value=mock_client):
        cards = await discover_from_urls(["http://unreachable.local"])

    assert cards == []


@pytest.mark.asyncio
async def test_discovers_multiple_urls_independently():
    """Each URL is probed independently; failures on one don't affect others."""
    call_count = 0
    cards_data = [MOCK_CARD, {**MOCK_CARD, "name": "second-agent"}]

    async def fake_get(url):
        nonlocal call_count
        idx = call_count
        call_count += 1
        if idx < 2:
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = cards_data[idx]
            return resp
        raise httpx.ConnectError("down")

    mock_client = AsyncMock()
    mock_client.get = fake_get
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("kazi.agents.discovery.httpx.AsyncClient", return_value=mock_client):
        cards = await discover_from_urls([
            "https://agent1.example.com",
            "https://agent2.example.com",
            "https://down.example.com",
        ])

    names = {c.name for c in cards}
    assert "test-agent" in names
    assert "second-agent" in names
    assert len(cards) == 2


@pytest.mark.asyncio
async def test_discover_empty_list_returns_empty():
    cards = await discover_from_urls([])
    assert cards == []


@pytest.mark.asyncio
async def test_fetches_well_known_agent_json_path():
    """The card URL must be <base>/.well-known/agent.json."""
    fetched_urls = []

    async def capture_get(url):
        fetched_urls.append(url)
        resp = MagicMock()
        resp.status_code = 404
        return resp

    mock_client = AsyncMock()
    mock_client.get = capture_get
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("kazi.agents.discovery.httpx.AsyncClient", return_value=mock_client):
        await discover_from_urls(["https://myagent.example.com"])

    assert len(fetched_urls) == 1
    assert fetched_urls[0].endswith("/.well-known/agent.json")


@pytest.mark.asyncio
async def test_base_url_trailing_slash_is_normalised():
    """Trailing slash on the base URL must not produce a double slash."""
    fetched_urls = []

    async def capture_get(url):
        fetched_urls.append(url)
        resp = MagicMock()
        resp.status_code = 404
        return resp

    mock_client = AsyncMock()
    mock_client.get = capture_get
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("kazi.agents.discovery.httpx.AsyncClient", return_value=mock_client):
        await discover_from_urls(["https://myagent.example.com/"])

    assert "//" not in fetched_urls[0].split("https://")[1]


@pytest.mark.asyncio
async def test_returns_agent_card_with_url_set():
    mock_client = _make_mock_client(status=200, json_data=MOCK_CARD)
    with patch("kazi.agents.discovery.httpx.AsyncClient", return_value=mock_client):
        cards = await discover_from_urls(["https://agents.example.com"])

    assert cards[0].url == "https://agents.example.com"


# ── scan_localhost ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_scan_localhost_uses_default_port_range():
    """Default range is 8000–8020 inclusive (21 ports)."""
    discovered_urls = []

    async def fake_discover(urls, timeout):
        discovered_urls.extend(urls)
        return []

    with patch("kazi.agents.discovery.discover_from_urls", side_effect=fake_discover):
        await scan_localhost()

    assert len(discovered_urls) == 21
    assert "http://localhost:8000" in discovered_urls
    assert "http://localhost:8020" in discovered_urls
    assert "http://localhost:8021" not in discovered_urls


@pytest.mark.asyncio
async def test_scan_localhost_uses_provided_ports():
    discovered_urls = []

    async def fake_discover(urls, timeout):
        discovered_urls.extend(urls)
        return []

    with patch("kazi.agents.discovery.discover_from_urls", side_effect=fake_discover):
        await scan_localhost(ports=[9000, 9001, 9002])

    assert len(discovered_urls) == 3
    assert "http://localhost:9000" in discovered_urls
    assert "http://localhost:9002" in discovered_urls


@pytest.mark.asyncio
async def test_scan_localhost_returns_discovered_cards():
    fake_card = AgentCard(name="local-agent", description="", url="http://localhost:8000")

    async def fake_discover(urls, timeout):
        return [fake_card]

    with patch("kazi.agents.discovery.discover_from_urls", side_effect=fake_discover):
        results = await scan_localhost(ports=[8000])

    assert len(results) == 1
    assert results[0].name == "local-agent"


@pytest.mark.asyncio
async def test_scan_localhost_empty_ports_returns_empty():
    async def fake_discover(urls, timeout):
        return []

    with patch("kazi.agents.discovery.discover_from_urls", side_effect=fake_discover):
        results = await scan_localhost(ports=[])

    assert results == []
