"""Tests for kazi.brain.memory — checkpointer factory and thread helpers."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

# ── get_checkpointer ──────────────────────────────────────────────────────────

def test_get_checkpointer_in_memory_returns_memory_saver():
    from langgraph.checkpoint.memory import MemorySaver

    from kazi.brain.memory import get_checkpointer

    cp = get_checkpointer("in_memory")
    assert isinstance(cp, MemorySaver)


def test_get_checkpointer_unknown_backend_returns_memory_saver():
    """Unrecognised backend falls through to in_memory."""
    from langgraph.checkpoint.memory import MemorySaver

    from kazi.brain.memory import get_checkpointer

    cp = get_checkpointer("unknown_backend")
    assert isinstance(cp, MemorySaver)


def test_get_checkpointer_empty_string_returns_memory_saver():
    from langgraph.checkpoint.memory import MemorySaver

    from kazi.brain.memory import get_checkpointer

    cp = get_checkpointer("")
    assert isinstance(cp, MemorySaver)


def test_get_checkpointer_sqlite_calls_from_conn_string():
    from kazi.brain.memory import get_checkpointer

    mock_saver = MagicMock()
    mock_cls = MagicMock()
    mock_cls.from_conn_string.return_value = mock_saver

    with patch.dict("sys.modules", {"langgraph.checkpoint.sqlite.aio": MagicMock(AsyncSqliteSaver=mock_cls)}):
        result = get_checkpointer("sqlite", "sqlite:///test.db")

    mock_cls.from_conn_string.assert_called_once_with("test.db")
    assert result is mock_saver


def test_get_checkpointer_sqlite_strips_uri_prefix():
    from kazi.brain.memory import get_checkpointer

    mock_cls = MagicMock()
    mock_cls.from_conn_string.return_value = MagicMock()

    with patch.dict("sys.modules", {"langgraph.checkpoint.sqlite.aio": MagicMock(AsyncSqliteSaver=mock_cls)}):
        get_checkpointer("sqlite", "sqlite:///path/to/db.db")

    call_arg = mock_cls.from_conn_string.call_args[0][0]
    assert call_arg == "path/to/db.db"
    assert "sqlite:///" not in call_arg


def test_get_checkpointer_sqlite_uses_default_path_when_empty():
    from kazi.brain.memory import get_checkpointer

    mock_cls = MagicMock()
    mock_cls.from_conn_string.return_value = MagicMock()

    with patch.dict("sys.modules", {"langgraph.checkpoint.sqlite.aio": MagicMock(AsyncSqliteSaver=mock_cls)}):
        get_checkpointer("sqlite", "")

    call_arg = mock_cls.from_conn_string.call_args[0][0]
    assert call_arg == "kazi_memory.db"


def test_get_checkpointer_redis_calls_from_conn_string():
    from kazi.brain.memory import get_checkpointer

    mock_cls = MagicMock()
    mock_cls.from_conn_string.return_value = MagicMock()

    with patch.dict("sys.modules", {"langgraph.checkpoint.redis.aio": MagicMock(AsyncRedisSaver=mock_cls)}):
        get_checkpointer("redis", "redis://localhost:6379/0")

    mock_cls.from_conn_string.assert_called_once_with("redis://localhost:6379/0")


def test_get_checkpointer_postgres_calls_from_conn_string():
    from kazi.brain.memory import get_checkpointer

    mock_cls = MagicMock()
    mock_cls.from_conn_string.return_value = MagicMock()

    with patch.dict("sys.modules", {"langgraph.checkpoint.postgres.aio": MagicMock(AsyncPostgresSaver=mock_cls)}):
        get_checkpointer("postgres", "postgresql://user:pass@host/db")

    mock_cls.from_conn_string.assert_called_once_with("postgresql://user:pass@host/db")


# ── get_thread_history ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_thread_history_returns_empty_when_no_snapshot():
    from kazi.brain.memory import get_thread_history

    mock_cp = AsyncMock()
    mock_cp.aget_tuple = AsyncMock(return_value=None)

    result = await get_thread_history(mock_cp, "thread-1")
    assert result == []


@pytest.mark.asyncio
async def test_get_thread_history_returns_empty_when_checkpointer_raises():
    from kazi.brain.memory import get_thread_history

    mock_cp = AsyncMock()
    mock_cp.aget_tuple = AsyncMock(side_effect=RuntimeError("backend unavailable"))

    result = await get_thread_history(mock_cp, "thread-1")
    assert result == []


@pytest.mark.asyncio
async def test_get_thread_history_returns_messages():
    from kazi.brain.memory import get_thread_history

    messages = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there"),
    ]
    snapshot = MagicMock()
    snapshot.checkpoint = {"channel_values": {"messages": messages}}

    mock_cp = AsyncMock()
    mock_cp.aget_tuple = AsyncMock(return_value=snapshot)

    result = await get_thread_history(mock_cp, "thread-1")
    assert len(result) == 2
    assert result[0] == {"role": "human", "content": "Hello"}
    assert result[1] == {"role": "ai", "content": "Hi there"}


@pytest.mark.asyncio
async def test_get_thread_history_respects_limit():
    from kazi.brain.memory import get_thread_history

    messages = [HumanMessage(content=f"msg {i}") for i in range(10)]
    snapshot = MagicMock()
    snapshot.checkpoint = {"channel_values": {"messages": messages}}

    mock_cp = AsyncMock()
    mock_cp.aget_tuple = AsyncMock(return_value=snapshot)

    result = await get_thread_history(mock_cp, "thread-1", limit=3)
    assert len(result) == 3
    # Should be the LAST 3 messages
    assert result[-1]["content"] == "msg 9"


@pytest.mark.asyncio
async def test_get_thread_history_no_limit_returns_all():
    from kazi.brain.memory import get_thread_history

    messages = [HumanMessage(content=f"msg {i}") for i in range(5)]
    snapshot = MagicMock()
    snapshot.checkpoint = {"channel_values": {"messages": messages}}

    mock_cp = AsyncMock()
    mock_cp.aget_tuple = AsyncMock(return_value=snapshot)

    result = await get_thread_history(mock_cp, "t")
    assert len(result) == 5


@pytest.mark.asyncio
async def test_get_thread_history_skips_messages_without_content():
    from kazi.brain.memory import get_thread_history

    class NoContentMsg:
        pass

    messages = [HumanMessage(content="real"), NoContentMsg()]
    snapshot = MagicMock()
    snapshot.checkpoint = {"channel_values": {"messages": messages}}

    mock_cp = AsyncMock()
    mock_cp.aget_tuple = AsyncMock(return_value=snapshot)

    result = await get_thread_history(mock_cp, "t")
    assert len(result) == 1
    assert result[0]["content"] == "real"


@pytest.mark.asyncio
async def test_get_thread_history_empty_channel_values():
    from kazi.brain.memory import get_thread_history

    snapshot = MagicMock()
    snapshot.checkpoint = {"channel_values": {}}  # no "messages" key

    mock_cp = AsyncMock()
    mock_cp.aget_tuple = AsyncMock(return_value=snapshot)

    result = await get_thread_history(mock_cp, "t")
    assert result == []


# ── clear_thread ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_clear_thread_calls_adelete():
    from kazi.brain.memory import clear_thread

    mock_cp = AsyncMock()
    mock_cp.adelete = AsyncMock()

    await clear_thread(mock_cp, "thread-to-delete")
    mock_cp.adelete.assert_called_once_with({"configurable": {"thread_id": "thread-to-delete"}})


@pytest.mark.asyncio
async def test_clear_thread_silently_skips_when_adelete_absent():
    """Backends like MemorySaver don't support delete — this must not raise."""
    from kazi.brain.memory import clear_thread

    mock_cp = MagicMock(spec=[])  # no adelete attribute

    await clear_thread(mock_cp, "thread-1")  # should not raise


@pytest.mark.asyncio
async def test_clear_thread_passes_correct_config_format():
    from kazi.brain.memory import clear_thread

    received = {}

    async def capture(config):
        received["config"] = config

    mock_cp = AsyncMock()
    mock_cp.adelete = capture

    await clear_thread(mock_cp, "my-thread-id")
    assert received["config"]["configurable"]["thread_id"] == "my-thread-id"
