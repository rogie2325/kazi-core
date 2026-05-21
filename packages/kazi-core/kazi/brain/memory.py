"""
Memory backends and helpers for conversation persistence.

The heavy lifting is done by LangGraph checkpointers; this module
provides thin factory functions and a helper to inspect stored threads.
"""
from __future__ import annotations

from typing import Optional


def get_checkpointer(backend: str = "in_memory", connection_string: str = ""):
    """
    Return a LangGraph checkpointer for the requested backend.

    backend: "in_memory" | "sqlite" | "redis" | "postgres"
    """
    if backend == "sqlite":
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        db_path = connection_string.replace("sqlite:///", "") or "kazi_memory.db"
        return AsyncSqliteSaver.from_conn_string(db_path)

    if backend == "redis":
        from langgraph.checkpoint.redis.aio import AsyncRedisSaver
        return AsyncRedisSaver.from_conn_string(connection_string)

    if backend == "postgres":
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        return AsyncPostgresSaver.from_conn_string(connection_string)

    from langgraph.checkpoint.memory import MemorySaver
    return MemorySaver()


async def get_thread_history(
    checkpointer,
    thread_id: str,
    limit: Optional[int] = None,
) -> list[dict]:
    """
    Return message history for a given thread as plain dicts.

    Only works with checkpointers that support .aget_tuple().
    """
    try:
        snapshot = await checkpointer.aget_tuple({"configurable": {"thread_id": thread_id}})
    except Exception:
        return []

    if snapshot is None:
        return []

    messages = snapshot.checkpoint.get("channel_values", {}).get("messages", [])
    result = [
        {
            "role": m.__class__.__name__.replace("Message", "").lower(),
            "content": m.content,
        }
        for m in messages
        if hasattr(m, "content")
    ]
    return result[-limit:] if limit else result


async def clear_thread(checkpointer, thread_id: str) -> None:
    """Delete all stored state for a thread (if the backend supports it)."""
    try:
        await checkpointer.adelete({"configurable": {"thread_id": thread_id}})
    except AttributeError:
        pass  # some backends don't support delete; silently skip
