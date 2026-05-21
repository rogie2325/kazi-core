"""Helpers for serialising/deserialising LangGraph state and tool results."""
from __future__ import annotations

import json
from typing import Any


def state_to_dict(state: dict) -> dict:
    """Convert a LangGraph AgentState to a JSON-serialisable dict."""
    out = {}
    for k, v in state.items():
        if k == "messages":
            out[k] = [_message_to_dict(m) for m in v]
        elif isinstance(v, (str, int, float, bool, type(None))):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def _message_to_dict(msg) -> dict:
    role_map = {
        "HumanMessage": "user",
        "AIMessage": "assistant",
        "SystemMessage": "system",
        "ToolMessage": "tool",
    }
    cls_name = type(msg).__name__
    return {
        "role": role_map.get(cls_name, cls_name.lower()),
        "content": msg.content if isinstance(msg.content, str) else json.dumps(msg.content),
    }


def safe_json(value: Any, fallback: str = "[non-serialisable]") -> str:
    """Serialise any value to a JSON string, falling back gracefully."""
    try:
        return json.dumps(value, default=str)
    except Exception:
        return fallback
