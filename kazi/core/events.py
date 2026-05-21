"""
Typed streaming event protocol for kazi.stream_events().

Every event has a ``type`` discriminator and a ``data`` string payload.
The optional ``metadata`` dict carries structured extras (tool args, costs, etc.).

Types
-----
token       LLM text token.  data = the token string.
tool_start  Tool invocation beginning.  data = tool name.
            metadata["args"] = the argument dict the LLM supplied.
tool_end    Tool finished.  data = tool name.
            metadata["result"] = first 200 chars of the result.
            metadata["cached"] = True when the result came from the tool cache.
done        Stream complete.  data = "".
            metadata["cost_usd"]  and metadata["tokens"] set when tracking.
error       Unrecoverable error.  data = human-readable error message.
"""
from __future__ import annotations

from typing import Any, Literal, TypedDict

from typing_extensions import NotRequired


class StreamEvent(TypedDict):
    """
    A typed event emitted by ``kazi.stream_events()``.

    Serialize to JSON with ``json.dumps(event)`` — all values are primitives.
    """
    type: Literal["token", "tool_start", "tool_end", "done", "error"]
    data: str
    metadata: NotRequired[dict[str, Any]]
