from __future__ import annotations

import operator
from collections.abc import Sequence
from typing import Annotated

from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """State schema carried through the LangGraph execution graph."""

    messages: Annotated[Sequence[BaseMessage], operator.add]
    thread_id: str
    current_step: str
    tool_calls_made: int
    max_tool_calls: int
    system_prompt: str | None
    final_answer: str | None
    metadata: dict
