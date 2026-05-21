from __future__ import annotations

import operator
from typing import Annotated, Optional, Sequence

from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """State schema carried through the LangGraph execution graph."""

    messages: Annotated[Sequence[BaseMessage], operator.add]
    thread_id: str
    current_step: str
    tool_calls_made: int
    max_tool_calls: int
    system_prompt: Optional[str]
    final_answer: Optional[str]
    metadata: dict
