"""Abstract LLM interface used internally for type annotations."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any


class BaseLLM(ABC):
    """Minimal interface that all Kazi LLM wrappers conform to."""

    @abstractmethod
    async def complete(self, prompt: str, **kwargs) -> str:
        """Single-turn completion."""

    @abstractmethod
    async def chat(self, messages: list[dict[str, Any]], **kwargs) -> str:
        """Multi-turn chat completion. Each message: {"role": str, "content": str}."""

    @abstractmethod
    def stream_chat(self, messages: list[dict[str, Any]], **kwargs) -> AsyncIterator[str]:
        """Streaming chat — yields token strings."""
