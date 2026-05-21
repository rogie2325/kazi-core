"""Abstract LLM interface used internally for type annotations."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator


class BaseLLM(ABC):
    """Minimal interface that all Kazi LLM wrappers conform to."""

    @abstractmethod
    async def complete(self, prompt: str, **kwargs) -> str:
        """Single-turn completion."""

    @abstractmethod
    async def chat(self, messages: list[dict], **kwargs) -> str:
        """Multi-turn chat completion. Each message: {"role": str, "content": str}."""

    @abstractmethod
    async def stream_chat(self, messages: list[dict], **kwargs) -> AsyncIterator[str]:
        """Streaming chat — yields token strings."""
