from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast

from kazi.llm.base import BaseLLM


class OpenAILLM(BaseLLM):
    """Thin async wrapper around the OpenAI chat completions API."""

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai package required: pip install 'kazi[openai]'")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def complete(self, prompt: str, **kwargs) -> str:
        return await self.chat([{"role": "user", "content": prompt}], **kwargs)

    async def chat(self, messages: list[dict[str, Any]], **kwargs) -> str:
        payload = cast(list[Any], messages)
        resp = await self._client.chat.completions.create(
            model=kwargs.get("model", self.model),
            messages=payload,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        return resp.choices[0].message.content or ""

    async def stream_chat(self, messages: list[dict[str, Any]], **kwargs) -> AsyncIterator[str]:
        payload = cast(list[Any], messages)
        stream = await self._client.chat.completions.create(
            model=kwargs.get("model", self.model),
            messages=payload,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            stream=True,
        )
        async for chunk in cast(AsyncIterator[Any], stream):
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
