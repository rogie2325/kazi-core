from __future__ import annotations

from typing import AsyncIterator, Optional

from kazi.llm.base import BaseLLM


class AnthropicLLM(BaseLLM):
    """Thin async wrapper around the Anthropic Messages API with prompt caching."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        api_key: Optional[str] = None,
    ) -> None:
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package required: pip install 'kazi[anthropic]'")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    async def complete(self, prompt: str, **kwargs) -> str:
        return await self.chat([{"role": "user", "content": prompt}], **kwargs)

    async def chat(self, messages: list[dict], **kwargs) -> str:
        # Separate system messages from the conversation
        system = next((m["content"] for m in messages if m["role"] == "system"), None)
        user_msgs = [m for m in messages if m["role"] != "system"]

        kwargs_merged = {
            "model": kwargs.get("model", self.model),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "messages": user_msgs,
        }
        if system:
            # Use cache_control on the system prompt for prompt caching
            kwargs_merged["system"] = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ]

        resp = await self._client.messages.create(**kwargs_merged)
        return resp.content[0].text if resp.content else ""

    async def stream_chat(self, messages: list[dict], **kwargs) -> AsyncIterator[str]:
        system = next((m["content"] for m in messages if m["role"] == "system"), None)
        user_msgs = [m for m in messages if m["role"] != "system"]

        kwargs_merged = {
            "model": kwargs.get("model", self.model),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "messages": user_msgs,
        }
        if system:
            kwargs_merged["system"] = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ]

        async with self._client.messages.stream(**kwargs_merged) as stream:
            async for text in stream.text_stream:
                yield text
