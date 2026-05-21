from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from kazi.llm.base import BaseLLM


class GoogleLLM(BaseLLM):
    """Thin async wrapper around Google Gemini via the generativeai SDK."""

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        api_key: str | None = None,
    ) -> None:
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("google-generativeai required: pip install 'kazi[google]'")

        if api_key:
            genai.configure(api_key=api_key)

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._genai = genai

    async def complete(self, prompt: str, **kwargs) -> str:
        return await self.chat([{"role": "user", "content": prompt}], **kwargs)

    async def chat(self, messages: list[dict[str, Any]], **kwargs) -> str:
        import asyncio

        model = self._genai.GenerativeModel(
            kwargs.get("model", self.model),
            generation_config=self._genai.types.GenerationConfig(
                temperature=kwargs.get("temperature", self.temperature),
                max_output_tokens=kwargs.get("max_tokens", self.max_tokens),
            ),
        )
        # Convert to Gemini format
        contents = [{"role": "user" if m["role"] != "model" else "model", "parts": [m["content"]]} for m in messages if m["role"] != "system"]
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(None, lambda: model.generate_content(contents))
        return resp.text or ""

    async def stream_chat(self, messages: list[dict[str, Any]], **kwargs) -> AsyncIterator[str]:
        import asyncio

        model = self._genai.GenerativeModel(kwargs.get("model", self.model))
        contents = [{"role": "user" if m["role"] != "model" else "model", "parts": [m["content"]]} for m in messages if m["role"] != "system"]
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(None, lambda: model.generate_content(contents, stream=True))
        for chunk in resp:
            if chunk.text:
                yield chunk.text
