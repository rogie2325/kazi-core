from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from kazi.llm.base import BaseLLM


class OllamaLLM(BaseLLM):
    """Thin async wrapper around Ollama (local model server)."""

    def __init__(
        self,
        model: str = "llama3.2",
        temperature: float = 0.1,
        base_url: str = "http://localhost:11434",
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.base_url = base_url.rstrip("/")

    async def complete(self, prompt: str, **kwargs) -> str:
        return await self.chat([{"role": "user", "content": prompt}], **kwargs)

    async def chat(self, messages: list[dict[str, Any]], **kwargs) -> str:
        import httpx

        payload = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "stream": False,
            "options": {"temperature": kwargs.get("temperature", self.temperature)},
        }
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "")

    async def stream_chat(self, messages: list[dict[str, Any]], **kwargs) -> AsyncIterator[str]:
        import httpx

        payload = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "stream": True,
            "options": {"temperature": kwargs.get("temperature", self.temperature)},
        }
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as resp:
                import json
                async for line in resp.aiter_lines():
                    if line.strip():
                        chunk = json.loads(line)
                        content = chunk.get("message", {}).get("content", "")
                        if content:
                            yield content
