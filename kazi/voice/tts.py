"""Text-to-speech abstraction — OpenAI TTS and ElevenLabs."""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Literal

from kazi.core.config import TTSProvider

logger = logging.getLogger(__name__)

OpenAIVoice = Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"]


async def synthesize(
    text: str,
    *,
    provider: TTSProvider = TTSProvider.OPENAI,
    api_key: str | None = None,
    # OpenAI options
    model: str = "tts-1",
    voice: str = "nova",
    speed: float = 1.0,
    # ElevenLabs options
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM",  # Rachel (default)
    elevenlabs_model: str = "eleven_turbo_v2",
) -> bytes:
    """Synthesize text to audio bytes (MP3)."""
    if provider == TTSProvider.OPENAI:
        return await _synthesize_openai(text, api_key=api_key, model=model, voice=voice, speed=speed)
    if provider == TTSProvider.ELEVENLABS:
        return await _synthesize_elevenlabs(
            text, api_key=api_key, voice_id=elevenlabs_voice_id, model=elevenlabs_model
        )
    raise ValueError(f"Unsupported TTS provider: {provider!r}")


async def synthesize_stream(
    text_iter,
    *,
    provider: TTSProvider = TTSProvider.OPENAI,
    api_key: str | None = None,
    model: str = "tts-1",
    voice: str = "nova",
    speed: float = 1.0,
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM",
    elevenlabs_model: str = "eleven_turbo_v2",
) -> AsyncIterator[bytes]:
    """
    Stream audio chunks as text tokens arrive from the LLM.

    text_iter   AsyncIterator[str] — typically kazi.stream()
    Yields      Raw audio bytes chunks (MP3) suitable for WebSocket/WebRTC.
    """
    if provider == TTSProvider.OPENAI:
        async for chunk in _stream_openai(
            text_iter, api_key=api_key, model=model, voice=voice, speed=speed
        ):
            yield chunk
    elif provider == TTSProvider.ELEVENLABS:
        async for chunk in _stream_elevenlabs(
            text_iter,
            api_key=api_key,
            voice_id=elevenlabs_voice_id,
            model=elevenlabs_model,
        ):
            yield chunk
    else:
        raise ValueError(f"Unsupported TTS provider: {provider!r}")


# ── OpenAI ────────────────────────────────────────────────────────────────────

async def _synthesize_openai(
    text: str, *, api_key: str | None, model: str, voice: str, speed: float
) -> bytes:
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise ImportError(
            "openai package required for OpenAI TTS. "
            "Install: pip install kazi-voice[openai]"
        )

    client = AsyncOpenAI(api_key=api_key)
    response = await client.audio.speech.create(
        model=model, voice=voice, input=text, speed=speed, response_format="mp3"
    )
    return response.content


async def _stream_openai(
    text_iter, *, api_key: str | None, model: str, voice: str, speed: float
) -> AsyncIterator[bytes]:
    """
    Buffer LLM tokens into sentence-sized chunks, then synthesize each chunk
    so audio starts arriving before the LLM finishes generating.
    """
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise ImportError(
            "openai package required for OpenAI TTS. "
            "Install: pip install kazi-voice[openai]"
        )

    client = AsyncOpenAI(api_key=api_key)
    buffer = ""

    async for token in text_iter:
        buffer += token
        # Flush on sentence boundaries to keep latency low
        if any(buffer.rstrip().endswith(p) for p in (".", "!", "?", "\n")):
            if buffer.strip():
                async with client.audio.speech.with_streaming_response.create(
                    model=model, voice=voice, input=buffer.strip(), speed=speed,
                    response_format="mp3",
                ) as resp:
                    async for chunk in resp.iter_bytes(chunk_size=4096):
                        yield chunk
            buffer = ""

    # Flush remainder
    if buffer.strip():
        async with client.audio.speech.with_streaming_response.create(
            model=model, voice=voice, input=buffer.strip(), speed=speed, response_format="mp3"
        ) as resp:
            async for chunk in resp.iter_bytes(chunk_size=4096):
                yield chunk


# ── ElevenLabs ────────────────────────────────────────────────────────────────

async def _synthesize_elevenlabs(
    text: str, *, api_key: str | None, voice_id: str, model: str
) -> bytes:
    try:
        from elevenlabs.client import AsyncElevenLabs
    except ImportError:
        raise ImportError(
            "elevenlabs package required for ElevenLabs TTS. "
            "Install: pip install kazi-voice[elevenlabs]"
        )

    client = AsyncElevenLabs(api_key=api_key)
    audio = await client.generate(text=text, voice=voice_id, model=model)
    chunks = []
    async for chunk in audio:
        chunks.append(chunk)
    return b"".join(chunks)


async def _stream_elevenlabs(
    text_iter, *, api_key: str | None, voice_id: str, model: str
) -> AsyncIterator[bytes]:
    try:
        from elevenlabs.client import AsyncElevenLabs
    except ImportError:
        raise ImportError(
            "elevenlabs package required for ElevenLabs TTS. "
            "Install: pip install kazi-voice[elevenlabs]"
        )

    client = AsyncElevenLabs(api_key=api_key)
    buffer = ""

    async for token in text_iter:
        buffer += token
        if any(buffer.rstrip().endswith(p) for p in (".", "!", "?", "\n")):
            if buffer.strip():
                audio_stream = await client.generate(
                    text=buffer.strip(), voice=voice_id, model=model, stream=True
                )
                async for chunk in audio_stream:
                    if chunk:
                        yield chunk
            buffer = ""

    if buffer.strip():
        audio_stream = await client.generate(
            text=buffer.strip(), voice=voice_id, model=model, stream=True
        )
        async for chunk in audio_stream:
            if chunk:
                yield chunk
