"""Speech-to-text abstraction — OpenAI Whisper and Deepgram."""
from __future__ import annotations

import io
import logging

from kazi.core.config import STTProvider

logger = logging.getLogger(__name__)


async def transcribe(
    audio: bytes,
    *,
    provider: STTProvider = STTProvider.OPENAI,
    api_key: str | None = None,
    model: str = "whisper-1",
    language: str | None = None,
    # Deepgram-specific
    deepgram_model: str = "nova-2",
) -> str:
    """
    Transcribe audio bytes to text.

    audio    Raw audio bytes — WAV, MP3, MP4, WEBM, OGG all accepted.
    Returns  Transcribed text string.
    """
    if provider == STTProvider.OPENAI:
        return await _transcribe_openai(audio, api_key=api_key, model=model, language=language)
    if provider == STTProvider.DEEPGRAM:
        return await _transcribe_deepgram(
            audio, api_key=api_key, model=deepgram_model, language=language
        )
    raise ValueError(f"Unsupported STT provider: {provider!r}")


async def _transcribe_openai(
    audio: bytes,
    *,
    api_key: str | None,
    model: str,
    language: str | None,
) -> str:
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise ImportError(
            "openai package required for OpenAI STT. "
            "Install: pip install kazi-voice[openai]"
        )

    client = AsyncOpenAI(api_key=api_key)
    audio_file = io.BytesIO(audio)
    audio_file.name = "audio.wav"

    kwargs: dict = {"model": model, "file": audio_file}
    if language:
        kwargs["language"] = language

    response = await client.audio.transcriptions.create(**kwargs)
    return response.text.strip()


async def _transcribe_deepgram(
    audio: bytes,
    *,
    api_key: str | None,
    model: str,
    language: str | None,
) -> str:
    try:
        from deepgram import DeepgramClient, PrerecordedOptions
    except ImportError:
        raise ImportError(
            "deepgram-sdk package required for Deepgram STT. "
            "Install: pip install kazi-voice[deepgram]"
        )

    client = DeepgramClient(api_key or "")
    options = PrerecordedOptions(model=model, smart_format=True, language=language or "en")
    payload = {"buffer": audio}
    response = await client.listen.asyncprerecorded.v("1").transcribe_file(payload, options)
    results = response.results
    if not results or not results.channels:
        return ""
    return results.channels[0].alternatives[0].transcript.strip()
