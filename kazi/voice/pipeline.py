"""VoicePipeline — ties STT → LLM → TTS into a single async interface."""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from kazi.voice.stt import transcribe
from kazi.voice.tts import synthesize, synthesize_stream

if TYPE_CHECKING:
    from kazi.core.config import VoiceConfig

logger = logging.getLogger(__name__)


class VoicePipeline:
    """
    Wraps a Kazi instance to add real-time voice I/O.

    The thread_id is shared with text chat — voice and chat sessions for the
    same user land in the same LangGraph checkpoint, giving Riley (or any
    sub-agent) identical memory regardless of whether the user spoke or typed.

    Usage::

        async with await Kazi.create(config) as kazi:
            # Push-to-talk: audio bytes in, audio bytes out
            audio_out = await kazi.run_voice(audio_bytes, thread_id="user:123")

            # Real-time streaming (WebSocket / WebRTC)
            async for chunk in kazi.stream_voice(audio_bytes, thread_id="user:123"):
                await ws.send_bytes(chunk)

            # Same thread — chat and voice share memory
            reply = await kazi.run("What did I just say?", thread_id="user:123")
    """

    def __init__(self, kazi, voice_config: VoiceConfig) -> None:
        self._kazi = kazi
        self._cfg = voice_config

    async def run(self, audio: bytes, *, thread_id: str = "default") -> bytes:
        """
        Full round-trip: transcribe audio → run LLM → synthesize reply.
        Returns MP3 audio bytes.
        """
        text = await self._transcribe(audio)
        logger.debug("Voice STT: %r", text[:120])

        reply = await self._kazi.run(text, thread_id=thread_id)
        logger.debug("Voice LLM reply: %r", reply[:120])

        audio_out = await synthesize(
            reply,
            provider=self._cfg.tts_provider,
            api_key=self._cfg.tts_api_key,
            model=self._cfg.tts_model,
            voice=self._cfg.tts_voice,
            speed=self._cfg.tts_speed,
            elevenlabs_voice_id=self._cfg.elevenlabs_voice_id,
            elevenlabs_model=self._cfg.elevenlabs_model,
        )
        return audio_out

    async def stream(self, audio: bytes, *, thread_id: str = "default") -> AsyncIterator[bytes]:
        """
        Low-latency streaming: transcribe → stream LLM tokens → TTS chunks arrive
        before the LLM finishes.  Yields MP3 audio chunks for WebSocket delivery.
        """
        text = await self._transcribe(audio)
        logger.debug("Voice STT (stream): %r", text[:120])

        token_stream = self._kazi.stream(text, thread_id=thread_id)
        async for audio_chunk in synthesize_stream(
            token_stream,
            provider=self._cfg.tts_provider,
            api_key=self._cfg.tts_api_key,
            model=self._cfg.tts_model,
            voice=self._cfg.tts_voice,
            speed=self._cfg.tts_speed,
            elevenlabs_voice_id=self._cfg.elevenlabs_voice_id,
            elevenlabs_model=self._cfg.elevenlabs_model,
        ):
            yield audio_chunk

    async def _transcribe(self, audio: bytes) -> str:
        return await transcribe(
            audio,
            provider=self._cfg.stt_provider,
            api_key=self._cfg.stt_api_key,
            model=self._cfg.stt_model,
            language=self._cfg.language,
        )
