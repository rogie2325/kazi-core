"""
Example 6: Real-time voice agent

Push-to-talk and streaming voice with cross-modal memory.
The same thread_id links voice and chat sessions — memory is shared.

Install::

    pip install kazi-core[anthropic]
    pip install kazi-voice[openai]          # OpenAI Whisper STT + TTS
    # or
    pip install kazi-voice[deepgram,elevenlabs]

Requires: ANTHROPIC_API_KEY (or OPENAI_API_KEY), OPENAI_API_KEY for voice
"""
import asyncio
import os

from kazi import Kazi, KaziConfig, LLMConfig, LLMProvider, VoiceConfig, STTProvider, TTSProvider


async def push_to_talk_example():
    """
    Full voice round-trip: audio bytes in → transcript → LLM → speech bytes out.
    Use this when you have the full audio clip before sending.
    """
    config = KaziConfig(
        llm=LLMConfig(provider=LLMProvider.ANTHROPIC, model="claude-sonnet-4-6"),
        voice=VoiceConfig(
            stt_provider=STTProvider.OPENAI,
            stt_model="whisper-1",
            tts_provider=TTSProvider.OPENAI,
            tts_voice="nova",
        ),
    )

    async with await Kazi.create(config) as kazi:
        # In production, read this from a microphone or uploaded audio file
        with open("./audio_sample.wav", "rb") as f:
            audio_bytes = f.read()

        # Voice turn — returns MP3 audio you can play or stream to the client
        reply_audio: bytes = await kazi.run_voice(
            audio_bytes,
            thread_id="user:alice:voice",
        )

        # Save to disk or send over WebSocket
        with open("reply.mp3", "wb") as f:
            f.write(reply_audio)
        print("Saved reply to reply.mp3")

        # Text turn on the same thread — agent remembers the voice conversation
        text_reply = await kazi.run(
            "What did I just ask you?",
            thread_id="user:alice:voice",  # same thread_id = shared memory
        )
        print(f"Text follow-up: {text_reply}")


async def streaming_voice_example():
    """
    Low-latency streaming: audio arrives chunk-by-chunk for WebSocket / WebRTC delivery.
    First audio chunk arrives before the LLM finishes — keeps latency under ~500ms.
    """
    config = KaziConfig(
        llm=LLMConfig(provider=LLMProvider.ANTHROPIC, model="claude-sonnet-4-6"),
        voice=VoiceConfig(
            stt_provider=STTProvider.OPENAI,
            tts_provider=TTSProvider.ELEVENLABS,
            elevenlabs_voice_id="21m00Tcm4TlvDq8ikWAM",  # Rachel
            tts_api_key=os.getenv("ELEVENLABS_API_KEY"),
        ),
    )

    async with await Kazi.create(config) as kazi:
        with open("./audio_sample.wav", "rb") as f:
            audio_bytes = f.read()

        # Pipe chunks directly to a WebSocket or WebRTC track
        chunks = []
        async for audio_chunk in kazi.stream_voice(
            audio_bytes,
            thread_id="user:bob:voice",
        ):
            chunks.append(audio_chunk)
            # In production: await websocket.send_bytes(audio_chunk)

        print(f"Received {len(chunks)} audio chunks")


async def fastapi_voice_server():
    """
    Minimal FastAPI + WebSocket voice server.
    kazi.as_app() includes the /voice WebSocket endpoint out of the box.

    Run with:  uvicorn examples.06_voice_agent:app --host 0.0.0.0 --port 8000
    Then connect from your React app: ws://localhost:8000/voice
    """
    import uvicorn
    from kazi import Kazi, KaziConfig, LLMConfig, LLMProvider, VoiceConfig, STTProvider, TTSProvider

    config = KaziConfig(
        llm=LLMConfig(provider=LLMProvider.ANTHROPIC, model="claude-sonnet-4-6"),
        voice=VoiceConfig(
            stt_provider=STTProvider.OPENAI,
            tts_provider=TTSProvider.OPENAI,
            tts_voice="nova",
        ),
    )

    kazi = await Kazi.create(config)
    app = kazi.as_app(
        api_key=os.getenv("KAZI_API_KEY", "secret"),
        cors_origins=["http://localhost:3000"],  # React dev server
    )
    # uvicorn.run(app, host="0.0.0.0", port=8000)
    print("App ready — uncomment uvicorn.run() to start the server")
    await kazi.close()


if __name__ == "__main__":
    # asyncio.run(push_to_talk_example())   # needs audio_sample.wav
    # asyncio.run(streaming_voice_example()) # needs audio_sample.wav
    asyncio.run(fastapi_voice_server())
