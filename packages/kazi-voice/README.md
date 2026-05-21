# kazi-voice

Real-time voice I/O for kazi. Adds STT (speech-to-text) and TTS (text-to-speech) to the Kazi orchestrator with cross-modal memory — a user's chat history and voice history are the same thing.

## Install

```bash
pip install kazi-voice[openai]       # OpenAI Whisper STT + OpenAI TTS
pip install kazi-voice[elevenlabs]   # ElevenLabs TTS (voice cloning, streaming)
pip install kazi-voice[deepgram]     # Deepgram STT (real-time, low-latency)
pip install kazi-voice[all]          # everything
```

## How cross-modal memory works

Memory is keyed by `thread_id` in the LangGraph checkpointer. Voice and chat sessions using the same `thread_id` share identical conversation history — the agent has no concept of modality. A user can speak to Riley, then open a chat with Riley, and Riley remembers both sides of the conversation.

```python
from kazi import Kazi, KaziConfig, LLMConfig, LLMProvider, VoiceConfig, STTProvider, TTSProvider

config = KaziConfig(
    llm=LLMConfig(provider=LLMProvider.ANTHROPIC, model="claude-sonnet-4-6"),
    voice=VoiceConfig(
        stt_provider=STTProvider.OPENAI,
        tts_provider=TTSProvider.ELEVENLABS,
        elevenlabs_voice_id="21m00Tcm4TlvDq8ikWAM",
    ),
)

async with await Kazi.create(config) as kazi:
    # Voice session
    audio_out = await kazi.run_voice(audio_bytes, thread_id="user:123")

    # Chat session — same thread_id, Riley remembers the voice conversation
    reply = await kazi.run("What did I just say?", thread_id="user:123")
```

## Streaming (WebSocket / WebRTC)

Audio chunks arrive before the LLM finishes generating — sentence-by-sentence synthesis keeps end-to-end latency under ~500ms.

```python
async def handle_websocket(ws, audio_bytes: bytes, thread_id: str):
    async for audio_chunk in kazi.stream_voice(audio_bytes, thread_id=thread_id):
        await ws.send_bytes(audio_chunk)
```

## Sub-agent voice

Each sub-agent in a `Supervisor` crew also supports voice. The agent's personality is maintained via its system prompt regardless of which LLM model handles the request.

```python
from kazi.agents import SubAgent, SubAgentConfig, Supervisor

riley = SubAgent(SubAgentConfig(
    name="Riley",
    role="Research & Intelligence",
    system_prompt="You are Riley, a research specialist...",
), kazi)

crew = Supervisor(agents=[riley, ...], kazi=kazi)

# Routes to the right agent, transcribes, synthesizes
audio_out = await crew.run_voice(audio_bytes, thread_id="user:123")
```

## Supported providers

| | STT | TTS |
|---|---|---|
| OpenAI | Whisper (`whisper-1`) | TTS-1 / TTS-1-HD — 6 voices |
| Deepgram | Nova-2 — real-time streaming | — |
| ElevenLabs | — | Turbo v2 — voice cloning, streaming |

## License

MIT
