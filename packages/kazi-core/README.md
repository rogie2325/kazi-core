# kazi-core

[![CI](https://github.com/erose2502/kazi-core/actions/workflows/ci.yml/badge.svg)](https://github.com/erose2502/kazi-core/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/kazi-core)](https://pypi.org/project/kazi-core/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Full AI orchestration engine — LangGraph multi-turn memory, tool execution, streaming, multi-agent supervision, and voice I/O in one library.

```python
import asyncio
from kazi import Kazi, KaziConfig, LLMConfig, LLMProvider

config = KaziConfig(llm=LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4o-mini"))

async def main():
    async with await Kazi.create(config) as kazi:
        reply = await kazi.run("What is the capital of France?")
        print(reply)  # Paris

asyncio.run(main())
```

## Install

```bash
pip install kazi-core[anthropic]   # Anthropic / Claude
pip install kazi-core[openai]      # OpenAI / GPT
pip install kazi-core[google]      # Google / Gemini
pip install kazi-core[all]         # all providers + redis + postgres
```

## Features

### Multi-turn memory
```python
async with await Kazi.create(config) as nx:
    await nx.run("My name is Alice.", thread_id="user-123")
    reply = await nx.run("What's my name?", thread_id="user-123")
    print(reply)  # Alice
```

### Streaming
```python
async with await Kazi.create(config) as nx:
    async for token in nx.stream("Tell me a story"):
        print(token, end="", flush=True)
```

### Typed event streaming (tool spinners, cost tickers)
```python
async with await Kazi.create(config) as nx:
    async for event in nx.stream_events("Search the web for kazi"):
        if event["type"] == "token":
            print(event["data"], end="")
        elif event["type"] == "tool_start":
            print(f"\n[calling {event['data']}...]")
```

### Tool registration
```python
from kazi import ToolDefinition, ToolParameter, ToolSource

async def get_weather(city: str) -> str:
    return f"Sunny and 72°F in {city}"

tool = ToolDefinition(
    name="get_weather",
    description="Returns current weather for a city.",
    parameters=[ToolParameter(name="city", type="string", description="City name", required=True)],
    source=ToolSource.NATIVE,
    handler=get_weather,
)

async with await Kazi.create(config) as nx:
    nx.add_tool(tool)
    reply = await nx.run("What's the weather in Tokyo?")
```

### Multi-agent supervision
```python
from kazi.agents import SubAgent, SubAgentConfig, Supervisor

async with await Kazi.create(config) as nx:
    researcher = SubAgent(SubAgentConfig(name="Researcher", role="Research and find facts"))
    writer = SubAgent(SubAgentConfig(name="Writer", role="Write clear summaries"))
    crew = Supervisor([researcher, writer], kazi=nx)
    reply = await crew.run("Research and summarise the history of the internet")
```

### Performance monitoring — auto-fire bad agents
```python
from kazi.agents import PerformanceMonitor

monitor = PerformanceMonitor(consecutive_threshold=3, on_fired=lambda name, reason: print(f"{name} fired: {reason}"))
crew = Supervisor([agent], kazi=nx, monitor=monitor)
```

### Branch threads (fork conversation history)
```python
async with await Kazi.create(config) as nx:
    await nx.run("The answer is 42.", thread_id="main")
    await nx.branch_thread("main", "branch-a")
    # branch-a starts with full context from main
    reply = await nx.run("What is the answer?", thread_id="branch-a")
```

### User profile system prompts
```python
from kazi import KaziConfig
from kazi.memory.profile import UserProfile

async with await Kazi.create(config) as nx:
    nx._profile_store.save("alice", {"name": "Alice", "language": "French"})
    reply = await nx.run("Bonjour!", user_id="alice")  # responds with Alice's context
```

### Batch runs
```python
async with await Kazi.create(config) as nx:
    results = await nx.batch_run(["Q1", "Q2", "Q3"], concurrency=3)
```

### Human-in-the-loop approval
```python
async def my_approval(tool_calls):
    for call in tool_calls:
        print(f"Approve {call['name']}({call['args']})? [y/n]")
        if input() != "y":
            return None
    return tool_calls

async with await Kazi.create(config) as nx:
    reply = await nx.run_with_approval("Delete old files", approval_callback=my_approval)
```

### Voice I/O (STT → LLM → TTS)
```python
from kazi import KaziConfig
from kazi.core.config import VoiceConfig

config = KaziConfig(llm=..., voice=VoiceConfig(stt_api_key="...", tts_api_key="..."))
async with await Kazi.create(config) as nx:
    audio_out = await nx.run_voice(audio_bytes, thread_id="user-123")
```

## Security

- **Prompt-injection detection** — built-in regex guard with `warn` and `block` modes
- **SQL read-only guard** — `SELECT`-only enforcement with multi-statement and `OUTFILE` blocking
- **Path traversal prevention** — user IDs sanitized before any filesystem access
- **Token budget** — hard per-run token cap to control costs

```python
from kazi.core.security import InjectionDetectionConfig, SecurityConfig
from kazi.core.config import KaziConfig

cfg = KaziConfig(security=SecurityConfig(injection=InjectionDetectionConfig(enabled=True, mode="block")))
```

## Model routing

```python
from kazi.core.config import LLMConfig, ModelRouter, RoutingConfig

config = KaziConfig(llm=LLMConfig(
    provider=LLMProvider.OPENAI,
    model="gpt-4o",
    router=ModelRouter(
        fallback=RoutingConfig(model="gpt-4o-mini"),       # on tool-call errors
        tool_call=RoutingConfig(model="gpt-4o-mini"),      # cheaper model for tool turns
        summarizer=RoutingConfig(model="gpt-4o-mini"),     # long-context summarization
    )
))
```

## Configuration reference

```python
from kazi import KaziConfig, LLMConfig, LLMProvider
from kazi.core.config import RAGConfig, VoiceConfig, SecurityConfig
from kazi.core.token_budget import TokenBudgetConfig

config = KaziConfig(
    llm=LLMConfig(
        provider=LLMProvider.OPENAI,
        model="gpt-4o-mini",
        api_key="sk-...",           # or set OPENAI_API_KEY env var
        temperature=0.7,
        max_tokens=4096,
    ),
    rag=RAGConfig(enabled=False),   # disable RAG if not needed
    budget=TokenBudgetConfig(max_tokens_per_run=50_000),
    security=SecurityConfig(),
)
```

## What's inside

| Component | Description |
|-----------|-------------|
| `kazi.Kazi` | Top-level async context manager |
| `kazi.brain.GraphBrain` | LangGraph execution engine with checkpointing |
| `kazi.core.registry.ToolRegistry` | Async-first tool registry with monitor wiring |
| `kazi.agents.Supervisor` | Multi-agent router with `PerformanceMonitor` integration |
| `kazi.agents.monitor.PerformanceMonitor` | Rolling-window failure tracker, auto-fires bad components |
| `kazi.core.security` | Injection guard, content policy, thread policy |
| `kazi.memory.profile.UserProfile` | Per-user JSON fact store for system-prompt preambles |
| `kazi.voice` | STT → LLM → TTS pipeline (OpenAI Whisper + ElevenLabs) |
| `kazi.serve.app` | FastAPI app with SSE streaming endpoints |

## Requirements

- Python 3.10+
- One of: `anthropic`, `openai`, `google-generativeai`
- LangGraph 1.1+

## License

MIT
