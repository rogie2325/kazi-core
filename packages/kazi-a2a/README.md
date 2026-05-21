# kazi-a2a

Google A2A (Agent-to-Agent) protocol integration for kazi. Discover and delegate to specialist agents over HTTP.

## Install

```bash
pip install kazi-a2a
```

## What's included

- A2A agent discovery — fetches agent cards from discovery endpoints and registers them as tools
- SSRF protection — private IP ranges, loopback, and non-HTTP schemes are blocked by default
- `allowed_hosts` allowlist — restrict delegation to trusted agent hostnames

## Usage

```python
from kazi import KaziConfig, A2AConfig

config = KaziConfig(
    a2a=A2AConfig(
        discovery_endpoints=["https://agents.internal/summarizer"],
        allowed_hosts=["agents.internal"],
    )
)
```

Discovered agents appear in the tool registry automatically and the orchestrator can delegate to them during a run.

## License

MIT
