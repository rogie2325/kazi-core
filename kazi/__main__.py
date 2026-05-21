"""
python -m kazi <command> [args]

Commands:
  validate <kazi.yaml>   Parse config, check provider keys, list registered tools.
  config-schema           Print the KaziConfig JSON Schema (for IDEs / LLM validation).
"""
from __future__ import annotations

import asyncio
import sys
import time


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        _usage()
        sys.exit(0)

    cmd = args[0]
    if cmd == "validate":
        asyncio.run(_cmd_validate(args[1:]))
    elif cmd == "config-schema":
        _cmd_config_schema()
    else:
        print(f"Unknown command: {cmd!r}")
        _usage()
        sys.exit(1)


def _usage() -> None:
    print("Usage: python -m kazi <command> [args]")
    print()
    print("Commands:")
    print("  validate <kazi.yaml>   Validate config and check provider connectivity")
    print("  config-schema           Print the KaziConfig JSON Schema")


def _cmd_config_schema() -> None:
    """Print the KaziConfig JSON Schema to stdout — pipe to a file or jq."""
    from kazi.core.schema import kazi_config_schema_json
    print(kazi_config_schema_json())


async def _cmd_validate(args: list[str]) -> None:
    if not args:
        print("Usage: python -m kazi validate <kazi.yaml>")
        sys.exit(1)

    path = args[0]

    # 1. Parse YAML
    try:
        from kazi.core.config import KaziConfig
        config = KaziConfig.from_yaml(path)
        _ok(f"Config parsed: {path}")
    except FileNotFoundError:
        _fail(f"File not found: {path}")
        sys.exit(1)
    except Exception as exc:
        _fail(f"Config parse error: {exc}")
        sys.exit(1)

    # 2. LLM summary
    print(f"  provider : {config.llm.provider.value}")
    print(f"  model    : {config.llm.model}")

    key = config.llm.resolved_api_key()
    if key:
        masked = key[:6] + "..." + key[-4:] if len(key) > 12 else "****"
        print(f"  api_key  : {masked}")
    else:
        _warn("api_key is not set — set KAZI_API_KEY or equivalent env var")

    # 3. RAG config
    if config.rag:
        print(f"  rag      : {config.rag.vector_store.value} / {config.rag.embedding_model}")

    # 4. Memory backend
    print(f"  memory   : {config.memory.backend.value}")

    # 5. MCP servers
    if config.mcp and config.mcp.servers:
        print(f"  mcp      : {list(config.mcp.servers.keys())}")

    # 6. A2A endpoints
    if config.a2a and config.a2a.discovery_endpoints:
        print(f"  a2a      : {config.a2a.discovery_endpoints}")

    # 7. Attempt a live startup (non-fatal — prints warnings on failure)
    print()
    print("Checking connectivity…")
    try:
        from kazi.core.orchestrator import Kazi
        t0 = time.monotonic()
        kazi = await Kazi.create(config)
        elapsed_ms = round((time.monotonic() - t0) * 1000)
        _ok(f"Kazi started in {elapsed_ms}ms — {len(kazi.registry)} tools registered")

        health = await kazi.health()
        for name, result in health.get("checks", {}).items():
            status = result.get("status", "?")
            if status == "ok":
                latency = result.get("latency_ms", "")
                suffix = f" ({latency}ms)" if latency else ""
                _ok(f"{name}{suffix}")
            else:
                _warn(f"{name}: {result.get('error', 'unknown error')}")

        await kazi.close()
        print()
        overall = health.get("status", "unknown")
        if overall == "healthy":
            _ok("All checks passed")
        elif overall == "degraded":
            _warn("Some checks failed — see above")
        else:
            _fail("Connectivity checks failed — see above")

    except ImportError as exc:
        _warn(f"Skipping live check (missing dependency: {exc})")
    except Exception as exc:
        _warn(f"Live check failed: {exc}")


def _ok(msg: str) -> None:
    print(f"\033[32m✓\033[0m {msg}")


def _warn(msg: str) -> None:
    print(f"\033[33m!\033[0m {msg}")


def _fail(msg: str) -> None:
    print(f"\033[31m✗\033[0m {msg}")


if __name__ == "__main__":
    main()
