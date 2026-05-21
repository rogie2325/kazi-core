from __future__ import annotations

import asyncio
import logging
import re as _re
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Literal

from kazi.core.audit import RunAudit, RunAuditResult, run_context
from kazi.core.config import KaziConfig
from kazi.core.cost import CostAccumulator, CostReport, RunCost, RunResult, TenantCostLedger
from kazi.core.registry import ToolRegistry

logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    from kazi.agents.a2a_client import A2ABridge
    from kazi.brain.graph_builder import GraphBrain
    from kazi.cache.semantic import SemanticCache
    from kazi.core.events import StreamEvent
    from kazi.data.index_manager import IndexManager
    from kazi.memory.profile import UserProfile
    from kazi.tools.mcp_client import MCPBridge
    from kazi.voice.pipeline import VoicePipeline

# Thread ID path-traversal guard — same whitelist as serve/app.py.
# Use an explicit ASCII whitelist (NOT \w) because Python's \w matches Unicode
# word characters like ¹ ϰ ｱ etc., which could survive sanitisation and break
# filesystem-safe assumptions in path-based checkpointer backends.
_SAFE_TID_RE = _re.compile(r"[^A-Za-z0-9_\-:.@]")


def _sanitize_thread_id(tid: str) -> str:
    r"""Whitelist-strip thread IDs: keep [A-Za-z0-9_\-:.@] only."""
    return _SAFE_TID_RE.sub("_", tid.replace("\x00", ""))[:256]


class Kazi:
    """
    Top-level orchestrator that wires all four layers together.

    Usage::

        kazi = await Kazi.create(config)
        response = await kazi.run("Summarise the Q3 report")
        await kazi.close()

    Or as an async context manager (recommended — guarantees cleanup)::

        async with await Kazi.create(config) as kazi:
            response = await kazi.run("...")

    Thread authentication (optional)::

        config = KaziConfig(
            security=SecurityConfig(
                threads=ThreadPolicy(
                    require_auth=True,
                    validator=lambda thread_id, token: verify_jwt(token, thread_id),
                )
            )
        )
        # now every run() call must pass user_token=
        await kazi.run("...", thread_id="user:123:session:abc", user_token=jwt)
    """

    def __init__(self, config: KaziConfig) -> None:
        self.config = config
        self.registry = ToolRegistry()
        self._brain: GraphBrain | None = None
        self._data: IndexManager | None = None
        self._mcp: MCPBridge | None = None
        self._a2a: A2ABridge | None = None
        self._voice: VoicePipeline | None = None
        self._profile_store: UserProfile | None = None
        self._semantic_cache: SemanticCache | None = None
        self._ready = False
        self._ledger = TenantCostLedger()

    # ── factory ───────────────────────────────────────────────────────────

    @classmethod
    async def create(cls, config: KaziConfig | None = None) -> Kazi:
        inst = cls(config or KaziConfig())
        await inst._startup()
        return inst

    async def _startup(self) -> None:
        security = self.config.security

        # 1. Long-term user profile store (always-on, zero overhead when not used)
        from kazi.memory.profile import UserProfile
        self._profile_store = UserProfile()

        # 2. Semantic cache — optional
        if self.config.semantic_cache is not None:
            from kazi.cache.semantic import SemanticCache
            self._semantic_cache = SemanticCache(self.config.semantic_cache)

        # 3. Voice pipeline — optional
        if self.config.voice is not None:
            from kazi.voice.pipeline import VoicePipeline
            self._voice = VoicePipeline(self, self.config.voice)

        # 4. Data layer — no I/O yet
        from kazi.data.index_manager import IndexManager
        self._data = IndexManager(self.config.rag, self.config.llm)

        # 5. MCP bridge
        if self.config.mcp.servers:
            from kazi.tools.mcp_client import MCPBridge
            self._mcp = MCPBridge(
                self.config.mcp,
                self.registry,
                security=security.mcp,
                validate_args=security.mcp.validate_args,
            )
            await self._mcp.connect_all()

        # 6. A2A bridge
        if self.config.a2a.discovery_endpoints:
            from kazi.agents.a2a_client import A2ABridge
            self._a2a = A2ABridge(self.config.a2a, self.registry, security=security)
            await self._a2a.discover_agents()

        # 7. Declarative tool imports from YAML config (no Python required)
        if self.config.tools_imports:
            self._process_tools_imports()

        # 8. Brain — built last so it sees the full registry
        from kazi.brain.graph_builder import GraphBrain
        self._brain = GraphBrain(self.config, self.registry)

        self._ready = True
        logger.info(
            "Kazi ready — %d tools registered across categories: %s",
            len(self.registry),
            self.registry.list_categories(),
        )

    # ── document ingestion ────────────────────────────────────────────────

    async def ingest(
        self,
        path: str,
        index_name: str = "default",
        *,
        tool_name: str | None = None,
        tool_description: str | None = None,
    ) -> None:
        self._assert_ready()
        data = self._require_data()
        await data.ingest_directory(path, index_name=index_name)
        tool_def = data.as_tool_definition(
            index_name, tool_name=tool_name, description=tool_description,
        )
        if tool_def.name in self.registry:
            self.registry.unregister(tool_def.name)
        self.registry.register(tool_def, category="rag")

    async def ingest_documents(
        self,
        documents: list[dict],
        index_name: str = "default",
        *,
        tool_name: str | None = None,
        tool_description: str | None = None,
    ) -> None:
        self._assert_ready()
        data = self._require_data()
        await data.ingest_documents(documents, index_name=index_name)
        tool_def = data.as_tool_definition(
            index_name, tool_name=tool_name, description=tool_description,
        )
        if tool_def.name in self.registry:
            self.registry.unregister(tool_def.name)
        self.registry.register(tool_def, category="rag")

    # ── tool management ───────────────────────────────────────────────────

    def add_tool(self, func_or_def, *, name=None, description=None, category="custom") -> None:
        """Register a tool. Accepts either a ToolDefinition or a plain Python function."""
        from kazi.core.registry import ToolDefinition
        if isinstance(func_or_def, ToolDefinition):
            self.registry.register(func_or_def, category=category)
        else:
            self.registry.register_function(
                func_or_def, name=name, description=description, category=category
            )

    def _process_tools_imports(self) -> None:
        """
        Apply every entry in ``config.tools_imports`` to this instance.

        Each entry is a dict with one of:
          - "import":  "pkg.module.func"        — register a single function
          - "module":  "pkg.module" + only=...  — bulk-scan a module
          - "openapi": "<url-or-dict>"          — import REST endpoints

        Errors are logged but do not abort startup — partial integration is
        more useful than a hard fail when one of N sources is misconfigured.
        """

        for i, entry in enumerate(self.config.tools_imports):
            try:
                if "import" in entry:
                    self._import_single_function(entry)
                elif "module" in entry:
                    self._import_module_scan(entry)
                elif "openapi" in entry:
                    self._import_openapi(entry)
                else:
                    logger.warning(
                        "tools[%d]: unknown directive (need 'import', 'module', or 'openapi'): %s",
                        i, entry,
                    )
            except Exception as exc:
                logger.error("tools[%d] import failed: %s", i, exc)

    def _import_single_function(self, entry: dict) -> None:
        """Handle {"import": "pkg.module.func", "name": ..., "category": ...}"""
        import importlib

        dotted = entry["import"]
        module_path, _, func_name = dotted.rpartition(".")
        if not module_path:
            raise ValueError(f"'import' must be a dotted path, got {dotted!r}")
        mod = importlib.import_module(module_path)
        fn = getattr(mod, func_name)
        self.add_tool(
            fn,
            name=entry.get("name"),
            description=entry.get("description"),
            category=entry.get("category", "imported"),
        )

    def _import_module_scan(self, entry: dict) -> None:
        """Handle {"module": "pkg.module", "only": [...], "exclude": [...]}"""
        import importlib

        from kazi.integration.scanner import register_module
        mod = importlib.import_module(entry["module"])
        register_module(
            self,
            mod,
            only=entry.get("only"),
            exclude=entry.get("exclude"),
            category=entry.get("category"),
            include_undecorated=entry.get("include_undecorated"),
        )

    def _import_openapi(self, entry: dict) -> None:
        """Handle {"openapi": "<url>", "base_url": "...", "allowlist": [...]}"""
        from kazi.integration.openapi_import import from_openapi_spec
        from_openapi_spec(
            self,
            entry["openapi"],
            base_url=entry.get("base_url"),
            auth_header=entry.get("auth_header"),
            allowlist=entry.get("allowlist"),
            denylist=entry.get("denylist"),
            category=entry.get("category", "openapi"),
            timeout=float(entry.get("timeout", 30.0)),
        )

    # ── cost budget helpers ───────────────────────────────────────────────

    async def _check_daily_budget(self, user_id: str, tenant_id: str) -> None:
        """Raise BudgetExceededError if the user or tenant has hit their daily cap."""
        await self._ledger.check_budget(
            tenant_id=tenant_id,
            user_id=user_id,
            limit_usd=self.config.max_daily_cost_per_user_usd,
        )

    async def _record_spend(self, user_id: str, tenant_id: str, cost: RunCost) -> None:
        await self._ledger.record(tenant_id=tenant_id, user_id=user_id, cost=cost)

    def _check_per_run_budget(self, cost_usd: float) -> None:
        limit = self.config.max_cost_per_run_usd
        if limit <= 0:
            return
        if cost_usd > limit:
            from kazi.core.exceptions import BudgetExceededError
            raise BudgetExceededError(
                f"Run cost ${cost_usd:.6f} exceeds per-run limit ${limit:.6f}."
            )

    async def get_spend_report(
        self,
        *,
        tenant_id: str | None = None,
        user_id: str | None = None,
        date: str | None = None,
    ) -> list[CostReport]:
        """
        Return today's spending report, optionally filtered by tenant or user.

        Use this to power cost dashboards, per-team budget alerts, or
        chargeback reports in your billing system::

            rows = await kazi.get_spend_report(tenant_id="acme")
            for row in rows:
                print(row)   # [2026-05-13] tenant='acme' — $0.042 | 3 runs
        """
        return await self._ledger.report(
            tenant_id=tenant_id, user_id=user_id, date=date
        )

    # ── inference ─────────────────────────────────────────────────────────

    async def run(
        self,
        message: str,
        *,
        thread_id: str = "default",
        max_tool_calls: int = 25,
        system_prompt: str | None = None,
        response_schema=None,
        track_cost: bool = False,
        user_token: str | None = None,
        images: list | None = None,
        user_id: str | None = None,
        tenant_id: str | None = None,
        audit: bool = False,
        shadow: bool = False,
    ) -> str | RunResult | RunAuditResult:
        """
        Run a conversation turn.

        message          The user's input text.
        thread_id        Conversation thread. Same thread_id shares memory across runs.
        max_tool_calls   Hard cap on tool iterations per run.
        system_prompt    Override the auto-generated system prompt for this run.
        response_schema  Pydantic model. Parses the reply into the schema via structured output.
        track_cost       Return RunResult(reply, cost) instead of a plain string.
        user_token       Required when ThreadPolicy.require_auth=True.
        images           List of image paths, URLs, or bytes for vision models.
        user_id          When set, injects the user's long-term profile into the system prompt
                         and enforces their daily spending cap.
        tenant_id        Restricts visible tools to those configured for this tenant in
                         KaziConfig.tenant_tools.
        audit            Collect a RunAudit alongside the reply.  When True, the return type
                         becomes RunAuditResult.  Use for validator dashboards and replay.
        shadow           Dry-run mode: tool calls are intercepted and replaced with a stub
                         so no side effects hit the real system.  The agent still reasons
                         over the stub result and produces a reply.  Combine with audit=True
                         to capture exactly which tools would have fired.
        """
        self._assert_ready()
        thread_id = _sanitize_thread_id(thread_id)

        # Tenant thread isolation: prefix thread_id with the tenant namespace so
        # tenant A cannot access tenant B's threads by guessing IDs.
        if tenant_id:
            _prefix = f"t:{_sanitize_thread_id(tenant_id)}:"
            if not thread_id.startswith(_prefix):
                thread_id = _prefix + thread_id

        self.config.security.threads.check(thread_id, user_token)

        # 1. Prompt injection detection
        self.config.security.injection.check(message)

        # 2. Daily budget check (before spending anything)
        await self._check_daily_budget(user_id or "", tenant_id or "")

        # 3. Semantic cache lookup
        cache_key_message = message  # preserve original for cache
        if self._semantic_cache is not None:
            cached_reply = await self._semantic_cache.get(message)
            if cached_reply is not None:
                from kazi.utils.metrics import record_cache_hit
                record_cache_hit("semantic")
                if track_cost:
                    return RunResult(reply=cached_reply, cost=RunCost())
                return cached_reply
            from kazi.utils.metrics import record_cache_miss
            record_cache_miss("semantic")

        # 4. Inject long-term user profile
        effective_prompt = system_prompt
        if user_id and self._profile_store:
            preamble = self._profile_store.as_system_preamble(user_id)
            if preamble:
                effective_prompt = (
                    f"{preamble}\n\n{system_prompt}" if system_prompt else preamble
                )

        # 5. Cost accumulator
        accumulator = CostAccumulator(self.config.llm.model) if track_cost else None

        # 6. Run the brain (inside audit / shadow context if requested)
        brain = self._require_brain()
        with run_context(
            audit=audit,
            shadow=shadow,
            thread_id=thread_id,
            tenant_id=tenant_id or "",
            user_id=user_id or "",
        ) as ctx:
            try:
                reply = await brain.run(
                    message,
                    thread_id=thread_id,
                    max_tool_calls=max_tool_calls,
                    system_prompt=effective_prompt,
                    images=images,
                    tenant_id=tenant_id,
                )
            except Exception as exc:
                if ctx.recorder is not None:
                    ctx.recorder.finalize(error=str(exc))
                raise
            run_audit: RunAudit | None = ctx.recorder.finalize() if ctx.recorder else None

        # 7. Structured output parsing
        if response_schema is not None:
            reply = await self._parse_structured(reply, response_schema)

        # 8. Output guardrails
        if self.config.guardrails is not None:
            from kazi.core.guardrails import check_output
            result = check_output(
                reply if isinstance(reply, str) else str(reply),
                self.config.guardrails,
            )
            reply = result.text

        # 9. Cost tracking, budget enforcement, spend recording
        run_cost: RunCost | None = None
        if track_cost:
            from kazi.core.token_budget import count_tokens
            from kazi.utils.metrics import record_cost, record_tokens
            out_tokens = count_tokens(reply if isinstance(reply, str) else str(reply))
            in_tokens = count_tokens(message)
            if accumulator:
                accumulator.record(in_tokens, out_tokens)
            run_cost = accumulator.to_run_cost() if accumulator else RunCost()
            self._check_per_run_budget(run_cost.cost_usd)
            await self._record_spend(user_id or "", tenant_id or "", run_cost)
            record_tokens(run_cost.model, run_cost.input_tokens, run_cost.output_tokens)
            record_cost(run_cost.model, run_cost.cost_usd, tenant_id or "")

        # Store in semantic cache (skip cache writes for shadow runs — the
        # stubbed tool results would poison the cache for real traffic)
        if self._semantic_cache is not None and isinstance(reply, str) and not shadow:
            await self._semantic_cache.set(cache_key_message, reply)

        # Compose return value based on audit / track_cost flags
        if audit and run_audit is not None:
            return RunAuditResult(reply=reply, audit=run_audit, cost=run_cost)
        if track_cost:
            return RunResult(reply=reply, cost=run_cost or RunCost())
        return reply

    async def _parse_structured(self, text: str, schema):
        """Use with_structured_output to parse `text` into `schema`."""
        from langchain_core.messages import HumanMessage as LCHuman
        llm = self._require_brain()._build_llm()
        structured_llm = llm.with_structured_output(schema)
        result = await structured_llm.ainvoke([
            LCHuman(content=f"Parse the following into the requested format:\n\n{text}")
        ])
        return result

    async def run_with_approval(
        self,
        message: str,
        *,
        thread_id: str = "default",
        max_tool_calls: int = 25,
        system_prompt: str | None = None,
        approval_callback,
        user_token: str | None = None,
    ) -> str:
        """
        Run with human-in-the-loop approval before every tool execution.

        approval_callback   async callable(tool_calls: list[dict]) -> list[dict] | None
                            Receives pending tool calls before execution.
                            Return the (possibly modified) list to approve.
                            Return None to skip all tool calls for this turn.

        Example::

            async def my_approval(tool_calls):
                for call in tool_calls:
                    print(f"Agent wants: {call['name']}({call['args']})")
                answer = input("Approve? [y/n]: ")
                return tool_calls if answer == "y" else None

            reply = await kazi.run_with_approval(
                "Delete the staging database",
                approval_callback=my_approval,
                thread_id="user:123",
            )
        """
        self._assert_ready()
        thread_id = _sanitize_thread_id(thread_id)
        self.config.security.threads.check(thread_id, user_token)
        self.config.security.injection.check(message)
        return await self._require_brain().run_with_approval(
            message,
            thread_id=thread_id,
            max_tool_calls=max_tool_calls,
            system_prompt=system_prompt,
            approval_callback=approval_callback,
        )

    async def batch_run(
        self,
        messages: list[str],
        *,
        concurrency: int = 5,
        thread_id_prefix: str = "batch",
        on_error: Literal["skip", "raise"] = "skip",
        **run_kwargs,
    ) -> list:
        """
        Run many prompts concurrently with a bounded concurrency limit.

        messages         List of user messages to process.
        concurrency      Max simultaneous runs (default 5 — tune to your rate limit).
        thread_id_prefix Each prompt gets its own thread: "{prefix}:{index}".
        on_error         "skip" — failed items become Exception objects in the result list.
                         "raise" — first failure propagates immediately.
        **run_kwargs     Forwarded to kazi.run() (system_prompt, max_tool_calls, etc.).
                         Do not pass thread_id — it is generated per-item.

        Returns a list parallel to ``messages``.  Each element is either the
        reply string (or RunResult when track_cost=True) or an Exception if that
        item failed and on_error="skip".

        Example::

            results = await kazi.batch_run(
                ["Summarise doc A", "Summarise doc B", "Summarise doc C"],
                concurrency=3,
                max_tool_calls=5,
            )
            for msg, res in zip(messages, results):
                if isinstance(res, Exception):
                    print(f"FAILED: {msg}: {res}")
                else:
                    print(f"OK: {res[:80]}")
        """
        self._assert_ready()
        sem = asyncio.Semaphore(concurrency)

        async def _run_one(index: int, msg: str):
            tid = f"{thread_id_prefix}:{index}"
            async with sem:
                try:
                    return await self.run(msg, thread_id=tid, **run_kwargs)
                except Exception as exc:
                    if on_error == "raise":
                        raise
                    logger.warning("batch_run item %d failed: %s", index, exc)
                    return exc

        return list(await asyncio.gather(
            *[_run_one(i, msg) for i, msg in enumerate(messages)]
        ))

    async def stream(
        self,
        message: str,
        *,
        thread_id: str = "default",
        max_tool_calls: int = 25,
        system_prompt: str | None = None,
        user_token: str | None = None,
        images: list | None = None,
        user_id: str | None = None,
        tenant_id: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream raw text tokens as they arrive from the LLM."""
        self._assert_ready()
        thread_id = _sanitize_thread_id(thread_id)
        self.config.security.threads.check(thread_id, user_token)
        self.config.security.injection.check(message)
        effective_prompt = system_prompt
        if user_id and self._profile_store:
            preamble = self._profile_store.as_system_preamble(user_id)
            if preamble:
                effective_prompt = (
                    f"{preamble}\n\n{system_prompt}" if system_prompt else preamble
                )
        async for chunk in self._require_brain().stream(
            message, thread_id=thread_id, max_tool_calls=max_tool_calls,
            system_prompt=effective_prompt, images=images, tenant_id=tenant_id,
        ):
            yield chunk

    async def stream_events(
        self,
        message: str,
        *,
        thread_id: str = "default",
        max_tool_calls: int = 25,
        system_prompt: str | None = None,
        user_token: str | None = None,
        images: list | None = None,
        user_id: str | None = None,
        tenant_id: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """
        Stream typed events as the agent executes.

        Yields StreamEvent dicts with ``type`` in:
          "token"       — LLM text token.  ``data`` = token string.
          "tool_start"  — tool invocation starting.  ``data`` = tool name.
                          ``metadata["args"]`` = argument dict.
          "tool_end"    — tool finished.  ``data`` = tool name.
                          ``metadata["result"]`` = first 200 chars of result.
          "done"        — stream complete.
          "error"       — unrecoverable error.  ``data`` = error message.

        Use this instead of ``stream()`` when your frontend needs to show
        tool-use spinners, typing indicators, or cost tickers in real time.

        Example (server-sent events over FastAPI)::

            async def token_generator():
                async for event in kazi.stream_events(message, thread_id=tid):
                    yield f"data: {json.dumps(event)}\\n\\n"
        """
        self._assert_ready()
        thread_id = _sanitize_thread_id(thread_id)
        self.config.security.threads.check(thread_id, user_token)
        self.config.security.injection.check(message)
        effective_prompt = system_prompt
        if user_id and self._profile_store:
            preamble = self._profile_store.as_system_preamble(user_id)
            if preamble:
                effective_prompt = (
                    f"{preamble}\n\n{system_prompt}" if system_prompt else preamble
                )
        async for event in self._require_brain().stream_events(
            message, thread_id=thread_id, max_tool_calls=max_tool_calls,
            system_prompt=effective_prompt, images=images, tenant_id=tenant_id,
        ):
            yield event

    async def branch_thread(
        self,
        source_thread_id: str,
        branch_thread_id: str,
    ) -> None:
        """
        Fork ``source_thread_id`` into a new ``branch_thread_id``.

        The branch starts with an identical copy of the source's conversation
        history.  Subsequent runs on either thread are fully independent —
        changes to one do not affect the other.

        Raises ValueError if the source thread has no saved state yet.

        Example::

            # Run several turns on "main"
            await kazi.run("Hello", thread_id="main")
            await kazi.run("Tell me about Python", thread_id="main")

            # Fork at this exact point
            await kazi.branch_thread("main", "main:experiment-A")
            await kazi.branch_thread("main", "main:experiment-B")

            # Both branches start from the same history but diverge independently
            reply_a = await kazi.run("How does GIL work?", thread_id="main:experiment-A")
            reply_b = await kazi.run("What is asyncio?",   thread_id="main:experiment-B")
        """
        self._assert_ready()
        source_thread_id = _sanitize_thread_id(source_thread_id)
        branch_thread_id = _sanitize_thread_id(branch_thread_id)
        await self._require_brain().branch_thread(source_thread_id, branch_thread_id)

    async def run_voice(
        self,
        audio: bytes,
        *,
        thread_id: str = "default",
        user_token: str | None = None,
    ) -> bytes:
        """
        Full voice round-trip: transcribe → run LLM → synthesize.
        Returns MP3 audio bytes.

        Requires VoiceConfig to be set on KaziConfig.
        Thread memory is shared with text chat sessions on the same thread_id.
        """
        self._assert_ready()
        self._assert_voice()
        self.config.security.threads.check(thread_id, user_token)
        voice = self._require_voice()
        return await voice.run(audio, thread_id=thread_id)

    async def stream_voice(
        self,
        audio: bytes,
        *,
        thread_id: str = "default",
        user_token: str | None = None,
    ) -> AsyncIterator[bytes]:
        """
        Low-latency streaming voice: transcribe → stream LLM tokens → TTS chunks.
        Yields MP3 audio chunks — pipe directly to a WebSocket or WebRTC data channel.

        Audio starts arriving before the LLM finishes, keeping latency under ~500ms.
        Thread memory is shared with text chat on the same thread_id.
        """
        self._assert_ready()
        self._assert_voice()
        self.config.security.threads.check(thread_id, user_token)
        voice = self._require_voice()
        async for chunk in voice.stream(audio, thread_id=thread_id):
            yield chunk

    # ── lifecycle ─────────────────────────────────────────────────────────

    def as_app(
        self,
        *,
        prefix: str = "",
        api_key: str | None = None,
        cors_origins=None,
        rate_limit_per_minute: int = 0,
        max_body_bytes: int = 1 * 1024 * 1024,
        allowed_ips=None,
        enable_audit_log: bool = True,
        max_concurrent_runs: int = 50,
        request_timeout_seconds: int = 120,
        graceful_shutdown_timeout: int = 30,
    ):
        """
        Return a FastAPI application that exposes this Kazi instance over HTTP.

        Routes (all prefixed with `prefix`):
          POST /run           — single-turn text chat (JSON)
          POST /stream        — SSE raw token streaming
          POST /events        — SSE typed event streaming (token/tool_start/tool_end/done)
          POST /ingest        — document ingestion
          WS   /voice         — real-time voice (requires VoiceConfig)
          GET  /health        — health check (no auth required)
          GET  /metrics       — usage metrics

        Requires: pip install kazi-core[serve]

        Example::

            import uvicorn
            app = kazi.as_app(api_key="secret", cors_origins=["https://yourapp.com"])
            uvicorn.run(app, host="0.0.0.0", port=8000)
        """
        from kazi.serve.app import build_app
        return build_app(
            self,
            prefix=prefix,
            api_key=api_key,
            cors_origins=cors_origins,
            rate_limit_per_minute=rate_limit_per_minute,
            max_body_bytes=max_body_bytes,
            allowed_ips=allowed_ips,
            enable_audit_log=enable_audit_log,
            max_concurrent_runs=max_concurrent_runs,
            request_timeout_seconds=request_timeout_seconds,
            graceful_shutdown_timeout=graceful_shutdown_timeout,
        )

    async def health(self) -> dict:
        """
        Check connectivity for all active subsystems.

        Returns a dict with shape::

            {
                "status": "healthy" | "degraded" | "unhealthy",
                "checks": {
                    "checkpointer": {"status": "ok", "latency_ms": 4},
                    "mcp:filesystem": {"status": "error", "error": "..."},
                    "vector_store":   {"status": "ok",   "latency_ms": 12},
                }
            }

        HTTP servers can expose this directly as a liveness/readiness probe.
        """
        import time

        checks: dict[str, dict] = {}

        # Checkpointer
        try:
            start = time.monotonic()
            checkpointer = self._require_brain()._get_checkpointer()
            if hasattr(checkpointer, "alist"):
                async for _ in checkpointer.alist({"configurable": {"thread_id": "__health__"}}):
                    break
            checks["checkpointer"] = {"status": "ok", "latency_ms": round((time.monotonic() - start) * 1000, 1)}
        except Exception as exc:
            checks["checkpointer"] = {"status": "error", "error": str(exc)}

        # MCP servers
        if self._mcp:
            for name, handle in self._mcp._handles.items():
                try:
                    start = time.monotonic()
                    _ = handle.tools
                    checks[f"mcp:{name}"] = {"status": "ok", "latency_ms": round((time.monotonic() - start) * 1000, 1)}
                except Exception as exc:
                    checks[f"mcp:{name}"] = {"status": "error", "error": str(exc)}

        # Vector store
        if self._data:
            try:
                start = time.monotonic()
                await self._data.ping()
                checks["vector_store"] = {"status": "ok", "latency_ms": round((time.monotonic() - start) * 1000, 1)}
            except Exception as exc:
                checks["vector_store"] = {"status": "error", "error": str(exc)}

        # A2A bridge
        if self._a2a:
            checks["a2a"] = {"status": "ok", "agents": len(self._a2a._agents) if hasattr(self._a2a, "_agents") else "?"}

        # Semantic cache
        if self._semantic_cache and self.config.semantic_cache is not None:
            checks["semantic_cache"] = {
                "status": "ok",
                "backend": self.config.semantic_cache.backend,
                "threshold": self.config.semantic_cache.similarity_threshold,
            }

        errors = [k for k, v in checks.items() if v.get("status") == "error"]
        if not checks:
            overall = "healthy"
        elif len(errors) == len(checks):
            overall = "unhealthy"
        elif errors:
            overall = "degraded"
        else:
            overall = "healthy"

        return {"status": overall, "checks": checks}

    async def close(self) -> None:
        if self._mcp:
            await self._mcp.disconnect_all()
        if self._a2a:
            await self._a2a.close()
        self._ready = False

    async def __aenter__(self) -> Kazi:
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()

    # ── layer access ─────────────────────────────────────────────────────

    @property
    def data(self):
        return self._data

    @property
    def mcp(self):
        return self._mcp

    @property
    def a2a(self):
        return self._a2a

    # ── helpers ───────────────────────────────────────────────────────────

    def _assert_ready(self) -> None:
        if not self._ready:
            raise RuntimeError(
                "Kazi is not initialised. Use `await Kazi.create(config)` "
                "instead of calling Kazi() directly."
            )

    def _require_brain(self) -> GraphBrain:
        if self._brain is None:
            raise RuntimeError("Kazi brain is not initialised")
        return self._brain

    def _require_data(self) -> IndexManager:
        if self._data is None:
            raise RuntimeError("Kazi data layer is not initialised")
        return self._data

    def _require_voice(self) -> VoicePipeline:
        if self._voice is None:
            raise RuntimeError("Voice pipeline is not initialised")
        return self._voice

    def _assert_voice(self) -> None:
        if self._voice is None:
            raise RuntimeError(
                "Voice is not configured. Add VoiceConfig to KaziConfig:\n"
                "  config = KaziConfig(voice=VoiceConfig(stt_provider=STTProvider.OPENAI, ...))\n"
                "  pip install kazi-voice[openai]"
            )
