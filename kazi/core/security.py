"""
Security configuration and middleware for kazi.

Design principle: secure by default, loosening requires explicit opt-in.

What this module covers
────────────────────────
1. MCPSecurityPolicy   — allowlist / denylist which MCP tools can be registered
2. ContentPolicy       — tag all external content as untrusted before it enters LLM context,
                         enforce result length limits, provide inspection hooks
3. ThreadPolicy        — bind thread IDs to a user identity, deny unauth'd access
4. SecurityConfig      — top-level container wired into KaziConfig

What this module does NOT cover (intentionally)
────────────────────────────────────────────────
- Network-level controls (TLS termination, firewalling) — handled by infrastructure
- Complete prompt-injection prevention — no library can guarantee this; we mitigate
- Docker/VM-grade sandbox isolation — the subprocess sandbox is hardened but not containerised
"""
from __future__ import annotations

import fnmatch
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)


def _require_callable(value, label: str) -> None:
    if value is not None and not callable(value):
        from kazi.core.exceptions import ConfigurationError
        raise ConfigurationError(f"{label} must be callable or None, got {type(value).__name__}")

# ── MCP ───────────────────────────────────────────────────────────────────────


@dataclass
class MCPSecurityPolicy:
    """
    Controls which tools discovered from MCP servers are allowed into the registry.

    Evaluation order: denylist beats allowlist.

    Examples::

        # Only allow file reads — nothing that writes or executes
        MCPSecurityPolicy(allowlist=["filesystem__read_*"])

        # Allow everything except dangerous ops
        MCPSecurityPolicy(denylist=["filesystem__delete_*", "shell__*"])

        # Exact list per server
        MCPSecurityPolicy(allowlist=["github__list_prs", "github__get_pr", "github__create_comment"])
    """

    allowlist: list[str] = field(default_factory=list)
    denylist: list[str] = field(default_factory=list)
    validate_args: bool = True  # validate LLM-supplied args against declared parameter schema

    def __post_init__(self) -> None:
        from kazi.core.exceptions import ConfigurationError
        for attr in ("allowlist", "denylist"):
            val = getattr(self, attr)
            if not isinstance(val, list):
                raise ConfigurationError(
                    f"MCPSecurityPolicy.{attr} must be a list of strings, got {type(val).__name__}"
                )
            for pattern in val:
                if not isinstance(pattern, str):
                    raise ConfigurationError(
                        f"MCPSecurityPolicy.{attr} patterns must be strings, got {type(pattern).__name__}"
                    )

    def is_allowed(self, tool_name: str) -> bool:
        for pattern in self.denylist:
            if fnmatch.fnmatch(tool_name, pattern):
                logger.warning("Tool '%s' blocked by denylist pattern '%s'", tool_name, pattern)
                return False
        if self.allowlist:
            allowed = any(fnmatch.fnmatch(tool_name, p) for p in self.allowlist)
            if not allowed:
                logger.warning("Tool '%s' not in MCP allowlist — skipping", tool_name)
            return allowed
        return True


# ── Content policy ────────────────────────────────────────────────────────────


_EXTERNAL_WRAP = (
    "<external_content source=\"{source}\">\n"
    "{content}\n"
    "</external_content>"
)

_TRUNCATION_NOTE = "\n[... truncated at {limit} chars — original length {original} chars]"


@dataclass
class ContentPolicy:
    """
    Controls how tool results enter the LLM context.

    tag_external_content
        Wraps every tool result in <external_content> XML tags.
        This creates a clear trust boundary visible to the LLM — injected
        instructions inside tool results are less likely to be followed when
        the LLM understands they came from an untrusted external source.
        This is a mitigation, not a guarantee. Set False only in fully
        controlled environments where every tool result is trusted.

    max_result_chars
        Hard cap on individual tool result length before it enters context.
        Prevents a single runaway result from consuming the whole context window.

    on_tool_call(tool_name, args) → args | None
        Called before every tool execution. Return None to block the call.
        Return a (possibly modified) args dict to allow it.

    on_tool_result(tool_name, result) → result | None
        Called after every tool execution. Return None to block the result
        from entering context (raises ToolBlockedError). Return a (possibly
        modified) string to allow it.
    """

    tag_external_content: bool = True
    max_result_chars: int = 50_000
    on_tool_call: Callable[[str, dict], dict | None] | None = None
    on_tool_result: Callable[[str, str], str | None] | None = None

    def __post_init__(self) -> None:
        from kazi.core.exceptions import ConfigurationError
        if self.max_result_chars <= 0:
            raise ConfigurationError(
                f"ContentPolicy.max_result_chars must be a positive integer, got {self.max_result_chars}"
            )
        _require_callable(self.on_tool_call, "ContentPolicy.on_tool_call")
        _require_callable(self.on_tool_result, "ContentPolicy.on_tool_result")

    def wrap(self, tool_name: str, result: str) -> str:
        """Apply length limit, optional hook, and optional tagging to a tool result."""
        from kazi.core.exceptions import ToolBlockedError

        # 1. Enforce length limit
        original_len = len(result)
        if original_len > self.max_result_chars:
            result = result[: self.max_result_chars] + _TRUNCATION_NOTE.format(
                limit=self.max_result_chars, original=original_len
            )

        # 2. Run user hook
        if self.on_tool_result is not None:
            modified = self.on_tool_result(tool_name, result)
            if modified is None:
                raise ToolBlockedError(
                    f"Tool result from '{tool_name}' blocked by on_tool_result hook"
                )
            result = modified

        # 3. Tag as external
        if self.tag_external_content:
            result = _EXTERNAL_WRAP.format(source=tool_name, content=result)

        return result

    def check_call(self, tool_name: str, args: dict) -> dict:
        """
        Run the on_tool_call hook.
        Returns (possibly modified) args, or raises ToolBlockedError if blocked.
        """
        from kazi.core.exceptions import ToolBlockedError

        if self.on_tool_call is not None:
            modified = self.on_tool_call(tool_name, args)
            if modified is None:
                raise ToolBlockedError(
                    f"Tool call '{tool_name}' blocked by on_tool_call hook"
                )
            return modified
        return args


# ── Thread policy ─────────────────────────────────────────────────────────────


@dataclass
class ThreadPolicy:
    """
    Controls access to conversation threads.

    When require_auth=True, every call to kazi.run() must supply a
    user_token kwarg. The validator callable decides whether that token
    authorises access to the given thread_id.

    Example — JWT-based ownership::

        def validate(thread_id: str, token: str | None) -> bool:
            if not token:
                return False
            claims = jwt.decode(token, PUBLIC_KEY, algorithms=["RS256"])
            # thread IDs are namespaced as "<user_id>:<session_id>"
            return thread_id.startswith(claims["sub"] + ":")

        ThreadPolicy(require_auth=True, validator=validate)
    """

    require_auth: bool = False
    validator: Callable[[str, str | None], bool] | None = None

    def __post_init__(self) -> None:
        from kazi.core.exceptions import ConfigurationError
        _require_callable(self.validator, "ThreadPolicy.validator")
        if self.require_auth and self.validator is None:
            raise ConfigurationError(
                "ThreadPolicy.require_auth=True requires a validator callable — "
                "set validator=lambda thread_id, token: <your auth logic>"
            )

    def check(self, thread_id: str, user_token: str | None) -> None:
        """Raise ThreadAuthError if access is denied."""
        if not self.require_auth:
            return
        from kazi.core.exceptions import ThreadAuthError

        if self.validator is None:
            raise ThreadAuthError(
                "ThreadPolicy.require_auth=True but no validator is configured"
            )
        if not self.validator(thread_id, user_token):
            raise ThreadAuthError(f"Access denied to thread '{thread_id}'")


# ── Prompt injection detection ────────────────────────────────────────────────

_DEFAULT_INJECTION_PATTERNS: list[str] = [
    r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+instructions",
    r"forget\s+(everything|all|prior|previous|your|the\s+above)",
    r"disregard\s+(all\s+)?(previous|prior|above|your)",
    r"you\s+are\s+now\s+(?:a|an|the)\s+\w+",
    r"pretend\s+(you\s+are|to\s+be)",
    r"act\s+as\s+(?:a|an|the)\s+\w+",
    r"your\s+new\s+(role|instructions|persona|system\s+prompt)",
    r"new\s+(system\s+prompt|instructions|persona)",
    r"\[SYSTEM\]",
    r"override\s+(your\s+)?(instructions|guidelines|constraints|rules)",
    r"jailbreak",
    r"DAN\s+mode",
]


@dataclass
class InjectionDetectionConfig:
    """
    Prompt-injection detection applied to every incoming user message.

    enabled          Master switch.
    mode             "warn" — log and continue; "block" — raise InjectionDetectedError.
    patterns         Additional regex patterns (merged with built-ins).
    use_built_ins    When False, only ``patterns`` are checked (skip built-ins).
    """
    enabled: bool = True
    mode: Literal["warn", "block"] = "warn"
    patterns: list[str] = field(default_factory=list)
    use_built_ins: bool = True

    def check(self, message: str) -> str | None:
        """
        Check ``message`` for injection patterns.

        Returns the first matching pattern label, or None if clean.
        Raises InjectionDetectedError when mode='block' and a match is found.
        """
        if not self.enabled:
            return None
        from kazi.core.exceptions import InjectionDetectedError

        to_check = (
            (_DEFAULT_INJECTION_PATTERNS + list(self.patterns))
            if self.use_built_ins
            else list(self.patterns)
        )
        for pattern in to_check:
            try:
                if re.search(pattern, message, re.IGNORECASE | re.DOTALL):
                    label = pattern[:50]
                    logger.warning("Possible prompt injection detected — pattern: %s", label)
                    if self.mode == "block":
                        raise InjectionDetectedError(
                            f"Message blocked: matched injection pattern '{label}'"
                        )
                    return label
            except re.error as exc:
                logger.error("Invalid injection-detection regex '%s': %s", pattern[:50], exc)
        return None


# ── Top-level container ───────────────────────────────────────────────────────


@dataclass
class SecurityConfig:
    """
    Top-level security configuration — wired into KaziConfig.

    Secure defaults:
      - External content is tagged
      - Tool results are length-capped at 50,000 chars
      - TLS is verified on all outbound connections
      - Thread auth is off (opt-in)
      - MCP has no allowlist/denylist (allow all — add restrictions per deployment)
    """

    content: ContentPolicy = field(default_factory=ContentPolicy)
    threads: ThreadPolicy = field(default_factory=ThreadPolicy)
    mcp: MCPSecurityPolicy = field(default_factory=MCPSecurityPolicy)
    injection: InjectionDetectionConfig = field(default_factory=InjectionDetectionConfig)
    verify_tls: bool = True
