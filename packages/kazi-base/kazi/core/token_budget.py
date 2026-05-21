"""
Token budget tracking and conversation summarisation.

Token counting
──────────────
Uses tiktoken when available (accurate for OpenAI models, reasonable for others).
Falls back to a character-ratio estimate (~4 chars per token) when tiktoken is
not installed, which is accurate enough for budget warnings.

Summarisation
─────────────
When a thread's message history exceeds `summarize_after_turns`, the oldest
messages are compressed into a single summary message. This keeps the context
window from growing unboundedly across long sessions without losing meaning.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class TokenBudgetConfig:
    """
    Per-run token budget and history compression settings.

    max_tokens_per_run
        Hard stop on total tokens (input + output) across all LLM calls in
        one kazi.run() invocation. Raises TokenBudgetExceeded when hit.
        None = no limit.

    warn_at_fraction
        Log a WARNING when this fraction of the budget is consumed. 0.8 = 80%.

    summarize_after_turns
        After this many message turns accumulate in a thread, compress the
        oldest half into a single summary. 0 = disabled.

    max_chars_per_tool_result
        Per-result character cap applied before the result enters context.
        Duplicates ContentPolicy.max_result_chars intentionally — the budget
        layer truncates for cost reasons, the security layer truncates for
        safety reasons. Both gates run.
    """

    max_tokens_per_run: Optional[int] = None
    warn_at_fraction: float = 0.8
    summarize_after_turns: int = 20
    max_chars_per_tool_result: int = 50_000


# ── Token counting ────────────────────────────────────────────────────────────


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Return an estimated token count for `text`."""
    try:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text, disallowed_special=()))
    except ImportError:
        return max(1, len(text) // 4)


def count_messages_tokens(messages: list, model: str = "gpt-4o") -> int:
    """Sum token counts across a list of LangChain message objects."""
    total = 0
    for msg in messages:
        content = getattr(msg, "content", "") or ""
        if isinstance(content, str):
            total += count_tokens(content, model)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    total += count_tokens(block.get("text", ""), model)
    return total


# ── Budget tracker ────────────────────────────────────────────────────────────


class TokenBudget:
    """
    Tracks token consumption during a single kazi.run() call.

    Usage::

        budget = TokenBudget(config, model="claude-sonnet-4-6")
        budget.charge(messages)        # charge for a batch of messages
        budget.charge_text(result_str) # charge for a single string
        # raises TokenBudgetExceeded automatically when limit hit
    """

    def __init__(self, config: TokenBudgetConfig, model: str = "gpt-4o") -> None:
        self.config = config
        self.model = model
        self._used = 0

    def charge(self, messages: list) -> None:
        self._used += count_messages_tokens(messages, self.model)
        self._check()

    def charge_text(self, text: str) -> None:
        self._used += count_tokens(text, self.model)
        self._check()

    def _check(self) -> None:
        if self.config.max_tokens_per_run is None:
            return
        fraction = self._used / self.config.max_tokens_per_run
        if fraction >= 1.0:
            from kazi.core.exceptions import TokenBudgetExceeded
            raise TokenBudgetExceeded(
                f"Token budget exceeded: {self._used:,} / {self.config.max_tokens_per_run:,} tokens"
            )
        if fraction >= self.config.warn_at_fraction:
            logger.warning(
                "Token budget %.0f%% consumed (%d / %d)",
                fraction * 100,
                self._used,
                self.config.max_tokens_per_run,
            )

    @property
    def used(self) -> int:
        return self._used

    def reset(self) -> None:
        self._used = 0


# ── Conversation summarisation ────────────────────────────────────────────────


async def maybe_summarise(
    messages: list,
    llm,
    config: TokenBudgetConfig,
) -> list:
    """
    Compress old messages if the history exceeds `summarize_after_turns`.

    Keeps the most recent `summarize_after_turns // 2` messages intact and
    replaces the remainder with a single summary SystemMessage. The LLM is
    called once to produce the summary.

    Returns the (possibly compressed) message list unchanged when under the limit.
    """
    if config.summarize_after_turns <= 0:
        return messages

    if len(messages) <= config.summarize_after_turns:
        return messages

    from langchain_core.messages import HumanMessage, SystemMessage

    keep = config.summarize_after_turns // 2
    to_compress = messages[:-keep]
    to_keep = messages[-keep:]

    history_text = "\n".join(
        f"{type(m).__name__.replace('Message', '')}: {m.content}"
        for m in to_compress
        if hasattr(m, "content") and isinstance(m.content, str)
    )

    try:
        summary_response = await llm.ainvoke([
            HumanMessage(
                content=(
                    "Summarise the following conversation history in 5 concise sentences, "
                    "preserving all key facts, decisions, and context:\n\n" + history_text
                )
            )
        ])
        summary_text = summary_response.content if hasattr(summary_response, "content") else str(summary_response)
        summary_msg = SystemMessage(content=f"[Conversation history summary]: {summary_text}")
        logger.info(
            "Summarised %d messages into 1 summary (%d messages retained)",
            len(to_compress),
            len(to_keep),
        )
        return [summary_msg] + to_keep
    except Exception as exc:
        logger.warning("Summarisation failed (%s) — using full history", exc)
        return messages
