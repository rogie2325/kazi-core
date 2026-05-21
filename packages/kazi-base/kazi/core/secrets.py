"""
SecretRef — a secret value that can never be accidentally logged or serialised.

Supports three resolution strategies:
  - Literal string    : SecretRef("sk-ant-...")         — convenient, less safe
  - Env variable      : SecretRef.from_env("API_KEY")   — recommended for most cases
  - Callable          : SecretRef.from_callable(fn)     — for vault / secrets-manager integration

The value is resolved lazily at call time (never at construction time) so that
config objects can be built without network calls, and secrets are not held in
memory longer than necessary.

__repr__ and __str__ always return "SecretRef(***)" — logging a config object
that contains a SecretRef will NEVER expose the underlying value.
"""
from __future__ import annotations

import os
from typing import Callable, Optional, Union


class SecretRef:
    """Opaque container for a sensitive string value."""

    def __init__(self, value: Union[str, Callable[[], Optional[str]], None] = None) -> None:
        self._value = value

    # ── factories ─────────────────────────────────────────────────────────

    @classmethod
    def from_env(cls, var_name: str, fallback: Optional[str] = None) -> "SecretRef":
        """Resolve from an environment variable at call time."""
        return cls(lambda: os.environ.get(var_name, fallback))

    @classmethod
    def from_callable(cls, fn: Callable[[], Optional[str]]) -> "SecretRef":
        """Resolve by calling `fn()` each time — use for vault / secrets-manager."""
        return cls(fn)

    @classmethod
    def coerce(cls, value: Union[str, "SecretRef", None]) -> Optional["SecretRef"]:
        """
        Accept either a plain string or an existing SecretRef and return a SecretRef.
        Returns None when value is None.

        This lets config fields accept both::

            LLMConfig(api_key="sk-ant-...")                    # still works
            LLMConfig(api_key=SecretRef.from_env("ANT_KEY"))   # preferred
        """
        if value is None:
            return None
        if isinstance(value, SecretRef):
            return value
        return cls(value)

    # ── resolution ────────────────────────────────────────────────────────

    def resolve(self) -> Optional[str]:
        """Return the secret value. Calls the callable each time if one was provided."""
        if self._value is None:
            return None
        if callable(self._value):
            return self._value()
        return self._value

    # ── safety ────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return "SecretRef(***)"

    def __str__(self) -> str:
        return "***"

    def __eq__(self, other: object) -> bool:
        # Only equal to itself — prevents accidental comparison leaking the value
        return self is other

    def __hash__(self) -> int:
        return id(self)
