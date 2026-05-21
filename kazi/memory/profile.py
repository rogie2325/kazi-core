"""
Long-term semantic user memory.

Stores cross-session user facts and preferences as a JSON file per user_id.
The profile is automatically injected as a preamble into every system prompt
so agents know the user's context, preferences, and history across sessions.

Quick start::

    from kazi import Kazi, KaziConfig, UserProfile

    profile = UserProfile()
    profile.update("alice", {"role": "data scientist", "prefers": "concise answers", "timezone": "UTC-5"})

    async with await Kazi.create(config) as kazi:
        reply = await kazi.run("Explain this SQL query", user_id="alice")
        # Agent knows alice is a data scientist and prefers concise answers
"""
from __future__ import annotations

import json
import logging
import re as _re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Whitelist: only alphanumeric, hyphens, underscores, dots, and @.
# Everything else (slashes, backslashes, null bytes, colons, etc.) → "_".
# Max 128 chars to bound filename length.
_SAFE_USER_ID_RE = _re.compile(r"[^\w\-@.]")


class UserProfile:
    """
    Per-user fact store backed by JSON files.

    Each user gets one file: <storage_dir>/<safe_user_id>.json

    Files are plain JSON dicts — human-readable, easy to debug,
    and trivially portable to any key-value store later.
    """

    def __init__(self, storage_dir: str = ".kazi_profiles") -> None:
        self._dir = Path(storage_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    # ── CRUD ──────────────────────────────────────────────────────────────

    def load(self, user_id: str) -> dict[str, Any]:
        """Return the full profile dict for `user_id`, or {} if not found."""
        p = self._path(user_id)
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to load profile for %r: %s", user_id, exc)
            return {}

    def save(self, user_id: str, profile: dict[str, Any]) -> None:
        """Overwrite the entire profile for `user_id`."""
        self._path(user_id).write_text(
            json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def update(self, user_id: str, facts: dict[str, Any]) -> None:
        """Merge `facts` into the existing profile (shallow update)."""
        profile = self.load(user_id)
        profile.update(facts)
        self.save(user_id, profile)

    def get(self, user_id: str, key: str, default: Any = None) -> Any:
        """Return a single fact from the profile."""
        return self.load(user_id).get(key, default)

    def delete_fact(self, user_id: str, key: str) -> None:
        """Remove one fact from the profile."""
        profile = self.load(user_id)
        profile.pop(key, None)
        self.save(user_id, profile)

    def delete(self, user_id: str) -> None:
        """Delete the entire profile for `user_id`."""
        p = self._path(user_id)
        if p.exists():
            p.unlink()

    def list_users(self) -> list[str]:
        """Return all user_ids that have a stored profile."""
        return [
            p.stem.replace("_", ":", 1)
            for p in self._dir.glob("*.json")
        ]

    # ── System prompt integration ─────────────────────────────────────────

    def as_system_preamble(self, user_id: str) -> str | None:
        """
        Return a compact string describing the user's profile,
        suitable for prepending to a system prompt.

        Returns None when the profile is empty (no overhead for unknown users).
        """
        profile = self.load(user_id)
        if not profile:
            return None
        lines = [f"User profile ({user_id}):"]
        for k, v in profile.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    # ── Internal ──────────────────────────────────────────────────────────

    def _path(self, user_id: str) -> Path:
        # Whitelist-sanitize: strip null bytes, then replace any char outside
        # [a-zA-Z0-9_\-@.] with "_", cap at 128 chars.
        safe = _SAFE_USER_ID_RE.sub("_", user_id.replace("\x00", ""))[:128]
        if not safe:
            safe = "_anonymous"
        path = self._dir / f"{safe}.json"
        # Defence-in-depth: confirm the resolved path stays inside storage dir.
        try:
            path.resolve().relative_to(self._dir.resolve())
        except ValueError:
            raise ValueError(f"Invalid user_id: path traversal detected for {user_id!r}")
        return path
