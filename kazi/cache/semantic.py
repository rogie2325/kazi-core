"""
Semantic response cache.

Embeds incoming prompts using an OpenAI embedding model and looks for a
near-duplicate response already stored in the cache.  On a hit it returns
the cached answer immediately — no LLM round-trip, no cost.

Typical savings: 40-60% of API spend in production apps where users phrase
the same question differently ("What's your refund policy?" vs
"How do I get a refund?").

Backends
--------
memory   In-process dict.  Fast, cleared on restart.  Default.
redis    Persistent across processes and restarts.  Requires ``redis`` package.

Install::

    pip install kazi-core[openai]        # embeddings
    pip install redis                                 # only for redis backend

Usage::

    from kazi.cache.semantic import SemanticCache, SemanticCacheConfig

    cache = SemanticCache(SemanticCacheConfig(similarity_threshold=0.95))

    hit = await cache.get("What is the capital of France?")
    if hit is None:
        reply = await kazi.run("What is the capital of France?")
        await cache.set("What is the capital of France?", reply)
    else:
        reply = hit   # free!
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_NUMPY_MISSING = "numpy is required for semantic caching: pip install numpy"


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class SemanticCacheConfig:
    """
    Configuration for the semantic response cache.

    enabled              Master switch.
    similarity_threshold Cosine similarity required for a cache hit (0–1).
                         0.95 catches near-identical rephrasing.
                         0.85 is more aggressive — may produce false hits.
    ttl_seconds          Entries expire after this many seconds (0 = never).
    backend              "memory" (in-process) or "redis" (persistent).
    redis_url            Redis URL — only used when backend="redis".
    embedding_model      OpenAI embedding model name.
    max_entries          Maximum entries to keep (memory backend, LRU-style eviction).
    namespace            Key prefix for Redis — isolate caches per app.
    """
    enabled: bool = True
    similarity_threshold: float = 0.95
    ttl_seconds: int = 3600
    backend: str = "memory"
    redis_url: str = "redis://localhost:6379"
    embedding_model: str = "text-embedding-3-small"
    max_entries: int = 1000
    namespace: str = "kazi"


# ── In-memory backend ─────────────────────────────────────────────────────────

class _MemoryStore:
    def __init__(self, max_entries: int) -> None:
        # List of (embedding: list[float], response: str, expires_at: float)
        self._entries: list[tuple[list[float], str, float]] = []
        self._max = max_entries

    def get(self, embedding: list[float], threshold: float, ttl: int) -> str | None:
        try:
            import numpy as np
        except ImportError:
            raise ImportError(_NUMPY_MISSING)

        now = time.monotonic()
        q = np.array(embedding, dtype="float32")
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return None

        best_score = -1.0
        best_resp: str | None = None

        for emb, resp, expires_at in self._entries:
            if ttl > 0 and 0 < expires_at < now:
                continue
            e = np.array(emb, dtype="float32")
            e_norm = np.linalg.norm(e)
            if e_norm == 0:
                continue
            score = float(np.dot(q, e) / (q_norm * e_norm))
            if score > best_score:
                best_score = score
                best_resp = resp

        if best_score >= threshold:
            logger.debug("Semantic cache hit (memory, score=%.3f)", best_score)
            return best_resp
        return None

    def set(self, embedding: list[float], response: str, ttl: int) -> None:
        expires_at = time.monotonic() + ttl if ttl > 0 else 0.0
        if len(self._entries) >= self._max:
            self._entries.pop(0)  # FIFO eviction
        self._entries.append((embedding, response, expires_at))

    def clear(self) -> None:
        self._entries.clear()


# ── Redis backend ─────────────────────────────────────────────────────────────

class _RedisStore:
    def __init__(self, url: str, namespace: str) -> None:
        self._url = url
        self._ns = namespace
        self._redis = None

    @property
    def _prefix(self) -> str:
        return f"{self._ns}:semcache:"

    def _client(self):
        if self._redis is None:
            try:
                import redis as _redis
                self._redis = _redis.from_url(self._url, decode_responses=True)
            except ImportError:
                raise ImportError("redis package required: pip install redis")
        return self._redis

    def get(self, embedding: list[float], threshold: float, ttl: int) -> str | None:
        try:
            import numpy as np
        except ImportError:
            raise ImportError(_NUMPY_MISSING)

        r = self._client()
        keys = r.keys(f"{self._prefix}*")
        if not keys:
            return None

        q = np.array(embedding, dtype="float32")
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return None

        best_score = -1.0
        best_resp: str | None = None

        for key in keys:
            raw = r.get(key)
            if raw is None:
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                continue
            e = np.array(entry["embedding"], dtype="float32")
            e_norm = np.linalg.norm(e)
            if e_norm == 0:
                continue
            score = float(np.dot(q, e) / (q_norm * e_norm))
            if score > best_score:
                best_score = score
                best_resp = entry["response"]

        if best_score >= threshold:
            logger.debug("Semantic cache hit (redis, score=%.3f)", best_score)
            return best_resp
        return None

    def set(self, embedding: list[float], response: str, ttl: int) -> None:
        r = self._client()
        # Key derived from first 16 floats for collision resistance without storing full vector twice
        key_data = json.dumps(embedding[:16]).encode()
        key = f"{self._prefix}{hashlib.sha256(key_data).hexdigest()[:24]}"
        payload = json.dumps({"embedding": embedding, "response": response})
        if ttl > 0:
            r.set(key, payload, ex=ttl)
        else:
            r.set(key, payload)

    def clear(self) -> None:
        r = self._client()
        for key in r.scan_iter(f"{self._prefix}*"):
            r.delete(key)


# ── Public API ────────────────────────────────────────────────────────────────

class SemanticCache:
    """
    Semantic response cache.  Thread-safe (async-safe) across concurrent runs.

    All errors are caught and logged — a cache failure never breaks a live request.
    """

    def __init__(self, config: SemanticCacheConfig) -> None:
        self.config = config
        self._store: _MemoryStore | _RedisStore = (
            _RedisStore(config.redis_url, config.namespace)
            if config.backend == "redis"
            else _MemoryStore(config.max_entries)
        )
        self._oai_client = None

    # ── Embedding ─────────────────────────────────────────────────────────

    def _get_client(self):
        if self._oai_client is None:
            try:
                from openai import AsyncOpenAI
                self._oai_client = AsyncOpenAI()
            except ImportError:
                raise ImportError(
                    "openai package required for semantic cache: "
                    "pip install kazi-core[openai]"
                )
        return self._oai_client

    async def _embed(self, text: str) -> list[float]:
        client = self._get_client()
        resp = await client.embeddings.create(
            model=self.config.embedding_model,
            input=text[:8192],  # embedding model input cap
        )
        return resp.data[0].embedding

    # ── Public methods ────────────────────────────────────────────────────

    async def get(self, prompt: str) -> str | None:
        """Return the cached response for `prompt`, or None on a miss."""
        if not self.config.enabled:
            return None
        try:
            embedding = await self._embed(prompt)
            return self._store.get(
                embedding,
                self.config.similarity_threshold,
                self.config.ttl_seconds,
            )
        except Exception as exc:
            logger.warning("Semantic cache lookup failed (non-fatal): %s", exc)
            return None

    async def set(self, prompt: str, response: str) -> None:
        """Store the response for `prompt` in the cache."""
        if not self.config.enabled:
            return
        try:
            embedding = await self._embed(prompt)
            self._store.set(embedding, response, self.config.ttl_seconds)
        except Exception as exc:
            logger.warning("Semantic cache store failed (non-fatal): %s", exc)

    def clear(self) -> None:
        """Flush all entries from the cache."""
        try:
            self._store.clear()
        except Exception as exc:
            logger.warning("Semantic cache clear failed: %s", exc)
