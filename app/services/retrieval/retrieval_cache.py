"""L2 retrieval-result cache (docs/cache-layering-plan.md layer table).

Identical (query, filters) pairs within one KB epoch replay the cached
doc set instead of re-running the Qdrant+BM25 legs. The plan's stated
value is protecting Qdrant — the shared stateful service that does not
scale by adding API replicas — especially the filter-miss
double-search amplification, where one user turn costs two vector
searches. Boundaries inherited from the plan:

- **Epoch-scoped namespace**: a successful document mutation
  invalidates every entry wholesale, same doctrine as the L0/L1
  answer caches.
- **Filter participation**: the metadata-filter signature is part of
  the entry key, so a filtered search can never replay another
  filter's docs.
- **Bounded store**: an entry cap keeps long-tail one-offs from
  flooding the namespace (same warning as the plan's L1 boundary).
- **Fail-open iron law**: any Redis failure (outage, timeout, corrupt
  payload) degrades to a miss — the cache protects retrieval, never
  gates it.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from app.services.chat.answer_cache import normalize_message
from app.services.retrieval.kb_epoch import EPOCH_UNAVAILABLE
from app.services.retrieval.metrics import (
    RETRIEVAL_CACHE_HITS,
    RETRIEVAL_CACHE_MISSES,
)

logger = logging.getLogger(__name__)

_NAMESPACE = "retrieval_results"


class RedisSeam(Protocol):
    """The face of an async Redis client this cache needs."""

    async def hset(self, name: str, key: str, value: str) -> None: ...
    async def hget(self, name: str, key: str) -> str | None: ...
    async def hlen(self, name: str) -> int: ...
    async def expire(self, name: str, ttl: int) -> None: ...


class RetrievalCacheService:
    """Replays cached retrieval results for repeated (query, filters)."""

    def __init__(
        self,
        redis_client: RedisSeam,
        *,
        epoch_provider: Callable[[], Awaitable[str]],
        settings: Any,
    ) -> None:
        """
        Args:
            redis_client: Async Redis client (or fake) exposing the
                hash commands above.
            epoch_provider: Async callable returning the current KB
                epoch (the same ``get_kb_epoch`` the answer caches
                key on), read per call so a KB update flushes this
                layer in lockstep.
            settings: App settings; read per call so runtime flag
                flips take effect without rebuilding.
        """
        self._redis = redis_client
        self._epoch_provider = epoch_provider
        self._settings = settings

    async def get(self, query: str, filters: dict[str, Any]) -> list[dict[str, Any]] | None:
        """Cached doc list for this query+filters, or None on miss."""
        if not self._enabled():
            return None
        try:
            key = await self._key(query, filters)
            if key is None:
                return None
            raw = await self._redis.hget(*key)
            if raw is None:
                RETRIEVAL_CACHE_MISSES.inc()
                return None
            docs = json.loads(raw)
            if not isinstance(docs, list) or not docs:
                # Empty or corrupt payload: treat as a miss; the
                # caller re-runs retrieval and the write path repairs.
                RETRIEVAL_CACHE_MISSES.inc()
                return None
            RETRIEVAL_CACHE_HITS.inc()
            return docs
        except Exception:  # noqa: BLE001 - fail-open iron law
            logger.warning("Retrieval cache read failed; re-running retrieval", exc_info=True)
            RETRIEVAL_CACHE_MISSES.inc()
            return None

    async def put(
        self,
        query: str,
        filters: dict[str, Any],
        docs: list[dict[str, Any]],
    ) -> None:
        """Cache one retrieval result. Fail-open: errors skip the write."""
        if not self._enabled() or not docs:
            return
        try:
            key = await self._key(query, filters)
            if key is None:
                return
            if await self._redis.hlen(key[0]) >= self._settings.RETRIEVAL_CACHE_MAX_ENTRIES:
                return
            payload = json.dumps(docs, ensure_ascii=False)
            await self._redis.hset(key[0], key[1], payload)
            await self._redis.expire(key[0], self._settings.RETRIEVAL_CACHE_TTL_SECONDS)
        except Exception:  # noqa: BLE001 - fail-open iron law
            logger.warning("Retrieval cache write failed; result not cached", exc_info=True)

    async def _key(self, query: str, filters: dict[str, Any]) -> tuple[str, str] | None:
        """(namespace, entry) key; None when the epoch is unavailable.

        The epoch scopes the whole hash (one DEL-equivalent flip on KB
        mutation); the entry digest folds the normalized query and the
        filter signature so different filters never collide.
        """
        epoch = await self._epoch_provider()
        if not epoch or epoch == EPOCH_UNAVAILABLE:
            return None
        filter_sig = json.dumps(filters, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(
            "\x00".join((normalize_message(query), filter_sig)).encode()
        ).hexdigest()
        return f"{_NAMESPACE}:{epoch}", digest

    def _enabled(self) -> bool:
        return bool(self._settings.RETRIEVAL_CACHE_ENABLED)
