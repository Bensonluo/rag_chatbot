"""KB epoch: a rotating content version for the vector knowledge base.

Every successful Qdrant mutation (ingest, delete) rotates a Redis-held
epoch value. Cache keys that embed the epoch — the L0 answer cache
today, retrieval-result caches if ever added — are invalidated
wholesale by the bump: no per-entry eviction, no tombstones, no
partial-invalidation bugs.

Why Redis and not the Postgres documents table: the upload path never
writes that table (it is metadata-only), and more importantly the epoch
must die together with the caches it guards. A Redis flush takes both
the epoch and every cache entry out at once — a total miss, which is
the safe direction — and because values are random UUIDs, a flush can
never resurrect stale entries under a recycled counter.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = logging.getLogger(__name__)

EPOCH_KEY = "kb:epoch"

# Returned when Redis is unreachable. It deliberately matches no cache
# key ever written (put paths refuse it), so an epoch outage degrades
# every scoped cache to 100% misses — fail-open toward the backend.
EPOCH_UNAVAILABLE = "epoch-unavailable"

_client: Redis[str] | None = None


def _get_client() -> Redis[str]:
    """Lazy module-level Redis client (one pooled connection set)."""
    global _client
    if _client is None:
        from redis import asyncio as aioredis

        from app.config.settings import get_settings

        _client = aioredis.from_url(get_settings().REDIS_URL, decode_responses=True)
    return _client


async def get_kb_epoch() -> str:
    """Current KB epoch, creating it on first read. Fail-open.

    Concurrent first-reads converge via SET NX: only one caller's
    candidate wins and every reader returns the stored value.
    """
    try:
        client = _get_client()
        epoch = await client.get(EPOCH_KEY)
        if epoch is not None:
            return str(epoch)
        candidate = uuid.uuid4().hex
        await client.set(EPOCH_KEY, candidate, nx=True)
        epoch = await client.get(EPOCH_KEY)
        return str(epoch) if epoch is not None else candidate
    except Exception:  # noqa: BLE001 - the epoch is never a dependency
        logger.warning("KB epoch read failed; epoch-scoped caches will miss", exc_info=True)
        return EPOCH_UNAVAILABLE


async def bump_kb_epoch() -> None:
    """Rotate the epoch, invalidating every epoch-scoped cache entry.

    Called after a successful Qdrant mutation. Fail-open: a failed bump
    leaves caches on the old epoch (entries may be stale until TTL),
    which is logged — an ingestion must never fail over invalidation.
    """
    try:
        client = _get_client()
        await client.set(EPOCH_KEY, uuid.uuid4().hex)
    except Exception:  # noqa: BLE001 - invalidation must not fail ingestion
        logger.warning("KB epoch bump failed; caches stay on the old epoch", exc_info=True)
