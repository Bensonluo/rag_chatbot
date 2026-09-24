"""Feedback-driven eviction across the cache pyramid.

The feedback endpoint is the user-verified quality signal: a downvote
marks an answer as "was not a resolution". Without eviction the
pyramid (L0 exact + L1 semantic) keeps replaying that rejected answer
in ~5ms until TTL/epoch rotation — at storm scale the cache re-serves
the failure to every near-duplicate ask, the exact opposite of the
one-shot north star.

Mechanics: cache keys cannot be derived from a downvoted message (the
L0 key digests the *query*; feedback only knows the *response*), so
every L0/L1 ``put`` records a reverse index::

    answer_resp_index:{epoch}   field=response_digest → JSON [answer keys]
    semantic_resp_index:{epoch} field=response_digest → JSON [hash fields]

Eviction is then O(1): one HGET per layer, then DEL/HDEL the listed
targets. Fail-open iron law throughout — eviction is a quality
optimization, never a dependency of the feedback endpoint.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any

from app.services.chat.metrics import CACHE_FEEDBACK_EVICTIONS
from app.services.retrieval.kb_epoch import EPOCH_UNAVAILABLE, get_kb_epoch

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = logging.getLogger(__name__)

_L0_INDEX = "answer_resp_index"
_L1_INDEX = "semantic_resp_index"
_L1_NAMESPACE = "semantic_answers"

_default_redis: Redis[str] | None = None


def response_digest(response: str) -> str:
    """Stable digest of a response text (the reverse-index field)."""
    return hashlib.sha256(response.encode()).hexdigest()


async def record_response_index(
    redis_client: Any,
    *,
    index_name: str,
    response: str,
    target: str,
    ttl_seconds: int,
) -> None:
    """Append one target to the response-digest reverse index.

    Read-modify-write: two racing puts can drop one target (last write
    wins). The dropped entry then survives until epoch/TTL rotation —
    exactly the pre-eviction status quo, i.e. the safe direction.
    """
    digest = response_digest(response)
    raw = await redis_client.hget(index_name, digest)
    try:
        targets: list[str] = json.loads(raw) if raw else []
        if not isinstance(targets, list):
            targets = []
    except ValueError:
        targets = []
    if target not in targets:
        targets.append(target)
    await redis_client.hset(index_name, digest, json.dumps(targets))
    await redis_client.expire(index_name, ttl_seconds)


async def evict_downvoted_response(response: str, *, redis_client: Any = None) -> int:
    """Remove every pyramid entry that replays this downvoted response.

    Returns the number of entries targeted across L0 and L1. Every
    failure mode (empty response, unreadable epoch, Redis outage,
    corrupt index payload) degrades to a no-op and never raises.
    """
    if not response:
        return 0
    try:
        epoch = await get_kb_epoch()
    except Exception:  # noqa: BLE001 - fail-open iron law
        logger.warning("Epoch read failed; skipping downvote eviction", exc_info=True)
        return 0
    if not epoch or epoch == EPOCH_UNAVAILABLE:
        return 0
    client = redis_client if redis_client is not None else _default_client()
    digest = response_digest(response)

    evicted = await _evict_layer(
        client, index_name=f"{_L0_INDEX}:{epoch}", digest=digest, layer="l0"
    )
    evicted += await _evict_layer(
        client,
        index_name=f"{_L1_INDEX}:{epoch}",
        digest=digest,
        layer="l1",
        hash_name=f"{_L1_NAMESPACE}:{epoch}",
    )
    return evicted


async def _evict_layer(
    client: Any,
    *,
    index_name: str,
    digest: str,
    layer: str,
    hash_name: str | None = None,
) -> int:
    """Evict one layer's indexed targets; L0 keys vs L1 hash fields."""
    try:
        raw = await client.hget(index_name, digest)
        if not raw:
            return 0
        targets = json.loads(raw)
        if not isinstance(targets, list) or not targets:
            await client.hdel(index_name, digest)  # corrupt entry: clear it
            return 0
        if hash_name is None:
            await client.delete(*targets)
        else:
            await client.hdel(hash_name, *targets)
        await client.hdel(index_name, digest)
        CACHE_FEEDBACK_EVICTIONS.labels(layer=layer).inc(len(targets))
        return len(targets)
    except Exception:  # noqa: BLE001 - fail-open iron law
        logger.warning(
            "Downvote eviction failed for layer %s; entries stay cached until epoch/TTL rotation",
            layer,
            exc_info=True,
        )
        return 0


def _default_client() -> Redis[str]:
    """Lazy shared client for the standalone (non-request) call path."""
    global _default_redis
    if _default_redis is None:
        from redis import asyncio as aioredis

        from app.config.settings import settings

        _default_redis = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    return _default_redis
