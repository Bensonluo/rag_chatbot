"""L0 exact-match answer cache (docs/cache-layering-plan.md, layer 0).

Caches complete grounded answers keyed on
``sha256(normalized message + KB epoch + model tag + persona tag)``. A
hit skips the whole pipeline (intent → retrieval → generation) and
replays the stored answer in ~5ms instead of seconds — the top of the
traffic pyramid for storm traffic, where thousands of users ask the
identical question.

Design invariants (the cache is an optimization, never a dependency):

- **Fail-open**: every Redis error degrades to a miss / skipped write
  plus a warning log. Nothing here can fail or meaningfully slow a turn.
- **Epoch-scoped**: the KB epoch participates in the key, so any
  successful document mutation invalidates every entry wholesale
  without eviction work.
- **Stateless turns only**: personalized turns (named user, slots in
  flight, staged actions, executed tools) are never written, so a hit
  can never replay one user's context at another.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.services.chat.cache_eviction import record_response_index
from app.services.chat.metrics import ANSWER_CACHE_HITS, ANSWER_CACHE_MISSES
from app.services.retrieval.kb_epoch import EPOCH_UNAVAILABLE, get_kb_epoch

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = logging.getLogger(__name__)

_WHITESPACE = re.compile(r"\s+")


def normalize_message(message: str) -> str:
    """Canonical query form: trimmed, whitespace-collapsed, casefolded.

    Variants that differ only in leading/trailing space, inner
    whitespace runs, or letter case ("Return  POLICY" vs "return
    policy") land on one key. Whitespace is collapsed to a single
    space, not removed — English needs its spaces — so CJK variants
    with incidental spaces (“退货 政策” vs “退货政策”) miss. Misses are
    always the safe direction for an exact cache.
    """
    return _WHITESPACE.sub(" ", message).strip().casefold()


@dataclass(frozen=True)
class CachedAnswer:
    """One replayable grounded turn."""

    response: str
    sources: list[str]
    intent: str


class AnswerCacheService:
    """Redis-backed exact-answer cache with fail-open semantics."""

    def __init__(
        self,
        *,
        redis_url: str,
        ttl_seconds: int,
        max_response_chars: int = 4000,
        model_tag: str = "",
        persona_tag: str = "",
    ) -> None:
        self._redis_url = redis_url
        self._ttl = ttl_seconds
        self._max_response_chars = max_response_chars
        self._model_tag = model_tag
        self._persona_tag = persona_tag
        # Lazy client; Redis typing kept out of the hot import path.
        self._redis: Redis[str] | None = None

    async def get(self, message: str) -> CachedAnswer | None:
        """Return the cached answer for this message, or None on miss.

        Any failure — Redis down, epoch unreadable, corrupt payload —
        counts and behaves as a miss: the caller routes into the normal
        pipeline. A miss ratio of 100% is the ceiling of safe behavior.
        """
        try:
            key, _ = await self._key_and_epoch(message)
            if key is None:
                ANSWER_CACHE_MISSES.inc()
                return None
            raw = await self._client().get(key)
            if raw is None:
                ANSWER_CACHE_MISSES.inc()
                return None
            payload = json.loads(raw)
            answer = CachedAnswer(
                response=str(payload.get("response", "")),
                sources=[str(s) for s in payload.get("sources", [])],
                intent=str(payload.get("intent", "")),
            )
            if not answer.response:
                ANSWER_CACHE_MISSES.inc()
                return None
            ANSWER_CACHE_HITS.inc()
            return answer
        except Exception:  # noqa: BLE001 - cache is never a dependency
            logger.warning("Answer cache read failed; treating as miss", exc_info=True)
            ANSWER_CACHE_MISSES.inc()
            return None

    async def put(
        self,
        message: str,
        *,
        response: str,
        sources: list[str],
        intent: str,
    ) -> None:
        """Store a grounded answer. Fail-open: errors are logged, never raised."""
        if not response or not sources or len(response) > self._max_response_chars:
            return
        try:
            key, epoch = await self._key_and_epoch(message)
            if key is None:
                # Epoch unavailable: refuse to populate a key namespace
                # that bypasses KB invalidation.
                return
            payload = json.dumps(
                {"response": response, "sources": sources, "intent": intent},
                ensure_ascii=False,
            )
            await self._client().set(key, payload, ex=self._ttl)
            # Reverse index for downvote eviction (cache_eviction): the
            # L0 key digests the query, but feedback only knows the
            # response text — so record key ← response digest.
            await record_response_index(
                self._client(),
                index_name=f"answer_resp_index:{epoch}",
                response=response,
                target=key,
                ttl_seconds=self._ttl,
            )
        except Exception:  # noqa: BLE001 - cache is never a dependency
            logger.warning("Answer cache write failed; answer simply not cached", exc_info=True)

    async def _key_and_epoch(self, message: str) -> tuple[str | None, str | None]:
        """Epoch-scoped digest key with its epoch; Nones when unavailable.

        The epoch is returned alongside the key so ``put`` can index
        the entry for downvote eviction without a second epoch read.
        """
        epoch = await get_kb_epoch()
        if epoch == EPOCH_UNAVAILABLE:
            return None, None
        digest = hashlib.sha256(
            "\x00".join(
                (normalize_message(message), epoch, self._model_tag, self._persona_tag)
            ).encode()
        ).hexdigest()
        return f"answer:{digest}", epoch

    def _client(self) -> Redis[str]:
        if self._redis is None:
            from redis import asyncio as aioredis

            self._redis = aioredis.from_url(self._redis_url, decode_responses=True)
        return self._redis

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.close()
            self._redis = None
