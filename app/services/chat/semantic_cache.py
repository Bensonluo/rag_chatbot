"""L1 semantic answer cache (docs/cache-layering-plan.md layer table).

Near-duplicate hot turns — the same question rephrased — replay a
cached grounded answer in milliseconds instead of re-running the full
pipeline. Boundaries inherited from the plan's risk analysis:

- Intent allowlist at write time: only small-talk/FAQ-class intents
  are cached. Paraphrased *policy* questions can embed very close
  while differing materially (“7天 vs 30天”), so policy/RAG answers
  stay L0-exact-only — semantic nearness is the plan's「唯一真风险层」.
- Epoch scoping: entries live in a per-KB-epoch namespace, so a
  knowledge-base update invalidates them wholesale (same doctrine as
  the L0 cache key).
- Bounded store: an entry cap keeps the namespace from being flooded
  by long-tail one-offs (the plan's personalization warning — cache
  space must not drown in no-hit entries).
- Fail-open iron law: any cache failure (Redis outage, embedding
  outage, corrupt entry) degrades to a miss — the cache is an
  optimization, never a dependency.

Scan design note: the lookup decodes every entry's vector and scores
it in-process. With the demo-node entry cap (hundreds) this is
milliseconds; at multi-node scale the seam to swap is the storage
scan for a vector index — the service's contract stays unchanged.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import math
from array import array
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from app.services.chat.answer_cache import CachedAnswer, normalize_message
from app.services.chat.cache_eviction import record_response_index
from app.services.retrieval.kb_epoch import EPOCH_UNAVAILABLE

logger = logging.getLogger(__name__)

_NAMESPACE = "semantic_answers"


class EmbeddingSeam(Protocol):
    """The face of an embedding service this cache needs."""

    async def embed(self, texts: list[str]) -> Any: ...


class RedisSeam(Protocol):
    """The face of an async Redis client this cache needs."""

    async def hset(self, name: str, key: str, value: str) -> None: ...
    async def hget(self, name: str, key: str) -> str | None: ...
    async def hgetall(self, name: str) -> dict[str, str]: ...
    async def hlen(self, name: str) -> int: ...
    async def expire(self, name: str, ttl: int) -> None: ...


def encode_vector(vector: list[float]) -> str:
    """Pack a float32 vector into a compact base64 string."""
    return base64.b64encode(array("f", vector).tobytes()).decode()


def decode_vector(raw: str) -> list[float]:
    """Unpack a base64 float32 vector; corrupt input raises."""
    arr = array("f")
    arr.frombytes(base64.b64decode(raw))
    return arr.tolist()


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity; dimension mismatch or zero vectors score 0."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = math.fsum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(math.fsum(x * x for x in a))
    norm_b = math.sqrt(math.fsum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticCacheService:
    """Serves cached answers to semantically near-duplicate turns."""

    def __init__(
        self,
        embedding_service: EmbeddingSeam,
        redis_client: RedisSeam,
        *,
        epoch_provider: Callable[[], Awaitable[str]],
        settings: Any,
    ) -> None:
        """
        Args:
            embedding_service: Provider with ``embed(texts)`` (the same
                seam FAQService uses); vectors come back on
                ``.embeddings``.
            redis_client: Async Redis client (or fake) exposing the
                hash commands above.
            epoch_provider: Async callable returning the current KB
                epoch (the same ``get_kb_epoch`` the L0 cache keys
                on), read per call so a KB update flushes this layer
                in lockstep with L0 — a construction-frozen epoch
                would silently outlive its KB.
            settings: App settings; read per call so tests (and
                runtime flag flips) take effect without rebuilding.
        """
        self._embeddings = embedding_service
        self._redis = redis_client
        self._epoch_provider = epoch_provider
        self._settings = settings

    async def get(self, message: str) -> CachedAnswer | None:
        """Best allowlisted entry above the similarity threshold, or None."""
        if not self._enabled():
            return None
        try:
            key, _ = await self._key_and_epoch()
            if key is None:
                return None
            query_vec = (await self._embeddings.embed([message])).embeddings[0]
            entries = await self._redis.hgetall(key)
            best: dict[str, Any] | None = None
            best_score = -1.0
            for raw in entries.values():
                try:
                    entry = json.loads(raw)
                    score = _cosine(query_vec, decode_vector(entry["vec"]))
                except (ValueError, KeyError, TypeError, binascii.Error):
                    continue  # corrupt entry: skip, never fail the lookup
                if score > best_score:
                    best, best_score = entry, score
            served = self._served_intents()
            if (
                best is None
                or best_score < self._settings.SEMANTIC_CACHE_SIMILARITY_THRESHOLD
                or best.get("intent") not in served
            ):
                return None
            return CachedAnswer(
                response=str(best["response"]),
                sources=[str(s) for s in best.get("sources", [])],
                intent=str(best["intent"]),
            )
        except Exception:  # noqa: BLE001 - fail-open iron law
            logger.warning("Semantic cache read failed; degrading to miss", exc_info=True)
            return None

    async def put(self, message: str, answer: CachedAnswer) -> None:
        """Cache one grounded answer for near-duplicate replay.

        Skips silently when disabled, when the intent is outside the
        allowlist, or when the store is at capacity — every skip is the
        safe direction.
        """
        if not self._enabled():
            return
        if answer.intent not in self._served_intents():
            return
        try:
            key, epoch = await self._key_and_epoch()
            if key is None:
                return
            if await self._redis.hlen(key) >= self._settings.SEMANTIC_CACHE_MAX_ENTRIES:
                return
            vec = (await self._embeddings.embed([message])).embeddings[0]
            payload = json.dumps(
                {
                    "response": answer.response,
                    "sources": answer.sources,
                    "intent": answer.intent,
                    "vec": encode_vector(list(vec)),
                },
                ensure_ascii=False,
            )
            field = normalize_message(message)
            await self._redis.hset(key, field, payload)
            await self._redis.expire(key, self._settings.SEMANTIC_CACHE_TTL_SECONDS)
            # Reverse index for downvote eviction (cache_eviction): the
            # entry is keyed by the query, but feedback only knows the
            # response text — so record field ← response digest.
            await record_response_index(
                self._redis,
                index_name=f"semantic_resp_index:{epoch}",
                response=answer.response,
                target=field,
                ttl_seconds=self._settings.SEMANTIC_CACHE_TTL_SECONDS,
            )
        except Exception:  # noqa: BLE001 - fail-open iron law
            logger.warning("Semantic cache write failed; skipping entry", exc_info=True)

    async def _key_and_epoch(self) -> tuple[str | None, str | None]:
        """Epoch-scoped namespace with its epoch; Nones when unavailable."""
        epoch = await self._epoch_provider()
        if not epoch or epoch == EPOCH_UNAVAILABLE:
            return None, None
        return f"{_NAMESPACE}:{epoch}", epoch

    def _enabled(self) -> bool:
        return bool(self._settings.SEMANTIC_CACHE_ENABLED)

    def _served_intents(self) -> frozenset[str]:
        raw = getattr(self._settings, "SEMANTIC_CACHE_SERVED_INTENTS", "")
        return frozenset(part.strip() for part in raw.split(",") if part.strip())
