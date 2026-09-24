"""
Cached embedding service wrapper using Redis.

Caches embeddings to reduce redundant API calls and computation.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any

from app.core.exceptions import ExternalServiceError

if TYPE_CHECKING:
    from redis.asyncio import Redis
from app.services.embeddings.base import EmbeddingResult, EmbeddingServiceBase

logger = logging.getLogger(__name__)


class CachedEmbeddingService(EmbeddingServiceBase):
    """
    Cached embedding service that wraps another embedding service.

    Uses Redis to cache embeddings, reducing redundant API calls
    and improving performance for frequently embedded texts.
    """

    def __init__(
        self,
        embedding_service: EmbeddingServiceBase,
        redis_url: str,
        cache_ttl: int = 604800,  # 7 days default
        prefix: str = "embedding",
    ) -> None:
        """
        Initialize cached embedding service.

        Args:
            embedding_service: Underlying embedding service to wrap
            redis_url: Redis connection URL
            cache_ttl: Cache time-to-live in seconds (default: 7 days)
            prefix: Redis key prefix (default: "embedding")
        """
        super().__init__(
            model=embedding_service.model,
            dimensions=embedding_service.dimensions,
        )

        self.embedding_service = embedding_service
        self.redis_url = redis_url
        self.cache_ttl = cache_ttl
        self.prefix = prefix

        # Redis client (lazy loaded)
        self._redis: Redis[str] | None = None

    async def _get_redis(self) -> Redis[str]:
        """Get or create Redis client."""
        if self._redis is None:
            try:
                import redis.asyncio as aioredis

                self._redis = await aioredis.from_url(
                    self.redis_url, encoding="utf-8", decode_responses=True
                )
            except ImportError as e:
                raise ExternalServiceError(
                    service="Redis", message="redis not installed. Run: pip install redis"
                ) from e
            except Exception as e:
                raise ExternalServiceError(
                    service="Redis", message=f"Failed to connect to Redis: {str(e)}"
                ) from e

        return self._redis

    def _generate_cache_key(self, text: str) -> str:
        """
        Generate a cache key for the given text.

        Args:
            text: Text to generate key for

        Returns:
            str: Redis cache key
        """
        # Create hash of text + model name
        content = f"{self.model}:{text}"
        hash_value = hashlib.sha256(content.encode()).hexdigest()

        return f"{self.prefix}:{self.model}:{hash_value}"

    async def _get_cached_embeddings(self, texts: list[str]) -> dict[int, list[float] | None]:
        """
        Get cached embeddings for multiple texts.

        Args:
            texts: List of texts to get from cache

        Returns:
            dict: Mapping of text index to cached embedding (or None if not cached)
        """
        redis = await self._get_redis()

        # Generate cache keys
        cache_keys = [self._generate_cache_key(text) for text in texts]

        # Batch get from Redis
        cached_values = await redis.mget(cache_keys)

        # Parse results
        result: dict[int, list[float] | None] = {}
        for idx, value in enumerate(cached_values):
            if value is not None:
                try:
                    parsed: Any = json.loads(value)
                    result[idx] = parsed
                except json.JSONDecodeError:
                    result[idx] = None
            else:
                result[idx] = None

        return result

    async def _set_cached_embeddings(self, texts: list[str], embeddings: list[list[float]]) -> None:
        """
        Cache embeddings for multiple texts.

        Args:
            texts: List of texts to cache
            embeddings: List of embeddings to cache
        """
        redis = await self._get_redis()

        # Prepare cache data
        cache_data: dict[str, str] = {}
        for text, embedding in zip(texts, embeddings, strict=True):
            key = self._generate_cache_key(text)
            value = json.dumps(embedding)
            cache_data[key] = value

        # Batch set in Redis with TTL
        if cache_data:
            pipeline = redis.pipeline()
            for key, value in cache_data.items():
                pipeline.setex(key, self.cache_ttl, value)
            await pipeline.execute()

    async def embed(self, texts: list[str]) -> EmbeddingResult:
        """
        Generate embeddings for a list of texts with caching.

        Args:
            texts: List of text strings to embed

        Returns:
            EmbeddingResult: Generated embeddings with metadata
        """
        if not texts:
            return EmbeddingResult(
                embeddings=[], model=self.model, dimensions=self.dimensions, tokens_used=0
            )

        # Try to get from cache first. The cache is an optimization
        # layer, never a dependency: a Redis outage (down, timeout,
        # eviction storm) degrades to a direct underlying call instead
        # of amplifying into a failed retrieval leg. Fail-open iron
        # law (docs/cache-layering-plan.md §4.3).
        cached: dict[int, list[float] | None] = dict.fromkeys(range(len(texts)), None)
        try:
            cached = await self._get_cached_embeddings(texts)
        except Exception:  # noqa: BLE001 - cache must never fail the embed
            logger.warning("Embedding cache read failed; embedding directly", exc_info=True)

        # Separate cached and uncached texts
        uncached_indices = [i for i, emb in cached.items() if emb is None]
        cached_indices = [i for i, emb in cached.items() if emb is not None]

        result_embeddings: list[list[float] | None] = [None] * len(texts)

        # Fill in cached embeddings
        for idx in cached_indices:
            result_embeddings[idx] = cached[idx]

        # Generate uncached embeddings
        if uncached_indices:
            uncached_texts = [texts[i] for i in uncached_indices]

            # Call underlying embedding service
            new_result = await self.embedding_service.embed(uncached_texts)

            # Fill in new embeddings
            for idx, embedding in zip(uncached_indices, new_result.embeddings, strict=True):
                result_embeddings[idx] = embedding

            # Cache the new embeddings (fail-open: a failed write only
            # means the next call re-embeds these texts)
            try:
                await self._set_cached_embeddings(uncached_texts, new_result.embeddings)
            except Exception:  # noqa: BLE001 - cache must never fail the embed
                logger.warning("Embedding cache write failed; continuing uncached", exc_info=True)

        # Every index is filled from cache or fresh generation; a None
        # here means a bookkeeping bug — fail loudly instead of returning
        # a misaligned vector list.
        if any(emb is None for emb in result_embeddings):
            raise ExternalServiceError(
                service="EmbeddingCache",
                message="Internal error: unresolved embeddings after cache fill",
            )

        final_embeddings = [emb for emb in result_embeddings if emb is not None]

        # Calculate tokens (estimate)
        tokens_used = sum(self.estimate_tokens(text) for text in texts)

        return EmbeddingResult(
            embeddings=final_embeddings,
            model=self.model,
            dimensions=self.dimensions,
            tokens_used=tokens_used,
        )

    async def embed_single(self, text: str) -> list[float]:
        """
        Generate embedding for a single text with caching.

        Args:
            text: Text string to embed

        Returns:
            List[float]: Embedding vector
        """
        result = await self.embed([text])
        return result.embeddings[0] if result.embeddings else []

    async def clear_cache(self) -> None:
        """
        Clear all cached embeddings.

        Warning: This will delete all embeddings from the cache.
        """
        redis = await self._get_redis()

        # Scan for all keys with our prefix
        pattern = f"{self.prefix}:{self.model}:*"
        keys: list[str] = []
        async for key in redis.scan_iter(match=pattern):
            keys.append(key)

        # Delete all keys
        if keys:
            await redis.delete(*keys)

    async def close(self) -> None:
        """
        Close the Redis connection.
        """
        if self._redis is not None:
            await self._redis.close()
            self._redis = None
