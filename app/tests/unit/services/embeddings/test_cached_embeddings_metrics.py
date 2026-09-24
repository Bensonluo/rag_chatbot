"""Embedding cache counter wiring: the pyramid's foundation becomes measurable.

L3 sits under every semantic operation — L1 semantic lookups embed the
query first, and every RAG turn embeds too. The service is fail-open in
both directions, so a dead cache is invisible in request outcomes:
these tests pin that hits/misses are counted per text (the honest,
traffic-proportional measure for a batch API) and that a failed read is
counted as a *failure*, never smeared into the miss count — a miss
means "healthy read found no entry", otherwise the hit rate misstates
cache health exactly when it matters.
"""

import json
from unittest.mock import AsyncMock, Mock

from prometheus_client import REGISTRY

from app.services.embeddings.base import EmbeddingResult, EmbeddingServiceBase
from app.services.embeddings.cached_embeddings import CachedEmbeddingService


def _underlying(embeddings: list[list[float]]) -> Mock:
    service = Mock(spec=EmbeddingServiceBase, model="bge-m3", dimensions=4)
    service.embed = AsyncMock(
        return_value=EmbeddingResult(
            embeddings=embeddings, model="bge-m3", dimensions=4, tokens_used=7
        )
    )
    service.estimate_tokens = lambda text: len(text)
    return service


def _redis_stub(cached_values: list[str | None]) -> Mock:
    redis = Mock()
    redis.mget = AsyncMock(return_value=cached_values)
    set_pipeline = Mock()
    set_pipeline.setex = Mock()
    set_pipeline.execute = AsyncMock(return_value=[True])
    redis.pipeline = Mock(return_value=set_pipeline)
    return redis


def _cached_service(underlying: Mock, redis: Mock) -> CachedEmbeddingService:
    service = CachedEmbeddingService(
        embedding_service=underlying, redis_url="redis://localhost:6379/0"
    )
    # Pre-seed the lazy client: _get_redis() returns it without connecting.
    service._redis = redis  # noqa: SLF001
    return service


def _counter(name: str, labels: dict[str, str] | None = None) -> float:
    value = REGISTRY.get_sample_value(name, labels)
    return value if value is not None else 0.0


class TestEmbeddingCacheMetrics:
    async def test_hit_and_miss_counters_track_per_text(self) -> None:
        vector = [0.1, 0.2, 0.3, 0.4]
        underlying = _underlying([[0.9, 0.9, 0.9, 0.9]])
        redis = _redis_stub([json.dumps(vector), None])  # first cached, second not

        hits_before = _counter("embedding_cache_hits_total")
        misses_before = _counter("embedding_cache_misses_total")
        result = await _cached_service(underlying, redis).embed(["hello", "fresh"])

        assert result.embeddings == [vector, [0.9, 0.9, 0.9, 0.9]]
        assert _counter("embedding_cache_hits_total") == hits_before + 1.0
        assert _counter("embedding_cache_misses_total") == misses_before + 1.0

    async def test_read_failure_counts_failure_not_misses(self) -> None:
        """A failed read must not masquerade as misses: the hit rate stays
        a conditional-on-healthy-read metric, and the failure counter is
        the mechanism-level signal the alert rides on."""
        underlying = _underlying([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        redis = _redis_stub([])
        redis.mget = AsyncMock(side_effect=ConnectionError("redis down"))

        hits_before = _counter("embedding_cache_hits_total")
        misses_before = _counter("embedding_cache_misses_total")
        failures_before = _counter("embedding_cache_failures_total", {"op": "read"})
        result = await _cached_service(underlying, redis).embed(["a", "b"])

        # Fail-open: both texts embedded directly, request unharmed.
        assert result.embeddings == [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        underlying.embed.assert_awaited_once()
        assert _counter("embedding_cache_hits_total") == hits_before
        assert _counter("embedding_cache_misses_total") == misses_before
        assert _counter("embedding_cache_failures_total", {"op": "read"}) == failures_before + 1.0

    async def test_write_failure_counts_failure(self) -> None:
        vector = [0.5, 0.5, 0.5, 0.5]
        underlying = _underlying([vector])
        redis = _redis_stub([None])
        redis.pipeline.return_value.execute = AsyncMock(side_effect=ConnectionError("redis down"))

        misses_before = _counter("embedding_cache_misses_total")
        failures_before = _counter("embedding_cache_failures_total", {"op": "write"})
        result = await _cached_service(underlying, redis).embed(["fresh"])

        assert result.embeddings == [vector]
        assert _counter("embedding_cache_misses_total") == misses_before + 1.0
        assert _counter("embedding_cache_failures_total", {"op": "write"}) == failures_before + 1.0

    async def test_empty_input_moves_no_counters(self) -> None:
        underlying = _underlying([])
        hits_before = _counter("embedding_cache_hits_total")
        misses_before = _counter("embedding_cache_misses_total")

        result = await _cached_service(underlying, _redis_stub([])).embed([])

        assert result.embeddings == []
        assert _counter("embedding_cache_hits_total") == hits_before
        assert _counter("embedding_cache_misses_total") == misses_before
