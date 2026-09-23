"""Tests for the cached and OpenAI embedding services."""

import json
from unittest.mock import AsyncMock, Mock

import pytest

from app.core.exceptions import ExternalServiceError
from app.services.embeddings.base import EmbeddingResult, EmbeddingServiceBase
from app.services.embeddings.cached_embeddings import CachedEmbeddingService
from app.services.embeddings.openai_embeddings import OpenAIEmbeddingService


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


class TestCachedEmbeddingService:
    async def test_full_cache_hit_skips_underlying_call(self) -> None:
        vector = [0.1, 0.2, 0.3, 0.4]
        underlying = _underlying([vector])
        redis = _redis_stub([json.dumps(vector), json.dumps(vector)])

        service = _cached_service(underlying, redis)
        result = await service.embed(["hello", "world"])

        assert result.embeddings == [vector, vector]
        underlying.embed.assert_not_awaited()

    async def test_miss_generates_and_caches_in_order(self) -> None:
        vector_a = [0.5, 0.5, 0.5, 0.0]
        vector_b = [0.0, 0.5, 0.5, 0.5]
        underlying = _underlying([vector_a, vector_b])
        redis = _redis_stub([None, None])

        service = _cached_service(underlying, redis)
        result = await service.embed(["hello", "world"])

        assert result.embeddings == [vector_a, vector_b]
        underlying.embed.assert_awaited_once_with(["hello", "world"])
        cache_writes = redis.pipeline.return_value.setex.call_args_list
        assert len(cache_writes) == 2

    async def test_mixed_hit_and_miss_preserves_alignment(self) -> None:
        cached_vector = [1.0, 0.0, 0.0, 0.0]
        fresh_vector = [0.0, 1.0, 0.0, 0.0]
        underlying = _underlying([fresh_vector])
        # First text cached, second missing.
        redis = _redis_stub([json.dumps(cached_vector), None])

        service = _cached_service(underlying, redis)
        result = await service.embed(["cached-text", "fresh-text"])

        # Index alignment: cached vector first, fresh vector second.
        assert result.embeddings[0] == cached_vector
        assert result.embeddings[1] == fresh_vector
        # Only the missing text is re-embedded.
        underlying.embed.assert_awaited_once_with(["fresh-text"])

    async def test_corrupt_cache_entry_is_treated_as_miss(self) -> None:
        vector = [0.9, 0.1, 0.0, 0.0]
        underlying = _underlying([vector])
        redis = _redis_stub(["not valid json {"])

        service = _cached_service(underlying, redis)
        result = await service.embed(["hello"])

        assert result.embeddings == [vector]
        underlying.embed.assert_awaited_once_with(["hello"])

    async def test_empty_input_short_circuits_without_redis(self) -> None:
        underlying = _underlying([])
        redis = _redis_stub([])
        service = _cached_service(underlying, redis)

        result = await service.embed([])

        assert result.embeddings == []
        assert result.tokens_used == 0
        redis.mget.assert_not_awaited()

    async def test_embed_single_returns_vector(self) -> None:
        vector = [0.3, 0.3, 0.3, 0.1]
        underlying = _underlying([vector])
        redis = _redis_stub([None])

        service = _cached_service(underlying, redis)
        assert await service.embed_single("hello") == vector


class TestOpenAIEmbeddingService:
    def _client_returning(
        self, data: list[tuple[int, list[float]]], total_tokens: int = 12
    ) -> Mock:
        client = Mock()
        response = Mock()
        response.data = [Mock(index=index, embedding=embedding) for index, embedding in data]
        response.usage = Mock(total_tokens=total_tokens)
        client.embeddings.create = AsyncMock(return_value=response)
        return client

    async def test_embed_sorts_results_by_index(self) -> None:
        # API may return embeddings out of request order.
        client = self._client_returning([(1, [0.0, 1.0]), (0, [1.0, 0.0])], total_tokens=5)

        service = OpenAIEmbeddingService(
            api_key="test-key", model="text-embedding-3-small", client=client
        )
        result = await service.embed(["first", "second"])

        assert result.embeddings == [[1.0, 0.0], [0.0, 1.0]]
        assert result.tokens_used == 5
        assert result.model == "text-embedding-3-small"

    async def test_empty_input_skips_api_call(self) -> None:
        client = self._client_returning([])

        service = OpenAIEmbeddingService(api_key="test-key", client=client)
        result = await service.embed([])

        assert result.embeddings == []
        client.embeddings.create.assert_not_awaited()

    async def test_api_failure_wrapped_as_external_error(self) -> None:
        client = Mock()
        client.embeddings.create = AsyncMock(side_effect=RuntimeError("boom"))

        service = OpenAIEmbeddingService(api_key="test-key", client=client)
        with pytest.raises(ExternalServiceError, match="Failed to generate embeddings"):
            await service.embed(["hello"])

    async def test_embed_single_returns_first_vector(self) -> None:
        client = self._client_returning([(0, [0.25, 0.75])])

        service = OpenAIEmbeddingService(api_key="test-key", client=client)
        assert await service.embed_single("hello") == [0.25, 0.75]

    async def test_default_dimensions_from_model_table(self) -> None:
        client = self._client_returning([])

        small = OpenAIEmbeddingService(api_key="k", model="text-embedding-3-small", client=client)
        large = OpenAIEmbeddingService(api_key="k", model="text-embedding-3-large", client=client)

        assert small.dimensions == 1536
        assert large.dimensions == 3072
