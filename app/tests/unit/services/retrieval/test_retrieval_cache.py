"""L2 retrieval-result cache: repeated queries stop round-tripping Qdrant.

Cache-layering plan L2 (docs/cache-layering-plan.md): identical
(query, filters) pairs within one KB epoch replay the cached doc set
instead of re-running the Qdrant+BM25 legs. The plan's stated value is
protecting Qdrant — especially the filter-miss double-search
amplification, where one user turn costs two vector searches. Pins the
layer's hard boundaries: epoch-scoped keys (a KB update invalidates
wholesale), filter-signature participation (a filtered search must
never replay another filter's docs), bounded store, fail-open on every
Redis failure, and hit/miss telemetry so the protection is measurable.
"""

import json
from typing import Any
from unittest.mock import AsyncMock, Mock

from prometheus_client import REGISTRY

from app.services.dialogue.nodes import NodeFactory
from app.services.retrieval.retrieval_cache import RetrievalCacheService


class FakeRedis:
    """In-memory hash store sufficient for the retrieval cache."""

    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}
        self.last_expire: tuple[str, int] | None = None

    async def hset(self, name: str, key: str, value: str) -> None:
        self.hashes.setdefault(name, {})[key] = value

    async def hget(self, name: str, key: str) -> str | None:
        return self.hashes.get(name, {}).get(key)

    async def hlen(self, name: str) -> int:
        return len(self.hashes.get(name, {}))

    async def expire(self, name: str, ttl: int) -> None:
        self.last_expire = (name, ttl)


class RaisingRedis:
    async def hset(self, name: str, key: str, value: str) -> None:
        raise ConnectionError("redis down")

    async def hget(self, name: str, key: str) -> str | None:
        raise ConnectionError("redis down")

    async def hlen(self, name: str) -> int:
        raise ConnectionError("redis down")

    async def expire(self, name: str, ttl: int) -> None:
        raise ConnectionError("redis down")


def _provider(epoch: str = "e1") -> Any:
    async def provider() -> str:
        return epoch

    return provider


def _settings_stub(enabled: bool = True, ttl: int = 300, max_entries: int = 512) -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(
        RETRIEVAL_CACHE_ENABLED=enabled,
        RETRIEVAL_CACHE_TTL_SECONDS=ttl,
        RETRIEVAL_CACHE_MAX_ENTRIES=max_entries,
    )


def _make_service(
    redis: Any = None,
    enabled: bool = True,
    max_entries: int = 512,
    epoch: str = "e1",
) -> RetrievalCacheService:
    return RetrievalCacheService(
        redis_client=redis if redis is not None else FakeRedis(),
        epoch_provider=_provider(epoch),
        settings=_settings_stub(enabled=enabled, max_entries=max_entries),
    )


def _docs() -> list[dict[str, Any]]:
    return [
        {"document_id": "d1", "content": "退货政策…", "score": 0.9, "metadata": None},
        {"document_id": "d2", "content": "运费说明…", "score": 0.7, "metadata": None},
    ]


def _metric(name: str) -> float:
    value = REGISTRY.get_sample_value(name)
    return value if value is not None else 0.0


class TestRetrievalCacheHits:
    async def test_identical_query_replays_docs(self):
        service = _make_service()
        await service.put("退货政策", filters={}, docs=_docs())

        cached = await service.get("退货政策", filters={})
        assert cached is not None
        assert [d["document_id"] for d in cached] == ["d1", "d2"]
        assert cached[0]["content"] == "退货政策…"

    async def test_whitespace_and_case_variants_share_key(self):
        service = _make_service()
        await service.put("  Return   POLICY ", filters={}, docs=_docs())

        assert await service.get("return policy", filters={}) is not None

    async def test_hit_and_miss_counters_move(self):
        service = _make_service()
        before_hits = _metric("retrieval_cache_hits_total")
        before_misses = _metric("retrieval_cache_misses_total")

        await service.get("未缓存的问题", filters={})
        await service.put("已缓存的问题", filters={}, docs=_docs())
        await service.get("已缓存的问题", filters={})

        assert _metric("retrieval_cache_hits_total") == before_hits + 1.0
        assert _metric("retrieval_cache_misses_total") == before_misses + 1.0


class TestRetrievalCacheBoundaries:
    async def test_filters_participate_in_key(self):
        service = _make_service()
        await service.put("查订单", filters={"order_id": "A"}, docs=_docs())

        assert await service.get("查订单", filters={"order_id": "B"}) is None

    async def test_epoch_change_invalidates(self):
        redis = FakeRedis()
        writer = _make_service(redis=redis, epoch="e1")
        await writer.put("退货政策", filters={}, docs=_docs())

        reader = _make_service(redis=redis, epoch="e2")
        assert await reader.get("退货政策", filters={}) is None

    async def test_unavailable_epoch_refuses_reads_and_writes(self):
        from app.services.retrieval.kb_epoch import EPOCH_UNAVAILABLE

        redis = FakeRedis()
        service = _make_service(redis=redis, epoch=EPOCH_UNAVAILABLE)
        await service.put("退货政策", filters={}, docs=_docs())

        assert await service.get("退货政策", filters={}) is None
        assert redis.hashes == {}

    async def test_disabled_is_full_noop(self):
        redis = FakeRedis()
        service = _make_service(redis=redis, enabled=False)
        await service.put("退货政策", filters={}, docs=_docs())

        assert redis.hashes == {}
        assert await service.get("退货政策", filters={}) is None

    async def test_full_store_refuses_new_writes(self):
        redis = FakeRedis()
        service = _make_service(redis=redis, max_entries=1)
        await service.put("第一条", filters={}, docs=_docs())
        await service.put("第二条", filters={}, docs=_docs())

        assert sum(len(fields) for fields in redis.hashes.values()) == 1

    async def test_empty_doclist_never_written(self):
        redis = FakeRedis()
        service = _make_service(redis=redis)
        await service.put("没有结果的查询", filters={}, docs=[])

        assert redis.hashes == {}


class TestRetrievalCacheFailOpen:
    async def test_redis_outage_never_raises(self):
        service = _make_service(redis=RaisingRedis())
        assert await service.get("任何查询", filters={}) is None
        await service.put("任何查询", filters={}, docs=_docs())  # must not raise

    async def test_corrupt_payload_is_a_miss(self):
        redis = FakeRedis()
        service = _make_service(redis=redis)
        await service.put("退货政策", filters={}, docs=_docs())
        # Corrupt the stored payload behind the service's back.
        name = next(iter(redis.hashes))
        field = next(iter(redis.hashes[name]))
        redis.hashes[name][field] = "not-json{"

        assert await service.get("退货政策", filters={}) is None


class TestRetrievalCacheCodec:
    async def test_roundtrip_preserves_doc_shape(self):
        service = _make_service()
        docs = _docs()
        await service.put("退货政策", filters={}, docs=docs)
        cached = await service.get("退货政策", filters={})
        assert cached is not None
        assert json.dumps(cached, ensure_ascii=False) == json.dumps(docs, ensure_ascii=False)


class StubRetrievalCache:
    """Recording stand-in for node-wiring tests."""

    def __init__(self, cached_docs: list[dict[str, Any]] | None) -> None:
        self.cached_docs = cached_docs
        self.gets: list[str] = []
        self.puts: list[tuple[str, list[dict[str, Any]]]] = []

    async def get(self, query: str, filters: dict[str, Any]) -> list[dict[str, Any]] | None:
        self.gets.append(query)
        return self.cached_docs

    async def put(self, query: str, filters: dict[str, Any], docs: list[dict[str, Any]]) -> None:
        self.puts.append((query, docs))


class _FakeSearchResults:
    """SearchResult-shaped objects for _search_result_to_dict."""

    def __init__(self, document_id: str, content: str, score: float) -> None:
        self.document_id = document_id
        self.content = content
        self.score = score
        self.metadata = None


class TestRagNodeWiring:
    async def test_cache_hit_skips_hybrid_search(self):
        stub_cache = StubRetrievalCache(cached_docs=_docs())
        hybrid = Mock()
        hybrid.search = AsyncMock(return_value=[])
        factory = NodeFactory(
            intent_detector=Mock(),
            slot_filler=None,
            tool_registry=Mock(),
            retrieval_pipeline={"hybrid_search": hybrid},
            retrieval_cache=stub_cache,
        )

        updates = await factory.rag_lookup_node(
            {"message": "退货政策是什么", "session_id": 1, "intent": "question"}
        )

        hybrid.search.assert_not_awaited()
        assert stub_cache.gets == ["退货政策是什么"]
        assert [d["document_id"] for d in updates["retrieved_docs"]] == ["d1", "d2"]
        assert updates["sources"] == ["d1", "d2"]

    async def test_miss_searches_then_writes_back(self):
        stub_cache = StubRetrievalCache(cached_docs=None)
        hybrid = Mock()
        hybrid.search = AsyncMock(return_value=[_FakeSearchResults("d1", "退货政策…", 0.9)])
        factory = NodeFactory(
            intent_detector=Mock(),
            slot_filler=None,
            tool_registry=Mock(),
            retrieval_pipeline={"hybrid_search": hybrid},
            retrieval_cache=stub_cache,
        )

        updates = await factory.rag_lookup_node(
            {"message": "退货政策是什么", "session_id": 1, "intent": "question"}
        )

        hybrid.search.assert_awaited_once()
        assert len(stub_cache.puts) == 1
        query, docs = stub_cache.puts[0]
        assert query == "退货政策是什么"
        assert [d["document_id"] for d in docs] == ["d1"]
        assert updates["sources"] == ["d1"]

    async def test_none_cache_is_pure_passthrough(self):
        hybrid = Mock()
        hybrid.search = AsyncMock(return_value=[])
        factory = NodeFactory(
            intent_detector=Mock(),
            slot_filler=None,
            tool_registry=Mock(),
            retrieval_pipeline={"hybrid_search": hybrid},
        )

        updates = await factory.rag_lookup_node(
            {"message": "退货政策是什么", "session_id": 1, "intent": "question"}
        )
        hybrid.search.assert_awaited_once()
        assert updates["retrieved_docs"] == []
