"""Rerank wiring in the RAG node: the documented pipeline stage, revived.

chat.py builds a reranker, budget-wraps it, and stores it in the
retrieval pipeline — but the dialogue graph never consumed it, so the
documented Reranking stage (检索 → 融合 → 重排 → 生成) silently never
ran. These tests pin the revived wiring: results are reranked before
conversion and L2 caching (the stored order IS the reranked order, so
cache hits replay it without paying the rerank again), a reranker
failure degrades to the hybrid order (fail-open, like every retrieval
leg), and a cache hit skips the reranker entirely.
"""

from typing import Any
from unittest.mock import AsyncMock, Mock

from app.services.dialogue.nodes import NodeFactory


class FakeSearchResult:
    """SearchResult-shaped objects for _search_result_to_dict."""

    def __init__(self, document_id: str, content: str, score: float) -> None:
        self.document_id = document_id
        self.content = content
        self.score = score
        self.metadata = None


class StubRetrievalCache:
    """Recording stand-in for node-wiring tests."""

    def __init__(self, cached_docs: list[dict[str, Any]] | None) -> None:
        self.cached_docs = cached_docs
        self.puts: list[tuple[str, list[dict[str, Any]]]] = []

    async def get(self, query: str, filters: dict[str, Any]) -> list[dict[str, Any]] | None:
        return self.cached_docs

    async def put(self, query: str, filters: dict[str, Any], docs: list[dict[str, Any]]) -> None:
        self.puts.append((query, docs))


def _factory(
    hybrid: Mock,
    reranker: Mock | None = None,
    cache: StubRetrievalCache | None = None,
) -> NodeFactory:
    pipeline: dict[str, Any] = {"hybrid_search": hybrid}
    if reranker is not None:
        pipeline["reranker"] = reranker
    return NodeFactory(
        intent_detector=Mock(),
        slot_filler=None,
        tool_registry=Mock(),
        retrieval_pipeline=pipeline,
        retrieval_cache=cache,
    )


async def _lookup(factory: NodeFactory) -> dict[str, Any]:
    return await factory.rag_lookup_node(
        {"message": "退货政策是什么", "session_id": 1, "intent": "question"}
    )


class TestRagRerankWiring:
    async def test_reranker_reorders_results_and_cache_stores_reranked_order(self):
        low = FakeSearchResult("d1", "普通条款", 0.9)
        top = FakeSearchResult("d2", "退货政策全文", 0.4)
        hybrid = Mock()
        hybrid.search = AsyncMock(return_value=[low, top])  # hybrid order: d1 first
        reranker = Mock()
        reranker.rerank = AsyncMock(return_value=[top, low])  # reranked: d2 first
        cache = StubRetrievalCache(cached_docs=None)

        updates = await _lookup(_factory(hybrid, reranker, cache))

        reranker.rerank.assert_awaited_once()
        assert [d["document_id"] for d in updates["retrieved_docs"]] == ["d2", "d1"]
        # The L2 entry replays the reranked order, not the hybrid one.
        _, put_docs = cache.puts[0]
        assert [d["document_id"] for d in put_docs] == ["d2", "d1"]

    async def test_reranker_failure_degrades_to_hybrid_order(self):
        """Fail-open: the reranker is an LLM-backed precision leg, never a
        dependency — a mid-rerank outage keeps the hybrid order serving."""
        first = FakeSearchResult("d1", "退货政策", 0.9)
        second = FakeSearchResult("d2", "次要文档", 0.5)
        hybrid = Mock()
        hybrid.search = AsyncMock(return_value=[first, second])
        reranker = Mock()
        reranker.rerank = AsyncMock(side_effect=RuntimeError("rerank outage"))
        cache = StubRetrievalCache(cached_docs=None)

        updates = await _lookup(_factory(hybrid, reranker, cache))

        assert [d["document_id"] for d in updates["retrieved_docs"]] == ["d1", "d2"]
        assert [d["document_id"] for d in cache.puts[0][1]] == ["d1", "d2"]

    async def test_cache_hit_skips_reranker(self):
        """The cached order already survived a rerank when it was stored —
        replaying it must not pay the rerank (an LLM call) again."""
        hybrid = Mock()
        hybrid.search = AsyncMock(return_value=[])
        reranker = Mock()
        reranker.rerank = AsyncMock(return_value=[])
        cache = StubRetrievalCache(
            cached_docs=[{"document_id": "d1", "content": "x", "score": 1.0}]
        )

        updates = await _lookup(_factory(hybrid, reranker, cache))

        reranker.rerank.assert_not_awaited()
        assert [d["document_id"] for d in updates["retrieved_docs"]] == ["d1"]

    async def test_no_reranker_in_pipeline_is_passthrough(self):
        first = FakeSearchResult("d1", "退货政策", 0.9)
        hybrid = Mock()
        hybrid.search = AsyncMock(return_value=[first])

        updates = await _lookup(_factory(hybrid))

        assert [d["document_id"] for d in updates["retrieved_docs"]] == ["d1"]
