"""Publication policy must protect every consumer, including cached text/BM25.

The policy double isolates wiring; real SQLite policy decisions are tested in
publication's own suite. Retrieval, RRF, BM25 and node consumption run normally.
"""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest

from app.config.settings import settings
from app.services.chat.answer_cache import CachedAnswer
from app.services.dialogue.nodes import NodeFactory
from app.services.graph.base import GraphSearchResult
from app.services.llm.base import LLMResponse
from app.services.retrieval.factory import RetrievalFactory
from app.services.retrieval.hybrid_search import HybridSearchService, KeywordSearch
from app.services.retrieval.vector_base import SearchResult, VectorSearchRequest


class PublicationPolicy:
    def __init__(self, unavailable: bool = False) -> None:
        self.unavailable = unavailable

    def check(self) -> None:
        if self.unavailable:
            raise RuntimeError("publication registry unavailable")

    async def filter_results(self, results: Any) -> list[Any]:
        self.check()
        return [r for r in results if (r.metadata or {}).get("version_id") == "active"]

    async def filter_documents(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.check()
        return [d for d in documents if (d.get("metadata") or {}).get("version_id") == "active"]

    async def filter_graph_results(self, results: Any) -> list[Any]:
        self.check()
        return [r for r in results if (r.metadata or {}).get("source_version_ids") == ["active"]]

    async def answer_allowed(self, sources: list[str], citations: list[dict[str, Any]]) -> bool:
        self.check()
        return bool(citations) and all(c.get("version_id") == "active" for c in citations)


def hit(version: str, content: str, chunk: str = "") -> SearchResult:
    return SearchResult(
        "policy",
        content,
        0.9,
        {"document_id": "policy", "version_id": version, "chunk_id": chunk or content},
    )


def document(version: str, content: str) -> dict[str, Any]:
    return {"document_id": "policy", "content": content, "metadata": {"version_id": version}}


def cached(version: str = "failed") -> CachedAnswer:
    return CachedAnswer(
        "UNPUBLISHED CACHED ANSWER",
        ["policy"],
        "chitchat",
        citations=[
            {
                "source_type": "document",
                "document_id": "policy",
                "version_id": version,
                "excerpt": "UNPUBLISHED",
            }
        ],
    )


def cache(answer: Any) -> Any:
    return SimpleNamespace(
        get=AsyncMock(return_value=answer),
        capture_epoch=AsyncMock(return_value="unchanged-epoch"),
        put=AsyncMock(),
    )


def factory(policy: Any, **kwargs: Any) -> tuple[NodeFactory, Any]:
    llm = Mock()
    llm.generate = AsyncMock(
        return_value=LLMResponse(content="Fresh safe greeting.", model="local")
    )
    return NodeFactory(
        intent_detector=Mock(),
        slot_filler=None,
        tool_registry=Mock(),
        llm_service=llm,
        retrieval_pipeline={"publication_guard": policy, **kwargs.pop("retrieval_pipeline", {})},
        **kwargs,
    ), llm


@pytest.fixture(autouse=True)
def fixed_generation(monkeypatch):
    monkeypatch.setattr(settings, "FACT_CLAIM_CHECK_ENABLED", False)
    monkeypatch.setattr(settings, "QUERY_REWRITE_ENABLED", False)


@pytest.mark.parametrize("unavailable", [False, True])
async def test_l0_rejects_the_entire_cached_body_when_publication_is_stale_or_unknown(unavailable):
    node, _ = factory(PublicationPolicy(unavailable), answer_cache=cache(cached()))
    result = await node.answer_cache_lookup_node({"message": "policy"})
    assert result["route_after_cache"] == "miss"
    assert "UNPUBLISHED" not in result.get("response", "")


async def test_l0_still_replays_published_cache_control():
    node, _ = factory(PublicationPolicy(), answer_cache=cache(cached("active")))
    assert (await node.answer_cache_lookup_node({"message": "hello"}))["route_after_cache"] == "hit"


@pytest.mark.parametrize("unavailable", [False, True])
async def test_l1_rejects_cached_text_without_turning_unknown_publication_into_a_hit(unavailable):
    node, llm = factory(PublicationPolicy(unavailable), semantic_cache=cache(cached()))
    result = await node.direct_response_node({"message": "Hello", "intent": "chitchat"})
    assert result["response"] == "Fresh safe greeting."
    llm.generate.assert_awaited_once()


async def test_l2_partly_filtered_cache_is_a_miss_and_refills_from_live_retrieval():
    hybrid = SimpleNamespace(search=AsyncMock(return_value=[hit("active", "FRESH ACTIVE POLICY")]))
    stored = cache(
        [document("active", "OLD CACHED ACTIVE"), document("failed", "UNPUBLISHED CACHE")]
    )
    node, _ = factory(
        PublicationPolicy(), retrieval_cache=stored, retrieval_pipeline={"hybrid_search": hybrid}
    )
    result = await node.rag_lookup_node({"message": "policy", "intent": "policy"})
    hybrid.search.assert_awaited_once()
    assert [d["content"] for d in result["retrieved_docs"]] == ["FRESH ACTIVE POLICY"]
    assert result["retrieval_degraded"] is False


async def test_l2_registry_outage_misses_cache_and_marks_fresh_attempt_degraded():
    hybrid = SimpleNamespace(search=AsyncMock(return_value=[hit("active", "CANNOT VERIFY")]))
    node, _ = factory(
        PublicationPolicy(True),
        retrieval_cache=cache([document("active", "CACHED")]),
        retrieval_pipeline={"hybrid_search": hybrid},
    )
    result = await node.rag_lookup_node({"message": "policy", "intent": "policy"})
    hybrid.search.assert_awaited_once()
    assert result["retrieved_docs"] == []
    assert result["retrieval_degraded"] is True


async def test_cached_and_fresh_graph_rows_cannot_bypass_publication_policy():
    graph = SimpleNamespace(
        search=AsyncMock(
            return_value=[
                GraphSearchResult(
                    "ACTIVE GRAPH", [], [], 1, "graph", {"source_version_ids": ["active"]}
                ),
                GraphSearchResult(
                    "MIXED UNPUBLISHED GRAPH",
                    [],
                    [],
                    0.9,
                    "graph",
                    {"source_version_ids": ["active", "failed"]},
                ),
            ]
        )
    )
    node, _ = factory(PublicationPolicy(), graph_retrieval_service=graph)
    result = await node.rag_lookup_node({"message": "compare", "intent": "relationship_query"})
    assert [d["content"] for d in result["retrieved_docs"]] == ["ACTIVE GRAPH"]


async def test_fresh_rag_custom_hybrid_is_checked_before_reranking_or_prompt_use():
    hybrid = SimpleNamespace(
        search=AsyncMock(
            return_value=[hit("failed", "UNPUBLISHED NEW"), hit("active", "PUBLISHED OLD")]
        )
    )
    reranker = SimpleNamespace(rerank=AsyncMock(side_effect=lambda results, request: results))
    node, _ = factory(
        PublicationPolicy(), retrieval_pipeline={"hybrid_search": hybrid, "reranker": reranker}
    )
    result = await node.rag_lookup_node({"message": "policy", "intent": "policy"})
    assert [d["content"] for d in result["retrieved_docs"]] == ["PUBLISHED OLD"]
    assert all(r.content != "UNPUBLISHED NEW" for r in reranker.rerank.await_args.args[0])


@pytest.mark.parametrize("vector_fails", [False, True])
async def test_real_bm25_and_rrf_filter_both_legs_before_delivery(vector_fails):
    keyword = KeywordSearch()
    await keyword.add_documents(
        [
            {
                "id": "failed",
                "content": "refund refund refund NEW UNPUBLISHED",
                "metadata": {"document_id": "policy", "version_id": "failed"},
            },
            {
                "id": "active",
                "content": "refund OLD PUBLISHED",
                "metadata": {"document_id": "policy", "version_id": "active"},
            },
        ]
    )
    vector = SimpleNamespace(
        search=AsyncMock(
            side_effect=RuntimeError("vector offline") if vector_fails else None,
            return_value=[hit("failed", "refund NEW UNPUBLISHED", "failed")],
        )
    )
    hybrid = HybridSearchService(cast(Any, vector), keyword)
    hybrid.publication_guard = cast(Any, PublicationPolicy())
    results = await hybrid.search(VectorSearchRequest("refund", top_k=1))
    assert len(results) == 1 and results[0].content == "refund OLD PUBLISHED"


async def test_guard_failure_never_falls_back_to_raw_keyword_hits():
    keyword = KeywordSearch()
    await keyword.add_documents(
        [
            {
                "id": "failed",
                "content": "refund UNVERIFIED",
                "metadata": {"document_id": "policy", "version_id": "failed"},
            }
        ]
    )
    hybrid = HybridSearchService(
        cast(Any, SimpleNamespace(search=AsyncMock(return_value=[]))), keyword
    )
    hybrid.publication_guard = cast(Any, PublicationPolicy(True))
    with pytest.raises(RuntimeError, match="registry unavailable"):
        await hybrid.search(VectorSearchRequest("refund"))


async def test_overfetch_recovers_published_hit_below_unpublished_top_slot(monkeypatch):
    monkeypatch.setattr(settings, "RETRIEVAL_CANDIDATE_POOL", 10)
    hits = [hit("failed", "UNPUBLISHED " + str(n)) for n in range(5)] + [
        hit("active", "DEEP PUBLISHED")
    ]
    vector = SimpleNamespace(search=AsyncMock(side_effect=lambda request: hits[: request.top_k]))
    hybrid = HybridSearchService(cast(Any, vector), KeywordSearch())
    hybrid.publication_guard = cast(Any, PublicationPolicy())
    result = await hybrid.search(
        VectorSearchRequest("refund", top_k=1, acl_filters={"tenant": "allowed"})
    )
    assert [r.content for r in result] == ["DEEP PUBLISHED"]
    request = vector.search.await_args.args[0]
    assert request.top_k >= 10 and request.filters == {"tenant": "allowed"}


async def test_factory_warmup_does_not_seed_unpublished_points_into_bm25():
    keyword = KeywordSearch()
    hybrid = HybridSearchService(cast(Any, SimpleNamespace()), keyword)
    hybrid.publication_guard = cast(Any, PublicationPolicy())
    vector = SimpleNamespace(
        list_chunks=AsyncMock(
            return_value=[
                {
                    "id": "active",
                    "content": "PUBLISHED",
                    "metadata": {"document_id": "policy", "version_id": "active"},
                },
                {
                    "id": "failed",
                    "content": "UNPUBLISHED",
                    "metadata": {"document_id": "policy", "version_id": "failed"},
                },
            ]
        )
    )
    count = await RetrievalFactory.warm_keyword_index(hybrid, cast(Any, vector))
    assert count == 1 and list(keyword.documents) == ["active"]


async def test_complete_factory_pipeline_carries_the_same_policy_to_retrieval_and_nodes():
    policy = PublicationPolicy()
    vector = SimpleNamespace(
        search=AsyncMock(return_value=[hit("failed", "NEW"), hit("active", "PUBLISHED")])
    )
    pipeline = RetrievalFactory.create_pipeline(
        cast(Any, vector), use_reranking=False, publication_guard=cast(Any, policy)
    )
    assert pipeline["publication_guard"] is policy
    result = await pipeline["hybrid_search"].search(VectorSearchRequest("refund", top_k=1))
    assert [r.content for r in result] == ["PUBLISHED"]


async def test_graph_publication_outage_is_degraded_without_exposing_raw_context():
    graph = SimpleNamespace(
        search=AsyncMock(
            return_value=[
                GraphSearchResult(
                    "UNVERIFIED GRAPH", [], [], 1, "graph", {"source_version_ids": ["active"]}
                )
            ]
        )
    )
    node, _ = factory(PublicationPolicy(True), graph_retrieval_service=graph)
    result = await node.rag_lookup_node({"message": "compare", "intent": "relationship_query"})
    assert result["retrieved_docs"] == []
    assert result["retrieval_degraded"] is True
