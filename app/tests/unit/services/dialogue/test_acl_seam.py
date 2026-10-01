"""Server-side retrieval ACL seam (review 2026-09-26, #1 final slice).

The review's requirement: 检索中的 ACL 由服务端注入，不能放进可取消的
业务筛选条件中. Until now every search filter was a *business* filter —
slot-extracted, whitelist-narrowed, and dropped wholesale by the
filter-miss fallback — so any future ACL expressed as a filter would be
client-cancellable by construction.

This file pins the seam contract (docs/design/per-user-architecture.md D5):

1. ``VectorSearchRequest.acl_filters`` is a separate channel, injected
   only by ``retrieval_acl_scope()`` from server-side context — never
   the request body, never slot extraction.
2. The FILTERABLE_METADATA_KEYS whitelist never applies to it (that
   whitelist is the *business* filter contract).
3. The filter-miss fallback in rag_lookup_node drops business filters
   but PRESERVES the ACL channel — recall protection must not become
   an authorization bypass.
4. HybridSearchService merges ACL + business filters (AND semantics,
   ACL wins on key collision) at the dispatch point, so neither leg
   can be asked to search without the caller's scope applied.
5. The L2 retrieval cache key folds the ACL in — entries are post-ACL
   doc sets, so a scope-ineligible hit must never replay to a caller.
6. The agent knowledge tool builds its own VectorSearchRequest — it
   carries the scope too (no search path bypasses the channel).

Today the scope is the empty constraint (every KB document is public),
so production behavior is unchanged; the tests inject a synthetic
scope to pin the invariants for when per-user documents exist (P2).
"""

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

from app.models.enums.intent import Intent
from app.services.chat.chat_service import ChatService
from app.services.dialogue.graph import build_dialogue_graph
from app.services.dialogue.tools import create_knowledge_tool
from app.services.llm.base import LLMMessage, LLMResponse, LLMServiceBase
from app.services.retrieval.hybrid_search import HybridSearchService, KeywordSearch
from app.services.retrieval.vector_base import VectorSearchRequest
from app.services.slot_filling.base import ExtractedSlot, SlotFillingResult

SCOPE = {"visibility": "public"}


class _StubLLM(LLMServiceBase):
    """Minimal generator: records calls, returns fixed content."""

    def __init__(self) -> None:
        super().__init__(api_key="test", model="test")

    async def generate(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        return LLMResponse(content="7-day no-reason returns.", model=self.model)

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        yield "7-day no-reason returns."

    def estimate_tokens(self, text: str) -> int:
        return len(text)

    async def count_tokens(self, messages: list[LLMMessage]) -> int:
        return sum(len(m.content) for m in messages)


class _FakeHit:
    """SearchResult-shaped object as hybrid_search returns."""

    def __init__(self, document_id: str, content: str, score: float) -> None:
        self.document_id = document_id
        self.content = content
        self.score = score
        self.metadata = None


def _product_filter_filler() -> Mock:
    """Slot filler whose extraction yields one whitelisted business
    filter (product=…) — the fallback-triggering kind."""
    slot = ExtractedSlot(
        slot_type="product",
        entity_type="product",
        value="nonexistent-product",
        normalized_value="nonexistent-product",
    )
    filler = Mock()
    filler.fill_slots = AsyncMock(return_value=SlotFillingResult(slots=[slot]))
    return filler


def _chat(
    hybrid: Mock,
    *,
    slot_filler: Mock | None = None,
    retrieval_cache: Mock | None = None,
) -> ChatService:
    detector = Mock()
    probe = Mock(intent=Intent.POLICY, confidence=0.9)
    detector.detect_with_confidence = AsyncMock(return_value=probe)
    return ChatService(
        graph=build_dialogue_graph(
            intent_detector=detector,
            slot_filler=slot_filler,
            tool_registry=Mock(),
            retrieval_pipeline={"hybrid_search": hybrid},
            llm_service=_StubLLM(),
            retrieval_cache=retrieval_cache,
        ),
    )


def _server_scope() -> Any:
    """Patch the scope constructor to a synthetic per-caller scope."""
    return patch(
        "app.services.retrieval.vector_base.retrieval_acl_scope",
        return_value=dict(SCOPE),
    )


class TestPipelineInjection:
    async def test_initial_request_carries_server_scope(self) -> None:
        """The scope rides its own channel on the primary search call."""
        hybrid = Mock()
        hybrid.search = AsyncMock(return_value=[_FakeHit("doc-1", "policy", 0.9)])
        chat = _chat(hybrid)

        with _server_scope():
            await chat.process_message(1, "What is the return window?", 0)

        request = hybrid.search.await_args.args[0]
        assert request.acl_filters == SCOPE

    async def test_fallback_preserves_acl_and_drops_business_filters(self) -> None:
        """The filter-miss retry drops the *business* filters (recall
        protection) but carries the ACL channel unchanged — the exact
        "cancellable business condition" the review forbids."""
        hybrid = Mock()
        hybrid.search = AsyncMock(side_effect=[[], [_FakeHit("doc-9", "rescued policy", 0.8)]])
        chat = _chat(hybrid, slot_filler=_product_filter_filler())

        with _server_scope():
            response = await chat.process_message(1, "What is the return window?", 0)

        assert hybrid.search.await_count == 2
        first, second = (c.args[0] for c in hybrid.search.await_args_list)
        # Business filters present on the primary call …
        assert first.filters == {"product": "nonexistent-product"}
        assert first.acl_filters == SCOPE
        # … dropped by the retry, while the ACL channel survives it.
        assert second.filters is None
        assert second.acl_filters == SCOPE
        # The unfiltered retry actually served the turn.
        assert response.sources == ["doc-9"]

    async def test_l2_cache_key_folds_acl_scope(self) -> None:
        """L2 entries are post-ACL doc sets: the key must fold the scope
        in, or a private-scope entry would replay to a public caller."""
        hybrid = Mock()
        hybrid.search = AsyncMock(return_value=[_FakeHit("doc-1", "policy", 0.9)])
        cache = Mock()
        cache.get = AsyncMock(return_value=None)
        cache.put = AsyncMock()
        chat = _chat(hybrid, slot_filler=_product_filter_filler(), retrieval_cache=cache)

        with _server_scope():
            await chat.process_message(1, "What is the return window?", 0)

        expected = {"product": "nonexistent-product", "visibility": "public"}
        assert cache.get.await_args.args[1] == expected
        assert cache.put.await_args.args[1] == expected


class TestHybridEnforcement:
    def test_merge_folds_acl_into_filters_acl_wins(self) -> None:
        """ACL authority: a colliding business key (even a forged one)
        cannot loosen the server-injected scope."""
        request = VectorSearchRequest(
            query="q",
            filters={"visibility": "private", "product": "widget"},
            acl_filters={"visibility": "public"},
        )
        merged = request.with_merged_filters()
        assert merged.filters == {"visibility": "public", "product": "widget"}

    def test_merge_without_acl_is_identity(self) -> None:
        """No scope (today's shared-public KB): the request passes
        through untouched — behavior is unchanged by the seam."""
        request = VectorSearchRequest(query="q", filters={"product": "widget"})
        assert request.with_merged_filters() is request

    async def test_both_legs_enforce_the_merged_scope(self) -> None:
        """The merge happens at the hybrid dispatch point: the vector
        leg sees the folded filters, and the keyword leg's metadata
        match excludes the private document from the fused result."""
        captured: list[VectorSearchRequest] = []

        class _CapturingVectorClient:
            async def search(self, request: VectorSearchRequest) -> list[Any]:
                captured.append(request)
                return []

        keyword = KeywordSearch()
        await keyword.add_documents(
            [
                {
                    "id": "pub-1",
                    "content": "return policy window is seven days",
                    "metadata": {"visibility": "public", "product": "widget"},
                },
                {
                    "id": "priv-1",
                    "content": "return policy window is confidential",
                    "metadata": {"visibility": "private", "product": "widget"},
                },
            ]
        )
        hybrid = HybridSearchService(
            vector_client=_CapturingVectorClient(),  # type: ignore[arg-type]
            keyword_search=keyword,
        )

        results = await hybrid.search(
            VectorSearchRequest(
                query="return policy window",
                top_k=5,
                filters={"product": "widget"},
                acl_filters=SCOPE,
            )
        )

        assert captured[0].filters == {"product": "widget", "visibility": "public"}
        assert [r.document_id for r in results] == ["pub-1"]


class TestKnowledgeToolInjection:
    async def test_agent_knowledge_tool_carries_scope(self) -> None:
        """The agent loop's search_knowledge_base builds its own request;
        it must carry the same server-injected scope (no bypass path)."""
        hybrid = Mock()
        hybrid.search = AsyncMock(return_value=[])
        tool = create_knowledge_tool(hybrid)

        with _server_scope():
            await tool.handler({"query": "return policy"})

        request = hybrid.search.await_args.args[0]
        assert request.acl_filters == SCOPE
