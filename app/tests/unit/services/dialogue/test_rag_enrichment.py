"""RAG enrichment: the injected slot filler drives retrieval.

Pinned contract: ``rag_lookup_node`` consumes ``self._slot_filler`` to
extract entities from the raw message — normalized slot values are
appended to the hybrid-search query (recall for BM25 + vector), and
entity hints are forwarded to graph retrieval via the
``entity_hints`` parameter. Extraction failure degrades to the raw
message; a disabled filler (None) leaves behavior unchanged.
"""

from unittest.mock import AsyncMock, Mock

from app.services.dialogue.nodes import NodeFactory
from app.services.slot_filling.base import ExtractedSlot, SlotFillingResult

SLOT_PRODUCT = ExtractedSlot(
    slot_type="product",
    entity_type="Product",
    value="iPhone 13",
    normalized_value="iPhone 13",
)
SLOT_ISSUE = ExtractedSlot(
    slot_type="issue",
    entity_type="Issue",
    value="不能开机",
    normalized_value="无法开机",
)


def _filler(*slots: ExtractedSlot) -> Mock:
    filler = Mock()
    filler.fill_slots = AsyncMock(return_value=SlotFillingResult(slots=list(slots), raw_query="q"))
    return filler


def _make_factory(
    hybrid: Mock | None = None,
    graph_service: Mock | None = None,
    filler: Mock | None = None,
) -> NodeFactory:
    return NodeFactory(
        intent_detector=Mock(),
        slot_filler=filler,
        tool_registry=Mock(),
        retrieval_pipeline={"hybrid_search": hybrid},
        llm_service=None,
        guardrail_service=None,
        graph_retrieval_service=graph_service,
        handoff_service=None,
    )


class TestVectorQueryEnrichment:
    async def test_slot_values_appended_to_query(self):
        hybrid = Mock()
        hybrid.search = AsyncMock(return_value=[])

        factory = _make_factory(hybrid=hybrid, filler=_filler(SLOT_PRODUCT, SLOT_ISSUE))
        await factory.rag_lookup_node({"message": "我的手机怎么了", "intent": "question"})

        request = hybrid.search.await_args.args[0]
        assert request.query.startswith("我的手机怎么了")
        assert "iPhone 13" in request.query
        assert "无法开机" in request.query

    async def test_value_already_in_message_not_duplicated(self):
        hybrid = Mock()
        hybrid.search = AsyncMock(return_value=[])

        factory = _make_factory(hybrid=hybrid, filler=_filler(SLOT_PRODUCT))
        await factory.rag_lookup_node({"message": "iPhone 13 不能用", "intent": "question"})

        assert hybrid.search.await_args.args[0].query == "iPhone 13 不能用"

    async def test_no_slots_keeps_plain_query(self):
        hybrid = Mock()
        hybrid.search = AsyncMock(return_value=[])

        factory = _make_factory(hybrid=hybrid, filler=_filler())
        await factory.rag_lookup_node({"message": "退货政策", "intent": "question"})

        assert hybrid.search.await_args.args[0].query == "退货政策"

    async def test_filler_failure_degrades_to_raw_message(self):
        hybrid = Mock()
        hybrid.search = AsyncMock(return_value=[])
        filler = Mock()
        filler.fill_slots = AsyncMock(side_effect=RuntimeError("boom"))

        factory = _make_factory(hybrid=hybrid, filler=filler)
        updates = await factory.rag_lookup_node({"message": "退货政策", "intent": "question"})

        assert hybrid.search.await_args.args[0].query == "退货政策"
        assert updates["retrieved_docs"] == []

    async def test_filler_none_keeps_plain_query(self):
        hybrid = Mock()
        hybrid.search = AsyncMock(return_value=[])

        factory = _make_factory(hybrid=hybrid, filler=None)
        await factory.rag_lookup_node({"message": "退货政策", "intent": "question"})

        assert hybrid.search.await_args.args[0].query == "退货政策"


class TestGraphEntityHints:
    async def test_hints_forwarded_to_graph_search(self):
        graph_service = Mock()
        graph_service.search = AsyncMock(return_value=[])

        factory = _make_factory(
            graph_service=graph_service, filler=_filler(SLOT_PRODUCT, SLOT_ISSUE)
        )
        await factory.rag_lookup_node({"message": "我的手机怎么了", "intent": "entity_lookup"})

        kwargs = graph_service.search.await_args.kwargs
        assert kwargs.get("entity_hints") == [
            {"type": "Product", "name": "iPhone 13"},
            {"type": "Issue", "name": "无法开机"},
        ]
        # Graph search also gets the enriched query.
        assert "iPhone 13" in graph_service.search.await_args.args[0]

    async def test_no_slots_passes_no_hints(self):
        graph_service = Mock()
        graph_service.search = AsyncMock(return_value=[])

        factory = _make_factory(graph_service=graph_service, filler=_filler())
        await factory.rag_lookup_node({"message": "退货政策", "intent": "entity_lookup"})

        assert graph_service.search.await_args.kwargs.get("entity_hints") is None

    async def test_filler_none_graph_unaffected(self):
        graph_service = Mock()
        graph_service.search = AsyncMock(return_value=[])

        factory = _make_factory(graph_service=graph_service, filler=None)
        await factory.rag_lookup_node({"message": "退货政策", "intent": "entity_lookup"})

        assert graph_service.search.await_args.args[0] == "退货政策"
