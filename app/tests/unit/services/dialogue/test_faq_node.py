"""FAQ node: hit/miss routing, provenance, guardrail, graph-level flow.

Pinned contract: a curated hit ends the turn with the pre-approved
answer (no retrieval, no LLM) and carries ``faq:<id>`` provenance;
every miss path — unwired service, below threshold, embedding failure
— continues into the RAG pipeline unchanged. The output guardrail
runs on hits like on every user-facing response.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from app.services.dialogue.nodes import NodeFactory
from app.services.dialogue.tools import create_default_tool_registry
from app.services.faq import FAQEntry, FAQService


class FakeEmbeddings:
    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self.mapping = dict(mapping)

    async def embed(self, texts):
        vectors = [self.mapping.get(t, [0.7, 0.7]) for t in texts]
        return SimpleNamespace(embeddings=vectors)


FAQ_HIT = FAQEntry(
    faq_id="returns_policy",
    question="退货政策是什么",
    answer="支持 7 天无理由退货。",
)
FAQ_OTHER = FAQEntry(
    faq_id="refund_timeline",
    question="退款多久到账",
    answer="1-3 个工作日原路退回。",
)
EMBED_MAP = {
    FAQ_HIT.question: [1.0, 0.0],
    FAQ_OTHER.question: [0.0, 1.0],
}


def _faq_service(threshold: float = 0.8) -> FAQService:
    return FAQService(FakeEmbeddings(EMBED_MAP), [FAQ_HIT, FAQ_OTHER], threshold)


def _make_factory(faq_service=None, guardrail_service=None) -> NodeFactory:
    intent_detector = Mock()
    intent_detector.detect_with_confidence = AsyncMock()
    return NodeFactory(
        intent_detector=intent_detector,
        slot_filler=Mock(),
        tool_registry=Mock(),
        retrieval_pipeline={},
        llm_service=None,
        guardrail_service=guardrail_service,
        graph_retrieval_service=None,
        handoff_service=None,
        faq_service=faq_service,
    )


def _guardrail(blocked: bool = False, sanitized: str | None = None) -> Mock:
    guardrail = Mock()
    guardrail.check_output = Mock(
        return_value=SimpleNamespace(was_blocked=blocked, sanitized_content=sanitized)
    )
    return guardrail


# ── Node-level behavior ────────────────────────────────────────────────────


class TestFaqLookupNode:
    async def test_hit_serves_curated_answer_with_provenance(self):
        factory = _make_factory(faq_service=_faq_service())

        updates = await factory.faq_lookup_node({"message": "退货政策是什么"})

        assert updates["route_after_faq"] == "hit"
        assert updates["response"] == FAQ_HIT.answer
        assert updates["sources"] == ["faq:returns_policy"]

    async def test_miss_continues_to_rag(self):
        factory = _make_factory(faq_service=_faq_service())

        updates = await factory.faq_lookup_node({"message": "这款手机的芯片型号是什么"})

        assert updates == {"route_after_faq": "miss"}

    async def test_unwired_service_misses(self):
        factory = _make_factory(faq_service=None)
        assert await factory.faq_lookup_node({"message": "退货政策是什么"}) == {
            "route_after_faq": "miss"
        }

    async def test_service_failure_misses(self):
        broken = Mock()
        broken.match = AsyncMock(side_effect=RuntimeError("boom"))
        factory = _make_factory(faq_service=broken)

        updates = await factory.faq_lookup_node({"message": "退货政策是什么"})
        assert updates == {"route_after_faq": "miss"}

    async def test_blocked_output_replaced(self):
        factory = _make_factory(
            faq_service=_faq_service(), guardrail_service=_guardrail(blocked=True)
        )

        updates = await factory.faq_lookup_node({"message": "退货政策是什么"})
        assert "安全检查" in updates["response"]

    async def test_sanitized_output_replaced(self):
        factory = _make_factory(
            faq_service=_faq_service(), guardrail_service=_guardrail(sanitized="脱敏后的答案")
        )

        updates = await factory.faq_lookup_node({"message": "退货政策是什么"})
        assert updates["response"] == "脱敏后的答案"

    def test_route_after_faq(self):
        assert NodeFactory.route_after_faq({"route_after_faq": "hit"}) == "hit"
        assert NodeFactory.route_after_faq({}) == "miss"


# ── Full graph: FAQ hit skips retrieval + LLM; miss runs RAG ───────────────


def _build_graph(faq_service, hybrid_search):
    from app.services.dialogue.graph import build_dialogue_graph

    detector = Mock()
    detector.detect_with_confidence = AsyncMock(
        return_value=Mock(intent=Mock(value="faq"), confidence=0.99)
    )
    llm = Mock()
    llm.generate = AsyncMock(return_value=SimpleNamespace(content="RAG 生成的回答"))
    return build_dialogue_graph(
        intent_detector=detector,
        slot_filler=None,
        tool_registry=create_default_tool_registry(),
        retrieval_pipeline={"hybrid_search": hybrid_search},
        llm_service=llm,
        faq_service=faq_service,
    )


class TestGraphFaqFlow:
    async def test_hit_never_touches_retrieval_or_llm(self):
        hybrid = Mock()
        hybrid.search = AsyncMock(return_value=[])
        graph = _build_graph(_faq_service(), hybrid)

        turn = await graph.ainvoke(
            {"message": "退货政策是什么", "session_id": 1, "user_id": 1},
            {"configurable": {"thread_id": "faq-hit-1"}},
        )

        assert turn["response"] == FAQ_HIT.answer
        assert turn["sources"] == ["faq:returns_policy"]
        hybrid.search.assert_not_awaited()

    async def test_miss_flows_into_rag(self):
        hybrid = Mock()
        hybrid.search = AsyncMock(return_value=[])
        graph = _build_graph(_faq_service(), hybrid)

        turn = await graph.ainvoke(
            {"message": "这款手机的芯片型号是什么", "session_id": 1, "user_id": 1},
            {"configurable": {"thread_id": "faq-miss-1"}},
        )

        hybrid.search.assert_awaited_once()
        assert turn["response"] == "RAG 生成的回答"
