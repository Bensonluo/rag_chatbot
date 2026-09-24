"""Funnel-layer telemetry contract: every serving layer is countable.

The funnel inversion makes deep-layer traffic (agent tool loops) the
product's real customer-service needs, and one-shot resolution its
highest-weight metric — but neither is optimizable while the funnel
itself is invisible. These tests pin that each routing path records
its layer, so the distribution and the resolution proxy (served vs
handoff turns) stay computable in one PromQL query.
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from prometheus_client import REGISTRY

from app.services.dialogue.funnel_metrics import (
    LAYER_AGENT_TOOL,
    LAYER_FAQ,
    LAYER_HANDOFF,
    LAYER_L0_CACHE,
    LAYER_RAG,
)
from app.services.dialogue.nodes import NodeFactory


def _layer_value(layer: str) -> float:
    value = REGISTRY.get_sample_value("chat_funnel_layers_total", {"layer": layer})
    return value if value is not None else 0.0


@pytest.fixture(autouse=True)
def _claim_gate_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """The claim gate has its own test file; keep this one about counting."""
    from app.config.settings import settings

    monkeypatch.setattr(settings, "FACT_CLAIM_CHECK_ENABLED", False)


def _make_factory(**overrides: Any) -> NodeFactory:
    defaults: dict[str, Any] = {
        "intent_detector": Mock(),
        "slot_filler": None,
        "tool_registry": Mock(),
        "guardrail_service": None,
    }
    defaults.update(overrides)
    return NodeFactory(**defaults)


class _FakeCache:
    def __init__(self, answer: Any) -> None:
        self.answer = answer

    async def get(self, message: str) -> Any:
        return self.answer


def _cached_answer() -> Any:
    from app.services.chat.answer_cache import CachedAnswer

    return CachedAnswer(
        response="签收后7天内可申请无理由退货。",
        sources=["doc-refund"],
        intent="question",
    )


class TestFunnelLayerCounting:
    async def test_l0_cache_hit_counts_l0_layer(self):
        factory = _make_factory(answer_cache=_FakeCache(_cached_answer()))

        await factory.answer_cache_lookup_node({"message": "退货政策"})

        before = _layer_value(LAYER_L0_CACHE)
        await factory.answer_cache_lookup_node({"message": "退货政策"})
        assert _layer_value(LAYER_L0_CACHE) == before + 1.0

    async def test_faq_hit_counts_faq_layer(self):
        faq = Mock()
        faq.match = AsyncMock(
            return_value=SimpleNamespace(faq_id="f1", answer="签收后支持7天无理由。")
        )
        factory = _make_factory(faq_service=faq)

        before = _layer_value(LAYER_FAQ)
        await factory.faq_lookup_node({"message": "能退吗"})
        assert _layer_value(LAYER_FAQ) == before + 1.0

    async def test_faq_miss_touches_no_layer(self):
        faq = Mock()
        faq.match = AsyncMock(return_value=None)
        factory = _make_factory(faq_service=faq)

        before = _layer_value(LAYER_FAQ)
        updates = await factory.faq_lookup_node({"message": "能退吗"})
        assert updates["route_after_faq"] == "miss"
        assert _layer_value(LAYER_FAQ) == before

    async def test_rag_with_docs_counts_rag_layer(self):
        result = SimpleNamespace(document_id="d1", content="政策", score=0.9)
        hybrid = Mock()
        hybrid.search = AsyncMock(return_value=[result])
        factory = _make_factory(retrieval_pipeline={"hybrid_search": hybrid})

        before = _layer_value(LAYER_RAG)
        state = await factory.rag_lookup_node({"message": "退款政策", "intent": "question"})
        assert state["retrieved_docs"]
        assert _layer_value(LAYER_RAG) == before + 1.0

    async def test_rag_with_zero_docs_touches_no_layer(self):
        hybrid = Mock()
        hybrid.search = AsyncMock(return_value=[])
        factory = _make_factory(retrieval_pipeline={"hybrid_search": hybrid})

        before = _layer_value(LAYER_RAG)
        await factory.rag_lookup_node({"message": "退款政策", "intent": "question"})
        assert _layer_value(LAYER_RAG) == before

    async def test_tool_success_counts_agent_tool_layer(self):
        registry = Mock()
        registry.get_tool_for_intent = Mock(return_value=None)
        registry.execute = AsyncMock(
            return_value=SimpleNamespace(success=True, data={"order_id": "A1"}, message="")
        )
        factory = _make_factory(tool_registry=registry)

        before = _layer_value(LAYER_AGENT_TOOL)
        updates = await factory.execute_tool_node(
            {"message": "查订单", "intent": "order_lookup", "filled_slots": {"order_id": "A1"}}
        )
        assert updates["tool_result"] == {"order_id": "A1"}
        assert _layer_value(LAYER_AGENT_TOOL) == before + 1.0

    async def test_handoff_counts_handoff_layer(self):
        factory = _make_factory()

        before = _layer_value(LAYER_HANDOFF)
        updates = await factory.handle_handoff_node({"message": "转人工", "session_id": 1})
        assert updates["intent"] == "handoff"
        assert _layer_value(LAYER_HANDOFF) == before + 1.0
