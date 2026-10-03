"""Tests for dialogue flow patterns found during E2E testing.

Covers three bug patterns:
1. State not resetting after tool execution
2. Skip-intent too aggressive (cancel/chitchat not detected)
3. Nonsense text assigned to slots via fallback

Bug 1 was the review's named fake test (review 2026-09-26, #10): the
old version hand-built a DialogueState dict and asserted the literals
it had just written into it — it documented the bug without running
any implementation. It now runs the compiled graph across two turns.
The ChatService-level composition (both transports, cross-session) is
pinned in app/tests/integration/test_chat_pipeline_real_graph.py.
"""

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver

from app.services.dialogue.graph import build_dialogue_graph
from app.services.dialogue.nodes import NodeFactory
from app.services.dialogue.state import DialogueState
from app.services.dialogue.tools import create_default_tool_registry
from app.services.guardrails.base import GuardrailService
from app.services.guardrails.input_guard import DefaultInputGuardrail
from app.services.intent.rule_based import RuleBasedIntentDetector
from app.services.llm.base import LLMMessage, LLMResponse, LLMServiceBase


def _make_factory() -> NodeFactory:
    """Create a NodeFactory with minimal mock services."""
    intent_detector = Mock()
    intent_detector.detect_with_confidence = AsyncMock()
    return NodeFactory(
        intent_detector=intent_detector,
        slot_filler=Mock(),
        tool_registry=Mock(),
        retrieval_pipeline={},
        llm_service=None,
        guardrail_service=None,
        graph_retrieval_service=None,
    )


# ── Bug 1: State not resetting after tool execution ──────────────────────


class _FlowLLM(LLMServiceBase):
    """Records generation prompts; answers by the context marker the
    real prompt scaffolding emits (tool result vs reference material)."""

    def __init__(self) -> None:
        super().__init__(api_key="test", model="test")
        self.prompts: list[str] = []

    def _answer(self, messages: list[LLMMessage]) -> str:
        prompt = messages[-1].content
        self.prompts.append(prompt)
        if "工具执行结果:" in prompt:
            return "这是本次订单操作的结果。"
        if "参考资料:" in prompt:
            return "支持七天无理由退货。"
        return "您好，有什么可以帮您？"

    async def generate(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        return LLMResponse(content=self._answer(messages), model=self.model)

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        yield self._answer(messages)

    def estimate_tokens(self, text: str) -> int:
        return len(text)

    async def count_tokens(self, messages: list[LLMMessage]) -> int:
        return sum(len(m.content) for m in messages)


class _PolicyHit:
    """SearchResult-shaped object as hybrid_search returns."""

    def __init__(self) -> None:
        self.document_id = "returns_policy_zh"
        self.content = "退货政策：七天无理由退货。"
        self.score = 0.9
        self.metadata = None


class TestStateResetAfterToolExecution:
    """After a tool executes, the next turn must start clean.

    Bug: the checkpoint preserved tool_result/filled_slots from a
    completed task, so the next unrelated question generated from the
    stale tool result. begin_turn resets turn-scoped fields before
    routing (71a1623); failure here is a regression of review #3.
    """

    async def test_next_turn_after_tool_starts_clean(self):
        """Two real turns through the compiled graph: a tool turn, then
        an unrelated policy question on the same checkpointed thread."""
        llm = _FlowLLM()

        def _hybrid() -> Mock:
            hybrid = Mock()
            hybrid.search = AsyncMock(return_value=[_PolicyHit()])
            return hybrid

        graph = build_dialogue_graph(
            intent_detector=RuleBasedIntentDetector(),
            slot_filler=None,
            tool_registry=create_default_tool_registry(),
            retrieval_pipeline={"hybrid_search": _hybrid()},
            llm_service=llm,
            guardrail_service=GuardrailService(input_guard=DefaultInputGuardrail()),
            checkpointer=MemorySaver(),
        )
        config: RunnableConfig = {"configurable": {"thread_id": "flow-reset"}}

        first = await graph.ainvoke(
            {"message": "查询订单 ORD1001", "session_id": 1, "user_id": 1}, config
        )
        assert first["tool_result"]  # sanity: turn 1 really executed the tool
        assert any("工具执行结果:" in p for p in llm.prompts)

        second = await graph.ainvoke(
            {"message": "退货政策是什么", "session_id": 1, "user_id": 1}, config
        )

        # Turn 2 generated from its own retrieval, not the stale tool result.
        assert "参考资料:" in llm.prompts[-1]
        assert "工具执行结果:" not in llm.prompts[-1]
        assert "七天无理由" in second["response"]

        # Checkpointed thread carries no turn-scoped residue.
        snapshot = await graph.aget_state(config)
        assert snapshot.values["executed_tools"] == []
        assert not snapshot.values["tool_result"]


# ── Bug 2: Skip-intent too aggressive ────────────────────────────────────


class TestSkipIntentAggressive:
    """should_skip_intent must not block cancel/chitchat detection."""

    @pytest.mark.parametrize(
        "message",
        [
            "我不了",
            "算了",
            "不要了",
            "不想退了",
        ],
    )
    def test_cancel_phrases_not_skipped(self, message):
        """Cancel-like phrases must go through full intent detection."""
        factory = _make_factory()
        state: DialogueState = {
            "message": message,
            "intent": "refund",
            "filled_slots": {"order_id": "12345"},
            "pending_slots": ["reason"],
        }
        result = factory.should_skip_intent(state)
        assert result == "full", f"'{message}' should go through full detection"

    @pytest.mark.parametrize(
        "message",
        [
            "今天天气怎么样",
            "你好",
            "帮我查订单",
            "我要退货",
        ],
    )
    def test_non_slot_messages_not_skipped(self, message):
        """Messages that aren't slot answers must go through full detection."""
        factory = _make_factory()
        state: DialogueState = {
            "message": message,
            "intent": "refund",
            "filled_slots": {"order_id": "12345"},
            "pending_slots": ["reason"],
        }
        result = factory.should_skip_intent(state)
        assert result == "full", f"'{message}' should go through full detection"

    def test_short_answer_is_skipped(self):
        """A short, direct answer to a slot prompt should be skipped."""
        factory = _make_factory()
        state: DialogueState = {
            "message": "质量问题",
            "intent": "refund",
            "filled_slots": {"order_id": "12345"},
            "pending_slots": ["reason"],
        }
        result = factory.should_skip_intent(state)
        assert result == "skip"

    @pytest.mark.parametrize(
        "message",
        [
            "投诉ne",
            "我要投诉",
            "我说我要投诉",
        ],
    )
    def test_complaint_switch_not_skipped(self, message):
        """Switching to complaint during refund must go through full detection."""
        factory = _make_factory()
        state: DialogueState = {
            "message": message,
            "intent": "refund",
            "filled_slots": {},
            "pending_slots": ["order_id", "reason"],
        }
        result = factory.should_skip_intent(state)
        assert result == "full", f"'{message}' should trigger full intent detection"


# ── Bug 3: Nonsense text assigned to slots ───────────────────────────────


class TestNonsenseFallback:
    """The slot fallback should not assign garbage text to slots."""

    @pytest.mark.parametrize(
        "message",
        [
            "让他物业和婉婷宏伟人宏伟、",
            "个人个文玮个",
            "啊啊啊啊啊啊",
            "123456789012345678901234567890",
        ],
    )
    def test_nonsense_not_assigned_to_order_id(self, message):
        """Nonsense/typed-garbage should not become an order_id."""
        from app.services.slot_filling.slot_types import extract_slots_from_message

        merged = extract_slots_from_message("refund", message, {})
        # order_id pattern requires "订单号" prefix or "order" keyword
        assert "order_id" not in merged, f"'{message}' should not match order_id"

    def test_fallback_only_for_reasonable_text(self):
        """Fallback assignment should filter out garbage."""
        # This tests the expected behavior of collect_slots_node fallback.
        # Currently, any text < 30 chars gets assigned.
        # This test documents what SHOULD change.
        from app.services.slot_filling.slot_types import extract_slots_from_message

        # Good: "质量问题" is a valid reason
        good = extract_slots_from_message("refund", "质量问题", {})
        # This SHOULD extract reason via pattern, and it does
        assert good.get("reason") == "质量问题"

    @pytest.mark.parametrize(
        "message",
        [
            "订单号12345",
            "订单12345",
            "order12345",
        ],
    )
    def test_valid_order_id_extracted(self, message):
        """Valid order IDs with prefix should be extracted by regex."""
        from app.services.slot_filling.slot_types import extract_slots_from_message

        merged = extract_slots_from_message("refund", message, {})
        assert "order_id" in merged


class TestOrderIdExtractionRobustness:
    """Finding ① audit (2026-09-30): the shared order-id pattern
    "(?:order|订单)\\s*([A-Za-z0-9]{3,})" is the REQUIRED order_id slot
    on refund/return/query_order/track_shipping — under an extractor
    outage it both misses real codes and invents fake ones.

    - "…my ORD1001 order arrived cracked" captures the word after
      "order" ("arrived") as the order id;
    - "ORD1001订单…" (code directly glued to CJK) extracts nothing,
      because the value charset stops at the first CJK char.
    """

    @pytest.mark.parametrize(
        "intent", ["refund", "return", "query_order", "track_shipping", "complaint"]
    )
    def test_english_narrative_extracts_the_code_not_the_verb(self, intent: str):
        from app.services.slot_filling.slot_types import extract_slots_from_message

        merged = extract_slots_from_message(
            intent,
            "the screen of my ORD1001 order arrived cracked",
            {},
        )

        assert merged.get("order_id") == "ORD1001"

    @pytest.mark.parametrize(
        "intent", ["refund", "return", "query_order", "track_shipping", "complaint"]
    )
    def test_cjk_adjacent_code_extracts(self, intent: str):
        from app.services.slot_filling.slot_types import extract_slots_from_message

        merged = extract_slots_from_message(intent, "ORD1001订单的包裹还没到", {})

        assert merged.get("order_id") == "ORD1001"

    def test_bare_digit_run_still_rejected(self):
        """A letterless digit run is typed garbage, not an order code
        (order codes carry letters; digit-only ids come with the
        订单号/order prefix)."""
        from app.services.slot_filling.slot_types import extract_slots_from_message

        for intent in ("refund", "complaint"):
            merged = extract_slots_from_message(intent, "123456789012345678901", {})
            assert "order_id" not in merged, intent
