"""Human handoff: intent priority, emotion escalation, and ticket flow.

Industry baseline (Zendesk / Intercom / Sierra): a customer who asks
for a human gets one; a customer showing repeated frustration gets one
proactively; and the agent lands mid-conversation with full context —
never “您好，请问有什么可以帮您”.
"""

import asyncio
from unittest.mock import AsyncMock, Mock

from app.models.enums.intent import Intent
from app.services.dialogue.nodes import NodeFactory
from app.services.dialogue.state import DialogueState
from app.services.intent.base import IntentResult


def _make_factory(handoff_service=None) -> NodeFactory:
    """NodeFactory with stub services and an optional handoff service."""
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
        handoff_service=handoff_service,
    )


def _detector_returning(intent: Intent, confidence: float = 0.9) -> Mock:
    detector = Mock()
    detector.detect_with_confidence = AsyncMock(
        return_value=IntentResult(intent=intent, confidence=confidence)
    )
    return detector


def _config_with_queue() -> tuple[dict, asyncio.Queue]:
    queue: asyncio.Queue = asyncio.Queue()
    return {"configurable": {"stream_queue": queue}}, queue


# ── Intent priority ────────────────────────────────────────────────────────


class TestHandoffIntentPriority:
    async def test_explicit_handoff_beats_task_resume(self):
        """“转人工” mid-refund must reach the human, not another slot prompt."""
        factory = _make_factory()
        factory._intent_detector = _detector_returning(Intent.HANDOFF)

        state: DialogueState = {
            "message": "算了，给我转人工",
            "intent": "refund",
            "prev_intent": "refund",
            "filled_slots": {"order_id": "ORD1001"},
            "pending_slots": ["reason"],
        }
        updates = await factory.detect_intent_node(state)

        assert updates["intent"] == "handoff"
        assert updates["handoff_reason"] == "explicit"

    async def test_emotion_escalation_beats_task_resume(self):
        """An angry user mid-task escalates even without handoff keywords."""
        factory = _make_factory()
        factory._intent_detector = _detector_returning(Intent.REFUND)

        state: DialogueState = {
            "message": "退款退了三次了还是没到账，气死我了",
            "prev_intent": "refund",
            "filled_slots": {"order_id": "ORD1001"},
            "pending_slots": ["reason"],
        }
        updates = await factory.detect_intent_node(state)

        assert updates["intent"] == "handoff"
        assert updates["handoff_reason"] == "emotion"

    async def test_cancel_still_wins_over_emotion(self):
        """“算了，太失望了” is the user leaving, not asking for a human."""
        factory = _make_factory()
        factory._intent_detector = _detector_returning(Intent.CANCEL)

        state: DialogueState = {"message": "算了，太失望了"}
        updates = await factory.detect_intent_node(state)

        assert updates["intent"] == "cancel"

    async def test_benign_question_does_not_escalate(self):
        """“第三次下单有优惠吗” is a question, not frustration."""
        factory = _make_factory()
        factory._intent_detector = _detector_returning(Intent.FAQ)

        state: DialogueState = {"message": "第三次下单有优惠吗"}
        updates = await factory.detect_intent_node(state)

        assert updates["intent"] == "faq"

    async def test_no_emotion_keywords_keeps_detected_intent(self):
        factory = _make_factory()
        factory._intent_detector = _detector_returning(Intent.QUERY_ORDER)

        state: DialogueState = {"message": "帮我查一下订单 ORD1001 的物流"}
        updates = await factory.detect_intent_node(state)

        assert updates["intent"] == "query_order"
        assert "handoff_reason" not in updates


# ── Routing ────────────────────────────────────────────────────────────────


class TestHandoffRouting:
    async def test_handoff_intent_routes_to_handoff(self):
        factory = _make_factory()
        state: DialogueState = {"intent": "handoff"}
        assert (await factory.route_intent_node(state))["route"] == "handoff"

    async def test_route_by_intent_returns_handoff_route(self):
        factory = _make_factory()
        state: DialogueState = {"intent": "handoff", "route": "handoff"}
        assert factory.route_by_intent(state) == "handoff"

    @staticmethod
    def test_should_skip_intent_sends_handoff_to_full_detection():
        """Mid-slot-collection “转人工” must not be captured as a slot value."""
        state: DialogueState = {
            "intent": "refund",
            "pending_slots": ["order_id"],
            "message": "我要转人工",
        }
        assert NodeFactory.should_skip_intent(state) == "full"

    @staticmethod
    def test_should_skip_intent_still_skips_ordinary_answers():
        state: DialogueState = {
            "intent": "refund",
            "pending_slots": ["order_id"],
            "message": "ORD10086",
        }
        assert NodeFactory.should_skip_intent(state) == "skip"


# ── handle_handoff_node ────────────────────────────────────────────────────


class TestHandleHandoffNode:
    async def test_creates_ticket_with_context_and_responds(self):
        handoff = Mock()
        handoff.create_ticket_for_session = AsyncMock(
            return_value={"ticket_id": 42, "queue_position": 3, "reused": False}
        )
        factory = _make_factory(handoff_service=handoff)
        config, queue = _config_with_queue()

        state: DialogueState = {
            "message": "给我转人工",
            "session_id": 7,
            "user_id": 11,
            "prev_intent": "refund",
            "filled_slots": {"order_id": "ORD1001"},
            "pending_slots": [],
            "handoff_reason": "explicit",
        }
        updates = await factory.handle_handoff_node(state, config)

        handoff.create_ticket_for_session.assert_awaited_once()
        call = handoff.create_ticket_for_session.await_args
        assert call.kwargs["session_id"] == 7
        assert call.kwargs["user_id"] == 11
        assert call.kwargs["reason"] == "explicit"
        assert call.kwargs["context"]["filled_slots"] == {"order_id": "ORD1001"}

        assert updates["handoff_ticket_id"] == 42
        assert "工单号 #42" in updates["response"]
        assert "排队人数：3 人" in updates["response"]
        assert updates["intent"] == "handoff"
        # Template response reaches the stream queue (not LLM-streamed).
        assert queue.get_nowait() == updates["response"]

    async def test_discards_staged_irreversible_action(self):
        """A staged refund must not ambush a later turn after handoff."""
        handoff = Mock()
        handoff.create_ticket_for_session = AsyncMock(
            return_value={"ticket_id": 42, "queue_position": 3, "reused": False}
        )
        factory = _make_factory(handoff_service=handoff)

        state: DialogueState = {
            "message": "转人工",
            "session_id": 7,
            "pending_confirmation": {"intent": "refund", "args": {"order_id": "X"}},
        }
        updates = await factory.handle_handoff_node(state)

        assert updates["pending_confirmation"] is None

    async def test_emotion_reason_acknowledges_feelings(self):
        handoff = Mock()
        handoff.create_ticket_for_session = AsyncMock(
            return_value={"ticket_id": 43, "queue_position": 1, "reused": False}
        )
        factory = _make_factory(handoff_service=handoff)

        state: DialogueState = {
            "message": "还是没收到退款，太失望了",
            "session_id": 7,
            "handoff_reason": "emotion",
        }
        updates = await factory.handle_handoff_node(state)

        assert "抱歉" in updates["response"]

    async def test_ticket_failure_still_responds(self):
        """DB down must not block the user from reaching a human."""
        handoff = Mock()
        handoff.create_ticket_for_session = AsyncMock(
            return_value={"ticket_id": None, "queue_position": None, "reused": False}
        )
        factory = _make_factory(handoff_service=handoff)

        state: DialogueState = {"message": "转人工", "session_id": 7}
        updates = await factory.handle_handoff_node(state)

        assert "转接人工客服" in updates["response"]
        assert "handoff_ticket_id" not in updates

    async def test_no_service_still_responds(self):
        """Handoff without a service wired (None) still acknowledges."""
        factory = _make_factory(handoff_service=None)
        state: DialogueState = {"message": "转人工", "session_id": 7}
        updates = await factory.handle_handoff_node(state)
        assert "转接人工客服" in updates["response"]

    async def test_reused_ticket_mentions_queue_membership(self):
        handoff = Mock()
        handoff.create_ticket_for_session = AsyncMock(
            return_value={"ticket_id": 42, "queue_position": None, "reused": True}
        )
        factory = _make_factory(handoff_service=handoff)

        state: DialogueState = {"message": "怎么还没人理我", "session_id": 7}
        updates = await factory.handle_handoff_node(state)

        assert "排队中" in updates["response"]
        assert "排队人数" not in updates["response"]


# ── Rule-based detector integration ────────────────────────────────────────


class TestRuleBasedHandoffDetection:
    def test_transfer_keyword_detected(self):
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()
        result = detector.detect_with_confidence("我要转人工")
        assert result.intent == Intent.HANDOFF

    def test_human_agent_english_detected(self):
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()
        result = detector.detect_with_confidence("let me talk to a human agent")
        assert result.intent == Intent.HANDOFF

    def test_handoff_outranks_co_occurring_complaint(self):
        """“投诉没用，给我转人工” must classify as handoff, not complaint."""
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()
        result = detector.detect_with_confidence("投诉没用，给我转人工")
        assert result.intent == Intent.HANDOFF


class TestHandoffContextIncludesBotAttempts:
    async def test_ticket_context_carries_what_bot_tried(self):
        """Industry context transfer: the human agent sees which tools
        the bot already ran and the last result — the user must not
        repeat the story."""
        handoff = Mock()
        handoff.create_ticket_for_session = AsyncMock(
            return_value={"ticket_id": 9, "queue_position": 1, "reused": False}
        )
        factory = _make_factory(handoff_service=handoff)
        config, _queue = _config_with_queue()
        trace = [
            {
                "tool": "query_order_status",
                "ok": True,
                "args": {"order_id": "ORD1001"},
                "summary": "...",
            }
        ]
        state: DialogueState = {
            "message": "算了，转人工",
            "session_id": 3,
            "user_id": 1,
            "handoff_reason": "explicit",
            "executed_tools": trace,
            "tool_result": {"order_id": "ORD1001", "status": "shipped"},
        }

        await factory.handle_handoff_node(state, config)

        context = handoff.create_ticket_for_session.await_args.kwargs["context"]
        assert context["bot_executed_tools"] == trace
        assert context["last_tool_result"] == {"order_id": "ORD1001", "status": "shipped"}

    async def test_ticket_context_defaults_when_bot_ran_nothing(self):
        handoff = Mock()
        handoff.create_ticket_for_session = AsyncMock(
            return_value={"ticket_id": 10, "queue_position": 1, "reused": False}
        )
        factory = _make_factory(handoff_service=handoff)
        config, _queue = _config_with_queue()

        state: DialogueState = {
            "message": "转人工",
            "session_id": 4,
            "user_id": 1,
            "handoff_reason": "explicit",
        }
        await factory.handle_handoff_node(state, config)

        context = handoff.create_ticket_for_session.await_args.kwargs["context"]
        assert context["bot_executed_tools"] == []
        assert context["last_tool_result"] is None
