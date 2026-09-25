"""Task terminality: an executed task is never suspended or resumed.

Doctrine (double-execution risk): once a task's tool has actually run —
a refund executed via confirm, an order query run directly, an agent
loop firing a tool — the task is terminal. It must not be pushed onto
the intent-switch stack and must never be auto-resumed by an
unrecognized message: resurrecting an executed irreversible action
risks a second refund the user never asked for.

Companion contract (no silent hijack): an "unknown" classification is
never a resume signal. Unrecognized input is answered on the direct
tier with a visible hint about the suspended task; resume happens only
on an affirmative ("确认" / "继续") or by restating the task.
"""

from typing import Any
from unittest.mock import AsyncMock, Mock

from app.models.enums.intent import Intent
from app.services.agent import AgentResult
from app.services.dialogue.nodes import NodeFactory
from app.services.dialogue.state import DialogueState
from app.services.dialogue.tools import create_default_tool_registry
from app.services.intent.base import IntentResult


def _make_factory(**overrides: Any) -> NodeFactory:
    """NodeFactory with stub services; task-relevant services injectable."""
    intent_detector = Mock()
    intent_detector.detect_with_confidence = AsyncMock()
    kwargs: dict[str, Any] = {
        "intent_detector": intent_detector,
        "slot_filler": Mock(),
        "tool_registry": create_default_tool_registry(),
        "retrieval_pipeline": {},
        "llm_service": None,
        "guardrail_service": None,
        "graph_retrieval_service": None,
    }
    kwargs.update(overrides)
    return NodeFactory(**kwargs)


def _detector_returning(intent: Intent, confidence: float = 0.9) -> Mock:
    detector = Mock()
    detector.detect_with_confidence = AsyncMock(
        return_value=IntentResult(intent=intent, confidence=confidence)
    )
    return detector


# ── The terminal flag: set wherever a tool actually runs ──────────────────


class TestExecutedFlagProducers:
    async def test_confirm_executing_staged_action_marks_task_executed(self):
        factory = _make_factory()

        state: DialogueState = {
            "intent": "confirm",
            "message": "确认",
            "user_id": 1,
            "pending_confirmation": {
                "intent": "refund",
                "args": {"order_id": "ORD1001", "reason": "质量问题"},
            },
        }
        updates = await factory._handle_meta_intent(state)

        assert updates["task_executed"] is True

    async def test_confirm_failure_still_marks_task_executed(self):
        """The tool ran and was refused (unknown order) — still terminal:
        re-running it can only produce the same failure or worse."""
        factory = _make_factory()

        state: DialogueState = {
            "intent": "confirm",
            "message": "确认",
            "user_id": 1,
            "pending_confirmation": {
                "intent": "refund",
                "args": {"order_id": "ORD9999", "reason": "质量问题"},
            },
        }
        updates = await factory._handle_meta_intent(state)

        assert "error" in updates["tool_result"]
        assert updates["task_executed"] is True

    async def test_direct_tool_run_marks_task_executed(self):
        factory = _make_factory()

        state: DialogueState = {
            "intent": "query_order",
            "filled_slots": {"order_id": "ORD1001"},
            "pending_slots": [],
            "user_id": 1,
        }
        updates = await factory.execute_tool_node(state)

        assert updates["tool_result"]["status"] == "已发货"
        assert updates["task_executed"] is True

    async def test_agent_tool_run_marks_task_executed(self):
        agent = Mock()
        agent.run = AsyncMock(
            return_value=AgentResult(
                response="已为您查询订单。",
                tool_trace=[{"tool": "query_order_status", "ok": True}],
            )
        )
        factory = _make_factory(agent_service=agent)

        updates = await factory.handle_agent_node({"message": "查订单", "user_id": 1})

        assert updates["task_executed"] is True

    async def test_agent_run_without_tools_does_not_mark_executed(self):
        agent = Mock()
        agent.run = AsyncMock(return_value=AgentResult(response="好的"))
        factory = _make_factory(agent_service=agent)

        updates = await factory.handle_agent_node({"message": "退款怎么办", "user_id": 1})

        assert not updates.get("task_executed")

    async def test_staging_reopens_the_task(self):
        """Staging (not executing) a new irreversible action means the
        task is back in flight — it must be suspendable/resumable again."""
        factory = _make_factory()

        state: DialogueState = {
            "intent": "refund",
            "filled_slots": {"order_id": "ORD1001", "reason": "质量问题"},
            "pending_slots": [],
            "task_executed": True,
            "user_id": 1,
        }
        updates = await factory.execute_tool_node(state)

        assert updates["pending_confirmation"]["args"]["order_id"] == "ORD1001"
        assert updates["task_executed"] is False


# ── handle_switch: executed tasks are not suspended ───────────────────────


class TestExecutedTaskNotSuspended:
    async def test_switch_after_execution_does_not_push(self):
        """The live-bug shape: refund executed (failed lookup), user asks
        for a human — the executed task must not land on the stack."""
        factory = _make_factory()

        state: DialogueState = {
            "intent": "handoff",
            "prev_intent": "refund",
            "filled_slots": {"order_id": "1111", "reason": "不好吃"},
            "pending_slots": [],
            "task_executed": True,
        }
        updates = await factory.handle_switch_node(state)

        assert "state_stack" not in updates
        assert updates["filled_slots"] == {}
        assert updates["pending_confirmation"] is None
        assert updates["task_executed"] is False

    async def test_switch_still_suspends_inflight_task(self):
        """A task that has NOT executed (mid-collection) is still pushed,
        so the documented switch/resume feature keeps working."""
        factory = _make_factory()

        state: DialogueState = {
            "intent": "return",
            "prev_intent": "refund",
            "filled_slots": {"order_id": "ORD1001"},
            "pending_slots": ["reason"],
        }
        updates = await factory.handle_switch_node(state)

        stack = updates["state_stack"]
        assert len(stack) == 1
        assert stack[0]["intent"] == "refund"
        assert stack[0]["filled_slots"] == {"order_id": "ORD1001"}

    async def test_suspending_a_task_discards_its_staged_action(self):
        """A staged (not yet confirmed) action belongs to the turn it was
        staged in. Suspend the task → discard the staging: a later bare
        "好的" must never execute a refund the user abandoned. Resuming
        the task re-stages and re-asks, which is the safe direction."""
        factory = _make_factory()

        state: DialogueState = {
            "intent": "return",
            "prev_intent": "refund",
            "filled_slots": {"order_id": "ORD1001", "reason": "质量问题"},
            "pending_slots": [],
            "pending_confirmation": {"intent": "refund", "args": {"order_id": "ORD1001"}},
        }
        updates = await factory.handle_switch_node(state)

        assert updates["pending_confirmation"] is None
        assert updates["state_stack"][0]["intent"] == "refund"


# ── Intent detection: unknown never resumes ───────────────────────────────


class TestUnknownNeverResumes:
    async def test_unknown_after_suspension_is_answered_not_resumed(self):
        factory = _make_factory()
        factory._intent_detector = _detector_returning(Intent.UNKNOWN)

        state: DialogueState = {
            "message": "今天天气如何",
            "prev_intent": "handoff",
            "state_stack": [
                {"intent": "refund", "filled_slots": {"order_id": "1111"}, "pending_slots": []}
            ],
        }
        updates = await factory.detect_intent_node(state)

        assert updates["intent"] == "unknown"

    async def test_affirmative_still_resumes_suspended_task(self):
        """ "继续" (confirm) remains the documented way back into a
        suspended task — B must not break the resume feature."""
        factory = _make_factory()
        factory._intent_detector = _detector_returning(Intent.CONFIRM)

        state: DialogueState = {
            "message": "继续",
            "prev_intent": "handoff",
            "state_stack": [
                {"intent": "refund", "filled_slots": {"order_id": "1111"}, "pending_slots": []}
            ],
        }
        updates = await factory.detect_intent_node(state)

        assert updates["intent"] == "refund"

    async def test_unknown_mid_collection_is_still_a_slot_answer(self):
        """Inside an active slot collection, unknown short input is the
        user answering the prompt — unchanged by this contract."""
        factory = _make_factory()
        factory._intent_detector = _detector_returning(Intent.UNKNOWN)

        # At detection time the checkpoint's "intent" still holds the
        # previous turn's classification — that is what the node reads
        # as the in-flight task.
        state: DialogueState = {
            "message": "不好吃",
            "intent": "refund",
            "filled_slots": {"order_id": "1111"},
            "pending_slots": ["reason"],
        }
        updates = await factory.detect_intent_node(state)

        assert updates["intent"] == "refund"


# ── Direct tier: surface the suspended task instead of hijacking ─────────


class TestDirectAnswerShowsResumeHint:
    async def test_direct_response_appends_resume_hint(self):
        factory = _make_factory()

        state: DialogueState = {
            "message": "今天天气如何",
            "intent": "unknown",
            "state_stack": [{"intent": "refund", "filled_slots": {}, "pending_slots": []}],
        }
        updates = await factory.direct_response_node(state)

        assert "退款" in updates["response"]
        assert "进行中" in updates["response"]

    async def test_direct_response_without_stack_has_no_hint(self):
        factory = _make_factory()

        state: DialogueState = {"message": "你好呀", "intent": "chitchat"}
        updates = await factory.direct_response_node(state)

        assert "进行中" not in updates["response"]


# ── Full graph: the live-demo regression, end to end ─────────────────────


class TestLiveBugRegression:
    async def test_weather_after_executed_refund_and_handoff(self):
        """The exact live-demo sequence: stage refund (order 1111) →
        confirm → tool fails → ask for a human → ask about the weather.
        The weather turn must NOT re-stage the executed refund."""
        from app.services.dialogue.graph import build_dialogue_graph

        detector = Mock()
        detector.detect_with_confidence = AsyncMock(
            side_effect=[
                IntentResult(intent=Intent.REFUND, confidence=0.99),
                IntentResult(intent=Intent.CONFIRM, confidence=0.99),
                IntentResult(intent=Intent.HANDOFF, confidence=0.99),
                IntentResult(intent=Intent.UNKNOWN, confidence=0.3),
            ]
        )
        graph = build_dialogue_graph(
            intent_detector=detector,
            slot_filler=None,
            tool_registry=create_default_tool_registry(),
            retrieval_pipeline={},
            llm_service=None,
        )
        config = {"configurable": {"thread_id": "weather-regression-1"}}

        turn1 = await graph.ainvoke(
            {
                "message": "我要退款，订单号1111，原因是不好吃",
                "session_id": 1,
                "user_id": 1,
            },
            config,
        )
        assert turn1["pending_confirmation"]["args"]["order_id"] == "1111"

        turn2 = await graph.ainvoke({"message": "确认"}, config)
        assert turn2["pending_confirmation"] is None
        assert "error" in turn2["tool_result"]  # order 1111 does not exist

        turn3 = await graph.ainvoke({"message": "那我要人工"}, config)
        assert turn3["intent"] == "handoff"
        assert turn3.get("state_stack") in (None, [])

        turn4 = await graph.ainvoke({"message": "今天天气如何"}, config)
        assert "即将为您执行" not in turn4["response"]
        assert not turn4.get("pending_confirmation")
        assert turn4.get("state_stack") in (None, [])
