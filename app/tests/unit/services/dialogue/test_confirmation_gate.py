"""Confirmation gate for irreversible tools + per-user order authorization.

Industry baseline (Sierra / Intercom Fin): an agent must never execute a
money-moving or otherwise irreversible action without an explicit user
confirmation, and tool calls must be scoped to the caller's own data.
"""

from unittest.mock import AsyncMock, Mock

from app.services.dialogue.nodes import NodeFactory
from app.services.dialogue.state import DialogueState
from app.services.dialogue.tools import (
    DEFAULT_TOOLS,
    MOCK_ORDERS,
    REFUND_AUTO_THRESHOLD,
    ToolRegistry,
    create_default_tool_registry,
)


def _make_factory(tool_registry=None) -> NodeFactory:
    """NodeFactory with stub services and a real (or injected) registry."""
    intent_detector = Mock()
    intent_detector.detect_with_confidence = AsyncMock()
    return NodeFactory(
        intent_detector=intent_detector,
        slot_filler=Mock(),
        tool_registry=tool_registry if tool_registry is not None else Mock(),
        retrieval_pipeline={},
        llm_service=None,
        guardrail_service=None,
        graph_retrieval_service=None,
    )


# ── Tool registry: confirmation flags ─────────────────────────────────────


class TestConfirmationFlags:
    def test_irreversible_tools_require_confirmation(self):
        flags = {tool.intent: tool.requires_confirmation for tool in DEFAULT_TOOLS}
        assert flags["refund"] is True
        assert flags["return"] is True

    def test_read_only_tools_execute_immediately(self):
        flags = {tool.intent: tool.requires_confirmation for tool in DEFAULT_TOOLS}
        assert flags["query_order"] is False
        assert flags["track_shipping"] is False
        assert flags["complaint"] is False


# ── execute_tool_node: the gate ───────────────────────────────────────────


class TestConfirmationGate:
    async def test_refund_stages_action_instead_of_executing(self):
        """Slots complete must NOT run the refund handler — only stage it."""
        registry = create_default_tool_registry()
        registry.execute = AsyncMock()  # would fail the test if awaited
        factory = _make_factory(tool_registry=registry)

        state: DialogueState = {
            "intent": "refund",
            "filled_slots": {"order_id": "ORD1001", "reason": "质量问题"},
            "pending_slots": [],
            "user_id": 1,
        }
        updates = await factory.execute_tool_node(state)

        registry.execute.assert_not_awaited()
        assert updates["pending_confirmation"]["intent"] == "refund"
        assert updates["pending_confirmation"]["args"]["order_id"] == "ORD1001"
        assert "确认" in updates["response"]
        assert "取消" in updates["response"]

    async def test_gate_overwrites_stale_pending_confirmation(self):
        """A leftover staged action must not bypass the gate for a new one."""
        registry = create_default_tool_registry()
        registry.execute = AsyncMock()
        factory = _make_factory(tool_registry=registry)

        state: DialogueState = {
            "intent": "refund",
            "filled_slots": {"order_id": "ORD1002", "reason": "不想要了"},
            "pending_slots": [],
            "pending_confirmation": {"intent": "refund", "args": {"order_id": "OLD"}},
            "user_id": 1,
        }
        updates = await factory.execute_tool_node(state)

        assert updates["pending_confirmation"]["args"]["order_id"] == "ORD1002"

    async def test_read_only_tool_runs_and_clears_pending(self):
        registry = create_default_tool_registry()
        factory = _make_factory(tool_registry=registry)

        state: DialogueState = {
            "intent": "query_order",
            "filled_slots": {"order_id": "ORD1001"},
            "pending_slots": [],
            "pending_confirmation": {"intent": "refund", "args": {}},
            "user_id": 1,
        }
        updates = await factory.execute_tool_node(state)

        assert updates["tool_result"]["status"] == "已发货"
        assert updates["pending_confirmation"] is None

    def test_after_execute_tool_routes_to_end_when_staged(self):
        assert (
            NodeFactory.after_execute_tool({"pending_confirmation": {"intent": "refund"}})
            == "confirm"
        )
        assert NodeFactory.after_execute_tool({"pending_confirmation": None}) == "done"
        assert NodeFactory.after_execute_tool({}) == "done"


# ── Meta intent: confirm executes, deny/cancel discards ──────────────────


class TestMetaIntentResolution:
    async def test_confirm_executes_staged_action(self):
        registry = create_default_tool_registry()
        factory = _make_factory(tool_registry=registry)

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

        assert updates["pending_confirmation"] is None
        assert updates["tool_result"]["status"] == "success"
        assert "RF" in updates["tool_result"]["refund_id"]

    async def test_confirm_with_foreign_order_fails_closed(self):
        """Staged args are re-checked against the caller at execution time."""
        registry = create_default_tool_registry()
        factory = _make_factory(tool_registry=registry)

        state: DialogueState = {
            "intent": "confirm",
            "message": "确认",
            "user_id": 1,
            "pending_confirmation": {
                "intent": "query_order",
                "args": {"order_id": "ORD2001"},  # owned by user 2
            },
        }
        updates = await factory._handle_meta_intent(state)

        assert "无权" in updates["tool_result"]["error"]

    async def test_deny_discards_staged_action(self):
        registry = create_default_tool_registry()
        registry.execute = AsyncMock()
        factory = _make_factory(tool_registry=registry)

        state: DialogueState = {
            "intent": "deny",
            "pending_confirmation": {"intent": "refund", "args": {}},
        }
        updates = await factory._handle_meta_intent(state)

        registry.execute.assert_not_awaited()
        assert updates["pending_confirmation"] is None
        assert "取消" in updates["response"]

    async def test_cancel_discards_staged_action(self):
        registry = create_default_tool_registry()
        registry.execute = AsyncMock()
        factory = _make_factory(tool_registry=registry)

        state: DialogueState = {
            "intent": "cancel",
            "pending_confirmation": {"intent": "return", "args": {}},
        }
        updates = await factory._handle_meta_intent(state)

        registry.execute.assert_not_awaited()
        assert updates["pending_confirmation"] is None


# ── Per-user authorization on order-scoped tools ─────────────────────────


class TestOrderOwnership:
    def setup_method(self):
        self.registry = create_default_tool_registry()

    async def test_own_order_succeeds(self):
        result = await self.registry.execute("query_order", {"order_id": "ORD1001"}, user_id=1)
        assert result.success is True
        assert result.data["total_amount"] == MOCK_ORDERS["ORD1001"]["total_amount"]

    async def test_other_users_order_is_denied(self):
        result = await self.registry.execute(
            "query_order",
            {"order_id": "ORD2001"},
            user_id=1,  # owner is user 2
        )
        assert result.success is False
        assert "无权" in result.message

    async def test_refund_on_foreign_order_is_denied(self):
        result = await self.registry.execute(
            "refund", {"order_id": "ORD2001", "reason": "质量问题"}, user_id=1
        )
        assert result.success is False

    async def test_unknown_order_reports_not_found(self):
        result = await self.registry.execute("query_order", {"order_id": "ORD9999"}, user_id=1)
        assert result.success is False
        assert "不存在" in result.message

    async def test_anonymous_caller_passes_in_demo_mode(self):
        result = await self.registry.execute("query_order", {"order_id": "ORD1001"})
        assert result.success is True

    async def test_refund_over_threshold_escalates(self):
        result = await self.registry.execute(
            "refund", {"order_id": "ORD3001", "reason": "质量问题"}, user_id=3
        )
        assert result.success is True  # tool ran, but escalated instead of paying
        assert result.data["status"] == "escalated"
        assert result.data["amount"] > REFUND_AUTO_THRESHOLD

    async def test_user_id_is_injected_into_handler_args(self):
        seen: list[dict] = []

        def spying_handler(args: dict) -> dict:
            seen.append(args)
            return {"ok": True}

        registry = ToolRegistry()
        from app.services.dialogue.tools import ToolDefinition

        registry.register(
            ToolDefinition(
                name="spy",
                intent="query_order",
                description="spy",
                required_slots=[],
                handler=spying_handler,
            )
        )
        await registry.execute("query_order", {"order_id": "ORD1001"}, user_id=7)
        assert seen[0]["user_id"] == 7


# ── Full graph: two-turn refund → confirm conversation ───────────────────


def _build_graph(second_intent: str):
    """Compile a graph whose detector returns refund, then ``second_intent``."""
    from app.services.dialogue.graph import build_dialogue_graph

    detector = Mock()
    detector.detect_with_confidence = AsyncMock(
        side_effect=[
            Mock(intent=Mock(value="refund"), confidence=0.99),
            Mock(intent=Mock(value=second_intent), confidence=0.99),
        ]
    )
    return build_dialogue_graph(
        intent_detector=detector,
        slot_filler=None,
        tool_registry=create_default_tool_registry(),
        retrieval_pipeline={},
        llm_service=None,
    )


class TestGraphConfirmationFlow:
    async def test_refund_then_confirm_executes_once(self):
        graph = _build_graph("confirm")
        config = {"configurable": {"thread_id": "confirm-flow-1"}}

        turn1 = await graph.ainvoke(
            {
                "message": "我要退款，订单号ORD1001，原因是质量问题",
                "session_id": 1,
                "user_id": 1,
            },
            config,
        )
        # Gate engaged: staged action, fixed-template question, no execution.
        assert turn1["pending_confirmation"]["args"]["order_id"] == "ORD1001"
        assert "确认" in turn1["response"]
        assert "tool_result" not in turn1 or not turn1.get("tool_result")

        turn2 = await graph.ainvoke({"message": "确认"}, config)
        assert turn2["pending_confirmation"] is None
        assert turn2["tool_result"]["status"] == "success"

    async def test_refund_then_deny_discards(self):
        graph = _build_graph("deny")
        config = {"configurable": {"thread_id": "deny-flow-1"}}

        turn1 = await graph.ainvoke(
            {
                "message": "我要退款，订单号ORD1002，原因是质量问题",
                "session_id": 1,
                "user_id": 1,
            },
            config,
        )
        assert turn1["pending_confirmation"] is not None

        turn2 = await graph.ainvoke({"message": "不要了"}, config)
        assert turn2["pending_confirmation"] is None
        assert "取消" in turn2["response"]

    async def test_refund_then_cancel_discards(self):
        graph = _build_graph("cancel")
        config = {"configurable": {"thread_id": "cancel-flow-1"}}

        turn1 = await graph.ainvoke(
            {
                "message": "我要退款，订单号ORD1002，原因是质量问题",
                "session_id": 1,
                "user_id": 1,
            },
            config,
        )
        assert turn1["pending_confirmation"] is not None

        turn2 = await graph.ainvoke({"message": "取消"}, config)
        assert turn2["pending_confirmation"] is None
        assert "取消" in turn2["response"]
