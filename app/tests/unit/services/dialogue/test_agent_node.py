"""Agent node: routing, fallback, staging, and output guardrail.

The A12 contract: task intents upgrade to the function-calling agent
when it is wired; every failure mode (provider without tool support,
LLM outage) falls back to the deterministic slot pipeline so the user
is still served. Safety invariants from the slot pipeline — the
confirmation gate, per-user authorization, output guardrail — apply
unchanged on the agent path.
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

from app.services.agent import AgentResult, AgentService
from app.services.dialogue.nodes import NodeFactory
from app.services.dialogue.state import DialogueState
from app.services.dialogue.tools import create_default_tool_registry
from app.services.guardrails.base import GuardrailService


def _make_factory(
    agent_service: AgentService | None = None,
    guardrail_service: GuardrailService | None = None,
    history_provider: Any = None,
) -> NodeFactory:
    """NodeFactory with stub services and an optional agent service."""
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
        agent_service=agent_service,
        history_provider=history_provider,
    )


def _agent_returning(result: AgentResult) -> Mock:
    agent = Mock()
    agent.run = AsyncMock(return_value=result)
    return agent


def _guardrail(blocked: bool = False, sanitized: str | None = None) -> Mock:
    guardrail = Mock()
    guardrail.check_output = Mock(
        return_value=SimpleNamespace(was_blocked=blocked, sanitized_content=sanitized)
    )
    return guardrail


# ── route_intent: agent claims task intents only when wired ────────────────


class TestRouteIntent:
    async def test_task_intent_routes_to_agent_when_wired(self):
        factory = _make_factory(agent_service=Mock())
        updates = await factory.route_intent_node({"intent": "refund"})
        assert updates == {"route": "agent"}

    async def test_task_intent_uses_slot_pipeline_when_not_wired(self):
        factory = _make_factory(agent_service=None)
        updates = await factory.route_intent_node({"intent": "refund"})
        assert updates == {"route": "task"}

    async def test_agent_never_claims_non_task_intents(self):
        factory = _make_factory(agent_service=Mock())
        for intent in ("greeting", "question", "handoff", "confirm"):
            updates = await factory.route_intent_node({"intent": intent})
            assert updates["route"] != "agent"


# ── handle_agent_node: success, fallback, staging, guardrail ───────────────


class TestHandleAgentSuccess:
    async def test_agent_answer_ends_turn(self):
        agent = _agent_returning(AgentResult(response="您的订单已发货。"))
        factory = _make_factory(agent_service=agent)

        updates = await factory.handle_agent_node({"message": "查订单", "user_id": 1})

        assert updates["route_after_agent"] == "agent_done"
        assert updates["response"] == "您的订单已发货。"

    async def test_success_clears_stale_pending_confirmation(self):
        agent = _agent_returning(AgentResult(response="好了"))
        factory = _make_factory(agent_service=agent)

        state: DialogueState = {
            "message": "查订单",
            "user_id": 1,
            "pending_confirmation": {"intent": "refund", "args": {}},
        }
        updates = await factory.handle_agent_node(state)

        assert updates["pending_confirmation"] is None

    async def test_staged_action_propagates(self):
        staged = {"intent": "refund", "args": {"order_id": "ORD1001"}}
        agent = _agent_returning(
            AgentResult(response="⚠️ 即将为您执行「退款」", pending_confirmation=staged)
        )
        factory = _make_factory(agent_service=agent)

        updates = await factory.handle_agent_node({"message": "退款", "user_id": 1})

        assert updates["pending_confirmation"] == staged
        assert updates["route_after_agent"] == "agent_done"

    async def test_context_note_carries_known_slots(self):
        agent = _agent_returning(AgentResult(response="好的"))
        factory = _make_factory(agent_service=agent)

        state: DialogueState = {
            "message": "退款",
            "user_id": 1,
            "filled_slots": {"order_id": "ORD1001"},
        }
        await factory.handle_agent_node(state)

        agent.run.assert_awaited_once()
        kwargs = agent.run.await_args.kwargs
        assert kwargs["user_id"] == 1
        assert kwargs["context_note"] == "对话中已知信息：order_id=ORD1001"

    async def test_no_slots_means_empty_context_note(self):
        agent = _agent_returning(AgentResult(response="好的"))
        factory = _make_factory(agent_service=agent)
        await factory.handle_agent_node({"message": "查订单"})
        assert agent.run.await_args.kwargs["context_note"] == ""


class TestHandleAgentFallback:
    async def test_provider_without_function_calling_falls_back(self):
        agent = Mock()
        agent.run = AsyncMock(side_effect=NotImplementedError("no tools"))
        factory = _make_factory(agent_service=agent)

        updates = await factory.handle_agent_node({"message": "查订单"})

        assert updates == {"route_after_agent": "agent_fallback"}
        # No response set — the slot pipeline owns the reply now.
        assert "response" not in updates

    async def test_agent_crash_falls_back(self):
        agent = Mock()
        agent.run = AsyncMock(side_effect=RuntimeError("provider outage"))
        factory = _make_factory(agent_service=agent)

        updates = await factory.handle_agent_node({"message": "查订单"})
        assert updates == {"route_after_agent": "agent_fallback"}

    async def test_missing_service_falls_back(self):
        factory = _make_factory(agent_service=None)
        updates = await factory.handle_agent_node({"message": "查订单"})
        assert updates == {"route_after_agent": "agent_fallback"}


class TestHandleAgentGuardrail:
    async def test_blocked_output_replaced_with_safe_message(self):
        agent = _agent_returning(AgentResult(response="call 13800000000 for help"))
        factory = _make_factory(agent_service=agent, guardrail_service=_guardrail(blocked=True))

        updates = await factory.handle_agent_node({"message": "查订单"})

        assert "安全检查" in updates["response"]

    async def test_sanitized_output_replaces_response(self):
        agent = _agent_returning(AgentResult(response="我的邮箱是 a@b.com"))
        factory = _make_factory(
            agent_service=agent, guardrail_service=_guardrail(sanitized="我的邮箱是[REDACTED]")
        )

        updates = await factory.handle_agent_node({"message": "查订单"})

        assert updates["response"] == "我的邮箱是[REDACTED]"


class TestRouteAfterAgent:
    def test_routes(self):
        assert NodeFactory.route_after_agent({"route_after_agent": "agent_done"}) == "agent_done"
        assert NodeFactory.route_after_agent({}) == "agent_fallback"


# ── Full graph: task intent through the agent, and its fallback ────────────


def _build_graph(agent_service: AgentService | None) -> Any:
    from app.services.dialogue.graph import build_dialogue_graph

    detector = Mock()
    detector.detect_with_confidence = AsyncMock(
        return_value=Mock(intent=Mock(value="refund"), confidence=0.99)
    )
    return build_dialogue_graph(
        intent_detector=detector,
        slot_filler=None,
        tool_registry=create_default_tool_registry(),
        retrieval_pipeline={},
        llm_service=None,
        agent_service=agent_service,
    )


class TestGraphAgentFlow:
    async def test_task_intent_answered_by_agent(self):
        agent = _agent_returning(AgentResult(response="已为您办理退款，3-5 个工作日到账。"))
        graph = _build_graph(agent)

        turn = await graph.ainvoke(
            {
                "message": "ORD1001 质量问题，退款",
                "session_id": 1,
                "user_id": 1,
            },
            {"configurable": {"thread_id": "agent-flow-1"}},
        )

        agent.run.assert_awaited_once()
        assert turn["response"] == "已为您办理退款，3-5 个工作日到账。"
        # The agent handled the whole turn: the slot pipeline never ran.
        assert "pending_confirmation" not in turn or turn["pending_confirmation"] is None
        assert not turn.get("tool_result")

    async def test_agent_failure_falls_back_to_slot_pipeline(self):
        """Provider without function calling → the deterministic gate still protects refund."""
        agent = Mock()
        agent.run = AsyncMock(side_effect=NotImplementedError)
        graph = _build_graph(agent)

        turn = await graph.ainvoke(
            {
                "message": "我要退款，订单号ORD1001，原因是质量问题",
                "session_id": 1,
                "user_id": 1,
            },
            {"configurable": {"thread_id": "agent-fallback-1"}},
        )

        # Same end state as the pure slot pipeline: staged action + fixed question.
        assert turn["pending_confirmation"]["args"]["order_id"] == "ORD1001"
        assert "确认" in turn["response"]


class TestHandleAgentTrace:
    async def test_agent_trace_lands_in_state(self):
        """The audit trail must be declared in DialogueState and carried
        into the graph output — undeclared keys are silently dropped."""
        trace = [{"tool": "get_recent_orders", "ok": True, "args": {}, "summary": "..."}]
        agent = _agent_returning(AgentResult(response="查到了", tool_trace=trace))
        factory = _make_factory(agent_service=agent)

        updates = await factory.handle_agent_node({"message": "我的订单", "user_id": 1})

        assert updates["executed_tools"] == trace


class TestHandleAgentTraceOnFallback:
    async def test_fallback_carries_no_trace(self):
        agent = Mock()
        agent.run = AsyncMock(side_effect=RuntimeError("llm down"))
        factory = _make_factory(agent_service=agent)

        updates = await factory.handle_agent_node({"message": "查订单", "user_id": 1})

        assert updates["route_after_agent"] == "agent_fallback"
        assert "executed_tools" not in updates


# ── Agent loop runs with prior turns ────────────────────────────────────────


class TestAgentHistory:
    async def test_history_provider_feeds_agent_run(self):
        """The agent loop receives the session's prior turns."""
        from app.services.llm.base import LLMMessage

        agent = _agent_returning(AgentResult(response="已处理"))

        async def provider(session_id: int) -> list[Any]:
            assert session_id == 42
            return [LLMMessage(role="user", content="昨天买的手机想退货")]

        factory = _make_factory(agent_service=agent, history_provider=provider)
        state: DialogueState = {
            "message": "那运费谁出？",
            "session_id": 42,
            "user_id": 7,
        }

        await factory.handle_agent_node(state)

        history = agent.run.await_args.kwargs["history"]
        assert [(m.role, m.content) for m in history] == [("user", "昨天买的手机想退货")]

    async def test_agent_history_failure_degrades_to_none(self):
        agent = _agent_returning(AgentResult(response="已处理"))

        async def provider(session_id: int) -> list[Any]:
            raise RuntimeError("history backend down")

        factory = _make_factory(agent_service=agent, history_provider=provider)
        state: DialogueState = {"message": "查订单", "session_id": 1, "user_id": 7}

        result = await factory.handle_agent_node(state)

        assert agent.run.await_args.kwargs["history"] is None
        assert result["route_after_agent"] == "agent_done"
