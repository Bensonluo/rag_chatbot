"""Agent escalation tool: the unresolvable deep-funnel user gets a human.

The agent loop's outcome set had no escalation path — a task the bot
cannot resolve (failed tool, out-of-policy request) was still answered
by the model, with only the claim gate catching fabricated resolutions
after the fact. This tool lets the loop route such users into the SLA
queue (HandoffService) mid-run, same as the intent-routed handoff
node, so the fastest resolution path for an unsolvable case is one
tool call away.

Pinned here: session_id is injected server-side by the registry (a
forged model-supplied value never reaches the ticket), the funnel
handoff layer is recorded (agent escalations must not undercount the
handoff share), the ticket is created for the authenticated session,
and a persistence failure is a failed tool outcome with honest retry
copy, never a fabricated successful transfer.
"""

import json
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from prometheus_client import REGISTRY

from app.models.enums.intent import TASK_INTENTS
from app.services.dialogue.tools import (
    ToolRegistry,
    create_escalate_tool,
)


def _layers() -> float:
    value = REGISTRY.get_sample_value("chat_funnel_layers_total", {"layer": "handoff"})
    return value if value is not None else 0.0


def _handoff(ticket_id: int | None, queue_position: int | None = 2) -> Mock:
    handoff = Mock()
    handoff.create_ticket_for_session = AsyncMock(
        return_value={
            "ticket_id": ticket_id,
            "queue_position": queue_position,
            "reused": False,
        }
    )
    return handoff


async def _execute(
    handoff: Mock,
    args: dict[str, Any],
    session_id: int | None = 42,
    user_id: int | None = 9,
) -> Any:
    registry = ToolRegistry()
    registry.register(create_escalate_tool(handoff))
    return await registry.execute("agent_escalate", args, user_id=user_id, session_id=session_id)


class TestSessionInjection:
    async def test_execute_injects_session_id_as_server_truth(self) -> None:
        def handler(args: dict[str, Any]) -> dict[str, Any]:
            return {"seen_session": args.get("session_id"), "seen_user": args.get("user_id")}

        registry = ToolRegistry()
        registry.register(
            type(
                "T",
                (),
                {
                    "name": "probe",
                    "intent": "probe_intent",
                    "description": "",
                    "required_slots": [],
                    "handler": staticmethod(handler),
                },
            )()
        )

        result = await registry.execute(
            "probe_intent",
            # Forged identities in model-supplied args must lose to
            # server truth, exactly like user_id.
            {"user_id": 666, "session_id": 999},
            user_id=9,
            session_id=42,
        )

        assert result.success is True
        assert result.data == {"seen_session": 42, "seen_user": 9}

    async def test_execute_without_session_strips_supplied_value(self) -> None:
        def handler(args: dict[str, Any]) -> dict[str, Any]:
            return {"seen_session": args.get("session_id")}

        registry = ToolRegistry()
        registry.register(
            type(
                "T",
                (),
                {
                    "name": "probe",
                    "intent": "probe_intent",
                    "description": "",
                    "required_slots": [],
                    "handler": staticmethod(handler),
                },
            )()
        )

        result = await registry.execute("probe_intent", {"session_id": 999}, user_id=1)

        assert result.data == {"seen_session": None}


class TestEscalateTool:
    def test_tool_is_agent_only_and_exported(self) -> None:
        tool = create_escalate_tool(_handoff(1))
        registry = ToolRegistry()
        registry.register(tool)

        schemas = registry.to_function_schemas()

        assert tool.name == "escalate_to_human"
        assert any(s["function"]["name"] == tool.name for s in schemas)
        # Registry key only — the slot pipeline routes on dialogue
        # intents and must never reach this tool.
        assert tool.intent not in TASK_INTENTS

    async def test_successful_escalation_creates_ticket_and_records_layer(self) -> None:
        handoff = _handoff(ticket_id=7, queue_position=2)
        before = _layers()

        result = await _execute(handoff, {"reason": "订单状态异常，工具无法处理"})

        assert result.success is True
        assert result.data["ticket_id"] == 7
        assert result.data["queue_position"] == 2
        # Session and user are the authenticated ones, not model-supplied.
        handoff.create_ticket_for_session.assert_awaited_once_with(
            session_id=42,
            user_id=9,
            reason="agent",
            context={"source": "agent_tool"},
        )
        # Agent escalations must count toward the handoff funnel layer —
        # otherwise the handoff share (containment inverse) undercounts.
        assert _layers() == before + 1.0

    async def test_reused_ticket_passes_through(self) -> None:
        handoff = _handoff(ticket_id=7)
        handoff.create_ticket_for_session.return_value = {
            "ticket_id": 7,
            "queue_position": 1,
            "reused": True,
        }

        result = await _execute(handoff, {"reason": "还是不行"})

        assert result.success is True
        assert result.data["reused"] is True

    async def test_persistence_failure_is_a_failed_tool_result(self) -> None:
        handoff = _handoff(ticket_id=None, queue_position=None)

        result = await _execute(handoff, {"reason": "无法解决"})

        assert result.success is False
        assert result.data["ticket_id"] is None
        assert "未能" in result.message

    async def test_missing_session_does_not_report_success(self) -> None:
        handoff = _handoff(ticket_id=7)
        result = await _execute(handoff, {}, session_id=None)
        assert result.success is False
        handoff.create_ticket_for_session.assert_not_awaited()

    async def test_unexpected_service_error_is_not_a_success(self) -> None:
        handoff = _handoff(ticket_id=7)
        handoff.create_ticket_for_session.side_effect = RuntimeError("test service unavailable")
        result = await _execute(handoff, {})
        assert result.success is False


class TestAgentServiceSessionThreading:
    async def test_run_threads_session_id_into_tool_execution(self) -> None:
        """session_id flows run → execute; the escalate tool is the first
        consumer, so the threading is contract, not implementation detail."""
        from app.services.agent.service import AgentService
        from app.services.llm.base import LLMMessage

        handoff = _handoff(ticket_id=5)
        registry = ToolRegistry()
        registry.register(create_escalate_tool(handoff))

        tool_round = Mock()
        tool_round.content = ""
        tool_round.tool_calls = [
            {"id": "c1", "name": "escalate_to_human", "arguments": json.dumps({"reason": "x"})}
        ]
        final_round = Mock()
        final_round.content = "已为您转人工"
        final_round.tool_calls = []

        llm = Mock()
        llm.generate_with_tools = AsyncMock(side_effect=[tool_round, final_round])

        service = AgentService(llm_service=llm, tool_registry=registry, max_steps=2)
        result = await service.run(
            user_message="处理不了",
            user_id=3,
            session_id=88,
        )

        assert "#5" in result.response
        assert result.handoff_ticket_id == 5
        llm.generate_with_tools.assert_awaited_once()
        assert handoff.create_ticket_for_session.await_args.kwargs["session_id"] == 88
        # History stayed well-formed (assistant tool_call + tool result).
        assert isinstance(llm.generate_with_tools.await_args.args[0][0], LLMMessage)

    @pytest.mark.parametrize("ticket_id", [None, 7])
    @pytest.mark.parametrize("message", ["请帮我处理", "Please help me"])
    async def test_escalation_terminates_without_model_rewrite_or_later_actions(
        self, ticket_id: int | None, message: str
    ) -> None:
        from app.services.agent.service import AgentService
        from app.services.dialogue.tools import ToolDefinition

        handoff = _handoff(ticket_id=ticket_id)
        registry = ToolRegistry()
        registry.register(create_escalate_tool(handoff))
        later_action = Mock(return_value={"done": True})
        registry.register(
            ToolDefinition(
                name="another_action",
                intent="another_action",
                description="later action",
                required_slots=[],
                handler=later_action,
                parameters_schema={"type": "object"},
            )
        )
        tool_round = Mock(
            content="",
            tool_calls=[
                {"id": "c1", "name": "escalate_to_human", "arguments": "{}"},
                {"id": "c2", "name": "another_action", "arguments": "{}"},
            ],
        )
        # A model that would invent success must never get a rewrite round.
        llm = Mock()
        llm.generate_with_tools = AsyncMock(
            side_effect=[
                tool_round,
                Mock(content="Transfer completed; refund submitted", tool_calls=[]),
            ]
        )
        metric_labels = {
            "tool": "escalate_to_human",
            "outcome": "success" if ticket_id else "error",
        }
        before = REGISTRY.get_sample_value("agent_tool_calls_total", metric_labels) or 0.0
        result = await AgentService(llm, registry, max_steps=2).run(message, session_id=42)

        llm.generate_with_tools.assert_awaited_once()
        later_action.assert_not_called()
        assert result.pending_confirmation is None
        assert result.executed_tools == ["escalate_to_human"]
        assert result.tool_trace[0]["ok"] is (ticket_id is not None)
        assert result.outcome == ("handoff" if ticket_id else "handoff_failed")
        assert REGISTRY.get_sample_value("agent_tool_calls_total", metric_labels) == before + 1.0
        if ticket_id:
            assert "#7" in result.response
        else:
            assert ("未能" if message.startswith("请") else "couldn't") in result.response

    async def test_last_model_round_can_still_handoff(self) -> None:
        from app.services.agent.service import AgentService

        registry = ToolRegistry()
        registry.register(create_escalate_tool(_handoff(ticket_id=7)))
        llm = Mock()
        llm.generate_with_tools = AsyncMock(
            return_value=Mock(
                content="",
                tool_calls=[
                    {"id": "c1", "name": "escalate_to_human", "arguments": "{}"},
                ],
            )
        )
        result = await AgentService(llm, registry, max_steps=1).run("help", session_id=42)
        assert result.outcome == "handoff"
        assert "#7" in result.response
