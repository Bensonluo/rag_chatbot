"""Agent-loop telemetry contract: the funnel's deepest layer is measurable.

The funnel counter says how many turns reached the agent layer; nothing
says how the layer performs. The north star weights deep-funnel tool-loop
users as the real customer-service need — these tests pin that every run
outcome (answered / staged / budget_exhausted / steps_exhausted /
fallback-to-slot-pipeline), every tool call result, and the per-run tool
call depth are counted, so layer health is computable, not anecdotal.
"""

import json
from typing import Any
from unittest.mock import AsyncMock, Mock

from prometheus_client import REGISTRY

from app.services.agent import AgentService
from app.services.llm.base import LLMResponse
from app.services.llm.budget import LLMBudgetExceeded


class ScriptedLLM:
    """Fake LLM replaying queued responses."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)

    async def generate_with_tools(self, messages: Any, tools: Any) -> LLMResponse:
        return self.responses.pop(0)


class RaisingLLM:
    """Fake LLM whose every call raises the injected exception."""

    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    async def generate_with_tools(self, messages: Any, tools: Any) -> LLMResponse:
        raise self.exc


def _tool_call_response(name: str, args: dict[str, Any], call_id: str = "c1") -> LLMResponse:
    return LLMResponse(
        content="",
        model="fake",
        tool_calls=[{"id": call_id, "name": name, "arguments": json.dumps(args)}],
    )


def _final_response(text: str) -> LLMResponse:
    return LLMResponse(content=text, model="fake")


def _outcome_value(outcome: str) -> float:
    value = REGISTRY.get_sample_value("agent_loop_outcomes_total", {"outcome": outcome})
    return value if value is not None else 0.0


def _tool_call_value(tool: str, outcome: str) -> float:
    value = REGISTRY.get_sample_value("agent_tool_calls_total", {"tool": tool, "outcome": outcome})
    return value if value is not None else 0.0


def _histo_count() -> float:
    value = REGISTRY.get_sample_value("agent_loop_tool_calls_count")
    return value if value is not None else 0.0


class TestAgentLoopOutcomeTelemetry:
    async def test_answered_run_counts_answered_outcome(self):
        from app.services.dialogue.tools import create_default_tool_registry

        llm = ScriptedLLM([_final_response("好的")])
        service = AgentService(llm_service=llm, tool_registry=create_default_tool_registry())

        before = _outcome_value("answered")
        await service.run("你好")
        assert _outcome_value("answered") == before + 1.0

    async def test_staged_confirmation_counts_staged_outcome(self):
        from app.services.dialogue.tools import create_default_tool_registry

        llm = ScriptedLLM(
            [_tool_call_response("process_refund", {"order_id": "ORD1001", "reason": "质量问题"})]
        )
        service = AgentService(llm_service=llm, tool_registry=create_default_tool_registry())

        before = _outcome_value("staged")
        await service.run("退款", user_id=1)
        assert _outcome_value("staged") == before + 1.0

    async def test_budget_exhaustion_counts_budget_outcome(self):
        from app.services.dialogue.tools import create_default_tool_registry

        service = AgentService(
            llm_service=RaisingLLM(LLMBudgetExceeded("spent")),
            tool_registry=create_default_tool_registry(),
        )

        before = _outcome_value("budget_exhausted")
        await service.run("查订单", user_id=1)
        assert _outcome_value("budget_exhausted") == before + 1.0

    async def test_steps_exhaustion_counts_steps_outcome(self):
        from app.services.dialogue.tools import create_default_tool_registry

        llm = ScriptedLLM([_tool_call_response("query_order_status", {"order_id": "ORD1001"})])
        service = AgentService(
            llm_service=llm, tool_registry=create_default_tool_registry(), max_steps=1
        )

        before = _outcome_value("steps_exhausted")
        await service.run("查订单", user_id=1)
        assert _outcome_value("steps_exhausted") == before + 1.0


class TestAgentToolCallTelemetry:
    async def test_successful_tool_call_counts_success(self):
        from app.services.dialogue.tools import create_default_tool_registry

        llm = ScriptedLLM(
            [
                _tool_call_response("query_order_status", {"order_id": "ORD1001"}),
                _final_response("已发货。"),
            ]
        )
        service = AgentService(llm_service=llm, tool_registry=create_default_tool_registry())

        before = _tool_call_value("query_order_status", "success")
        before_runs = _histo_count()
        await service.run("ORD1001 发货了吗", user_id=1)
        assert _tool_call_value("query_order_status", "success") == before + 1.0
        assert _histo_count() == before_runs + 1.0

    async def test_failed_tool_call_counts_error(self):
        from app.services.dialogue.tools import create_default_tool_registry

        # ORD1001 belongs to user 1: caller 999 fails the ownership check.
        llm = ScriptedLLM(
            [
                _tool_call_response("query_order_status", {"order_id": "ORD1001"}),
                _final_response("查询失败，抱歉。"),
            ]
        )
        service = AgentService(llm_service=llm, tool_registry=create_default_tool_registry())

        before = _tool_call_value("query_order_status", "error")
        await service.run("ORD1001 发货了吗", user_id=999)
        assert _tool_call_value("query_order_status", "error") == before + 1.0


class TestAgentFallbackTelemetry:
    async def test_node_fallback_counts_fallback_outcome(self):
        from app.services.dialogue.nodes import NodeFactory

        agent = Mock()
        agent.run = AsyncMock(side_effect=RuntimeError("llm down"))
        factory = NodeFactory(
            intent_detector=Mock(),
            slot_filler=None,
            tool_registry=Mock(),
            guardrail_service=None,
            agent_service=agent,
        )

        before = _outcome_value("fallback")
        updates = await factory.handle_agent_node({"message": "查订单", "session_id": 1})
        assert updates["route_after_agent"] == "agent_fallback"
        assert _outcome_value("fallback") == before + 1.0
