"""Claim gate wiring in generate_response_node (Phase A2).

The generation node is the seam where LLM-proposed text gets verified
against the curated fact table before it leaves the graph: policy-claim
violations are rewritten to grounded statements, unexecuted-action
assertions are softened to guidance. Tool-grounded responses skip the
gate (their numbers come from executed tool output by construction) and
the whole gate is setting-gated for blast-radius control.
"""

from typing import Any
from unittest.mock import AsyncMock, Mock, patch

from app.services.dialogue.nodes import NodeFactory


def _make_factory() -> NodeFactory:
    return NodeFactory(
        intent_detector=Mock(),
        slot_filler=None,
        tool_registry=Mock(),
    )


def _logic_returning(response: str) -> AsyncMock:  # noqa: UP047 - AsyncMock instance
    logic = AsyncMock(return_value={"response": response})
    return logic


async def _run_node(factory: NodeFactory, state: dict[str, Any]) -> dict[str, Any]:
    return await factory.generate_response_node(state)  # type: ignore[arg-type]


REFUND_ASK = {"message": "退款多久到账", "intent": "question", "session_id": 1, "user_id": 1}
WRONG_SLA = "退款将在 10 个工作日内到账，请耐心等待。"
RIGHT_SLA = "退款将原路退回，支付宝 1-3 个工作日到账。"


class TestClaimGateRewrites:
    async def test_violating_sla_response_is_rewritten(self):
        factory = _make_factory()
        factory._generate_response_logic = _logic_returning(WRONG_SLA)  # type: ignore[method-assign]
        updates = await _run_node(factory, dict(REFUND_ASK))
        assert "10 个工作日" not in updates["response"]
        assert "1-3 个工作日" in updates["response"]

    async def test_clean_response_untouched(self):
        factory = _make_factory()
        factory._generate_response_logic = _logic_returning(RIGHT_SLA)  # type: ignore[method-assign]
        updates = await _run_node(factory, dict(REFUND_ASK))
        assert updates["response"] == RIGHT_SLA

    async def test_unexecuted_action_softened(self):
        factory = _make_factory()
        factory._generate_response_logic = _logic_returning("已为您办理退款，请查收。")  # type: ignore[method-assign]
        updates = await _run_node(factory, {"message": "我要退款", "session_id": 1})
        assert "已为您办理退款" not in updates["response"]

    async def test_no_anchor_message_untouched(self):
        """Chitchat pulls an empty subgraph — nothing to check, text
        passes through even if it contains numbers."""
        factory = _make_factory()
        response = "今天天气不错，适合出行 3 公里散步。"
        factory._generate_response_logic = _logic_returning(response)  # type: ignore[method-assign]
        updates = await _run_node(factory, {"message": "讲个笑话", "session_id": 1})
        assert updates["response"] == response


class TestClaimGateSkips:
    async def test_tool_result_path_skips_gate(self):
        """Tool responses quote tool output (order status, processing
        days) — grounded by construction, must not be rewritten."""
        factory = _make_factory()
        response = "退款已提交，预计 3-5 个工作日完成处理。"
        factory._generate_response_logic = _logic_returning(response)  # type: ignore[method-assign]
        state = {"message": "退款多久到账", "tool_result": {"status": "submitted"}, "session_id": 1}
        updates = await _run_node(factory, state)
        assert updates["response"] == response

    async def test_disabled_setting_disables_gate(self):
        from app.config.settings import settings

        factory = _make_factory()
        factory._generate_response_logic = _logic_returning(WRONG_SLA)  # type: ignore[method-assign]
        with patch.object(settings, "FACT_CLAIM_CHECK_ENABLED", False):
            updates = await _run_node(factory, dict(REFUND_ASK))
        assert updates["response"] == WRONG_SLA

    async def test_blocked_state_skips_gate(self):
        factory = _make_factory()
        factory._generate_response_logic = _logic_returning(WRONG_SLA)  # type: ignore[method-assign]
        updates = await _run_node(factory, {**REFUND_ASK, "blocked": True})
        assert updates["response"] == WRONG_SLA


class TestClaimGateMetrics:
    async def test_check_increments_counters(self):
        from prometheus_client import REGISTRY

        factory = _make_factory()
        factory._generate_response_logic = _logic_returning(WRONG_SLA)  # type: ignore[method-assign]
        before_checks = REGISTRY.get_sample_value("claim_checks_total") or 0.0
        before_violations = (
            REGISTRY.get_sample_value("claim_violations_total", {"reason": "numeric_mismatch"})
            or 0.0
        )
        await _run_node(factory, dict(REFUND_ASK))
        assert REGISTRY.get_sample_value("claim_checks_total") == before_checks + 1
        assert (
            REGISTRY.get_sample_value("claim_violations_total", {"reason": "numeric_mismatch"})
            == before_violations + 1
        )


class TestClaimGateSettings:
    def test_gate_defaults_to_enabled(self):
        from app.config.settings import settings

        assert settings.FACT_CLAIM_CHECK_ENABLED is True


class TestAgentPathClaimGate:
    """Agent responses are LLM paraphrases of tool output, not fixed
    formats — misstating tool numbers and fabricating completions are
    documented agent hallucination modes (arXiv 2026 agent-hallucination
    surveys), so policy claims are verified on this path too. A
    completed-action assertion is trusted only when tool_trace proves
    the tool actually executed."""

    @staticmethod
    def _agent_factory(
        response: str,
        tool_trace: list[dict[str, Any]] | None = None,
    ) -> NodeFactory:
        from app.services.agent.service import AgentResult

        factory = _make_factory()
        agent = Mock()
        agent.run = AsyncMock(
            return_value=AgentResult(
                response=response,
                pending_confirmation=None,
                tool_trace=tool_trace or [],
            )
        )
        factory._agent_service = agent
        return factory

    async def test_agent_wrong_sla_is_rewritten(self):
        factory = self._agent_factory("退款将在 10 个工作日内到账。")
        updates = await factory.handle_agent_node(dict(REFUND_ASK))  # type: ignore[arg-type]
        assert "10 个工作日" not in updates["response"]
        assert "1-3 个工作日" in updates["response"]

    async def test_agent_completed_action_without_tools_softened(self):
        factory = self._agent_factory("已为您办理退款，请耐心等待。")
        updates = await factory.handle_agent_node({"message": "我要退款", "session_id": 1})
        assert "已为您办理退款" not in updates["response"]

    async def test_agent_executed_tool_legitimizes_action_assertion(self):
        factory = self._agent_factory(
            "已为您提交退款申请，预计 3-5 个工作日完成处理。",
            tool_trace=[{"name": "submit_refund", "status": "success"}],
        )
        updates = await factory.handle_agent_node({"message": "我要退款", "session_id": 1})
        assert "已为您提交退款申请" in updates["response"]

    async def test_agent_tool_execution_does_not_exempt_numbers(self):
        factory = self._agent_factory(
            "已为您提交退款申请，预计 10 个工作日完成处理。",
            tool_trace=[{"name": "submit_refund", "status": "success"}],
        )
        updates = await factory.handle_agent_node({"message": "我要退款", "session_id": 1})
        assert "10 个工作日" not in updates["response"]
        assert "3-5 个工作日" in updates["response"]

    async def test_agent_clean_response_untouched(self):
        response = "退款已提交，预计 3-5 个工作日完成处理。"
        factory = self._agent_factory(
            response, tool_trace=[{"name": "submit_refund", "status": "success"}]
        )
        updates = await factory.handle_agent_node({"message": "我要退款", "session_id": 1})
        assert updates["response"] == response
