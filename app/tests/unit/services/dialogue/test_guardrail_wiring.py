"""Tests for guardrail PII wiring in the dialogue graph nodes.

Regression tests for the audit finding that sanitized_content was
discarded on input and check_output was never invoked on output.
"""
from unittest.mock import AsyncMock

from app.services.dialogue.nodes import NodeFactory
from app.services.guardrails.base import GuardrailResult

RAW_INPUT = "我的手机号是13800138000，帮我查下订单"
REDACTED_INPUT = "我的手机号是[手机号]，帮我查下订单"
RAW_OUTPUT = "好的，已为您查询，联系电13800138000会有专员回访"
REDACTED_OUTPUT = "好的，已为您查询，联系电[手机号]会有专员回访"


class _FakeGuardrail:
    """Configurable stand-in for the guardrail service."""

    def __init__(self, input_result=None, output_result=None):
        self._input_result = input_result
        self._output_result = output_result

    def check_input(self, content: str) -> GuardrailResult:
        if self._input_result is not None:
            return self._input_result
        return GuardrailResult(
            passed=True, action="allow", original_content=content, sanitized_content=content
        )

    def check_output(self, content: str) -> GuardrailResult:
        if self._output_result is not None:
            return self._output_result
        return GuardrailResult(
            passed=True, action="allow", original_content=content, sanitized_content=content
        )


def _make_factory(guardrail) -> NodeFactory:
    return NodeFactory(
        intent_detector=None,
        slot_filler=None,
        tool_registry=None,
        guardrail_service=guardrail,
    )


class TestGuardrailInputWiring:
    """Input-side PII redaction must reach the dialogue state"""

    async def test_redacted_input_replaces_state_message(self):
        """When input is redacted, the sanitized text flows into the state"""
        # Arrange
        guardrail = _FakeGuardrail(
            input_result=GuardrailResult(
                passed=True,
                action="redact",
                original_content=RAW_INPUT,
                sanitized_content=REDACTED_INPUT,
            )
        )
        factory = _make_factory(guardrail)

        # Act
        updates = await factory.guardrail_node({"message": RAW_INPUT})

        # Assert
        assert updates == {"message": REDACTED_INPUT}

    async def test_clean_input_returns_no_updates(self):
        """A passing, unredacted input leaves the state untouched"""
        # Arrange
        factory = _make_factory(_FakeGuardrail())

        # Act
        updates = await factory.guardrail_node({"message": "你好"})

        # Assert
        assert updates == {}

    async def test_no_guardrail_service_is_passthrough(self):
        """Without a guardrail service the node is a no-op"""
        # Arrange
        factory = _make_factory(None)

        # Act
        updates = await factory.guardrail_node({"message": RAW_INPUT})

        # Assert
        assert updates == {}


class TestGuardrailOutputWiring:
    """Output-side checks must run on generated responses"""

    def _factory_with_direct(self, guardrail, direct_response: str) -> NodeFactory:
        factory = _make_factory(guardrail)
        factory._generate_direct = AsyncMock(return_value={"response": direct_response})
        return factory

    async def test_redacted_output_replaces_response(self):
        """PII in the generated response is redacted before leaving the graph"""
        # Arrange
        guardrail = _FakeGuardrail(
            output_result=GuardrailResult(
                passed=True,
                action="redact",
                original_content=RAW_OUTPUT,
                sanitized_content=REDACTED_OUTPUT,
            )
        )
        factory = self._factory_with_direct(guardrail, RAW_OUTPUT)

        # Act
        updates = await factory.generate_response_node({"message": "查询订单", "intent": "chitchat"})

        # Assert
        assert updates["response"] == REDACTED_OUTPUT

    async def test_blocked_output_replaced_with_safe_message(self):
        """An unsafe response is replaced, never delivered raw"""
        # Arrange
        guardrail = _FakeGuardrail(
            output_result=GuardrailResult(
                passed=False,
                action="block",
                original_content=RAW_OUTPUT,
                sanitized_content="",
                violations=["unsafe_content"],
            )
        )
        factory = self._factory_with_direct(guardrail, RAW_OUTPUT)

        # Act
        updates = await factory.generate_response_node({"message": "查询订单", "intent": "chitchat"})

        # Assert
        assert updates["response"] == "抱歉，该回复未能通过安全检查，请重新提问。"

    async def test_clean_output_passes_through(self):
        """A clean response is delivered unchanged"""
        # Arrange
        factory = self._factory_with_direct(_FakeGuardrail(), "您的订单已发货")

        # Act
        updates = await factory.generate_response_node({"message": "查询订单", "intent": "chitchat"})

        # Assert
        assert updates["response"] == "您的订单已发货"

    async def test_no_guardrail_service_output_unchanged(self):
        """Without a guardrail service the raw response is returned"""
        # Arrange
        factory = self._factory_with_direct(None, RAW_OUTPUT)

        # Act
        updates = await factory.generate_response_node({"message": "查询订单", "intent": "chitchat"})

        # Assert
        assert updates["response"] == RAW_OUTPUT
