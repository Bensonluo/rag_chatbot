"""Order-id format guard for staged irreversible actions.

Regression coverage for the 2026-10-28 demo session: the agent accepted a
dismissive phrase ("Tune them out") as an order number and staged a refund
confirmation around it. A malformed id must now re-ask instead of staging.
"""

import json
from typing import Any

import pytest

from app.services.agent import AgentService
from app.services.dialogue.tools import create_default_tool_registry, valid_order_id_token
from app.services.llm.base import LLMResponse


class ScriptedLLM:
    """Fake LLM replaying queued responses."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)

    async def generate_with_tools(self, messages: Any, tools: Any) -> LLMResponse:
        return self.responses.pop(0)


def _tool_call(name: str, args: dict[str, Any]) -> LLMResponse:
    return LLMResponse(
        content="",
        model="fake",
        tool_calls=[{"id": "c1", "name": name, "arguments": json.dumps(args)}],
    )


def _make_service(responses: list[LLMResponse]) -> AgentService:
    return AgentService(
        llm_service=ScriptedLLM(responses),
        tool_registry=create_default_tool_registry(),
        max_steps=3,
    )


class TestValidOrderIdToken:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("ORD1001", True),  # demo store id
            ("ORD-1001", True),
            ("o_3f9a2b8c1d4e5f6a7b8c9d0e1f2a3b4c", True),  # provider id (uuid hex)
            ("mo_abcdef123456", True),
            ("ord1001", True),  # case tolerance
            ("Tune them out", False),  # multi-word phrase (2026-10-28 incident)
            ("refund", False),  # digit-free token
            ("", False),
            (None, False),
            ("ORD 1001", False),  # embedded space
            ("100%", False),  # stray symbol
        ],
    )
    def test_token_shapes(self, value: object, expected: bool) -> None:
        assert valid_order_id_token(value) is expected


class TestRefundStagingGuard:
    async def test_garbage_order_id_never_stages_en(self):
        result = await _make_service(
            [_tool_call("process_refund", {"order_id": "Tune them out", "reason": "don't like it"})]
        ).run("I want a refund, order Tune them out")

        assert result.pending_confirmation is None
        assert result.outcome == "answered"
        # Single re-ask question, ends with "?", names a concrete id shape.
        assert result.response.count("?") == 1
        assert "ORD1001" in result.response
        assert "signed in" not in result.response

    async def test_garbage_order_id_never_stages_zh(self):
        result = await _make_service(
            [_tool_call("process_return", {"order_id": "退了吧退了吧", "reason": "不想要"})]
        ).run("退货，订单号就是退了吧退了吧")

        assert result.pending_confirmation is None
        assert "订单号" in result.response
        assert "？" in result.response or "?" in result.response

    async def test_valid_demo_id_still_stages(self):
        result = await _make_service(
            [_tool_call("process_refund", {"order_id": "ORD1001", "reason": "wrong item"})]
        ).run("refund ORD1001 please")

        assert result.pending_confirmation is not None
        assert result.outcome == "staged"
        assert result.pending_confirmation["args"]["order_id"] == "ORD1001"

    async def test_provider_id_still_stages(self):
        # Provider-shaped ids must pass the guard (the mock handler reroutes
        # them afterwards — staging the gate is still correct).
        provider_id = "o_" + "a1" * 16
        result = await _make_service(
            [_tool_call("process_refund", {"order_id": provider_id, "reason": "broken"})]
        ).run(f"refund {provider_id}")

        assert result.pending_confirmation is not None
        assert result.pending_confirmation["args"]["order_id"] == provider_id

    async def test_missing_order_id_also_re_asks(self):
        result = await _make_service([_tool_call("process_refund", {"reason": "no id given"})]).run(
            "I want a refund"
        )

        assert result.pending_confirmation is None
        assert "?" in result.response


class TestComplaintSanitizesOrderId:
    """Non-money twin of the staging guard: a garbage order reference is
    dropped from the ticket instead of persisted for the human agent."""

    async def test_garbage_order_id_dropped_from_ticket(self):
        result = await _make_service(
            [
                _tool_call(
                    "submit_complaint",
                    {"category": "物流", "description": "慢", "order_id": "Tune them out"},
                ),
                _final("已提交投诉"),
            ]
        ).run("投诉物流太慢，订单 Tune them out")

        assert result.executed_tools == ["submit_complaint"]
        assert result.outcome == "answered"

    async def test_handler_drops_garbage_keeps_valid(self):
        from app.services.dialogue.tools import mock_complaint

        garbage = mock_complaint(
            {"category": "物流", "description": "慢", "order_id": "退了吧退了吧"}
        )
        assert garbage["order_id"] == ""

        valid = mock_complaint({"category": "物流", "description": "慢", "order_id": "ORD1001"})
        assert valid["order_id"] == "ORD1001"

        absent = mock_complaint({"category": "物流", "description": "慢"})
        assert absent["order_id"] == ""


def _final(text: str) -> LLMResponse:
    return LLMResponse(content=text, model="fake")
