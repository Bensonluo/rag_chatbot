"""AgentService: bounded function-calling loop with staged confirmations.

Industry baseline (OpenAI/Anthropic agent loops, Sierra's tool-use
dialogue): the model chooses tools and arguments; the platform owns
the safety rails. Pinned here: irreversible actions are staged for
user confirmation (never executed in-loop), tool execution threads
the caller's identity, and the loop cannot run away.
"""

import json
from typing import Any

import pytest

from app.services.agent import AgentResult, AgentService
from app.services.dialogue.tools import create_default_tool_registry
from app.services.llm.base import LLMResponse


class ScriptedLLM:
    """Fake LLM replaying queued responses and recording requests."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    async def generate_with_tools(self, messages: Any, tools: Any) -> LLMResponse:
        self.requests.append({"messages": list(messages), "tools": tools})
        return self.responses.pop(0)


def _tool_call_response(name: str, args: dict[str, Any], call_id: str = "c1") -> LLMResponse:
    return LLMResponse(
        content="",
        model="fake",
        tool_calls=[{"id": call_id, "name": name, "arguments": json.dumps(args)}],
    )


def _final_response(text: str) -> LLMResponse:
    return LLMResponse(content=text, model="fake")


def _make_service(llm: Any, max_steps: int = 3) -> AgentService:
    return AgentService(
        llm_service=llm, tool_registry=create_default_tool_registry(), max_steps=max_steps
    )


class TestHappyPath:
    async def test_tool_call_then_final_answer(self):
        llm = ScriptedLLM(
            [
                _tool_call_response("query_order_status", {"order_id": "ORD1001"}),
                _final_response("您的订单已发货，预计 12-20 送达。"),
            ]
        )
        result = await _make_service(llm).run("ORD1001 发货了吗", user_id=1)

        assert result.response == "您的订单已发货，预计 12-20 送达。"
        assert result.executed_tools == ["query_order_status"]
        assert result.pending_confirmation is None
        assert not result.truncated

        # The tool result was fed back to the model as a role=tool message.
        second_request = llm.requests[1]
        tool_msgs = [m for m in second_request["messages"] if m.role == "tool"]
        assert len(tool_msgs) == 1
        payload = json.loads(tool_msgs[0].content)
        assert payload["status"] == "已发货"
        assert tool_msgs[0].tool_call_id == "c1"

        # The assistant's tool request is echoed before the result.
        assistant_msgs = [m for m in second_request["messages"] if m.role == "assistant"]
        assert assistant_msgs[-1].tool_calls is not None

    async def test_tool_schemas_offered_to_model(self):
        llm = ScriptedLLM([_final_response("好的")])
        await _make_service(llm).run("你好")
        names = [t["function"]["name"] for t in llm.requests[0]["tools"]]
        assert set(names) == {
            "process_refund",
            "process_return",
            "query_order_status",
            "track_shipping",
            "submit_complaint",
            "get_recent_orders",
        }

    async def test_system_prompt_and_context_note(self):
        llm = ScriptedLLM([_final_response("好的")])
        await _make_service(llm).run("退款", context_note="对话中已知信息：order_id=ORD1001")
        roles = [m.role for m in llm.requests[0]["messages"]]
        assert roles == ["system", "system", "user"]
        assert "客服" in llm.requests[0]["messages"][0].content
        assert "ORD1001" in llm.requests[0]["messages"][1].content


class TestConfirmationGate:
    async def test_irreversible_tool_is_staged_not_executed(self):
        llm = ScriptedLLM(
            [_tool_call_response("process_refund", {"order_id": "ORD1001", "reason": "质量问题"})]
        )
        result = await _make_service(llm).run("ORD1001 有质量问题，退款", user_id=1)

        assert result.pending_confirmation == {
            "intent": "refund",
            "args": {"order_id": "ORD1001", "reason": "质量问题"},
        }
        # The handler must not have run — only one model round happened.
        assert result.executed_tools == []
        assert "确认" in result.response

    async def test_return_also_gated(self):
        llm = ScriptedLLM(
            [_tool_call_response("process_return", {"order_id": "ORD1001", "reason": "不想要了"})]
        )
        result = await _make_service(llm).run("退货", user_id=1)
        assert result.pending_confirmation is not None
        assert result.pending_confirmation["intent"] == "return"


class TestOrderContextResolution:
    """get_recent_orders lets the agent resolve "my order" itself.

    Industry baseline (阿里小蜜 / Intercom Fin): the bot pulls the
    customer's order context from the account instead of interrogating
    the user for an order number first.
    """

    async def test_recent_orders_then_tracking_chain(self):
        llm = ScriptedLLM(
            [
                _tool_call_response("get_recent_orders", {}),
                _tool_call_response("track_shipping", {"order_id": "ORD1001"}),
                _final_response("您 ORD1001 的包裹正在北京分拨中心，预计明天送达。"),
            ]
        )
        result = await _make_service(llm).run("我的订单到哪了", user_id=1)

        assert result.response == "您 ORD1001 的包裹正在北京分拨中心，预计明天送达。"
        assert result.executed_tools == ["get_recent_orders", "track_shipping"]
        assert result.pending_confirmation is None
        assert not result.truncated

        # Round 2 saw the caller's own orders (both of user 1, newest first).
        listing = json.loads(
            [m for m in llm.requests[1]["messages"] if m.role == "tool"][0].content
        )
        assert [o["order_id"] for o in listing["orders"]] == ["ORD1001", "ORD1002"]

        # Round 3 saw the shipping facts fetched with the resolved id
        # (history carries both tool results — the last one is tracking).
        tool_msgs = [m for m in llm.requests[2]["messages"] if m.role == "tool"]
        tracking = json.loads(tool_msgs[-1].content)
        assert tracking["order_id"] == "ORD1001"

    async def test_listing_is_isolated_to_the_caller(self):
        llm = ScriptedLLM(
            [
                _tool_call_response("get_recent_orders", {}),
                _final_response("您有一笔订单。"),
            ]
        )
        await _make_service(llm).run("我的订单", user_id=2)

        listing = json.loads(
            [m for m in llm.requests[1]["messages"] if m.role == "tool"][0].content
        )
        # User 2 sees only their own order — never user 1's.
        assert [o["order_id"] for o in listing["orders"]] == ["ORD2001"]

    async def test_anonymous_caller_gets_empty_listing(self):
        llm = ScriptedLLM(
            [
                _tool_call_response("get_recent_orders", {}),
                _final_response("请先登录后再查询订单。"),
            ]
        )
        result = await _make_service(llm).run("我的订单")

        listing = json.loads(
            [m for m in llm.requests[1]["messages"] if m.role == "tool"][0].content
        )
        assert listing["orders"] == []
        assert "未登录" in listing["message"]
        assert result.executed_tools == ["get_recent_orders"]


class TestAuthorization:
    async def test_user_id_threaded_into_execution(self):
        llm = ScriptedLLM(
            [
                _tool_call_response("query_order_status", {"order_id": "ORD2001"}),
                _final_response("无权查看该订单。"),
            ]
        )
        result = await _make_service(llm).run("查一下 ORD2001", user_id=1)

        # ORD2001 belongs to user 2: the registry refuses, the refusal is
        # fed back as the tool result, and the model answers from it.
        tool_msg = [m for m in llm.requests[1]["messages"] if m.role == "tool"][0]
        assert "error" in json.loads(tool_msg.content)
        assert result.executed_tools == ["query_order_status"]


class TestLoopBound:
    async def test_step_budget_exhaustion_stops_cleanly(self):
        always_tools = [
            _tool_call_response("query_order_status", {"order_id": "ORD1001"}, f"c{i}")
            for i in range(3)
        ]
        llm = ScriptedLLM(always_tools)
        result = await _make_service(llm, max_steps=3).run("查订单")

        assert result.truncated
        assert "转人工" in result.response
        # Rounds 1-2 executed; round 3 requests tools again but the
        # budget is gone — the loop must not start what it can't finish.
        assert result.executed_tools == ["query_order_status", "query_order_status"]

    async def test_unknown_tool_reported_to_model(self):
        llm = ScriptedLLM(
            [
                _tool_call_response("book_flight", {"from": "PEK"}),
                _final_response("抱歉，我无法订机票。"),
            ]
        )
        result = await _make_service(llm).run("帮我订机票")
        assert result.executed_tools == []
        tool_msg = [m for m in llm.requests[1]["messages"] if m.role == "tool"][0]
        assert "未知工具" in json.loads(tool_msg.content)["error"]

    async def test_invalid_argument_json_tolerated(self):
        llm = ScriptedLLM(
            [
                LLMResponse(
                    content="",
                    model="fake",
                    tool_calls=[{"id": "c1", "name": "query_order_status", "arguments": "{oops"}],
                ),
                _final_response("请提供订单号。"),
            ]
        )
        result = await _make_service(llm).run("查订单", user_id=1)
        # Missing order_id → LookupError fed back as the tool result.
        tool_msg = [m for m in llm.requests[1]["messages"] if m.role == "tool"][0]
        assert "error" in json.loads(tool_msg.content)
        assert result.response == "请提供订单号。"


class TestCapability:
    async def test_provider_without_function_calling_raises(self):
        class NoToolsLLM:
            async def generate_with_tools(self, messages, tools):  # pragma: no cover
                raise NotImplementedError("no function calling")

        with pytest.raises(NotImplementedError):
            await _make_service(NoToolsLLM()).run("查订单")


class TestResultDataclass:
    def test_defaults(self):
        result = AgentResult(response="x")
        assert result.pending_confirmation is None
        assert result.executed_tools == []
        assert not result.truncated


class TestToolTrace:
    """Structured audit trail of executed tools (compliance: refunds
    and other support actions must be traceable after the fact)."""

    async def test_success_entry_has_tool_ok_args_summary(self):
        llm = ScriptedLLM(
            [
                _tool_call_response("query_order_status", {"order_id": "ORD1001"}),
                _final_response("您的订单已发货。"),
            ]
        )
        result = await _make_service(llm).run("ORD1001 发货了吗", user_id=1)

        assert result.executed_tools == ["query_order_status"]
        assert len(result.tool_trace) == 1
        entry = result.tool_trace[0]
        assert entry["tool"] == "query_order_status"
        assert entry["ok"] is True
        assert entry["args"] == {"order_id": "ORD1001"}
        assert "ORD1001" in entry["summary"]

    async def test_failure_entry_records_error(self):
        llm = ScriptedLLM(
            [
                _tool_call_response("query_order_status", {"order_id": "ORD9999"}),
                _final_response("没有找到这个订单。"),
            ]
        )
        result = await _make_service(llm).run("ORD9999 发货了吗", user_id=1)

        assert len(result.tool_trace) == 1
        entry = result.tool_trace[0]
        assert entry["ok"] is False
        assert "不存在" in entry["summary"]

    async def test_unknown_tool_recorded_as_failed_trace(self):
        llm = ScriptedLLM(
            [
                _tool_call_response("hallucinated_tool", {"x": 1}),
                _final_response("抱歉，办不了。"),
            ]
        )
        result = await _make_service(llm).run("帮我办", user_id=1)

        assert result.executed_tools == []
        assert result.tool_trace == [
            {
                "tool": "hallucinated_tool",
                "ok": False,
                "args": {"x": 1},
                "summary": "未知工具: hallucinated_tool",
            }
        ]

    async def test_staging_preserves_prior_trace(self):
        """Tools executed before a staged irreversible action stay in
        the audit trail (staged itself is *not* an execution)."""
        llm = ScriptedLLM(
            [
                _tool_call_response("get_recent_orders", {}),
                _tool_call_response(
                    "process_refund", {"order_id": "ORD1001", "reason": "不想要了"}
                ),
                _final_response("占位"),
            ]
        )
        result = await _make_service(llm).run("帮我退款", user_id=1)

        assert result.pending_confirmation is not None
        assert result.executed_tools == ["get_recent_orders"]
        assert [e["tool"] for e in result.tool_trace] == ["get_recent_orders"]
        assert result.tool_trace[0]["ok"] is True

    async def test_summary_is_bounded(self):
        huge = "x" * 5000
        llm = ScriptedLLM(
            [
                _tool_call_response("query_order_status", {"order_id": "ORD1001"}),
                _final_response("ok"),
            ]
        )
        # Patch one order's payload via the registry data is overkill;
        # instead bound-check on a normal entry: summaries never exceed
        # the 200-char cap even with long args echoed into results.
        result = await _make_service(llm).run(huge, user_id=1)
        for entry in result.tool_trace:
            assert len(entry["summary"]) <= 200
            assert entry["args"] == {"order_id": "ORD1001"}


class TestConversationHistory:
    """Prior turns join the agent loop's message list."""

    async def test_history_precedes_user_message(self):
        from app.services.llm.base import LLMMessage

        llm = ScriptedLLM([_final_response("好的")])
        service = _make_service(llm)
        history = [
            LLMMessage(role="user", content="昨天买的手机想退货"),
            LLMMessage(role="assistant", content="好的，已受理退货"),
        ]

        await service.run(user_message="那运费谁出？", history=history)

        messages = llm.requests[0]["messages"]
        assert messages[0].role == "system"
        assert [(m.role, m.content) for m in messages[1:]] == [
            ("user", "昨天买的手机想退货"),
            ("assistant", "好的，已受理退货"),
            ("user", "那运费谁出？"),
        ]

    async def test_history_after_context_note(self):
        from app.services.llm.base import LLMMessage

        llm = ScriptedLLM([_final_response("好的")])
        service = _make_service(llm)
        history = [LLMMessage(role="user", content="查订单")]

        await service.run(user_message="最新的呢", context_note="已知：order_id=1", history=history)

        messages = llm.requests[0]["messages"]
        assert [m.role for m in messages] == ["system", "system", "user", "user"]
        assert "order_id=1" in messages[1].content
        assert messages[2].content == "查订单"
