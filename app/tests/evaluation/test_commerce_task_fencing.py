"""Shopping task fencing with a provider LLM and real SQL domain/provider.

Scripted/native planners model LLM decisions explicitly. No production phrase
parser participates in shopping semantics.
"""

import json
from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy import event, func, select
from sqlalchemy.exc import OperationalError

from app.models.database.commerce import MockCommerceObject, PurchaseOperation
from app.models.database.session import ChatSession
from app.services.dialogue.tools import MOCK_ORDERS
from app.services.llm.base import LLMMessage, LLMResponse
from app.tests.evaluation.test_guest_commerce_flow import (
    ShopHttp,
    _boot,
    _business,
    _chat,
    _counts,
    _history,
    _search,
    _status,
    _stream_parse,
    _ui,
)
from app.tests.evaluation.test_guest_commerce_flow import shop as shop
from app.tests.evaluation.test_guest_conversations import no_external_dns as no_external_dns
from app.tests.evaluation.test_guest_conversations import visitors as visitors
from app.tests.evaluation.test_human_conversation import case as case


class NativePlanner:
    def __init__(
        self,
        tool: str,
        arguments: dict[str, Any],
        *,
        fail_first: bool = False,
        fail_after_tool: bool = True,
    ) -> None:
        self.tool = tool
        self.arguments = arguments
        self.fail_first = fail_first
        self.fail_after_tool = fail_after_tool
        self.calls = 0

    async def generate(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        del messages, kwargs
        return LLMResponse(
            content=json.dumps(
                {"route": "shopping", "confidence": 0.9, "reason": "native"},
                ensure_ascii=False,
            ),
            model="native",
        )

    async def generate_with_tools(
        self, messages: list[LLMMessage], tools: list[dict[str, Any]]
    ) -> LLMResponse:
        del tools
        self.calls += 1
        if messages and messages[-1].role == "tool":
            if self.fail_after_tool:
                raise RuntimeError("isolated-native-shopping-planner-failure")
            return LLMResponse(content="已根据工具结果处理。", model="native")
        if self.fail_first:
            raise RuntimeError("isolated-native-shopping-planner-failure")
        return LLMResponse(
            content="",
            model="native",
            tool_calls=[
                {
                    "id": str(uuid4()),
                    "name": self.tool,
                    "arguments": json.dumps(self.arguments),
                }
            ],
        )


async def _epoch(shop: ShopHttp, browser: dict[str, Any]) -> int:
    async with shop.browsers.case.maker() as db:
        row = await db.get(ChatSession, browser["session"]["id"])
        assert row is not None
        return row.automation_epoch


async def _stage_owned_refund(
    shop: ShopHttp, browser: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> int:
    from app.models.database.guest import GuestVisitor

    async with shop.browsers.case.maker() as db:
        visitor = await db.get(GuestVisitor, browser["visitor_id"])
        assert visitor is not None
        order = {**MOCK_ORDERS["ORD1001"], "user_id": visitor.user_id}
    monkeypatch.setitem(MOCK_ORDERS, "ORD1001", order)
    epoch = await _epoch(shop, browser)
    for text in ("I want a refund", "ORD1001", "damaged item"):
        response = await _chat(
            shop.browsers.first, browser, text, shop=shop, plan=None, route="support"
        )
        assert response.status_code == 200
    thread = str(browser["session"]["id"]) + (f":automation:{epoch}" if epoch else "")
    state = await shop.browsers.case.chat.graph.aget_state({"configurable": {"thread_id": thread}})
    assert state.values.get("pending_confirmation")
    return epoch


async def _old_proposal_cannot_execute(
    shop: ShopHttp, browser: dict[str, Any], old_epoch: int
) -> None:
    epoch = await _epoch(shop, browser)
    thread = str(browser["session"]["id"]) + (f":automation:{epoch}" if epoch else "")
    before = len(shop.browsers.case.calls.refunds)
    response = await _chat(
        shop.browsers.first, browser, "yes", shop=shop, plan=None, route="support"
    )
    assert response.status_code == 200
    assert len(shop.browsers.case.calls.refunds) == before
    state = await shop.browsers.case.chat.graph.aget_state({"configurable": {"thread_id": thread}})
    assert not state.values.get("pending_confirmation")


async def test_successful_tool_then_native_failure_preserves_fence_and_recoverable_cart(
    shop: ShopHttp, monkeypatch: pytest.MonkeyPatch
) -> None:
    browser = await _boot(shop.browsers.first)
    view = await _search(shop, shop.browsers.first, browser)
    old_epoch = await _stage_owned_refund(shop, browser, monkeypatch)
    before_history = await _history(shop.browsers.first, browser)
    original_provider = shop.conversation.provider
    planner = NativePlanner(
        "commerce_update_cart_lines",
        {
            "purchase_id": view["purchase_id"],
            "lines": [{"sku_id": "ANC-02", "quantity": 2}],
            "expected_version": view["version"],
        },
    )
    shop.conversation.provider = planner
    response = await _chat(shop.browsers.first, browser, "我选白色2个", stream=True)
    _, _, traces = _stream_parse(response.text)
    assert response.text.endswith("data: [STREAM_ERROR]\n\ndata: [DONE]\n\n")
    assert ("cs.tool", "end") in [(event["stage"], event["status"]) for event in traces]
    assert ("cs.generation", "error") in [(event["stage"], event["status"]) for event in traces]
    assert (traces[-1]["stage"], traces[-1]["status"]) == ("cs.pipeline", "error")
    assert planner.calls == 2
    assert await _history(shop.browsers.first, browser) == before_history
    recovered = await _status(shop.browsers.first, browser, view["purchase_id"])
    assert recovered.status_code == 200
    actual = _ui(recovered.json())
    assert actual["stage"] == "cart" and actual["version"] > view["version"]
    assert await _counts(shop) == {"order": 0, "payment": 0}
    shop.conversation.provider = original_provider
    await _old_proposal_cannot_execute(shop, browser, old_epoch)


async def test_native_first_call_failure_fences_old_proposal_without_business_effect(
    shop: ShopHttp, monkeypatch: pytest.MonkeyPatch
) -> None:
    browser = await _boot(shop.browsers.first)
    await _search(shop, shop.browsers.first, browser)
    old_epoch = await _stage_owned_refund(shop, browser, monkeypatch)
    before_history = await _history(shop.browsers.first, browser)
    original_provider = shop.conversation.provider
    planner = NativePlanner("commerce_update_cart_lines", {}, fail_first=True)
    shop.conversation.provider = planner
    response = await _chat(shop.browsers.first, browser, "继续选购", stream=True)
    _, _, traces = _stream_parse(response.text)
    assert response.text.endswith("data: [STREAM_ERROR]\n\ndata: [DONE]\n\n")
    assert ("cs.generation", "error") in [(event["stage"], event["status"]) for event in traces]
    assert (traces[-1]["stage"], traces[-1]["status"]) == ("cs.pipeline", "error")
    assert planner.calls == 1
    assert await _history(shop.browsers.first, browser) == before_history
    shop.conversation.provider = original_provider
    await _old_proposal_cannot_execute(shop, browser, old_epoch)


async def test_message_insert_failure_keeps_fence_and_recoverable_cart(
    shop: ShopHttp, monkeypatch: pytest.MonkeyPatch
) -> None:
    browser = await _boot(shop.browsers.first)
    view = await _search(shop, shop.browsers.first, browser)
    old_epoch = await _stage_owned_refund(shop, browser, monkeypatch)
    before_history = await _history(shop.browsers.first, browser)
    original_provider = shop.conversation.provider
    engine = shop.browsers.case.maker.kw["bind"].sync_engine
    failed: list[str] = []

    def fail_insert(_connection, _cursor, statement, _parameters, _context, _many):
        if statement.lstrip().upper().startswith("INSERT INTO MESSAGES"):
            failed.append("Message INSERT")
            raise OperationalError("isolated Message insert", {}, RuntimeError("unavailable"))

    shop.conversation.provider = NativePlanner(
        "commerce_update_cart_lines",
        {
            "purchase_id": view["purchase_id"],
            "lines": [{"sku_id": "ANC-02", "quantity": 2}],
            "expected_version": view["version"],
        },
        fail_after_tool=False,
    )
    event.listen(engine, "before_cursor_execute", fail_insert)
    try:
        response = await _chat(shop.browsers.first, browser, "白色2个", stream=True)
    finally:
        event.remove(engine, "before_cursor_execute", fail_insert)
    assert response.status_code == 200 and response.text.endswith("data: [DONE]\n\n")
    assert failed and "event: completion" not in response.text
    assert await _history(shop.browsers.first, browser) == before_history
    recovered = await _status(shop.browsers.first, browser, view["purchase_id"])
    actual = _ui(recovered.json())
    assert actual["stage"] == "cart" and actual["version"] > view["version"]
    shop.conversation.provider = original_provider
    await _old_proposal_cannot_execute(shop, browser, old_epoch)


async def test_fence_sql_failure_stops_agent_before_any_business_effect(
    shop: ShopHttp, monkeypatch: pytest.MonkeyPatch
) -> None:
    browser = await _boot(shop.browsers.first)
    await _search(shop, shop.browsers.first, browser)
    old_epoch = await _stage_owned_refund(shop, browser, monkeypatch)
    before_history = await _history(shop.browsers.first, browser)
    before_business = await _business(shop)
    planner = NativePlanner("commerce_update_cart_lines", {}, fail_first=True)
    shop.conversation.provider = planner
    engine = shop.browsers.case.maker.kw["bind"].sync_engine
    refused: list[str] = []

    def fail_epoch_update(_connection, _cursor, statement, _parameters, _context, _many):
        if (
            statement.lstrip().upper().startswith("UPDATE CHAT_SESSIONS")
            and "automation_epoch" in statement
        ):
            refused.append("automation epoch UPDATE")
            raise OperationalError("isolated fence", {}, RuntimeError("unavailable"))

    event.listen(engine, "before_cursor_execute", fail_epoch_update)
    try:
        response = await _chat(shop.browsers.first, browser, "购买耳机")
    finally:
        event.remove(engine, "before_cursor_execute", fail_epoch_update)
    assert response.status_code == 503
    assert refused and planner.calls == 0
    assert await _epoch(shop, browser) == old_epoch
    assert await _business(shop) == before_business
    assert await _history(shop.browsers.first, browser) == before_history


async def test_policy_detour_does_not_fence_or_change_purchase(
    shop: ShopHttp, monkeypatch: pytest.MonkeyPatch
) -> None:
    browser = await _boot(shop.browsers.first)
    await _search(shop, shop.browsers.first, browser)
    old_epoch = await _stage_owned_refund(shop, browser, monkeypatch)
    before_business = await _business(shop)
    policy = await _chat(
        shop.browsers.first, browser, "退货政策是什么？", shop=shop, route="support"
    )
    assert policy.status_code == 200 and policy.json().get("ui") is None
    assert await _epoch(shop, browser) == old_epoch
    assert await _business(shop) == before_business


async def test_same_rag_task_can_still_confirm_owned_refund(
    shop: ShopHttp, monkeypatch: pytest.MonkeyPatch
) -> None:
    browser = await _boot(shop.browsers.first)
    await _search(shop, shop.browsers.first, browser)
    old_epoch = await _stage_owned_refund(shop, browser, monkeypatch)
    before = len(shop.browsers.case.calls.refunds)
    response = await _chat(shop.browsers.first, browser, "yes", shop=shop, route="support")
    assert response.status_code == 200
    assert len(shop.browsers.case.calls.refunds) == before + 1
    assert await _epoch(shop, browser) == old_epoch


async def test_shopping_agent_writes_are_bounded_to_one_operation_generation(
    shop: ShopHttp,
) -> None:
    browser = await _boot(shop.browsers.first)
    view = await _search(shop, shop.browsers.first, browser)
    shop.conversation.provider = NativePlanner(
        "commerce_update_cart_lines",
        {
            "purchase_id": view["purchase_id"],
            "lines": [{"sku_id": "ANC-02", "quantity": 2}],
            "expected_version": view["version"],
        },
        fail_after_tool=False,
    )
    response = await _chat(shop.browsers.first, browser, "白色两个")
    assert response.status_code == 200
    async with shop.browsers.case.maker() as db:
        operations = await db.scalar(
            select(func.count())
            .select_from(PurchaseOperation)
            .where(PurchaseOperation.kind == "update_cart")
        )
        objects = await db.scalar(
            select(func.count())
            .select_from(MockCommerceObject)
            .where(MockCommerceObject.kind == "cart")
        )
        assert operations == 1 and objects == 1
