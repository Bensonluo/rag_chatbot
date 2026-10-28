"""An explicit public shopping task fences old proposals despite later failures.

Real Cookie HTTP, SQL domain/provider, registry and AgentService are reused.
Only native model replies and a timing-local Message INSERT fault are controlled.
The purchase loop itself advances only through /guest/chat text — an order is
created by the explicit quoted-stage phrase after the state is displayed again.
"""

import json
from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy import event, select
from sqlalchemy.exc import OperationalError

from app.models.database.commerce import MockCommerceObject, PurchaseOperation, PurchaseSession
from app.models.database.guest import GuestVisitor
from app.models.database.session import ChatSession
from app.services.dialogue.tools import MOCK_ORDERS
from app.services.llm.base import LLMMessage, LLMResponse
from app.tests.evaluation.test_guest_commerce_flow import (
    ShopHttp,
    _boot,
    _chat,
    _counts,
    _history,
    _search,
    _status,
    _ui,
)
from app.tests.evaluation.test_guest_commerce_flow import shop as shop
from app.tests.evaluation.test_guest_conversations import no_external_dns as no_external_dns
from app.tests.evaluation.test_guest_conversations import visitors as visitors
from app.tests.evaluation.test_human_conversation import case as case


class NativePlanner:
    def __init__(self, tool: str, arguments: dict[str, Any], *, fail_first: bool = False) -> None:
        self.tool = tool
        self.arguments = arguments
        self.fail_first = fail_first
        self.calls = 0

    async def generate_with_tools(
        self, messages: list[LLMMessage], tools: list[dict[str, Any]]
    ) -> LLMResponse:
        self.calls += 1
        if self.fail_first or messages[-1].role == "tool":
            raise RuntimeError("isolated-native-shopping-planner-failure")
        return LLMResponse(
            content="",
            model="local-shopping-fault-planner",
            tool_calls=[
                {"id": str(uuid4()), "name": self.tool, "arguments": json.dumps(self.arguments)}
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
    async with shop.browsers.case.maker() as db:
        visitor = await db.get(GuestVisitor, browser["visitor_id"])
        assert visitor is not None
        # External-order fixture data belongs to this real visitor. Authorization
        # and refund execution stay real; an unrelated owner's denial cannot mask
        # accidental execution of this legitimately staged proposal.
        order = {**MOCK_ORDERS["ORD1001"], "user_id": visitor.user_id}
    monkeypatch.setitem(MOCK_ORDERS, "ORD1001", order)
    epoch = await _epoch(shop, browser)
    for text in ("I want a refund", "ORD1001", "damaged item"):
        assert (await _chat(shop.browsers.first, browser, text)).status_code == 200
    thread = str(browser["session"]["id"]) + (f":automation:{epoch}" if epoch else "")
    state = await shop.browsers.case.chat.graph.aget_state({"configurable": {"thread_id": thread}})
    assert state.values.get("pending_confirmation")
    return epoch


async def _business(shop: ShopHttp) -> dict[str, Any]:
    async with shop.browsers.case.maker() as db:
        rows = (
            await db.execute(
                select(
                    PurchaseSession.id,
                    PurchaseSession.version,
                    PurchaseSession.stage,
                    PurchaseSession.sku_id,
                    PurchaseSession.quantity,
                    PurchaseSession.cart_snapshot,
                    PurchaseSession.quote_snapshot,
                    PurchaseSession.order_id,
                    PurchaseSession.payment_id,
                    PurchaseSession.active_operation_id,
                ).order_by(PurchaseSession.id)
            )
        ).all()
        operations = (
            await db.execute(
                select(
                    PurchaseOperation.kind, PurchaseOperation.status, PurchaseOperation.result
                ).order_by(PurchaseOperation.id)
            )
        ).all()
        provider = (
            await db.execute(
                select(
                    MockCommerceObject.id, MockCommerceObject.kind, MockCommerceObject.payload
                ).order_by(MockCommerceObject.id)
            )
        ).all()
    return {
        "purchases": tuple(tuple(row) for row in rows),
        "operations": tuple(tuple(row) for row in operations),
        "provider": tuple(tuple(row) for row in provider),
    }


async def _assert_old_proposal_cannot_execute(
    shop: ShopHttp, browser: dict[str, Any], old_epoch: int
) -> None:
    epoch = await _epoch(shop, browser)
    assert epoch == old_epoch + 1
    before = len(shop.browsers.case.calls.refunds)
    # "yes" reaches the real RAG meta-intent instead of being intercepted as a
    # shopping status request. It must not approve a prior task's proposal.
    assert (await _chat(shop.browsers.first, browser, "yes")).status_code == 200
    assert len(shop.browsers.case.calls.refunds) == before
    thread = str(browser["session"]["id"]) + f":automation:{epoch}"
    state = await shop.browsers.case.chat.graph.aget_state({"configurable": {"thread_id": thread}})
    assert not state.values.get("pending_confirmation")


@pytest.mark.parametrize("kind", ["cart", "quote"])
async def test_successful_shopping_tool_then_native_failure_preserves_fence_and_recoverable_facts(
    shop: ShopHttp, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    browser = await _boot(shop.browsers.first)
    view = await _search(shop, shop.browsers.first, browser)
    selected = await _chat(shop.browsers.first, browser, "黑色2个")
    assert selected.status_code == 200, selected.text
    view = _ui(selected.json())
    assert view["stage"] == "cart"
    old_epoch = await _stage_owned_refund(shop, browser, monkeypatch)
    before_history = await _history(shop.browsers.first, browser)
    planner = NativePlanner(
        "commerce_set_cart_line" if kind == "cart" else "commerce_prepare_checkout",
        {
            "purchase_id": view["purchase_id"],
            "expected_version": view["version"],
            **({"sku_id": "ANC-BLACK", "quantity": 2} if kind == "cart" else {}),
        },
    )
    shop.conversation.provider = planner
    response = await _chat(
        shop.browsers.first, browser, "我选黑色2个" if kind == "cart" else "继续结算", stream=True
    )
    assert response.status_code == 200
    assert response.text == "data: [STREAM_ERROR]\n\ndata: [DONE]\n\n"
    assert planner.calls == 2
    assert await _history(shop.browsers.first, browser) == before_history
    recovered = await _status(shop.browsers.first, browser, view["purchase_id"])
    assert recovered.status_code == 200
    actual = _ui(recovered.json())
    assert actual["stage"] == ("cart" if kind == "cart" else "quoted")
    assert actual["version"] > view["version"] and actual["purchase_id"] == view["purchase_id"]
    cart = next(block for block in actual["blocks"] if block["type"] == "cart")
    assert (cart["cart"]["lines"][0]["sku_id"], cart["cart"]["lines"][0]["quantity"]) == (
        "ANC-BLACK",
        2,
    )
    assert await _counts(shop) == {"order": 0, "payment": 0}
    await _assert_old_proposal_cannot_execute(shop, browser, old_epoch)
    if kind == "quote":
        # A status read only exposes the known quote. Re-displaying it by text
        # and confirming explicitly still creates exactly one order/payment
        # entry after the model failed mid-turn.
        shop.conversation.provider = None
        redisplayed = await _chat(shop.browsers.first, browser, "查询订单")
        assert redisplayed.status_code == 200, redisplayed.text
        assert _ui(redisplayed.json())["stage"] == "quoted"
        accepted = await _chat(shop.browsers.first, browser, "确认下单")
        assert accepted.status_code == 200, accepted.text
        assert _ui(accepted.json())["stage"] == "awaiting_payment"
        assert await _counts(shop) == {"order": 1, "payment": 1}


async def test_native_first_call_failure_fences_old_proposal_without_business_or_message_effects(
    shop: ShopHttp, monkeypatch: pytest.MonkeyPatch
) -> None:
    browser = await _boot(shop.browsers.first)
    view = await _search(shop, shop.browsers.first, browser)
    old_epoch = await _stage_owned_refund(shop, browser, monkeypatch)
    before_history = await _history(shop.browsers.first, browser)
    before_business = await _business(shop)
    planner = NativePlanner("commerce_set_cart_line", {}, fail_first=True)
    shop.conversation.provider = planner
    response = await _chat(shop.browsers.first, browser, "想买耳机", stream=True)
    assert response.status_code == 200
    assert response.text == "data: [STREAM_ERROR]\n\ndata: [DONE]\n\n"
    assert planner.calls == 1
    assert await _history(shop.browsers.first, browser) == before_history
    assert await _business(shop) == before_business
    assert (await _status(shop.browsers.first, browser, view["purchase_id"])).status_code == 200
    await _assert_old_proposal_cannot_execute(shop, browser, old_epoch)


async def test_message_insert_failure_keeps_the_shopping_fence_and_recoverable_cart(
    shop: ShopHttp, monkeypatch: pytest.MonkeyPatch
) -> None:
    browser = await _boot(shop.browsers.first)
    view = await _search(shop, shop.browsers.first, browser)
    old_epoch = await _stage_owned_refund(shop, browser, monkeypatch)
    before_history = await _history(shop.browsers.first, browser)
    engine = shop.browsers.case.maker.kw["bind"].sync_engine
    failed_inserts: list[str] = []

    def fail_message_insert(_connection, _cursor, statement, _parameters, _context, _many):
        if statement.lstrip().upper().startswith("INSERT INTO MESSAGES"):
            failed_inserts.append("Message INSERT")
            raise OperationalError("isolated Message insert", {}, RuntimeError("write unavailable"))

    event.listen(engine, "before_cursor_execute", fail_message_insert)
    try:
        response = await _chat(shop.browsers.first, browser, "黑色2个", stream=True)
    finally:
        event.remove(engine, "before_cursor_execute", fail_message_insert)
    assert response.status_code == 200 and response.text.endswith("data: [DONE]\n\n")
    assert failed_inserts and "event: completion" not in response.text
    assert "[STREAM_ERROR]" not in response.text
    assert await _history(shop.browsers.first, browser) == before_history
    actual_response = await _status(shop.browsers.first, browser, view["purchase_id"])
    assert actual_response.status_code == 200
    actual = _ui(actual_response.json())
    assert actual["stage"] == "cart" and actual["version"] > view["version"]
    cart = next(block for block in actual["blocks"] if block["type"] == "cart")
    assert (cart["cart"]["lines"][0]["sku_id"], cart["cart"]["lines"][0]["quantity"]) == (
        "ANC-BLACK",
        2,
    )
    assert await _counts(shop) == {"order": 0, "payment": 0}
    await _assert_old_proposal_cannot_execute(shop, browser, old_epoch)


async def test_fence_sql_failure_stops_the_model_and_purchase_before_any_business_effect(
    shop: ShopHttp, monkeypatch: pytest.MonkeyPatch
) -> None:
    browser = await _boot(shop.browsers.first)
    await _search(shop, shop.browsers.first, browser)
    old_epoch = await _stage_owned_refund(shop, browser, monkeypatch)
    before_history = await _history(shop.browsers.first, browser)
    before_business = await _business(shop)
    planner = NativePlanner("commerce_set_cart_line", {}, fail_first=True)
    shop.conversation.provider = planner
    engine = shop.browsers.case.maker.kw["bind"].sync_engine
    refused: list[str] = []

    def fail_epoch_update(_connection, _cursor, statement, _parameters, _context, _many):
        if (
            statement.lstrip().upper().startswith("UPDATE CHAT_SESSIONS")
            and "automation_epoch" in statement
        ):
            refused.append("automation epoch UPDATE")
            raise OperationalError("isolated task fence", {}, RuntimeError("write unavailable"))

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
    assert await _counts(shop) == {"order": 0, "payment": 0}


async def test_policy_detour_does_not_create_a_shopping_task_fence_or_change_the_cart(
    shop: ShopHttp, monkeypatch: pytest.MonkeyPatch
) -> None:
    browser = await _boot(shop.browsers.first)
    await _search(shop, shop.browsers.first, browser)
    old_epoch = await _stage_owned_refund(shop, browser, monkeypatch)
    before_business = await _business(shop)
    policy = await _chat(shop.browsers.first, browser, "退货政策是什么？")
    assert policy.status_code == 200 and policy.json().get("ui") is None
    assert await _epoch(shop, browser) == old_epoch
    assert await _business(shop) == before_business


async def test_same_rag_task_can_still_confirm_its_legitimate_owned_refund(
    shop: ShopHttp, monkeypatch: pytest.MonkeyPatch
) -> None:
    browser = await _boot(shop.browsers.first)
    await _search(shop, shop.browsers.first, browser)
    old_epoch = await _stage_owned_refund(shop, browser, monkeypatch)
    before = len(shop.browsers.case.calls.refunds)
    assert (await _chat(shop.browsers.first, browser, "yes")).status_code == 200
    assert len(shop.browsers.case.calls.refunds) == before + 1
    assert await _epoch(shop, browser) == old_epoch


@pytest.mark.parametrize("invalid", ["foreign", "old-key"])
async def test_rejected_chat_scope_does_not_fence_a_task(
    shop: ShopHttp, monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    browser = await _boot(shop.browsers.first)
    await _search(shop, shop.browsers.first, browser)
    old_epoch = await _stage_owned_refund(shop, browser, monkeypatch)
    client = shop.browsers.first
    scope = browser
    if invalid == "foreign":
        await _boot(shop.browsers.second)
        client = shop.browsers.second
    else:
        scope = {**browser, "session": {**browser["session"], "session_key": str(uuid4())}}
    before_history = await _history(shop.browsers.first, browser)
    before_business = await _business(shop)
    response = await client.post(
        "/api/v1/guest/chat",
        headers={"Origin": "http://localhost:3000", "X-Rag-Guest": "1"},
        json={
            "session_id": scope["session"]["id"],
            "expected_session_key": scope["session"]["session_key"],
            "message": "确认下单",
        },
    )
    assert response.status_code == (404 if invalid == "foreign" else 409)
    assert await _epoch(shop, browser) == old_epoch
    assert await _business(shop) == before_business
    assert await _history(shop.browsers.first, browser) == before_history
