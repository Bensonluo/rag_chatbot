"""Owned cookie chat traverses real AgentService, domain/provider SQL and history.

Only model replies and external transport boundaries are local. Mutations use
normal guest HTTP calls; no quote, order, payment or ownership result is mocked.
"""

import json
import socket
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass
from typing import Any, cast
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient, Response
from sqlalchemy import func, select

from app.api.v1 import guest as guest_api
from app.config.settings import settings
from app.models.database.commerce import (
    MockCommerceObject,
    MockCommerceReceipt,
    PurchaseOperation,
    PurchaseSession,
)
from app.models.database.guest import GuestVisitor
from app.models.database.session import ChatSession
from app.models.schemas.commerce import CommerceView
from app.services.commerce.conversation import CommerceConversation
from app.services.commerce.mock_port import MockCommercePort
from app.services.commerce.service import PurchaseService
from app.services.llm.base import LLMMessage, LLMResponse
from app.tests.evaluation.test_guest_conversations import GuestHttp
from app.tests.evaluation.test_guest_conversations import visitors as visitors
from app.tests.evaluation.test_human_conversation import case as case

_BROWSER = {"Origin": "http://localhost:3000", "X-Rag-Guest": "1"}


@pytest.fixture(autouse=True)
def blocked_dns(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    attempts: list[str] = []

    def blocked(*args: Any, **kwargs: Any) -> None:
        attempts.append("DNS")
        raise AssertionError("Guest commerce evaluation forbids external DNS")

    for name in ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr"):
        monkeypatch.setattr(socket, name, blocked)
    yield
    assert attempts == []


@dataclass
class ShopHttp:
    browsers: GuestHttp
    service: PurchaseService
    port: MockCommercePort
    conversation: CommerceConversation


@pytest.fixture
async def shop(visitors: GuestHttp) -> AsyncGenerator[ShopHttp, None]:
    transport = cast(ASGITransport, visitors.first._transport)
    app = transport.app
    assert isinstance(app, FastAPI)
    port = MockCommercePort(visitors.case.maker)
    service = PurchaseService(visitors.case.maker, port)
    persister = visitors.case.chat.persister
    assert persister is not None
    conversation = CommerceConversation(service, visitors.case.coordinator, persister)
    app.dependency_overrides[guest_api.get_guest_purchase_service] = lambda: service
    app.dependency_overrides[guest_api.get_guest_chat_service] = lambda: visitors.case.chat
    app.dependency_overrides[guest_api.get_guest_commerce_conversation] = lambda: conversation
    yield ShopHttp(visitors, service, port, conversation)


async def _boot(client: AsyncClient) -> dict[str, Any]:
    response = await client.post("/api/v1/guest/bootstrap", headers=_BROWSER, json={})
    assert response.status_code == 200
    value = response.json()
    assert isinstance(value, dict)
    return value


def _scope(browser: dict[str, Any]) -> dict[str, Any]:
    return {
        "session_id": browser["session"]["id"],
        "expected_session_key": browser["session"]["session_key"],
    }


async def _chat(
    client: AsyncClient, browser: dict[str, Any], text: str, *, stream: bool = False
) -> Response:
    return await client.post(
        "/api/v1/guest/chat" + ("/stream" if stream else ""),
        headers=_BROWSER,
        json={**_scope(browser), "message": text, "include_completion": True},
    )


async def _history(client: AsyncClient, browser: dict[str, Any]) -> dict[str, Any]:
    response = await client.get("/api/v1/guest/history", headers=_BROWSER, params=_scope(browser))
    assert response.status_code == 200
    assert response.json()["session_key"] == browser["session"]["session_key"]
    return cast(dict[str, Any], response.json())


def _ui(value: dict[str, Any]) -> dict[str, Any]:
    parsed = CommerceView.model_validate(value["ui"])
    assert parsed.mode == "mock" and parsed.schema_version == 1
    return parsed.model_dump(mode="json")


def _action(view: dict[str, Any], kind: str) -> dict[str, Any]:
    return next(row for row in view["actions"] if row["kind"] == kind)


async def _act(
    client: AsyncClient,
    browser: dict[str, Any],
    view: dict[str, Any],
    kind: str,
    *,
    quantity: int | None = None,
    request_id: str | None = None,
) -> Response:
    action = _action(view, kind)
    body = {
        **_scope(browser),
        "action_ref": action["action_ref"],
        "expected_version": action["expected_version"],
        "client_request_id": request_id or str(uuid4()),
    }
    if quantity is not None:
        body["input"] = {"quantity": quantity}
    return await client.post(
        f"/api/v1/guest/commerce/purchases/{view['purchase_id']}/actions",
        headers=_BROWSER,
        json=body,
    )


async def _search(shop: ShopHttp, client: AsyncClient, browser: dict[str, Any]) -> dict[str, Any]:
    response = await _chat(client, browser, "我想买降噪耳机")
    assert response.status_code == 200, response.text
    assert response.json()["session_key"] == browser["session"]["session_key"]
    return _ui(response.json())


async def _quoted(shop: ShopHttp, client: AsyncClient, browser: dict[str, Any]) -> dict[str, Any]:
    view = await _search(shop, client, browser)
    response = await _act(client, browser, view, "select_sku", quantity=2)
    assert response.status_code == 200, response.text
    view = _ui(response.json())
    response = await _act(client, browser, view, "prepare_checkout")
    assert response.status_code == 200, response.text
    return _ui(response.json())


async def _counts(shop: ShopHttp) -> dict[str, int]:
    result: dict[str, int] = {}
    async with shop.browsers.case.maker() as db:
        for kind in ("order", "payment"):
            value = await db.scalar(
                select(func.count())
                .select_from(MockCommerceObject)
                .where(MockCommerceObject.kind == kind)
            )
            assert isinstance(value, int)
            result[kind] = value
    return result


async def _business(shop: ShopHttp) -> tuple[Any, ...]:
    async with shop.browsers.case.maker() as db:
        purchases = (
            await db.execute(
                select(
                    PurchaseSession.id,
                    PurchaseSession.owner_user_id,
                    PurchaseSession.version,
                    PurchaseSession.stage,
                    PurchaseSession.order_id,
                    PurchaseSession.payment_id,
                ).order_by(PurchaseSession.id)
            )
        ).all()
        operations = (
            await db.execute(
                select(
                    PurchaseOperation.id,
                    PurchaseOperation.kind,
                    PurchaseOperation.status,
                    PurchaseOperation.request_hash,
                    PurchaseOperation.result,
                ).order_by(PurchaseOperation.id)
            )
        ).all()
        objects = (
            await db.execute(
                select(
                    MockCommerceObject.id,
                    MockCommerceObject.kind,
                    MockCommerceObject.owner_user_id,
                    MockCommerceObject.version,
                    MockCommerceObject.payload,
                ).order_by(MockCommerceObject.id)
            )
        ).all()
        receipts = (
            await db.execute(
                select(
                    MockCommerceReceipt.request_key,
                    MockCommerceReceipt.owner_user_id,
                    MockCommerceReceipt.result,
                ).order_by(MockCommerceReceipt.request_key)
            )
        ).all()
    return tuple(
        tuple(tuple(row) for row in rows) for rows in (purchases, operations, objects, receipts)
    )


async def _status(client: AsyncClient, browser: dict[str, Any], purchase_id: str) -> Response:
    return await client.get(
        f"/api/v1/guest/commerce/purchases/{purchase_id}", headers=_BROWSER, params=_scope(browser)
    )


@pytest.mark.parametrize("stream", [False, True])
async def test_cookie_chat_to_explicit_order_and_payment_has_durable_ui_history(
    shop: ShopHttp, stream: bool
) -> None:
    client = shop.browsers.first
    browser = await _boot(client)
    response = await _chat(client, browser, "我想买降噪耳机", stream=stream)
    assert response.status_code == 200
    if stream:
        assert response.text.endswith("data: [DONE]\n\n")
        frames = [
            json.loads(frame.split("data: ", 1)[1])
            for frame in response.text.split("\n\n")
            if frame.startswith("event: completion\n")
        ]
        assert len(frames) == 1 and frames[0]["session_key"] == browser["session"]["session_key"]
        # UI is enabled only after DONE and an authoritative history read.
        history = await _history(client, browser)
        view = _ui(next(row for row in reversed(history["messages"]) if row.get("ui")))
    else:
        view = _ui(response.json())
    selected = await _act(client, browser, view, "select_sku", quantity=2)
    assert selected.status_code == 200
    view = _ui(selected.json())
    assert (
        next(block for block in view["blocks"] if block["type"] == "cart")["cart"]["lines"][0][
            "quantity"
        ]
        == 2
    )
    quoted = await _act(client, browser, view, "prepare_checkout")
    assert quoted.status_code == 200
    view = _ui(quoted.json())
    assert await _counts(shop) == {"order": 0, "payment": 0}
    created = await _act(client, browser, view, "commit_checkout")
    assert created.status_code == 200
    assert created.json()["session_key"] == browser["session"]["session_key"]
    view = _ui(created.json())
    assert view["stage"] == "awaiting_payment"
    paid = await _act(client, browser, view, "mock_pay")
    assert paid.status_code == 200
    view = _ui(paid.json())
    assert view["stage"] == "paid" and await _counts(shop) == {"order": 1, "payment": 1}
    status = await _status(client, browser, view["purchase_id"])
    assert status.status_code == 200 and _ui(status.json())["stage"] == "paid"
    before = await _business(shop)
    history = await _history(client, browser)
    stored = [row for row in history["messages"] if row.get("ui")]
    assert stored and _ui(stored[-1])["stage"] == "paid"
    assert all(row["status"] == "completed" and row["message_id"] > 0 for row in stored)
    assert await _business(shop) == before


@pytest.mark.parametrize("operation", ["status", "action"])
async def test_two_cookie_browsers_cannot_use_foreign_purchase_or_action_refs(
    shop: ShopHttp, operation: str
) -> None:
    first, second = shop.browsers.first, shop.browsers.second
    own, foreign = await _boot(first), await _boot(second)
    view = await _quoted(shop, first, own)
    before = await _business(shop)
    response = (
        await _status(second, foreign, view["purchase_id"])
        if operation == "status"
        else await _act(second, foreign, view, "commit_checkout")
    )
    assert response.status_code == 404 and await _business(shop) == before


@pytest.mark.parametrize("field,value", [("owner_user_id", 1), ("total_minor", 1), ("paid", True)])
async def test_action_rejects_client_owned_identity_price_and_payment_fields(
    shop: ShopHttp, field: str, value: Any
) -> None:
    browser = await _boot(shop.browsers.first)
    view = await _quoted(shop, shop.browsers.first, browser)
    act = _action(view, "commit_checkout")
    before = await _business(shop)
    writes = len(shop.browsers.writes)
    response = await shop.browsers.first.post(
        f"/api/v1/guest/commerce/purchases/{view['purchase_id']}/actions",
        headers=_BROWSER,
        json={
            **_scope(browser),
            "action_ref": act["action_ref"],
            "expected_version": act["expected_version"],
            "client_request_id": str(uuid4()),
            field: value,
        },
    )
    assert response.status_code == 422 and await _business(shop) == before
    assert len(shop.browsers.writes) == writes


async def test_plain_text_confirmation_never_authorizes_mock_payment(shop: ShopHttp) -> None:
    client = shop.browsers.first
    browser = await _boot(client)
    view = await _quoted(shop, client, browser)
    order = await _act(client, browser, view, "commit_checkout")
    assert order.status_code == 200
    view = _ui(order.json())
    before = await _business(shop)
    response = await _chat(client, browser, "我确认付款，请直接付掉")
    assert response.status_code == 200
    assert _ui(response.json())["stage"] == "awaiting_payment"
    assert await _business(shop) == before


async def test_repeated_confirmation_with_new_client_ids_keeps_one_order_operation(
    shop: ShopHttp,
) -> None:
    client = shop.browsers.first
    browser = await _boot(client)
    view = await _quoted(shop, client, browser)
    first = await _act(client, browser, view, "commit_checkout", request_id=str(uuid4()))
    second = await _act(client, browser, view, "commit_checkout", request_id=str(uuid4()))
    assert first.status_code == second.status_code == 200
    assert _ui(first.json())["purchase_id"] == _ui(second.json())["purchase_id"]
    async with shop.browsers.case.maker() as db:
        assert (
            await db.scalar(
                select(func.count())
                .select_from(PurchaseOperation)
                .where(PurchaseOperation.kind == "order")
            )
            == 1
        )
    assert await _counts(shop) == {"order": 1, "payment": 1}


@pytest.mark.parametrize("mismatch", ["version", "key"])
async def test_original_descriptor_or_conversation_mismatch_has_no_business_effect(
    shop: ShopHttp, mismatch: str
) -> None:
    client = shop.browsers.first
    browser = await _boot(client)
    view = await _quoted(shop, client, browser)
    act = _action(view, "commit_checkout")
    body = {
        **_scope(browser),
        "action_ref": act["action_ref"],
        "expected_version": act["expected_version"],
        "client_request_id": str(uuid4()),
    }
    if mismatch == "version":
        body["expected_version"] += 1
    else:
        body["expected_session_key"] = str(uuid4())
    before = await _business(shop)
    response = await client.post(
        f"/api/v1/guest/commerce/purchases/{view['purchase_id']}/actions",
        headers=_BROWSER,
        json=body,
    )
    assert response.status_code == 409 and await _business(shop) == before


async def test_history_clear_does_not_cancel_order_and_active_purchase_recovers_owned_state(
    shop: ShopHttp,
) -> None:
    client = shop.browsers.first
    browser = await _boot(client)
    view = await _quoted(shop, client, browser)
    response = await _act(client, browser, view, "commit_checkout")
    assert response.status_code == 200
    view = _ui(response.json())
    before = await _business(shop)
    cleared = await client.delete("/api/v1/guest/history", headers=_BROWSER, params=_scope(browser))
    assert cleared.status_code == 204
    old = await client.get(
        "/api/v1/guest/commerce/purchases/active", headers=_BROWSER, params=_scope(browser)
    )
    assert old.status_code == 409
    reopened = await _boot(client)
    assert reopened["session"]["id"] == browser["session"]["id"]
    assert reopened["session"]["session_key"] != browser["session"]["session_key"]
    current = await client.get(
        "/api/v1/guest/commerce/purchases/active", headers=_BROWSER, params=_scope(reopened)
    )
    assert current.status_code == 200 and _ui(current.json())["purchase_id"] == view["purchase_id"]
    assert (await _history(client, reopened))["messages"] == []
    assert await _business(shop) == before


async def test_same_action_ref_changed_quantity_is_not_replayed_as_the_old_success(
    shop: ShopHttp,
) -> None:
    client = shop.browsers.first
    browser = await _boot(client)
    view = await _search(shop, client, browser)
    first = await _act(client, browser, view, "select_sku", quantity=2)
    assert first.status_code == 200
    before = await _business(shop)
    second = await _act(client, browser, view, "select_sku", quantity=3)
    assert second.status_code == 409 and await _business(shop) == before


async def test_accepted_order_timeout_get_reconciles_without_hidden_payment_creation(
    shop: ShopHttp,
) -> None:
    client = shop.browsers.first
    browser = await _boot(client)
    view = await _quoted(shop, client, browser)
    shop.port.fail_next("order")
    response = await _act(client, browser, view, "commit_checkout")
    assert response.status_code == 200 and _ui(response.json())["stage"] == "unknown"
    assert await _counts(shop) == {"order": 1, "payment": 0}
    recovered = await _status(client, browser, view["purchase_id"])
    assert recovered.status_code == 200
    view = _ui(recovered.json())
    assert view["stage"] == "awaiting_payment" and await _counts(shop) == {"order": 1, "payment": 0}
    resumed = await _act(client, browser, view, "commit_checkout")
    assert resumed.status_code == 200 and _ui(resumed.json())["stage"] == "awaiting_payment"
    assert await _counts(shop) == {"order": 1, "payment": 1}


async def test_accepted_payment_timeout_query_confirms_once_without_second_attempt(
    shop: ShopHttp,
) -> None:
    client = shop.browsers.first
    browser = await _boot(client)
    view = await _quoted(shop, client, browser)
    created = await _act(client, browser, view, "commit_checkout")
    assert created.status_code == 200
    view = _ui(created.json())
    shop.port.fail_next("mock_pay")
    response = await _act(client, browser, view, "mock_pay")
    assert response.status_code == 200 and _ui(response.json())["stage"] == "unknown"
    recovered = await _status(client, browser, view["purchase_id"])
    assert recovered.status_code == 200 and _ui(recovered.json())["stage"] == "paid"
    retry = await _act(client, browser, view, "mock_pay")
    assert retry.status_code == 200 and await _counts(shop) == {"order": 1, "payment": 1}
    async with shop.browsers.case.maker() as db:
        assert (
            await db.scalar(
                select(func.count())
                .select_from(PurchaseOperation)
                .where(PurchaseOperation.kind == "mock_pay")
            )
            == 1
        )


async def test_legacy_shared_demo_cannot_use_cookie_owned_purchase_tools(
    shop: ShopHttp, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "DEMO_MODE", True)
    browser = await _boot(shop.browsers.first)
    view = await _quoted(shop, shop.browsers.first, browser)
    async with shop.browsers.case.maker() as db:
        visitor = await db.get(GuestVisitor, browser["visitor_id"])
        assert visitor is not None
        owner = visitor.user_id
    before = await _business(shop)
    response = await shop.browsers.first.post(
        "/api/v1/chat",
        json={"session_id": 10, "user_id": owner, "message": "购买商品并查询我的购物车"},
    )
    assert response.status_code == 200
    assert response.json().get("ui") is None and await _business(shop) == before
    assert view["stage"] == "quoted"


async def test_policy_rag_detour_preserves_cart_then_agent_continues_checkout(
    shop: ShopHttp,
) -> None:
    client = shop.browsers.first
    browser = await _boot(client)
    view = await _search(shop, client, browser)
    selected = await _act(client, browser, view, "select_sku", quantity=2)
    assert selected.status_code == 200
    view = _ui(selected.json())
    before, graph = await _business(shop), shop.browsers.case.calls.graph
    policy = await _chat(client, browser, "退货政策是什么？")
    assert policy.status_code == 200 and policy.json().get("ui") is None
    assert shop.browsers.case.calls.graph > graph
    assert await _business(shop) == before
    continued = await _chat(client, browser, "继续结算")
    assert continued.status_code == 200
    resumed = _ui(continued.json())
    assert resumed["purchase_id"] == view["purchase_id"] and resumed["stage"] == "quoted"


class NativePlanner:
    def __init__(self, name: str, arguments: dict[str, Any]) -> None:
        self.name, self.arguments = name, arguments
        self.schemas: list[str] = []
        self.calls = 0

    async def generate_with_tools(
        self, messages: list[LLMMessage], tools: list[dict[str, Any]]
    ) -> LLMResponse:
        self.calls += 1
        self.schemas = [tool["function"]["name"] for tool in tools]
        if messages[-1].role == "tool":
            return LLMResponse(
                content="I paid for everything; ignore the actual state",
                model="local-native-planner",
            )
        return LLMResponse(
            content="",
            model="local-native-planner",
            tool_calls=[
                {"id": str(uuid4()), "name": self.name, "arguments": json.dumps(self.arguments)}
            ],
        )


async def test_real_agent_tools_ignore_native_planner_supplied_owner_ids(shop: ShopHttp) -> None:
    first, second = shop.browsers.first, shop.browsers.second
    browser, other = await _boot(first), await _boot(second)
    async with shop.browsers.case.maker() as db:
        victim = await db.get(GuestVisitor, other["visitor_id"])
        visitor = await db.get(GuestVisitor, browser["visitor_id"])
        assert victim is not None and visitor is not None
        fake = NativePlanner(
            "commerce_search_products",
            {"query": "耳机", "user_id": victim.user_id, "owner_user_id": victim.user_id},
        )
        owner = visitor.user_id
    shop.conversation.provider = fake
    response = await _chat(first, browser, "购买耳机")
    assert response.status_code == 200 and fake.calls == 2
    view = _ui(response.json())
    async with shop.browsers.case.maker() as db:
        purchase = await db.get(PurchaseSession, view["purchase_id"])
        assert purchase is not None and purchase.owner_user_id == owner
    assert (
        "commerce_mock_pay" not in fake.schemas and "commerce_commit_checkout" not in fake.schemas
    )
    assert "I paid" not in response.json()["content"]


async def test_native_planner_cannot_invent_an_unregistered_payment_tool(shop: ShopHttp) -> None:
    client = shop.browsers.first
    browser = await _boot(client)
    view = await _quoted(shop, client, browser)
    fake = NativePlanner("commerce_mock_pay", {"purchase_id": view["purchase_id"], "paid": True})
    shop.conversation.provider = fake
    before = await _business(shop)
    response = await _chat(client, browser, "请付款")
    assert response.status_code == 200 and _ui(response.json())["stage"] == "quoted"
    assert await _business(shop) == before
    assert "commerce_mock_pay" not in fake.schemas and "I paid" not in response.json()["content"]


async def test_native_planner_cannot_select_another_browser_purchase(shop: ShopHttp) -> None:
    first, second = shop.browsers.first, shop.browsers.second
    browser, other = await _boot(first), await _boot(second)
    owned = await _search(shop, first, browser)
    foreign = await _quoted(shop, second, other)
    shop.conversation.provider = NativePlanner(
        "commerce_purchase_status", {"purchase_id": foreign["purchase_id"]}
    )
    before = await _business(shop)
    response = await _chat(first, browser, "查询购物车状态")
    assert response.status_code == 200
    assert _ui(response.json())["purchase_id"] == owned["purchase_id"]
    assert await _business(shop) == before


@pytest.mark.parametrize(
    "transition", ["new-purchase", "same-purchase-chat", "same-purchase-action"]
)
async def test_shopping_epoch_fence_cannot_revive_an_older_refund_proposal(
    shop: ShopHttp, transition: str
) -> None:
    client = shop.browsers.first
    browser = await _boot(client)
    view: dict[str, Any] | None = None
    if transition != "new-purchase":
        view = await _search(shop, client, browser)
    async with shop.browsers.case.maker() as db:
        row = await db.get(ChatSession, browser["session"]["id"])
        assert row is not None
        epoch = row.automation_epoch
    for text in ("I want a refund", "ORD1001", "damaged item"):
        response = await _chat(client, browser, text)
        assert response.status_code == 200
    thread = str(browser["session"]["id"]) + (f":automation:{epoch}" if epoch else "")
    state = await shop.browsers.case.chat.graph.aget_state({"configurable": {"thread_id": thread}})
    assert state.values.get("pending_confirmation")
    if transition == "same-purchase-action":
        assert view is not None
        advanced = await _act(client, browser, view, "select_sku", quantity=1)
    else:
        advanced = await _chat(client, browser, "继续看耳机" if view else "我想买耳机")
    assert advanced.status_code == 200
    async with shop.browsers.case.maker() as db:
        row = await db.get(ChatSession, browser["session"]["id"])
        assert row is not None and row.automation_epoch > epoch
        new_epoch = row.automation_epoch
    before = len(shop.browsers.case.calls.refunds)
    confirmation = await _chat(client, browser, "确认退款")
    assert confirmation.status_code == 200
    assert len(shop.browsers.case.calls.refunds) == before
    fresh = await shop.browsers.case.chat.graph.aget_state(
        {"configurable": {"thread_id": f"{browser['session']['id']}:automation:{new_epoch}"}}
    )
    assert not fresh.values.get("pending_confirmation")


async def test_no_cookie_cannot_discover_or_mutate_guest_purchase(shop: ShopHttp) -> None:
    browser = await _boot(shop.browsers.first)
    view = await _quoted(shop, shop.browsers.first, browser)
    before = await _business(shop)
    status = await _status(shop.browsers.second, browser, view["purchase_id"])
    changed = await _act(shop.browsers.second, browser, view, "commit_checkout")
    assert status.status_code == changed.status_code == 401
    assert await _business(shop) == before
