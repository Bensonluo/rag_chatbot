"""Provider-LLM shopping agent contract over real cookie HTTP and SQL mocks.

Tests script model decisions explicitly. They do not encode production phrase
rules: semantic routing, product references, quantities, and confirmation
intent are represented as the model's chosen tool calls.
"""

import json
import socket
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient, Response
from sqlalchemy import func, select

from app.api.v1 import guest as guest_api
from app.models.database.commerce import (
    MockCommerceObject,
    MockCommerceReceipt,
    PurchaseAction,
    PurchaseOperation,
    PurchaseSession,
)
from app.models.schemas.commerce import CommerceView
from app.services.commerce.conversation import CommerceConversation, CommercePlanner
from app.services.commerce.mock_port import MockCommercePort
from app.services.commerce.service import PurchaseService
from app.services.llm.base import LLMMessage, LLMResponse
from app.tests.evaluation.test_guest_conversations import GuestHttp
from app.tests.evaluation.test_guest_conversations import visitors as visitors
from app.tests.evaluation.test_human_conversation import case as case

_BROWSER = {"Origin": "http://localhost:3000", "X-Rag-Guest": "1"}


@pytest.fixture(autouse=True)
def blocked_dns(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None]:
    attempts: list[str] = []

    def blocked(*args: Any, **kwargs: Any) -> None:
        attempts.append("DNS")
        raise AssertionError("Guest commerce evaluation forbids external DNS")

    for name in ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr"):
        monkeypatch.setattr(socket, name, blocked)
    yield
    assert attempts == []


class ShopClock:
    def __init__(self) -> None:
        self.now = datetime.now(UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: int) -> None:
        self.now += timedelta(seconds=seconds)


class ScriptedShoppingLLM:
    """Explicit test-scripted model decisions; production uses GLM tool calls."""

    def __init__(self) -> None:
        self.routes: list[str] = []
        self.plans: list[dict[str, Any] | None] = []

    async def generate(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        del messages, kwargs
        route = self.routes.pop(0) if self.routes else "shopping"
        return LLMResponse(content=json.dumps({"route": route, "reason": "script"}), model="test")

    async def generate_with_tools(
        self, messages: list[LLMMessage], tools: list[dict[str, Any]]
    ) -> LLMResponse:
        del tools
        if messages and messages[-1].role == "tool":
            return LLMResponse(content="已根据外部工具结果处理。", model="test")
        plan = self.plans.pop(0) if self.plans else None
        if plan is None:
            return LLMResponse(content="请补充商品、偏好和数量。", model="test")
        return LLMResponse(
            content="",
            model="test",
            tool_calls=[
                {
                    "id": str(uuid4()),
                    "name": "commerce_" + plan["tool"],
                    "arguments": json.dumps(plan["arguments"], ensure_ascii=False),
                }
            ],
        )


@dataclass
class ShopHttp:
    browsers: GuestHttp
    service: PurchaseService
    port: MockCommercePort
    conversation: CommerceConversation
    clock: ShopClock
    provider: ScriptedShoppingLLM


@pytest.fixture
async def shop(visitors: GuestHttp) -> AsyncGenerator[ShopHttp, None]:
    transport = cast(ASGITransport, visitors.first._transport)
    app = transport.app
    assert isinstance(app, FastAPI)
    clock = ShopClock()
    port = MockCommercePort(visitors.case.maker, clock=clock)
    service = PurchaseService(visitors.case.maker, port, clock=clock)
    persister = visitors.case.chat.persister
    assert persister is not None
    provider = ScriptedShoppingLLM()
    conversation = CommerceConversation(
        service, visitors.case.coordinator, persister, provider=provider
    )
    app.dependency_overrides[guest_api.get_guest_purchase_service] = lambda: service
    app.dependency_overrides[guest_api.get_guest_chat_service] = lambda: visitors.case.chat
    app.dependency_overrides[guest_api.get_guest_commerce_conversation] = lambda: conversation
    try:
        yield ShopHttp(visitors, service, port, conversation, clock, provider)
    finally:
        app.dependency_overrides.pop(guest_api.get_guest_purchase_service, None)
        app.dependency_overrides.pop(guest_api.get_guest_chat_service, None)
        app.dependency_overrides.pop(guest_api.get_guest_commerce_conversation, None)


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
    client: AsyncClient,
    browser: dict[str, Any],
    text: str,
    *,
    stream: bool = False,
    shop: ShopHttp | None = None,
    plan: dict[str, Any] | None = None,
    route: str = "shopping",
) -> Response:
    if shop is not None:
        shop.provider.routes.append(route)
        if route == "shopping":
            shop.provider.plans.append(plan)
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


def _stream_parse(body: str) -> tuple[str, list[dict[str, Any]]]:
    chunks: list[str] = []
    completions: list[dict[str, Any]] = []
    for frame in filter(None, body.split("\n\n")):
        if frame.startswith("event: completion"):
            for line in frame.split("\n"):
                if line.startswith("data: "):
                    completions.append(json.loads(line.split("data: ", 1)[1]))
        elif frame.startswith("data: ["):
            continue
        elif frame.startswith("data: "):
            chunks.append(
                "\n".join(
                    line.split("data: ", 1)[1]
                    for line in frame.split("\n")
                    if line.startswith("data: ")
                )
            )
    return "\n".join(chunks), completions


async def _search(shop: ShopHttp, client: AsyncClient, browser: dict[str, Any]) -> dict[str, Any]:
    response = await _chat(
        client,
        browser,
        "我想买降噪耳机",
        shop=shop,
        plan={"tool": "search_products", "arguments": {"query": "降噪耳机"}},
    )
    assert response.status_code == 200, response.text
    return _ui(response.json())


async def _quoted(shop: ShopHttp, client: AsyncClient, browser: dict[str, Any]) -> dict[str, Any]:
    view = await _search(shop, client, browser)
    selected = await _chat(
        client,
        browser,
        "第二款黑色，两副",
        shop=shop,
        plan={
            "tool": "set_cart_line",
            "arguments": {
                "purchase_id": view["purchase_id"],
                "sku_id": "ANC-BLACK",
                "quantity": 2,
                "expected_version": view["version"],
            },
        },
    )
    assert selected.status_code == 200, selected.text
    view = _ui(selected.json())
    line = next(block for block in view["blocks"] if block["type"] == "cart")["cart"]["lines"][0]
    assert (line["sku_id"], line["quantity"]) == ("ANC-BLACK", 2)
    checkout = await _chat(
        client,
        browser,
        "结算",
        shop=shop,
        plan={
            "tool": "prepare_checkout",
            "arguments": {
                "purchase_id": view["purchase_id"],
                "expected_version": view["version"],
            },
        },
    )
    assert checkout.status_code == 200, checkout.text
    view = _ui(checkout.json())
    assert view["stage"] == "quoted"
    return view


async def _order(
    shop: ShopHttp, client: AsyncClient, browser: dict[str, Any], view: dict[str, Any]
) -> dict[str, Any]:
    ordered = await _chat(
        client,
        browser,
        "确认下单",
        shop=shop,
        plan={"tool": "commit_checkout", "arguments": {}},
    )
    assert ordered.status_code == 200, ordered.text
    return _ui(ordered.json())


async def _pay(
    shop: ShopHttp, client: AsyncClient, browser: dict[str, Any], view: dict[str, Any]
) -> dict[str, Any]:
    del view
    paid = await _chat(
        client,
        browser,
        "确认模拟付款",
        shop=shop,
        plan={"tool": "mock_pay", "arguments": {}},
    )
    assert paid.status_code == 200, paid.text
    return _ui(paid.json())


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


async def _operations(shop: ShopHttp, kind: str | None = None) -> int:
    async with shop.browsers.case.maker() as db:
        statement = select(func.count()).select_from(PurchaseOperation)
        if kind is not None:
            statement = statement.where(PurchaseOperation.kind == kind)
        value = await db.scalar(statement)
        assert isinstance(value, int)
        return value


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
async def test_provider_agent_completes_purchase_with_tools_only(
    shop: ShopHttp, stream: bool
) -> None:
    client = shop.browsers.first
    browser = await _boot(client)
    view = await _quoted(shop, client, browser)
    assert (
        "总价：¥598.00"
        in (
            await _chat(
                client,
                browser,
                "状态",
                shop=shop,
                plan={
                    "tool": "purchase_status",
                    "arguments": {"purchase_id": view["purchase_id"]},
                },
            )
        ).json()["content"]
        if not stream
        else True
    )

    ordered = await _order(shop, client, browser, view)
    assert ordered["stage"] == "awaiting_payment"
    order_id = (
        ordered["order_id"]
        if "order_id" in ordered
        else next(block for block in ordered["blocks"] if block["type"] == "payment")["order_id"]
    )
    paid = await _pay(shop, client, browser, ordered)
    assert paid["stage"] == "paid"
    assert await _counts(shop) == {"order": 1, "payment": 1}
    assert order_id


async def test_llm_clarification_without_tool_has_no_business_write(shop: ShopHttp) -> None:
    client = shop.browsers.first
    browser = await _boot(client)
    response = await _chat(client, browser, "帮我直接购买下单", shop=shop, plan=None)
    assert response.status_code == 200
    assert "请补充商品、偏好和数量。" in response.json()["content"]
    assert await _operations(shop) == 0
    assert await _counts(shop) == {"order": 0, "payment": 0}


async def test_model_cannot_order_before_a_quote(shop: ShopHttp) -> None:
    client = shop.browsers.first
    browser = await _boot(client)
    await _search(shop, client, browser)
    response = await _chat(
        client,
        browser,
        "确认下单",
        shop=shop,
        plan={"tool": "commit_checkout", "arguments": {}},
    )
    assert response.status_code == 200
    assert _ui(response.json())["stage"] == "shopping"
    assert await _counts(shop) == {"order": 0, "payment": 0}


async def test_expired_quote_requires_new_quote_and_agent_confirmation(shop: ShopHttp) -> None:
    client = shop.browsers.first
    browser = await _boot(client)
    view = await _quoted(shop, client, browser)
    shop.clock.advance(301)
    stale = await _chat(
        client, browser, "确认下单", shop=shop, plan={"tool": "commit_checkout", "arguments": {}}
    )
    assert stale.status_code == 200
    assert _ui(stale.json())["stage"] == "quoted"
    assert await _counts(shop) == {"order": 0, "payment": 0}
    refreshed = await _chat(
        client,
        browser,
        "重新结算",
        shop=shop,
        plan={
            "tool": "prepare_checkout",
            "arguments": {
                "purchase_id": view["purchase_id"],
                "expected_version": view["version"],
            },
        },
    )
    assert _ui(refreshed.json())["stage"] == "quoted"
    ordered = await _order(shop, client, browser, _ui(refreshed.json()))
    assert ordered["stage"] == "awaiting_payment"


async def test_repeated_agent_confirmations_remain_idempotent(shop: ShopHttp) -> None:
    client = shop.browsers.first
    browser = await _boot(client)
    view = await _quoted(shop, client, browser)
    first = await _order(shop, client, browser, view)
    second = await _order(shop, client, browser, view)
    assert first["purchase_id"] == second["purchase_id"]
    assert second["stage"] == "awaiting_payment"
    paid = await _pay(shop, client, browser, second)
    again = await _pay(shop, client, browser, paid)
    assert again["stage"] == "paid"
    assert await _operations(shop, "order") == 1
    assert await _operations(shop, "mock_pay") == 1


async def test_cross_visitor_agent_never_touches_first_purchase(shop: ShopHttp) -> None:
    first, second = shop.browsers.first, shop.browsers.second
    own, foreign = await _boot(first), await _boot(second)
    view = await _quoted(shop, first, own)
    before = await _business(shop)
    response = await _chat(second, foreign, "确认下单", shop=shop, plan=None)
    assert response.status_code == 200
    assert await _business(shop) == before
    assert (await _status(second, foreign, view["purchase_id"])).status_code == 404


async def test_policy_detour_preserves_cart_and_agent_continues(shop: ShopHttp) -> None:
    client = shop.browsers.first
    browser = await _boot(client)
    view = await _search(shop, client, browser)
    selected = await _chat(
        client,
        browser,
        "第二款黑色，两副",
        shop=shop,
        plan={
            "tool": "set_cart_line",
            "arguments": {
                "purchase_id": view["purchase_id"],
                "sku_id": "ANC-BLACK",
                "quantity": 2,
                "expected_version": view["version"],
            },
        },
    )
    assert selected.status_code == 200
    before = await _business(shop)
    policy = await _chat(client, browser, "退货政策是什么？", shop=shop, plan=None, route="support")
    assert policy.status_code == 200 and policy.json().get("ui") is None
    assert await _business(shop) == before
    cart_view = _ui(selected.json())
    continued = await _chat(
        client,
        browser,
        "继续结算",
        shop=shop,
        plan={
            "tool": "prepare_checkout",
            "arguments": {
                "purchase_id": cart_view["purchase_id"],
                "expected_version": cart_view["version"],
            },
        },
    )
    assert _ui(continued.json())["stage"] == "quoted"


async def test_order_timeout_query_never_pays_or_duplicates(shop: ShopHttp) -> None:
    client = shop.browsers.first
    browser = await _boot(client)
    view = await _quoted(shop, client, browser)
    shop.port.fail_next("order")
    ordered = await _chat(
        client, browser, "确认下单", shop=shop, plan={"tool": "commit_checkout", "arguments": {}}
    )
    assert _ui(ordered.json())["stage"] == "unknown"
    recovered = await _chat(
        client,
        browser,
        "查询订单",
        shop=shop,
        plan={"tool": "purchase_status", "arguments": {"purchase_id": view["purchase_id"]}},
    )
    assert _ui(recovered.json())["stage"] == "awaiting_payment"
    assert await _counts(shop) == {"order": 1, "payment": 0}
    paid = await _pay(shop, client, browser, _ui(recovered.json()))
    assert paid["stage"] == "paid"
    assert await _operations(shop, "order") == 1


async def test_payment_timeout_query_confirms_once(shop: ShopHttp) -> None:
    client = shop.browsers.first
    browser = await _boot(client)
    view = await _quoted(shop, client, browser)
    ordered = await _order(shop, client, browser, view)
    shop.port.fail_next("mock_pay")
    attempted = await _chat(
        client, browser, "确认模拟付款", shop=shop, plan={"tool": "mock_pay", "arguments": {}}
    )
    assert _ui(attempted.json())["stage"] == "unknown"
    recovered = await _chat(
        client,
        browser,
        "查询订单",
        shop=shop,
        plan={"tool": "purchase_status", "arguments": {"purchase_id": ordered["purchase_id"]}},
    )
    assert _ui(recovered.json())["stage"] == "paid"
    assert await _operations(shop, "mock_pay") == 1


async def test_history_clear_does_not_cancel_order(shop: ShopHttp) -> None:
    client = shop.browsers.first
    browser = await _boot(client)
    view = await _quoted(shop, client, browser)
    ordered = await _order(shop, client, browser, view)
    before = await _business(shop)
    cleared = await client.delete("/api/v1/guest/history", headers=_BROWSER, params=_scope(browser))
    assert cleared.status_code == 204
    reopened = await _boot(client)
    assert reopened["session"]["session_key"] != browser["session"]["session_key"]
    recovered = await _chat(
        client,
        reopened,
        "查询刚才订单",
        shop=shop,
        plan={"tool": "purchase_status", "arguments": {"purchase_id": ordered["purchase_id"]}},
    )
    assert _ui(recovered.json())["purchase_id"] == ordered["purchase_id"]
    assert await _business(shop) == before


async def test_no_mechanical_fallback_or_action_protocol(shop: ShopHttp) -> None:
    client = shop.browsers.first
    browser = await _boot(client)
    view = await _quoted(shop, client, browser)
    assert "actions" not in view
    await _order(shop, client, browser, view)
    async with shop.browsers.case.maker() as db:
        assert await db.scalar(select(func.count()).select_from(PurchaseAction)) == 0
    transport = cast(ASGITransport, client._transport)
    app = transport.app
    assert isinstance(app, FastAPI)
    assert not any(
        "/commerce/purchases/" in path and path.endswith("/actions")
        for path in app.openapi()["paths"]
    )


async def test_missing_provider_disables_shopping_without_mechanical_fallback(
    visitors: GuestHttp,
) -> None:
    transport = cast(ASGITransport, visitors.first._transport)
    app = transport.app
    assert isinstance(app, FastAPI)
    port = MockCommercePort(visitors.case.maker)
    service = PurchaseService(visitors.case.maker, port)
    persister = visitors.case.chat.persister
    assert persister is not None
    conversation = CommerceConversation(service, visitors.case.coordinator, persister)
    app.dependency_overrides[guest_api.get_guest_commerce_conversation] = lambda: conversation
    try:
        assert await conversation.should_handle(1, "我想买耳机") is False
        with pytest.raises(RuntimeError, match="tool-calling LLM"):
            CommercePlanner(None)
    finally:
        app.dependency_overrides.pop(guest_api.get_guest_commerce_conversation, None)
