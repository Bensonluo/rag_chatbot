"""Owned cookie chat traverses real AgentService, domain/provider SQL and history.

The purchase loop advances only through /guest/chat(/stream) text: an order is
created only by the explicit quoted-stage phrase 确认下单, and settled only by a
separate awaiting-payment phrase 确认模拟付款. Replies must carry the catalog,
subtotal, total, order and payment facts. Views are display facts only — the
button action protocol (actions/action_ref/client_request_id) is deleted.
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
from app.config.settings import settings
from app.models.database.commerce import (
    MockCommerceObject,
    MockCommerceReceipt,
    PurchaseAction,
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


class ShopClock:
    """Controllable commerce time; the surrounding stack keeps real time."""

    def __init__(self) -> None:
        self.now = datetime.now(UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: int) -> None:
        self.now += timedelta(seconds=seconds)


@dataclass
class ShopHttp:
    browsers: GuestHttp
    service: PurchaseService
    port: MockCommercePort
    conversation: CommerceConversation
    clock: ShopClock


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
    conversation = CommerceConversation(service, visitors.case.coordinator, persister)
    app.dependency_overrides[guest_api.get_guest_purchase_service] = lambda: service
    app.dependency_overrides[guest_api.get_guest_chat_service] = lambda: visitors.case.chat
    app.dependency_overrides[guest_api.get_guest_commerce_conversation] = lambda: conversation
    yield ShopHttp(visitors, service, port, conversation, clock)


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


def _content(payload: dict[str, Any]) -> str:
    value = payload.get("content")
    assert isinstance(value, str) and value.strip()
    return value


def _stream_parse(body: str) -> tuple[str, list[dict[str, Any]]]:
    """Rebuild the reply text and the completion receipts from SSE frames."""
    chunks: list[str] = []
    completions: list[dict[str, Any]] = []
    for frame in filter(None, body.split("\n\n")):
        if frame.startswith("event: completion"):
            for line in frame.split("\n"):
                if line.startswith("data: "):
                    completions.append(json.loads(line.split("data: ", 1)[1]))
        elif frame.startswith("data: ["):
            continue  # [DONE]/[STREAM_ERROR] carry no reply text.
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
    response = await _chat(client, browser, "我想买降噪耳机")
    assert response.status_code == 200, response.text
    assert response.json()["session_key"] == browser["session"]["session_key"]
    return _ui(response.json())


async def _quoted(shop: ShopHttp, client: AsyncClient, browser: dict[str, Any]) -> dict[str, Any]:
    """Search, pick the second (black) item twice, then ask for checkout — text only."""
    view = await _search(shop, client, browser)
    selected = await _chat(client, browser, "第二款黑色，两副")
    assert selected.status_code == 200, selected.text
    view = _ui(selected.json())
    line = next(block for block in view["blocks"] if block["type"] == "cart")["cart"]["lines"][0]
    assert (line["sku_id"], line["quantity"]) == ("ANC-BLACK", 2)
    checkout = await _chat(client, browser, "结算")
    assert checkout.status_code == 200, checkout.text
    view = _ui(checkout.json())
    assert view["stage"] == "quoted"
    return view


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


async def _purchase(shop: ShopHttp, purchase_id: str) -> PurchaseSession:
    async with shop.browsers.case.maker() as db:
        row = await db.get(PurchaseSession, purchase_id)
        assert row is not None
        return row


async def _status(client: AsyncClient, browser: dict[str, Any], purchase_id: str) -> Response:
    return await client.get(
        f"/api/v1/guest/commerce/purchases/{purchase_id}", headers=_BROWSER, params=_scope(browser)
    )


@pytest.mark.parametrize("stream", [False, True])
async def test_chat_text_journey_orders_and_pays_only_on_explicit_phrases(
    shop: ShopHttp, stream: bool
) -> None:
    client = shop.browsers.first
    browser = await _boot(client)

    async def say(text: str) -> str:
        response = await _chat(client, browser, text, stream=stream)
        assert response.status_code == 200, response.text
        if stream:
            assert response.text.endswith("data: [DONE]\n\n")
            content, completions = _stream_parse(response.text)
            assert completions, "every streamed turn ends with a completion receipt"
            assert completions[-1]["session_key"] == browser["session"]["session_key"]
            assert content.strip()
            return content
        payload = response.json()
        assert payload["session_key"] == browser["session"]["session_key"]
        return _content(payload)

    catalog = await say("我想买降噪耳机")
    assert "ANC-BLACK" in catalog and "ANC-WHITE" in catalog  # Catalog facts travel as text.
    active = await client.get(
        "/api/v1/guest/commerce/purchases/active", headers=_BROWSER, params=_scope(browser)
    )
    assert active.status_code == 200
    purchase_id = _ui(active.json())["purchase_id"]

    carted = await say("第二款黑色，两副")
    assert "小计" in carted and "598" in carted  # 2 × ¥299.00.
    quoted = await say("结算")
    assert "小计" in quoted and "总价" in quoted and "598" in quoted

    ordered = await say("确认下单")
    assert "订单" in ordered
    row = await _purchase(shop, purchase_id)
    assert row.stage == "awaiting_payment" and row.order_id
    assert row.order_id in ordered  # The reply carries the created order fact.
    assert await _counts(shop) == {"order": 1, "payment": 1}

    paid = await say("确认模拟付款")
    assert row.order_id in paid and "付款" in paid
    assert (await _purchase(shop, purchase_id)).stage == "paid"
    assert await _counts(shop) == {"order": 1, "payment": 1}

    queried = await say("查询订单")
    assert row.order_id in queried and "598" in queried
    before = await _business(shop)
    status = await _status(client, browser, purchase_id)
    assert status.status_code == 200 and _ui(status.json())["stage"] == "paid"
    assert await _business(shop) == before


async def test_commerce_views_and_database_no_longer_expose_button_actions(shop: ShopHttp) -> None:
    client = shop.browsers.first
    browser = await _boot(client)
    view = await _quoted(shop, client, browser)
    assert "actions" not in view
    ordered = await _chat(client, browser, "确认下单")
    assert ordered.status_code == 200, ordered.text
    view = _ui(ordered.json())
    assert view["stage"] == "awaiting_payment" and "actions" not in view
    async with shop.browsers.case.maker() as db:
        assert await db.scalar(select(func.count()).select_from(PurchaseAction)) == 0


async def test_purchase_action_endpoint_no_longer_exists(shop: ShopHttp) -> None:
    browser = await _boot(shop.browsers.first)
    view = await _quoted(shop, shop.browsers.first, browser)
    app = cast(ASGITransport, shop.browsers.first._transport).app
    assert isinstance(app, FastAPI)
    paths = set(app.openapi()["paths"])
    assert not any("/commerce/purchases/" in path and path.endswith("/actions") for path in paths)
    before = await _business(shop)
    response = await shop.browsers.first.post(
        f"/api/v1/guest/commerce/purchases/{view['purchase_id']}/actions",
        headers=_BROWSER,
        json={
            **_scope(browser),
            "action_ref": str(uuid4()),
            "expected_version": 0,
            "client_request_id": str(uuid4()),
        },
    )
    assert response.status_code in {404, 405}
    assert await _business(shop) == before


@pytest.mark.parametrize(
    "phrase",
    ["确认", "好的", "继续", "确认下单并付款", "确认模拟付款", "都确认了，直接下单付款吧"],
)
async def test_vague_or_combined_confirmation_phrases_never_write_business_facts(
    shop: ShopHttp, phrase: str
) -> None:
    client = shop.browsers.first
    browser = await _boot(client)
    await _quoted(shop, client, browser)
    before = await _business(shop)
    response = await _chat(client, browser, phrase)
    assert response.status_code == 200, response.text
    assert response.json()["ui"]["stage"] == "quoted"
    assert await _business(shop) == before
    assert await _operations(shop) == 2  # Only the cart and quote operations exist.
    assert await _counts(shop) == {"order": 0, "payment": 0}


async def test_order_phrase_without_a_quote_writes_nothing(shop: ShopHttp) -> None:
    client = shop.browsers.first
    browser = await _boot(client)
    await _search(shop, client, browser)
    carted = await _chat(client, browser, "第二款黑色，两副")
    assert carted.status_code == 200, carted.text
    view = _ui(carted.json())
    assert view["stage"] == "cart"
    before = await _business(shop)
    response = await _chat(client, browser, "确认下单")
    assert response.status_code == 200, response.text
    assert response.json()["ui"]["stage"] == "cart"  # No quote exists: nothing to confirm.
    assert await _business(shop) == before
    assert await _counts(shop) == {"order": 0, "payment": 0}


async def test_expired_quote_confirmation_requires_a_fresh_quote(shop: ShopHttp) -> None:
    client = shop.browsers.first
    browser = await _boot(client)
    await _quoted(shop, client, browser)
    shop.clock.advance(301)
    before = await _business(shop)
    stale = await _chat(client, browser, "确认下单")
    assert stale.status_code == 200, stale.text
    assert stale.json()["ui"]["stage"] == "quoted"
    assert await _business(shop) == before  # An expired confirmation writes nothing.
    assert await _counts(shop) == {"order": 0, "payment": 0}
    refreshed = await _chat(client, browser, "结算")
    assert refreshed.status_code == 200, refreshed.text
    assert _ui(refreshed.json())["stage"] == "quoted"
    confirmed = await _chat(client, browser, "确认下单")
    assert confirmed.status_code == 200, confirmed.text
    assert _ui(confirmed.json())["stage"] == "awaiting_payment"
    assert await _counts(shop) == {"order": 1, "payment": 1}


async def test_cart_change_after_quote_requires_a_fresh_confirmation(shop: ShopHttp) -> None:
    client = shop.browsers.first
    browser = await _boot(client)
    view = await _quoted(shop, client, browser)
    stale_quote = next(b for b in view["blocks"] if b["type"] == "quote")["quote"]["quote_id"]
    changed = await _chat(client, browser, "改成白色一个")
    assert changed.status_code == 200, changed.text
    view = _ui(changed.json())
    assert view["stage"] == "cart" and await _counts(shop) == {"order": 0, "payment": 0}
    refreshed = await _chat(client, browser, "结算")
    assert refreshed.status_code == 200, refreshed.text
    view = _ui(refreshed.json())
    fresh_quote = next(b for b in view["blocks"] if b["type"] == "quote")["quote"]["quote_id"]
    assert view["stage"] == "quoted" and fresh_quote != stale_quote
    confirmed = await _chat(client, browser, "确认下单")
    assert confirmed.status_code == 200, confirmed.text
    assert _ui(confirmed.json())["stage"] == "awaiting_payment"
    assert await _counts(shop) == {"order": 1, "payment": 1}


async def test_repeated_explicit_confirmations_settle_exactly_one_order(shop: ShopHttp) -> None:
    client = shop.browsers.first
    browser = await _boot(client)
    await _quoted(shop, client, browser)
    first = await _chat(client, browser, "确认下单")
    second = await _chat(client, browser, "确认下单")
    assert first.status_code == second.status_code == 200
    assert _ui(first.json())["purchase_id"] == _ui(second.json())["purchase_id"]
    assert _ui(second.json())["stage"] == "awaiting_payment"
    assert await _operations(shop, "order") == 1
    assert await _counts(shop) == {"order": 1, "payment": 1}
    paid = await _chat(client, browser, "确认模拟付款")
    again = await _chat(client, browser, "确认模拟付款")
    assert paid.status_code == again.status_code == 200
    assert _ui(again.json())["stage"] == "paid"
    assert await _operations(shop, "mock_pay") == 1
    assert await _counts(shop) == {"order": 1, "payment": 1}


async def test_cross_visitor_confirmation_text_cannot_touch_another_browser_purchase(
    shop: ShopHttp,
) -> None:
    first, second = shop.browsers.first, shop.browsers.second
    own, foreign = await _boot(first), await _boot(second)
    view = await _quoted(shop, first, own)
    for phrase in ("确认下单", "确认模拟付款"):
        response = await _chat(second, foreign, phrase)
        assert response.status_code == 200, response.text
        foreign_view = _ui(response.json())
        # The other browser only ever touches its own empty purchase.
        assert foreign_view["purchase_id"] != view["purchase_id"]
        assert foreign_view["stage"] == "shopping"
    assert (await _purchase(shop, view["purchase_id"])).stage == "quoted"
    assert await _counts(shop) == {"order": 0, "payment": 0}
    status = await _status(second, foreign, view["purchase_id"])
    assert status.status_code == 404


async def test_stale_session_key_confirmation_has_no_business_effect(shop: ShopHttp) -> None:
    client = shop.browsers.first
    browser = await _boot(client)
    await _quoted(shop, client, browser)
    before = await _business(shop)
    stale = await client.post(
        "/api/v1/guest/chat",
        headers=_BROWSER,
        json={**_scope(browser), "expected_session_key": str(uuid4()), "message": "确认下单"},
    )
    assert stale.status_code == 409 and await _business(shop) == before


async def test_history_clear_does_not_cancel_order_and_active_purchase_recovers_owned_state(
    shop: ShopHttp,
) -> None:
    client = shop.browsers.first
    browser = await _boot(client)
    await _quoted(shop, client, browser)
    ordered = await _chat(client, browser, "确认下单")
    assert ordered.status_code == 200, ordered.text
    ordered_view = _ui(ordered.json())
    assert ordered_view["stage"] == "awaiting_payment"
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
    assert current.status_code == 200
    recovered = _ui(current.json())
    assert recovered["purchase_id"] == ordered_view["purchase_id"]
    assert recovered["stage"] == "awaiting_payment"
    assert await _business(shop) == before


async def test_accepted_order_timeout_text_recovery_settles_the_original_operation_only(
    shop: ShopHttp,
) -> None:
    client = shop.browsers.first
    browser = await _boot(client)
    view = await _quoted(shop, client, browser)
    shop.port.fail_next("order")
    ordered = await _chat(client, browser, "确认下单")
    assert ordered.status_code == 200, ordered.text
    assert _ui(ordered.json())["stage"] == "unknown"
    assert await _counts(shop) == {"order": 1, "payment": 0}
    recovered = await _chat(client, browser, "查询订单")
    assert recovered.status_code == 200, recovered.text
    view = _ui(recovered.json())
    assert view["stage"] == "awaiting_payment"
    assert await _counts(shop) == {"order": 1, "payment": 0}  # Recovery never pays.
    paid = await _chat(client, browser, "确认模拟付款")
    assert paid.status_code == 200, paid.text
    assert _ui(paid.json())["stage"] == "paid"
    again = await _chat(client, browser, "确认模拟付款")
    assert again.status_code == 200 and _ui(again.json())["stage"] == "paid"
    assert await _operations(shop, "order") == 1 and await _operations(shop, "mock_pay") == 1
    assert await _counts(shop) == {"order": 1, "payment": 1}


async def test_accepted_payment_timeout_text_query_confirms_once_without_second_attempt(
    shop: ShopHttp,
) -> None:
    client = shop.browsers.first
    browser = await _boot(client)
    await _quoted(shop, client, browser)
    created = await _chat(client, browser, "确认下单")
    assert created.status_code == 200, created.text
    assert _ui(created.json())["stage"] == "awaiting_payment"
    shop.port.fail_next("mock_pay")
    attempted = await _chat(client, browser, "确认模拟付款")
    assert attempted.status_code == 200, attempted.text
    assert _ui(attempted.json())["stage"] == "unknown"
    recovered = await _chat(client, browser, "查询订单")
    assert recovered.status_code == 200, recovered.text
    assert _ui(recovered.json())["stage"] == "paid"
    retried = await _chat(client, browser, "确认模拟付款")
    assert retried.status_code == 200 and _ui(retried.json())["stage"] == "paid"
    assert await _operations(shop, "mock_pay") == 1
    assert await _counts(shop) == {"order": 1, "payment": 1}


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
    selected = await _chat(client, browser, "第二款黑色，两副")
    assert selected.status_code == 200, selected.text
    view = _ui(selected.json())
    before, graph = await _business(shop), shop.browsers.case.calls.graph
    policy = await _chat(client, browser, "退货政策是什么？")
    assert policy.status_code == 200 and policy.json().get("ui") is None
    assert shop.browsers.case.calls.graph > graph
    assert await _business(shop) == before
    continued = await _chat(client, browser, "继续结算")
    assert continued.status_code == 200, continued.text
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


async def test_native_planner_cannot_pay_without_an_explicit_user_authorization(
    shop: ShopHttp,
) -> None:
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


@pytest.mark.parametrize("transition", ["new-purchase", "same-purchase-chat", "same-purchase-cart"])
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
    if transition == "same-purchase-cart":
        advanced = await _chat(client, browser, "第二款黑色，两副")
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
    text = await shop.browsers.second.post(
        "/api/v1/guest/chat",
        headers=_BROWSER,
        json={**_scope(browser), "message": "确认下单"},
    )
    assert status.status_code == text.status_code == 401
    assert await _business(shop) == before
