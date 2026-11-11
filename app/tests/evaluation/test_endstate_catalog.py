"""End-state catalog eval: scripted journeys pin FINAL authoritative state.

Table-driven over :data:`ENDSTATE_ROWS`; each row asserts the durable end
state of one wave-1 feature (DB rows, receipts, tickets, HTTP status codes)
rather than intermediate narration. Journeys reuse the guest-commerce HTTP
harness and the in-memory MCP client pattern; all model decisions are
scripted, all clocks injectable — deterministic, no real LLM.
"""

import socket
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator
from datetime import UTC, datetime
from typing import Any, cast

import pytest
from fastapi import FastAPI
from httpx import ASGITransport
from mcp import Client
from mcp.client.session import ClientRequestContext
from mcp.types import ElicitRequestParams, ElicitResult
from sqlalchemy import func, select

from app.api.v1 import guest as guest_api
from app.commerce_provider.catalog import VirtualCatalog
from app.commerce_provider.core import CommerceProviderCore
from app.commerce_provider.faults import FaultController
from app.commerce_provider.mcp_server import CommerceMcpSettings, create_mcp_server
from app.commerce_provider.storage import Database
from app.config.settings import settings
from app.models.database.commerce import (
    MockCommerceReceipt,
    PurchaseConfirmation,
    PurchaseSession,
)
from app.models.database.ticket import PRIORITY_HIGH, HandoffTicket
from app.models.database.visitor_preference import (
    VisitorPreference,
    VisitorPreferenceAudit,
)
from app.services.chat.chat_service import ChatService
from app.services.commerce.conversation import rule_route_decision
from app.services.commerce.service import REASON_SPEND_LIMIT
from app.tests.evaluation.endstate_rows import ENDSTATE_ROWS, EndStateRow
from app.tests.evaluation.test_guest_commerce_flow import (
    ShopHttp,
    _boot,
    _business,
    _chat,
    _counts,
    _operations,
    _order,
    _pay,
    _quoted,
    _ui,
)
from app.tests.evaluation.test_guest_commerce_flow import shop as shop
from app.tests.evaluation.test_guest_conversations import GuestHttp
from app.tests.evaluation.test_guest_conversations import visitors as visitors
from app.tests.evaluation.test_human_conversation import case as case
from app.tests.evaluation.test_injection_redteam_corpus import PrefStore
from app.tests.evaluation.test_injection_redteam_corpus import pref_store as pref_store

_BROWSERS = {"Origin": "http://localhost:3000", "X-Rag-Guest": "1"}


@pytest.fixture(autouse=True)
def _no_external_sockets(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    def blocked(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("end-state catalog forbids external sockets")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    yield


@pytest.fixture
async def mcp_server(tmp_path: Any) -> AsyncGenerator[Any, None]:
    """In-memory MCP server built from the provider primitives (no shared conftest)."""

    database = Database(str(tmp_path / "endstate_mcp.sqlite3"))
    await database.startup()
    core = CommerceProviderCore(
        database,
        VirtualCatalog(),
        clock=lambda: datetime(2026, 11, 11, 12, 0, 0, tzinfo=UTC),
        quote_ttl_seconds=300,
        faults=FaultController(database, enabled=True),
    )
    yield create_mcp_server(CommerceMcpSettings(auth_token="test-token"), core=core)


async def _accept(context: ClientRequestContext, params: ElicitRequestParams) -> ElicitResult:
    del context, params
    return ElicitResult(action="accept", content={"confirmed": True})


async def _decline(context: ClientRequestContext, params: ElicitRequestParams) -> ElicitResult:
    del context, params
    return ElicitResult(action="decline")


async def _first_sku(client: Client, query: str = "显示器") -> str:
    search = await client.call_tool(
        "search_products", {"query": query, "owner_user_id": 1, "page_size": 1}
    )
    assert not search.is_error
    return str(search.structured_content["items"][0]["sku_id"])


def _block(view: dict[str, Any], kind: str) -> dict[str, Any]:
    return next(block for block in view["blocks"] if block["type"] == kind)


async def _receipt_kinds(shop: ShopHttp) -> dict[str, int]:
    async with shop.browsers.case.maker() as db:
        rows = (
            await db.execute(
                select(MockCommerceReceipt.kind, func.count()).group_by(MockCommerceReceipt.kind)
            )
        ).all()
    return dict(rows)


# --- journey implementations -----------------------------------------------------


async def _row_happy_path_paid(shop: ShopHttp, visitors: GuestHttp, **kwargs: Any) -> None:
    del visitors, kwargs
    client = shop.browsers.first
    browser = await _boot(client)
    view = await _quoted(shop, client, browser)
    ordered = await _order(shop, client, browser, view)
    assert ordered["stage"] == "awaiting_payment"
    assert _block(ordered, "payment")["order_id"]

    paid = await _pay(shop, client, browser, ordered)
    assert paid["stage"] == "paid"

    # Authoritative final state, straight from the durable rows.
    async with shop.browsers.case.maker() as db:
        row = await db.get(PurchaseSession, paid["purchase_id"])
        assert row is not None
        assert row.stage == "paid"
        assert row.order_id is not None and row.payment_id is not None
        assert row.active_slot is None  # a paid purchase releases the visitor slot
    assert await _counts(shop) == {"order": 1, "payment": 1}
    assert await _operations(shop, "order") == 1
    assert await _operations(shop, "mock_pay") == 1

    # Replay both confirmations: same end state, no second receipt of any kind.
    replayed_order = await _order(shop, client, browser, paid)
    replayed_pay = await _pay(shop, client, browser, replayed_order)
    assert replayed_pay["stage"] == "paid"
    assert replayed_pay["purchase_id"] == paid["purchase_id"]
    assert await _counts(shop) == {"order": 1, "payment": 1}
    assert await _operations(shop, "order") == 1
    assert await _operations(shop, "mock_pay") == 1
    assert await _receipt_kinds(shop) == {
        "update_cart": 1,
        "quote": 1,
        "order": 1,
        "payment": 1,
        "mock_pay": 1,
    }


async def _row_decline_preserves_cart(shop: ShopHttp, visitors: GuestHttp, **kwargs: Any) -> None:
    del visitors, kwargs
    client = shop.browsers.first
    browser = await _boot(client)
    view = await _quoted(shop, client, browser)
    quoted_lines = [
        (line["sku_id"], line["quantity"]) for line in _block(view, "cart")["cart"]["lines"]
    ]
    assert quoted_lines

    declined = await _chat(client, browser, "先不要，我再想想", shop=shop, plan=None)
    assert declined.status_code == 200, declined.text
    after = _ui(declined.json())
    assert after["stage"] == "quoted"
    assert [
        (line["sku_id"], line["quantity"]) for line in _block(after, "cart")["cart"]["lines"]
    ] == quoted_lines
    # The approval card is still armed: the confirmation was never consumed.
    assert after["approval"] is not None
    assert after["approval"]["action"] == "confirm_order"

    assert await _counts(shop) == {"order": 0, "payment": 0}
    assert await _operations(shop, "order") == 0
    assert await _operations(shop, "mock_pay") == 0
    async with shop.browsers.case.maker() as db:
        confirmation = await db.get(PurchaseConfirmation, _block(view, "quote")["confirmation_id"])
        assert confirmation is not None and confirmation.operation_id is None


async def _row_spend_limit_refusal_ticket(
    shop: ShopHttp,
    visitors: GuestHttp,
    monkeypatch: pytest.MonkeyPatch,
    **kwargs: Any,
) -> None:
    del visitors, kwargs
    monkeypatch.setattr(settings, "COMMERCE_SPEND_LIMIT_SINGLE_CENTS", 100)  # ¥1.00
    client = shop.browsers.first
    browser = await _boot(client)
    view = await _quoted(shop, client, browser)  # quotes ¥671.00

    refused = await _chat(
        client, browser, "确认下单", shop=shop, plan={"tool": "commit_checkout", "arguments": {}}
    )
    assert refused.status_code == 200, refused.text
    after = _ui(refused.json())
    assert after["stage"] == "quoted"

    limit = next(block for block in after["blocks"] if block["type"] == "spend_limit")
    assert limit["limit_kind"] == "single"
    assert limit["limit_minor"] == 100
    assert limit["attempted_minor"] > 100
    assert limit["ticket_id"]
    assert "未创建订单" in limit["message"]

    # No order was created anywhere: no objects, no order operation.
    assert await _counts(shop) == {"order": 0, "payment": 0}
    assert await _operations(shop, "order") == 0
    async with shop.browsers.case.maker() as db:
        tickets = (
            await db.scalars(
                select(HandoffTicket).where(HandoffTicket.reason == REASON_SPEND_LIMIT)
            )
        ).all()
        assert len(tickets) == 1
        assert tickets[0].priority == PRIORITY_HIGH
        assert tickets[0].session_id == browser["session"]["id"]
        confirmation = await db.get(PurchaseConfirmation, _block(view, "quote")["confirmation_id"])
        assert confirmation is not None and confirmation.operation_id is None


async def _row_quote_ttl_requote(shop: ShopHttp, visitors: GuestHttp, **kwargs: Any) -> None:
    del visitors, kwargs
    client = shop.browsers.first
    browser = await _boot(client)
    view = await _quoted(shop, client, browser)
    stale_confirmation = _block(view, "quote")["confirmation_id"]
    stale_expiry = view["approval"]["expires_at"]

    shop.clock.advance(301)  # past the 300s quote TTL
    stale = await _chat(
        client, browser, "确认下单", shop=shop, plan={"tool": "commit_checkout", "arguments": {}}
    )
    assert stale.status_code == 200, stale.text
    expired_view = _ui(stale.json())
    assert expired_view["stage"] == "quoted"
    assert expired_view["approval"] is None  # an expired quote disarms the card
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
    assert refreshed.status_code == 200, refreshed.text
    requoted = _ui(refreshed.json())
    assert requoted["stage"] == "quoted"
    fresh_confirmation = _block(requoted, "quote")["confirmation_id"]
    assert fresh_confirmation != stale_confirmation
    assert requoted["approval"]["expires_at"] > stale_expiry

    ordered = await _order(shop, client, browser, requoted)
    paid = await _pay(shop, client, browser, ordered)
    assert paid["stage"] == "paid"
    assert await _counts(shop) == {"order": 1, "payment": 1}
    assert await _operations(shop, "order") == 1
    assert await _operations(shop, "mock_pay") == 1


async def _row_routing_fallback_support_http(
    shop: ShopHttp,
    visitors: GuestHttp,
    monkeypatch: pytest.MonkeyPatch,
    **kwargs: Any,
) -> None:
    del visitors, kwargs
    client = shop.browsers.first
    browser = await _boot(client)
    before = await _business(shop)

    async def router_down(messages: Any, **kwargs_inner: Any) -> Any:
        del messages, kwargs_inner
        raise RuntimeError("router LLM exploded")

    monkeypatch.setattr(shop.provider, "generate", router_down)
    response = await _chat(client, browser, "你好呀，今天天气怎么样", shop=shop, route="support")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload.get("ui") is None  # support lane, no commerce view

    # Selection-only degradation: nothing in the business state moved.
    assert await _business(shop) == before
    assert await _operations(shop) == 0


_ROUTING_TABLE: tuple[tuple[str, str], ...] = (
    ("你好", "support"),
    ("今天天气怎么样", "support"),
    ("嗯嗯", "support"),
    ("hello there", "support"),
    ("随便看看", "support"),
    ("买的东西坏了要退", "support"),  # support terms (坏了/退) outweigh the lone 买
    ("帮我看看订单物流", "support"),  # tie (订单 vs 物流) defaults to support
    ("我想买降噪耳机", "shopping"),
    ("add to cart please", "shopping"),
)


async def _row_routing_rule_table(shop: ShopHttp, visitors: GuestHttp, **kwargs: Any) -> None:
    del shop, visitors, kwargs
    for message, lane in _ROUTING_TABLE:
        decided, reason = rule_route_decision(message)
        assert decided == lane, f"{message!r} must route {lane}, got {decided}"
        assert reason
        assert decided in {"shopping", "support"}  # bounded lanes only


async def _row_mcp_decline_before_mutation(
    shop: ShopHttp, visitors: GuestHttp, mcp_server: Any, **kwargs: Any
) -> None:
    del shop, visitors, kwargs
    async with Client(mcp_server, elicitation_callback=_decline) as client:
        sku_id = await _first_sku(client)
        lines = [{"sku_id": sku_id, "quantity": 1}]
        declined = await client.call_tool("update_cart", {"owner_user_id": 1, "lines": lines})
        assert declined.is_error
        assert "declined" in str(declined.content).lower()

    # The declined call mutated nothing: the accepted retry still starts at
    # cart_version 1 — had the decline leaked a mutation, this would be 2.
    async with Client(mcp_server, elicitation_callback=_accept) as client:
        accepted = await client.call_tool(
            "update_cart", {"owner_user_id": 1, "lines": [{"sku_id": sku_id, "quantity": 1}]}
        )
        assert not accepted.is_error
        assert accepted.structured_content["cart_version"] == 1


async def _row_mcp_accept_idempotent_replay(
    shop: ShopHttp, visitors: GuestHttp, mcp_server: Any, **kwargs: Any
) -> None:
    del shop, visitors, kwargs
    async with Client(mcp_server, elicitation_callback=_accept) as client:
        sku_id = await _first_sku(client)
        lines = [{"sku_id": sku_id, "quantity": 2}]
        first = await client.call_tool("update_cart", {"owner_user_id": 1, "lines": lines})
        assert not first.is_error
        cart = first.structured_content
        assert cart["cart_version"] == 1

        replay = await client.call_tool("update_cart", {"owner_user_id": 1, "lines": lines})
        assert replay.structured_content == cart  # structural key replays the receipt

        quote = await client.call_tool(
            "prepare_quote",
            {
                "owner_user_id": 1,
                "cart_id": cart["cart_id"],
                "expected_cart_version": cart["cart_version"],
            },
        )
        quoted = quote.structured_content
        order = await client.call_tool(
            "create_order", {"owner_user_id": 1, "quote_id": quoted["quote_id"]}
        )
        ordered = order.structured_content
        assert ordered["status"] == "awaiting_payment"

        payment = await client.call_tool(
            "sandbox_pay", {"owner_user_id": 1, "order_id": ordered["order_id"]}
        )
        assert payment.structured_content["status"] == "paid"
        pay_replay = await client.call_tool(
            "sandbox_pay", {"owner_user_id": 1, "order_id": ordered["order_id"]}
        )
        assert pay_replay.structured_content == payment.structured_content


async def _row_prefs_redline_keys_rejected(
    shop: ShopHttp, visitors: GuestHttp, pref_store: PrefStore, **kwargs: Any
) -> None:
    del shop, visitors, kwargs
    for key in (
        "quota.max_order",
        "limit.daily_spend",
        "permission.admin_tools",
        "policy.refund_override",
        "POLICY.spend_cap",  # redline matching is case-insensitive
    ):
        with pytest.raises(ValueError):
            await pref_store.service.save(pref_store.visitor_id, key, {"v": 1})
    assert await pref_store.service.load_active(pref_store.visitor_id) == []

    async with pref_store.service._maker() as db:  # noqa: SLF001 - eval-only probe
        preferences = (await db.scalars(select(VisitorPreference))).all()
        audits = (await db.scalars(select(VisitorPreferenceAudit))).all()
    assert preferences == []
    assert audits == []


async def _row_prefs_confirmed_not_downgraded(
    shop: ShopHttp, visitors: GuestHttp, pref_store: PrefStore, **kwargs: Any
) -> None:
    del shop, visitors, kwargs
    confirmed = await pref_store.service.save(
        pref_store.visitor_id, "display_preference", {"size": "27英寸"}, source="confirmed"
    )
    assert confirmed.source == "confirmed"

    inferred_attempt = await pref_store.service.save(
        pref_store.visitor_id, "display_preference", {"size": "32英寸"}, source="inferred"
    )
    # The inferred write is a no-op that returns the confirmed entry unchanged.
    assert inferred_attempt.pref_value == {"size": "27英寸"}
    assert inferred_attempt.source == "confirmed"

    active = await pref_store.service.load_active(pref_store.visitor_id)
    assert [entry.pref_value for entry in active] == [{"size": "27英寸"}]
    assert [entry.source for entry in active] == ["confirmed"]

    # A confirmed write is an upgrade path, not a downgrade: it still updates.
    upgraded = await pref_store.service.save(
        pref_store.visitor_id, "display_preference", {"size": "34英寸"}, source="confirmed"
    )
    assert upgraded.pref_value == {"size": "34英寸"}
    assert upgraded.source == "confirmed"


async def _row_legacy_history_503(shop: ShopHttp, visitors: GuestHttp, **kwargs: Any) -> None:
    del kwargs
    client = shop.browsers.first
    browser = await _boot(client)
    scope = {
        "session_id": browser["session"]["id"],
        "expected_session_key": browser["session"]["session_key"],
    }

    transport = cast(ASGITransport, client._transport)  # noqa: SLF001 - harness access
    app = transport.app
    assert isinstance(app, FastAPI)
    previous = app.dependency_overrides.get(guest_api.get_guest_chat_service)
    legacy = ChatService(graph=None, coordinator=visitors.case.coordinator)
    app.dependency_overrides[guest_api.get_guest_chat_service] = lambda: legacy
    try:
        refused = await client.get("/api/v1/guest/history", headers=_BROWSERS, params=scope)
        assert refused.status_code == 503, refused.text
    finally:
        if previous is not None:
            app.dependency_overrides[guest_api.get_guest_chat_service] = previous
        else:
            app.dependency_overrides.pop(guest_api.get_guest_chat_service, None)

    # The normally wired service keeps answering 200 on the same endpoint.
    healthy = await client.get("/api/v1/guest/history", headers=_BROWSERS, params=scope)
    assert healthy.status_code == 200, healthy.text


# --- table driver ------------------------------------------------------------------

Journey = Callable[..., Awaitable[None]]

_JOURNEYS: dict[str, Journey] = {
    "happy_path_paid": _row_happy_path_paid,
    "decline_preserves_cart": _row_decline_preserves_cart,
    "spend_limit_refusal_ticket": _row_spend_limit_refusal_ticket,
    "quote_ttl_requote": _row_quote_ttl_requote,
    "routing_fallback_support_http": _row_routing_fallback_support_http,
    "routing_rule_table": _row_routing_rule_table,
    "mcp_decline_before_mutation": _row_mcp_decline_before_mutation,
    "mcp_accept_idempotent_replay": _row_mcp_accept_idempotent_replay,
    "prefs_redline_keys_rejected": _row_prefs_redline_keys_rejected,
    "prefs_confirmed_not_downgraded": _row_prefs_confirmed_not_downgraded,
    "legacy_history_503": _row_legacy_history_503,
}


@pytest.mark.parametrize("row", ENDSTATE_ROWS, ids=[row.row_id for row in ENDSTATE_ROWS])
async def test_endstate_row(
    row: EndStateRow,
    shop: ShopHttp,
    visitors: GuestHttp,
    pref_store: PrefStore,
    mcp_server: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journey = _JOURNEYS.get(row.row_id)
    assert journey is not None, f"end-state row {row.row_id} has no journey"
    await journey(
        shop=shop,
        visitors=visitors,
        pref_store=pref_store,
        mcp_server=mcp_server,
        monkeypatch=monkeypatch,
    )


def test_catalog_rows_are_wired_and_unique() -> None:
    ids = [row.row_id for row in ENDSTATE_ROWS]
    assert len(ids) == len(set(ids))
    assert set(ids) == set(_JOURNEYS)
    assert len(ENDSTATE_ROWS) >= 10
