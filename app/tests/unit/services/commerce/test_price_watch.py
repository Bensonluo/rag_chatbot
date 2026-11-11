"""Proactive price watch: pre-authorized condition, event-driven trigger, notify-only.

Contract (red until the product changes):
- A watch rule is the visitor's explicit pre-authorization ("SKU X at or
  below ¥Y -> ask me"); storage is per-visitor, capped and expiring, and
  re-watching a SKU re-arms the same row instead of accumulating rules.
- Triggers are evaluated ONLY when price facts for a watched SKU are
  (re)read — the search / get_product tool reads and the session-start
  refresh. There is no background scheduler (documented for production).
- A satisfied condition flips the rule to ``triggered``; claiming the ask
  is exactly-once (``asked``) and yields a structured card whose every
  field is copied from the durable rule row and the provider's price
  facts — never model prose.
- Watch rules are notify-only: they never create cart, order or payment
  state, and the visitor's answer still goes through the existing
  quote -> order -> payment confirmation state machine.
"""

import socket
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.exceptions import ValidationError
from app.models.database.base import Base
from app.models.database.guest import GuestVisitor
from app.models.database.user import User
from app.models.schemas.commerce import CommerceView, WatchCard
from app.services.commerce.mock_port import MockCommercePort
from app.services.commerce.service import PurchaseService
from app.services.commerce.tools import (
    CATALOG_DATA_TAGS,
    SESSION_MEMORY_TAGS,
    create_commerce_tools,
)
from app.services.commerce.watch import PriceWatchService, WatchEntry

OWNER = 1
OTHER = 2
VISITOR = "11111111-1111-4111-8111-111111111111"
OTHER_VISITOR = "22222222-2222-4222-8222-222222222222"


class Clock:
    now = datetime(2026, 11, 11, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
async def shop(tmp_path, monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Commerce unit suite forbids external sockets/DNS")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/watch.sqlite")

    @event.listens_for(engine.sync_engine, "connect")
    def fk(connection, _):
        connection.execute("PRAGMA foreign_keys=ON")

    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    async with maker() as db:
        db.add_all(
            [
                User(id=OWNER, email="watch-one@unit.invalid", hashed_password="!"),
                User(id=OTHER, email="watch-two@unit.invalid", hashed_password="!"),
            ]
        )
        # Flush first: the users->guest_visitors FK has no relationship()
        # for the unit-of-work sorter to see, so ordering must be explicit.
        await db.flush()
        db.add_all(
            [
                GuestVisitor(id=VISITOR, user_id=OWNER, token_hash="a" * 64, expires_at=Clock.now),
                GuestVisitor(
                    id=OTHER_VISITOR,
                    user_id=OTHER,
                    token_hash="b" * 64,
                    expires_at=Clock.now,
                ),
            ]
        )
        await db.commit()
    clock = Clock()
    port = MockCommercePort(maker, clock=clock, quote_ttl_seconds=30)
    purchases = PurchaseService(maker, port, clock=clock, quote_ttl_seconds=30)
    watches = PriceWatchService(maker, clock=clock, max_rules=2)
    yield purchases, watches, maker, clock
    await engine.dispose()


# ── Service contract ────────────────────────────────────────────────────────


async def test_create_lists_rules_and_visitors_are_isolated(shop):
    _, watches, _, _ = shop
    entry = await watches.create(
        VISITOR, "ANC-02", sku_name="轻音降噪耳机 · 白色", target_price_minor=25000
    )
    assert isinstance(entry, WatchEntry)
    assert entry.status == "active"
    assert entry.condition == "at_or_below"
    assert entry.target_price_minor == 25000
    assert entry.created_at is not None
    assert entry.expires_at is not None and entry.expires_at > entry.created_at
    listed = await watches.list_rules(VISITOR)
    assert [rule.sku_id for rule in listed] == ["ANC-02"]
    # A second visitor sees nothing: rules never leak across visitors.
    assert await watches.list_rules(OTHER_VISITOR) == []


async def test_rewatch_re_arms_the_same_row_without_consuming_cap(shop):
    _, watches, maker, _ = shop
    first = await watches.create(VISITOR, "ANC-02", target_price_minor=25000)
    second = await watches.create(VISITOR, "anc-02", target_price_minor=20000)
    assert second.watch_id == first.watch_id
    assert second.target_price_minor == 20000
    assert second.status == "active"
    # One row per (visitor, sku); the cap still has room for one more SKU.
    third = await watches.create(VISITOR, "KEY-02", target_price_minor=15000)
    assert third.watch_id != first.watch_id


async def test_active_rule_cap_is_enforced(shop):
    _, watches, _, _ = shop
    await watches.create(VISITOR, "ANC-02", target_price_minor=25000)
    await watches.create(VISITOR, "KEY-02", target_price_minor=15000)
    with pytest.raises(Exception, match="最多"):
        await watches.create(VISITOR, "MON-02", target_price_minor=90000)
    # Re-arming an existing rule stays allowed at the cap.
    rearmed = await watches.create(VISITOR, "ANC-02", target_price_minor=24000)
    assert rearmed.status == "active"


async def test_expired_rules_are_hidden_and_never_trigger(shop):
    _, watches, _, clock = shop
    await watches.create(VISITOR, "ANC-02", target_price_minor=25000)
    clock.now = clock.now + timedelta(days=31)
    assert await watches.list_rules(VISITOR) == []
    assert await watches.active_rules(VISITOR) == []
    triggered = await watches.evaluate(VISITOR, {"ANC-02": (10000, "轻音降噪耳机 · 白色")})
    assert triggered == []


async def test_evaluate_triggers_only_at_or_below_target(shop):
    _, watches, _, _ = shop
    await watches.create(VISITOR, "ANC-02", target_price_minor=25000)
    above = await watches.evaluate(VISITOR, {"ANC-02": (26000, "轻音降噪耳机 · 白色")})
    assert above == []
    hit = await watches.evaluate(VISITOR, {"ANC-02": (25000, "轻音降噪耳机 · 白色")})
    assert len(hit) == 1 and hit[0].status == "triggered"
    assert hit[0].triggered_price_minor == 25000
    assert hit[0].triggered_at is not None
    # Already triggered rules are not re-triggered by later reads.
    again = await watches.evaluate(VISITOR, {"ANC-02": (24000, "轻音降噪耳机 · 白色")})
    assert again == []


async def test_evaluate_ignores_unwatched_skus_and_refreshes_names(shop):
    _, watches, _, _ = shop
    await watches.create(VISITOR, "ANC-02", sku_name="stale", target_price_minor=30000)
    hit = await watches.evaluate(VISITOR, {"KEY-02": (10000, "机械键盘 · 87键")})
    assert hit == []
    hit = await watches.evaluate(VISITOR, {"ANC-02": (29900, "轻音降噪耳机 · 白色")})
    assert hit[0].sku_name == "轻音降噪耳机 · 白色"


async def test_claim_ask_is_exactly_once_and_copies_server_facts(shop):
    _, watches, _, _ = shop
    await watches.create(
        VISITOR, "ANC-02", sku_name="轻音降噪耳机 · 白色", target_price_minor=30000
    )
    await watches.evaluate(VISITOR, {"ANC-02": (28000, "轻音降噪耳机 · 白色")})
    card = await watches.claim_ask(VISITOR)
    assert isinstance(card, WatchCard)
    assert card.sku_id == "ANC-02"
    assert card.sku_name == "轻音降噪耳机 · 白色"
    assert card.target_price_minor == 30000
    assert card.current_price_minor == 28000
    assert card.currency == "CNY"
    assert card.condition == "at_or_below"
    assert card.action == "ask_order"
    assert card.triggered_at is not None
    # Exactly once: the ask was delivered, no second card can be claimed.
    assert await watches.claim_ask(VISITOR) is None
    statuses = {rule.sku_id: rule.status for rule in await watches.list_rules(VISITOR)}
    assert statuses == {"ANC-02": "asked"}


async def test_cancel_disarms_a_rule(shop):
    _, watches, _, _ = shop
    await watches.create(VISITOR, "ANC-02", target_price_minor=25000)
    assert await watches.cancel(VISITOR, "ANC-02") is True
    assert await watches.active_rules(VISITOR) == []
    assert await watches.cancel(VISITOR, "ANC-02") is False
    triggered = await watches.evaluate(VISITOR, {"ANC-02": (10000, "轻音降噪耳机 · 白色")})
    assert triggered == []


async def test_visitor_id_comes_from_the_trusted_owner_mapping(shop):
    _, watches, _, _ = shop
    assert await watches.visitor_id_for(OWNER) == VISITOR
    assert await watches.visitor_id_for(OTHER) == OTHER_VISITOR
    assert await watches.visitor_id_for(999) is None


async def test_create_rejects_invalid_input(shop):
    _, watches, _, _ = shop
    with pytest.raises(ValidationError):
        await watches.create(VISITOR, "", target_price_minor=25000)
    with pytest.raises(ValidationError):
        await watches.create(VISITOR, "X" * 65, target_price_minor=25000)
    with pytest.raises(ValidationError):
        await watches.create(VISITOR, "ANC-02", target_price_minor=0)


# ── Agent tool contract ─────────────────────────────────────────────────────


async def run_tool(
    shop,
    intent: str,
    args: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    purchases, watches, _, _ = shop
    registry = create_commerce_tools(purchases, OWNER, [], None, price_watches=watches)
    result = await registry.execute(intent, args, user_id=OWNER)
    return result.success, (result.data if result.success else {"error": result.message})


async def test_watch_price_tool_creates_a_rule_with_provider_facts(shop):
    _, watches, _, _ = shop
    ok, payload = await run_tool(
        shop, "commerce:watch_price", {"sku_id": "ANC-02", "target_price_yuan": 250}
    )
    assert ok, payload
    rules = await watches.active_rules(VISITOR)
    assert [rule.sku_id for rule in rules] == ["ANC-02"]
    assert rules[0].target_price_minor == 25000
    # The name snapshot is the provider's authoritative name, not model prose.
    assert rules[0].sku_name == "轻音降噪耳机 · 白色"
    # The tool result states the notify-only contract explicitly.
    assert "不会自动下单" in str(payload)


async def test_watch_price_tool_requires_a_guest_visitor(shop):
    purchases, watches, _, _ = shop
    registry = create_commerce_tools(purchases, OTHER + 10, [], None, price_watches=watches)
    result = await registry.execute(
        "commerce:watch_price",
        {"sku_id": "ANC-02", "target_price_yuan": 250},
        user_id=OTHER + 10,
    )
    assert result.success is False
    assert "访客" in result.message


async def test_watch_price_tool_triggers_immediately_when_condition_holds(shop):
    _, watches, _, _ = shop
    ok, payload = await run_tool(
        shop, "commerce:watch_price", {"sku_id": "ANC-02", "target_price_yuan": 999}
    )
    assert ok, payload
    # The just-read provider price already satisfies the condition.
    rules = await watches.list_rules(VISITOR)
    assert rules[0].status == "triggered"
    assert rules[0].triggered_price_minor == 30400
    assert "已触发" in str(payload)


async def test_list_and_cancel_watch_tools(shop):
    _, watches, _, _ = shop
    ok, payload = await run_tool(
        shop, "commerce:watch_price", {"sku_id": "ANC-02", "target_price_yuan": 250}
    )
    assert ok
    ok, listing = await run_tool(shop, "commerce:list_watches", {})
    assert ok, listing
    body = listing["price_watches"]
    assert body.startswith(SESSION_MEMORY_TAGS[0]) and "ANC-02" in body
    ok, cancelled = await run_tool(shop, "commerce:cancel_watch", {"sku_id": "ANC-02"})
    assert ok and cancelled["cancelled"] is True
    assert await watches.active_rules(VISITOR) == []


async def test_catalog_read_evaluates_watches_and_notifies_the_model(shop):
    purchases, watches, _, _ = shop
    await watches.create(VISITOR, "ANC-02", target_price_minor=99900)
    views: list[CommerceView] = []
    registry = create_commerce_tools(purchases, OWNER, views, None, price_watches=watches)
    result = await registry.execute(
        "commerce:search_products", {"query": "降噪耳机"}, user_id=OWNER
    )
    assert result.success, result.message
    payload = result.data
    assert payload["catalog_data"].startswith(CATALOG_DATA_TAGS[0])
    # The trigger notice rides the tool result inside the session-memory
    # delimiters so the model can mention it as data, never as instructions.
    assert "price_watch" in payload
    notice = payload["price_watch"]
    assert notice.startswith(SESSION_MEMORY_TAGS[0]) and "ANC-02" in notice
    rules = await watches.list_rules(VISITOR)
    assert rules[0].status == "triggered"


async def test_watch_tools_never_touch_purchase_state(shop):
    purchases, watches, _, _ = shop
    before = await purchases.active(OWNER)
    ok, _ = await run_tool(
        shop, "commerce:watch_price", {"sku_id": "ANC-02", "target_price_yuan": 999}
    )
    assert ok
    ok, _ = await run_tool(shop, "commerce:list_watches", {})
    assert ok
    after = await purchases.active(OWNER)
    assert (before is None) == (after is None)
    if before is not None:
        assert after is not None and after.purchase_id == before.purchase_id
        assert after.version == before.version
        assert after.stage == before.stage
