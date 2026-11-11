"""Order tracking timeline assembled from authoritative rows.

Target contract under TDD: ``build_timeline`` rebuilds one purchase's full
event chain — purchase opening, every cart version with contents, quote with
totals/TTL/expiry, order creation, payment creation, payment completion and
every idempotency ledger receipt — strictly from durable rows
(``commerce_purchases`` / ``commerce_operations`` / ``commerce_confirmations``),
owner-scoped like every other read. Unknown or foreign purchase ids return a
clean ``found=False`` result instead of raising. ``register_timeline_tools``
exposes the same chain as a read-only agent tool inside the established
delimited provider-data framing.
"""

import socket
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import event, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.exceptions import ValidationError
from app.models.database.base import Base
from app.models.database.commerce import (
    PurchaseConfirmation,
    PurchaseOperation,
    PurchaseSession,
)
from app.models.database.user import User
from app.models.schemas.commerce import CartLineSelection
from app.services.commerce.mock_port import MockCommercePort, as_utc
from app.services.commerce.service import PurchaseService
from app.services.commerce.timeline import (
    OrderTimeline,
    TimelineEvent,
    build_timeline,
    register_timeline_tools,
)
from app.services.dialogue.tools import ToolRegistry

QUOTE_TTL = 30

FULL_PAID_KINDS = [
    "purchase_opened",
    "cart_updated",
    "cart_updated",
    "quote_prepared",
    "order_created",
    "payment_created",
    "payment_completed",
]


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
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/commerce.sqlite")

    @event.listens_for(engine.sync_engine, "connect")
    def fk(connection, _):
        connection.execute("PRAGMA foreign_keys=ON")

    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    async with maker() as db:
        db.add_all(
            [
                User(id=1, email="timeline-one@unit.invalid", hashed_password="!"),
                User(id=2, email="timeline-two@unit.invalid", hashed_password="!"),
            ]
        )
        await db.commit()
    clock = Clock()
    port = MockCommercePort(maker, clock=clock, quote_ttl_seconds=QUOTE_TTL)
    service = PurchaseService(maker, port, clock=clock, quote_ttl_seconds=QUOTE_TTL)
    yield service, maker, clock, port
    await engine.dispose()


async def paid_journey(service: PurchaseService, owner: int = 1) -> str:
    """Two cart versions → quote → order → payment → paid."""

    view = await service.start(owner)
    view = await service.update_cart_lines(
        owner,
        view.purchase_id,
        [CartLineSelection(sku_id="MAT-01", quantity=1)],
        view.version,
    )
    view = await service.update_cart_lines(
        owner,
        view.purchase_id,
        [CartLineSelection(sku_id="ANC-02", quantity=2)],
        view.version,
    )
    view = await service.prepare_checkout(owner, view.purchase_id, view.version)
    view = await service.commit_checkout(owner, view.purchase_id, view.version)
    view = await service.mock_pay(owner, view.purchase_id, view.version)
    assert view.stage == "paid"
    return view.purchase_id


async def quoted_journey(service: PurchaseService, owner: int = 1) -> str:
    view = await service.start(owner)
    view = await service.update_cart_lines(
        owner,
        view.purchase_id,
        [CartLineSelection(sku_id="ANC-02", quantity=2)],
        view.version,
    )
    view = await service.prepare_checkout(owner, view.purchase_id, view.version)
    assert view.stage == "quoted"
    return view.purchase_id


async def pin_all_timestamps(
    maker: async_sessionmaker[AsyncSession], purchase_id: str, when: datetime
) -> None:
    """Pin every authoritative row timestamp to one instant (naive UTC)."""

    naive = when.replace(tzinfo=None)
    async with maker() as db:
        await db.execute(
            update(PurchaseSession)
            .where(PurchaseSession.id == purchase_id)
            .values(created_at=naive, updated_at=naive)
        )
        await db.execute(
            update(PurchaseOperation)
            .where(PurchaseOperation.purchase_id == purchase_id)
            .values(created_at=naive, updated_at=naive)
        )
        await db.execute(
            update(PurchaseConfirmation)
            .where(PurchaseConfirmation.purchase_id == purchase_id)
            .values(created_at=naive, updated_at=naive)
        )
        await db.commit()


def event_of(timeline: OrderTimeline, kind: str) -> TimelineEvent:
    return next(event for event in timeline.events if event.kind == kind)


async def operations_of(
    maker: async_sessionmaker[AsyncSession], purchase_id: str
) -> list[PurchaseOperation]:
    async with maker() as db:
        return list(
            (
                await db.scalars(
                    select(PurchaseOperation).where(PurchaseOperation.purchase_id == purchase_id)
                )
            ).all()
        )


async def test_full_paid_journey_rebuilds_chain_from_rows(shop):
    service, maker, clock, _ = shop
    purchase_id = await paid_journey(service)
    timeline = await build_timeline(maker, 1, purchase_id, clock=clock)
    assert timeline.found is True and timeline.stage == "paid"
    assert [event.kind for event in timeline.events] == FULL_PAID_KINDS
    stamps = [event.timestamp for event in timeline.events]
    assert stamps == sorted(stamps)
    # Every idempotency ledger receipt appears exactly once, keyed by request key.
    ops = await operations_of(maker, purchase_id)
    assert len(ops) == 6
    by_key = {event.request_key: event for event in timeline.events if event.request_key}
    assert set(by_key) == {op.id for op in ops}
    amount_field = {
        "update_cart": "subtotal_minor",
        "quote": "total_minor",
        "order": "total_minor",
        "payment": "total_minor",
        "mock_pay": "total_minor",
    }
    for op in ops:
        event = by_key[op.id]
        assert event.generation == op.generation
        assert op.result is not None
        # Operative status: the receipt snapshot's business status when the
        # operation completed (order/payment carry one), else the ledger status.
        assert event.status == (op.result.get("status") or op.status)
        assert event.amount_minor == op.result[amount_field[op.kind]]
    # Amounts agree with the durable confirmation, not with any narration.
    async with maker() as db:
        purchase = await db.get(PurchaseSession, purchase_id)
        confirmation = await db.scalar(
            select(PurchaseConfirmation).where(PurchaseConfirmation.purchase_id == purchase_id)
        )
    assert purchase is not None and confirmation is not None
    quote_event = event_of(timeline, "quote_prepared")
    assert quote_event.amount_minor == confirmation.snapshot["total_minor"]
    for kind in ("order_created", "payment_created", "payment_completed"):
        assert event_of(timeline, kind).amount_minor == confirmation.snapshot["total_minor"]
    assert event_of(timeline, "order_created").order_id == purchase.order_id
    assert event_of(timeline, "payment_created").payment_id == purchase.payment_id
    assert event_of(timeline, "payment_completed").payment_id == purchase.payment_id
    # The final cart version carries its authoritative contents.
    cart_events = [event for event in timeline.events if event.kind == "cart_updated"]
    assert [line.sku_id for line in cart_events[-1].lines or []] == ["ANC-02"]
    assert cart_events[-1].cart_version == 2 and cart_events[0].cart_version == 1


async def test_quoted_only_journey_carries_totals_ttl_and_expiry(shop):
    service, maker, clock, _ = shop
    purchase_id = await quoted_journey(service)
    await pin_all_timestamps(maker, purchase_id, clock.now)
    timeline = await build_timeline(maker, 1, purchase_id, clock=clock)
    assert [event.kind for event in timeline.events] == [
        "purchase_opened",
        "cart_updated",
        "quote_prepared",
    ]
    async with maker() as db:
        confirmation = await db.scalar(
            select(PurchaseConfirmation).where(PurchaseConfirmation.purchase_id == purchase_id)
        )
    assert confirmation is not None
    quote_event = event_of(timeline, "quote_prepared")
    assert quote_event.expires_at is not None
    assert as_utc(quote_event.expires_at) == as_utc(confirmation.expires_at)
    assert quote_event.ttl_seconds == QUOTE_TTL
    assert quote_event.expired is False
    assert quote_event.lines and quote_event.lines[0].quantity == 2
    assert quote_event.amount_minor == confirmation.snapshot["total_minor"]
    assert not any("payment" in event.kind or "order" in event.kind for event in timeline.events)


async def test_quote_expired_flag_flips_with_injected_clock(shop):
    service, maker, clock, _ = shop
    purchase_id = await quoted_journey(service)
    await pin_all_timestamps(maker, purchase_id, clock.now)
    clock.now = clock.now + timedelta(seconds=QUOTE_TTL + 1)
    timeline = await build_timeline(maker, 1, purchase_id, clock=clock)
    assert event_of(timeline, "quote_prepared").expired is True


async def test_awaiting_payment_journey_stops_at_payment_creation(shop):
    service, maker, clock, _ = shop
    view = await service.start(1)
    view = await service.update_cart_lines(
        1, view.purchase_id, [CartLineSelection(sku_id="ANC-02", quantity=1)], view.version
    )
    view = await service.prepare_checkout(1, view.purchase_id, view.version)
    view = await service.commit_checkout(1, view.purchase_id, view.version)
    assert view.stage == "awaiting_payment"
    timeline = await build_timeline(maker, 1, view.purchase_id, clock=clock)
    assert timeline.stage == "awaiting_payment"
    assert [event.kind for event in timeline.events] == [
        "purchase_opened",
        "cart_updated",
        "quote_prepared",
        "order_created",
        "payment_created",
    ]
    payment_block = next(block for block in view.blocks if block.type == "payment")
    assert event_of(timeline, "payment_created").payment_id == payment_block.payment_id


async def test_abandoned_cart_journey_shows_cart_only(shop):
    service, maker, clock, _ = shop
    view = await service.start(1)
    view = await service.update_cart_lines(
        1, view.purchase_id, [CartLineSelection(sku_id="MAT-01", quantity=2)], view.version
    )
    timeline = await build_timeline(maker, 1, view.purchase_id, clock=clock)
    assert timeline.found is True and timeline.stage == "cart"
    assert [event.kind for event in timeline.events] == ["purchase_opened", "cart_updated"]
    assert event_of(timeline, "cart_updated").amount_minor == sum(
        line.subtotal_minor for line in event_of(timeline, "cart_updated").lines or []
    )


async def test_unknown_purchase_id_returns_clean_empty_timeline(shop):
    _, maker, clock, _ = shop
    timeline = await build_timeline(maker, 1, "no-such-purchase", clock=clock)
    assert timeline.found is False
    assert timeline.events == []
    assert timeline.stage is None
    assert timeline.purchase_id == "no-such-purchase"


async def test_timeline_never_leaks_across_owners(shop):
    service, maker, clock, _ = shop
    purchase_id = await paid_journey(service, owner=1)
    foreign = await build_timeline(maker, 2, purchase_id, clock=clock)
    assert foreign.found is False and foreign.events == []
    own = await build_timeline(maker, 1, purchase_id, clock=clock)
    assert own.found is True and own.events


async def test_replayed_idempotent_writes_appear_once(shop):
    service, maker, clock, _ = shop
    purchase_id = await paid_journey(service)
    before = await build_timeline(maker, 1, purchase_id, clock=clock)
    # Replay the consumed confirmation and reconcile status repeatedly: the
    # idempotent replay must not mint duplicate chain events.
    replayed = await service.commit_checkout(1, purchase_id, 0)
    assert replayed.stage == "paid"
    await service.status(1, purchase_id)
    await service.status(1, purchase_id)
    after = await build_timeline(maker, 1, purchase_id, clock=clock)
    assert [(event.kind, event.request_key) for event in after.events] == [
        (event.kind, event.request_key) for event in before.events
    ]
    assert [event.kind for event in after.events].count("order_created") == 1
    assert [event.kind for event in after.events].count("payment_created") == 1
    assert [event.kind for event in after.events].count("payment_completed") == 1


async def test_clock_tie_ordering_is_deterministic_and_causal(shop):
    service, maker, clock, _ = shop
    purchase_id = await paid_journey(service)
    await pin_all_timestamps(maker, purchase_id, clock.now)
    first = await build_timeline(maker, 1, purchase_id, clock=clock)
    second = await build_timeline(maker, 1, purchase_id, clock=clock)
    assert [event.kind for event in first.events] == FULL_PAID_KINDS
    assert first.model_dump() == second.model_dump()
    # The tie is real: order/payment/mock_pay share one generation while the
    # earlier cart/quote operations carry *higher* op generations — only the
    # kind-phase rank keeps the chain causal when timestamps collide.
    quote_generation = first.events[3].generation
    order_generation = first.events[4].generation
    assert quote_generation is not None and order_generation is not None
    assert quote_generation > order_generation
    assert first.events[4].generation == first.events[5].generation == first.events[6].generation


async def test_failed_cart_attempt_recorded_without_amounts(shop):
    service, maker, clock, port = shop
    view = await service.start(1)
    port.fail_next("update_cart", mode="reject")
    view = await service.update_cart_lines(
        1, view.purchase_id, [CartLineSelection(sku_id="ANC-02", quantity=1)], view.version
    )
    assert view.stage == "failed"
    timeline = await build_timeline(maker, 1, view.purchase_id, clock=clock)
    assert [event.kind for event in timeline.events] == [
        "purchase_opened",
        "cart_update_failed",
    ]
    failed = event_of(timeline, "cart_update_failed")
    ops = await operations_of(maker, view.purchase_id)
    assert failed.request_key == ops[0].id
    assert failed.status == "failed"
    assert failed.amount_minor is None and failed.lines is None


async def test_timeline_tool_is_framed_read_only_and_owner_scoped(shop):
    service, maker, clock, _ = shop
    purchase_id = await paid_journey(service)
    registry = ToolRegistry()
    register_timeline_tools(registry, maker, 1, clock=clock)
    tool = registry.get_tool_by_name("commerce_get_order_timeline")
    assert tool is not None
    assert tool.requires_confirmation is False
    result = await tool.handler({"purchase_id": purchase_id})
    assert result["source"] == "provider-data"
    assert result["notice"]
    assert result["order_timeline"].startswith("<order_timeline>")
    assert result["order_timeline"].endswith("</order_timeline>")
    assert f'"purchase_id": "{purchase_id}"' in result["order_timeline"]
    assert '"found": true' in result["order_timeline"]
    # The owner is bound in the closure, never a model argument.
    other = ToolRegistry()
    register_timeline_tools(other, maker, 2, clock=clock)
    foreign = await other.get_tool_by_name("commerce_get_order_timeline").handler(  # type: ignore[union-attr]
        {"purchase_id": purchase_id}
    )
    assert '"found": false' in foreign["order_timeline"]
    with pytest.raises(ValidationError):
        await tool.handler({"purchase_id": ""})
