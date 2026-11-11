"""Scoped-delegation spend limits on order creation.

Target contract under TDD: when a single-order or rolling 24h paid spend
limit is configured, ``commit_checkout`` refuses to create the order,
raises a high-priority human handoff ticket (reason ``spend_limit``), and
returns a structured refusal view. Limits off (0) keep the existing
flow; once a human resolves the ticket the delegation continues and the
order proceeds.
"""

import socket
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import event, select, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.models.database.base import Base
from app.models.database.commerce import PurchaseSession
from app.models.database.session import ChatSession
from app.models.database.ticket import HandoffTicket
from app.models.database.user import User
from app.models.schemas.commerce import CartLineSelection, CommerceView
from app.services.commerce.mock_port import MockCommercePort
from app.services.commerce.service import PurchaseService

REASON_SPEND_LIMIT = "spend_limit"


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
                User(id=1, email="limit-one@unit.invalid", hashed_password="!"),
                User(id=2, email="limit-two@unit.invalid", hashed_password="!"),
            ]
        )
        db.add(ChatSession(id=900, user_id=1, session_metadata={"access_mode": "private"}))
        db.add(ChatSession(id=901, user_id=2, session_metadata={"access_mode": "private"}))
        await db.commit()
    clock = Clock()
    port = MockCommercePort(maker, clock=clock, quote_ttl_seconds=30)
    yield maker, clock, port
    await engine.dispose()


def service(
    maker: async_sessionmaker, clock: Clock, port: MockCommercePort, *, single: int, daily: int
) -> PurchaseService:
    return PurchaseService(
        maker,
        port,
        clock=clock,
        quote_ttl_seconds=30,
        spend_limit_single_minor=single,
        spend_limit_daily_minor=daily,
    )


async def quoted(
    view_service: PurchaseService, owner: int = 1, sku: str = "ANC-02"
) -> CommerceView:
    view = await view_service.start(owner)
    view = await view_service.update_cart_lines(
        owner,
        view.purchase_id,
        [CartLineSelection(sku_id=sku, quantity=2)],
        view.version,
    )
    return await view_service.prepare_checkout(owner, view.purchase_id, view.version)


async def ticket_count(maker: async_sessionmaker, reason: str = REASON_SPEND_LIMIT) -> int:
    async with maker() as db:
        rows = (await db.scalars(select(HandoffTicket).where(HandoffTicket.reason == reason))).all()
        return len(rows)


async def stamp_paid_purchase(maker: async_sessionmaker, purchase_id: str, when: datetime) -> None:
    """Pin a paid purchase's updated_at (naive UTC storage, like sqlite).

    The DB column uses CURRENT_TIMESTAMP while the service clock is fixed
    in tests; pinning keeps the 24h window assertion deterministic.
    """

    async with maker() as db:
        await db.execute(
            update(PurchaseSession)
            .where(PurchaseSession.id == purchase_id)
            .values(updated_at=when.replace(tzinfo=None))
        )
        await db.commit()


def limit_block(view: CommerceView):
    return next((block for block in view.blocks if block.type == "spend_limit"), None)


async def test_single_limit_refuses_order_and_escalates(shop):
    maker, clock, port = shop
    purchases = service(maker, clock, port, single=100, daily=0)
    view = await quoted(purchases)
    refused = await purchases.commit_checkout(1, view.purchase_id, view.version)
    # No order was created: the purchase stays quoted with no payment block.
    assert refused.stage == "quoted"
    assert all(block.type != "payment" for block in refused.blocks)
    block = limit_block(refused)
    assert block is not None, "refusal must be structured, not just text"
    assert block.limit_kind == "single"
    assert block.limit_minor == 100
    assert block.attempted_minor > 100
    assert block.ticket_id is not None
    assert await ticket_count(maker) == 1
    async with maker() as db:
        row = await db.scalar(select(HandoffTicket))
        assert row is not None and row.reason == REASON_SPEND_LIMIT
        assert row.session_id == 900  # the owner's chat session
        assert row.priority == 1  # high, same tier as refund threshold


async def test_daily_limit_accumulates_paid_orders_in_24h_window(shop):
    maker, clock, port = shop
    # Unlimited first purchase: pay one order, then a second order in the
    # same 24h window must trip the accumulated daily cap.
    first_service = service(maker, clock, port, single=0, daily=0)
    first = await quoted(first_service)
    first = await first_service.commit_checkout(1, first.purchase_id, first.version)
    paid = await first_service.mock_pay(1, first.purchase_id, first.version)
    assert paid.stage == "paid"
    paid_total = next(b for b in paid.blocks if b.type == "payment").total_minor
    await stamp_paid_purchase(maker, paid.purchase_id, clock.now - timedelta(hours=1))

    capped = service(maker, clock, port, single=0, daily=paid_total + 1)
    second = await quoted(capped, sku="ANC-03")  # different SKU: ANC-02 stock is spent
    refused = await capped.commit_checkout(1, second.purchase_id, second.version)
    block = limit_block(refused)
    assert block is not None and block.limit_kind == "daily"
    assert block.window_paid_minor == paid_total
    assert block.limit_minor == paid_total + 1
    assert block.attempted_minor > 0
    assert await ticket_count(maker) == 1


async def test_daily_limit_ignores_paid_orders_outside_window(shop):
    maker, clock, port = shop
    unlimited = service(maker, clock, port, single=0, daily=0)
    first = await quoted(unlimited)
    first = await unlimited.commit_checkout(1, first.purchase_id, first.version)
    paid = await unlimited.mock_pay(1, first.purchase_id, first.version)
    assert paid.stage == "paid"
    paid_total = next(b for b in paid.blocks if b.type == "payment").total_minor
    # 26 hours old, the paid order no longer counts toward the rolling cap.
    await stamp_paid_purchase(maker, paid.purchase_id, clock.now - timedelta(hours=26))
    # A cap the stale paid order would trip if the window still counted it.
    second = await quoted(unlimited, sku="ANC-03")
    second_total = next(b for b in second.blocks if b.type == "quote").quote.total_minor
    capped = service(maker, clock, port, single=0, daily=paid_total + second_total - 1)
    view = await capped.commit_checkout(1, second.purchase_id, second.version)
    assert view.stage == "awaiting_payment"
    assert limit_block(view) is None


async def test_disabled_limits_do_not_change_the_existing_flow(shop):
    maker, clock, port = shop
    purchases = service(maker, clock, port, single=0, daily=0)
    view = await quoted(purchases)
    view = await purchases.commit_checkout(1, view.purchase_id, view.version)
    assert view.stage == "awaiting_payment"
    assert limit_block(view) is None
    assert await ticket_count(maker) == 0


async def test_under_limit_order_proceeds_and_no_ticket_exists(shop):
    maker, clock, port = shop
    view = await quoted(service(maker, clock, port, single=0, daily=0))
    total = next(b for b in view.blocks if b.type == "quote").quote.total_minor
    purchases = service(maker, clock, port, single=total, daily=total)
    view = await quoted(purchases)
    result = await purchases.commit_checkout(1, view.purchase_id, view.version)
    assert result.stage == "awaiting_payment"
    assert limit_block(result) is None
    assert await ticket_count(maker) == 0


async def test_resolved_ticket_delegates_and_order_proceeds(shop):
    maker, clock, port = shop
    purchases = service(maker, clock, port, single=100, daily=0)
    view = await quoted(purchases)
    refused = await purchases.commit_checkout(1, view.purchase_id, view.version)
    assert limit_block(refused) is not None
    assert await ticket_count(maker) == 1
    # A human resolves the escalation (agent workspace does exactly this update).
    async with maker() as db:
        await db.execute(
            update(HandoffTicket)
            .values(status="resolved", assigned_to=2)
            .where(HandoffTicket.reason == REASON_SPEND_LIMIT)
        )
        await db.commit()
    # The purchase is still quoted and unexpired: retry after resolve proceeds.
    current = await purchases.status(1, view.purchase_id)
    assert current.stage == "quoted"
    delegated = await purchases.commit_checkout(1, view.purchase_id, current.version)
    assert delegated.stage == "awaiting_payment"
    assert limit_block(delegated) is None
    assert await ticket_count(maker) == 1  # no duplicate escalation


async def test_refusal_does_not_double_escalate_on_repeat(shop):
    maker, clock, port = shop
    purchases = service(maker, clock, port, single=100, daily=0)
    view = await quoted(purchases)
    first = await purchases.commit_checkout(1, view.purchase_id, view.version)
    assert limit_block(first) is not None
    current = await purchases.status(1, view.purchase_id)
    second = await purchases.commit_checkout(1, view.purchase_id, current.version)
    assert limit_block(second) is not None
    assert await ticket_count(maker) == 1


async def test_limits_are_per_owner(shop):
    maker, clock, port = shop
    purchases = service(maker, clock, port, single=100, daily=0)
    owner_one = await quoted(purchases, owner=1)
    refused = await purchases.commit_checkout(1, owner_one.purchase_id, owner_one.version)
    assert limit_block(refused) is not None
    # Owner 2 is a different visitor scope: the same limits do not block a
    # fresh purchase by someone else (their own single limit still applies).
    owner_two = await quoted(purchases, owner=2)
    view = await purchases.commit_checkout(2, owner_two.purchase_id, owner_two.version)
    assert limit_block(view) is not None  # same single cap applies to them too
    async with maker() as db:
        rows = (
            await db.scalars(
                select(HandoffTicket).where(HandoffTicket.reason == REASON_SPEND_LIMIT)
            )
        ).all()
        assert {row.session_id for row in rows} == {900, 901}
