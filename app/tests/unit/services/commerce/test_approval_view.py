"""Structured approval card projection on the commerce view.

Target contract under TDD: while a purchase waits for the customer's
explicit order / payment confirmation, the view carries one structured
``approval`` object whose every field comes from the durable quote
confirmation — the LLM never authors card data. The card disappears once
its window closes and never leaks across visitors.
"""

import socket
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.exceptions import NotFoundError
from app.models.database.base import Base
from app.models.database.commerce import PurchaseConfirmation
from app.models.database.user import User
from app.models.schemas.commerce import ApprovalCard, CartLineSelection, CommerceView
from app.services.commerce.mock_port import MockCommercePort, as_utc
from app.services.commerce.service import PurchaseService


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
                User(id=1, email="approval-one@unit.invalid", hashed_password="!"),
                User(id=2, email="approval-two@unit.invalid", hashed_password="!"),
            ]
        )
        await db.commit()
    clock = Clock()
    port = MockCommercePort(maker, clock=clock, quote_ttl_seconds=30)
    service = PurchaseService(maker, port, clock=clock, quote_ttl_seconds=30)
    yield service, maker, clock
    await engine.dispose()


async def quoted(service: PurchaseService) -> CommerceView:
    view = await service.start(1)
    view = await service.update_cart_lines(
        1,
        view.purchase_id,
        [CartLineSelection(sku_id="ANC-02", quantity=2)],
        view.version,
    )
    return await service.prepare_checkout(1, view.purchase_id, view.version)


def approval_of(view: CommerceView) -> ApprovalCard | None:
    return view.approval


async def confirmation_row(maker: async_sessionmaker[AsyncSession], view: CommerceView):
    block = next(block for block in view.blocks if block.type == "quote")
    async with maker() as db:
        return await db.get(PurchaseConfirmation, block.confirmation_id)


async def test_quoted_view_carries_complete_approval_card(shop):
    service, maker, _ = shop
    view = await quoted(service)
    assert view.stage == "quoted"
    approval = approval_of(view)
    assert isinstance(approval, ApprovalCard)
    assert approval.purchase_id == view.purchase_id
    assert approval.stage == "quoted"
    assert approval.action == "confirm_order"
    assert approval.action_label == "确认下单"
    # Line details mirror the quote snapshot exactly: one row per SKU with
    # quantity and unit price from the provider's durable confirmation.
    assert [(line.sku_id, line.quantity) for line in approval.lines] == [("ANC-02", 2)]
    line = approval.lines[0]
    assert line.name and line.unit_price_minor >= 0 and line.subtotal_minor >= 0
    confirmation = await confirmation_row(maker, view)
    assert confirmation is not None
    # Card total is the durable confirmation's total (the same snapshot an
    # order must match), not any model-narrated amount.
    assert approval.total_minor == confirmation.snapshot["total_minor"]
    assert approval.currency == "CNY"
    # quote_hash binds the card to the exact confirmation the order must match.
    assert approval.quote_hash == confirmation.snapshot_hash
    assert as_utc(approval.expires_at) == as_utc(confirmation.expires_at)
    assert approval.purchase_version == view.version
    assert approval.generation is not None and approval.generation >= 0


async def test_card_total_equals_quote_total_exactly(shop):
    service, _, _ = shop
    view = await quoted(service)
    quote = next(block for block in view.blocks if block.type == "quote").quote
    approval = approval_of(view)
    assert approval is not None
    assert approval.total_minor == quote.total_minor
    assert approval.currency == quote.currency
    assert [(line.sku_id, line.quantity, line.unit_price_minor) for line in approval.lines] == [
        (line.sku_id, line.quantity, line.unit_price_minor) for line in quote.lines
    ]


async def test_awaiting_payment_view_carries_payment_approval_card(shop):
    service, maker, _ = shop
    view = await quoted(service)
    confirmation = await confirmation_row(maker, view)
    assert confirmation is not None
    view = await service.commit_checkout(1, view.purchase_id, view.version)
    assert view.stage == "awaiting_payment"
    approval = approval_of(view)
    assert isinstance(approval, ApprovalCard)
    assert approval.stage == "awaiting_payment"
    assert approval.action == "confirm_payment"
    assert approval.action_label == "确认支付"
    assert approval.purchase_id == view.purchase_id
    # Payment card still quotes the ordered amount from the same confirmation.
    assert approval.total_minor == confirmation.snapshot["total_minor"]
    assert approval.quote_hash == confirmation.snapshot_hash


async def test_approval_disappears_after_quote_expiry(shop):
    service, maker, clock = shop
    view = await quoted(service)
    assert approval_of(view) is not None
    confirmation = await confirmation_row(maker, view)
    assert confirmation is not None
    clock.now = as_utc(confirmation.expires_at) + timedelta(seconds=1)
    later = await service.status(1, view.purchase_id)
    assert later.stage == "quoted"
    assert approval_of(later) is None


async def test_stages_outside_confirmation_windows_have_no_approval(shop):
    service, _, _ = shop
    view = await service.start(1)
    assert view.stage == "shopping" and approval_of(view) is None
    view = await service.update_cart_lines(
        1, view.purchase_id, [CartLineSelection(sku_id="ANC-02", quantity=1)], view.version
    )
    assert view.stage == "cart" and approval_of(view) is None
    quoted_view = await service.prepare_checkout(1, view.purchase_id, view.version)
    paid = await service.commit_checkout(1, quoted_view.purchase_id, quoted_view.version)
    paid = await service.mock_pay(1, paid.purchase_id, paid.version)
    assert paid.stage == "paid" and approval_of(paid) is None


async def test_approval_never_leaks_across_visitors(shop):
    service, _, _ = shop
    view = await quoted(service)
    assert view.approval is not None
    # A different owner cannot even load the purchase, so no card can leak.
    with pytest.raises(NotFoundError):
        await service.status(2, view.purchase_id)
    own = await service.start(2)
    assert own.approval is None
