"""Refund flow, order-id timeline resolution and the recent-orders listing.

Regression coverage for the 2026-11-11 live E2E findings: the timeline tool
only accepted internal purchase ids while customers only ever hold the
provider's order id (o_…/mo_…), "my orders" listings knew nothing about real
store purchases, and refunds of store orders had no authoritative path.
"""

import socket
from datetime import UTC, datetime

import pytest
from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.exceptions import ConflictError
from app.models.database.base import Base
from app.models.database.commerce import PurchaseOperation, PurchaseSession
from app.models.database.user import User
from app.models.schemas.commerce import CartLineSelection, CommerceView
from app.services.commerce.mock_port import MockCommercePort
from app.services.commerce.service import PurchaseService
from app.services.commerce.timeline import (
    build_timeline,
    register_recent_orders_tools,
    register_timeline_tools,
)
from app.services.dialogue.tools import ToolRegistry, mock_refund


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
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/refund.sqlite")

    @event.listens_for(engine.sync_engine, "connect")
    def fk(connection, _):
        connection.execute("PRAGMA foreign_keys=ON")

    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    async with maker() as db:
        db.add(User(id=1, email="refund-one@unit.invalid", hashed_password="!"))
        db.add(User(id=2, email="refund-two@unit.invalid", hashed_password="!"))
        await db.commit()
    clock = Clock()
    port = MockCommercePort(maker, clock=clock, quote_ttl_seconds=30)
    service = PurchaseService(maker, port, clock=clock, quote_ttl_seconds=30)
    yield service, maker, clock, port
    await engine.dispose()


async def paid_journey(service: PurchaseService, owner: int = 1) -> CommerceView:
    view = await service.start(owner)
    view = await service.update_cart_lines(
        owner,
        view.purchase_id,
        [CartLineSelection(sku_id="ANC-02", quantity=1)],
        view.version,
    )
    view = await service.prepare_checkout(owner, view.purchase_id, view.version)
    view = await service.commit_checkout(owner, view.purchase_id, view.version)
    view = await service.mock_pay(owner, view.purchase_id, view.version)
    assert view.stage == "paid"
    return view


async def test_refund_paid_order_once(shop):
    service, maker, _clock, port = shop
    view = await paid_journey(service)
    refunded = await service.refund(1, view.purchase_id, view.version)
    assert refunded.stage == "refunded"

    # A repeated confirmation replays durable state — exactly one refund op.
    replay = await service.refund(1, view.purchase_id, refunded.version)
    assert replay.stage == "refunded"
    async with maker() as db:
        refund_ops = (
            await db.scalars(
                select(PurchaseOperation).where(
                    PurchaseOperation.kind == "refund",
                    PurchaseOperation.purchase_id == view.purchase_id,
                )
            )
        ).all()
    assert len(refund_ops) == 1

    # Provider-level idempotency across DIFFERENT request keys: same receipt.
    order_id = await order_id_for(maker, view.purchase_id)
    first = await port.refund_order(1, order_id, "rk-a")
    second = await port.refund_order(1, order_id, "rk-b")
    assert first.refund_id == second.refund_id
    assert first.status == "refunded"


async def order_id_for(maker: async_sessionmaker[AsyncSession], purchase_id: str) -> str:
    async with maker() as db:
        row: PurchaseSession | None = await db.scalar(
            select(PurchaseSession).where(PurchaseSession.id == purchase_id)
        )
    assert row is not None and row.order_id is not None
    return row.order_id


async def test_refund_requires_paid_stage(shop):
    service, _maker, _clock, _port = shop
    view = await service.start(1)
    view = await service.update_cart_lines(
        1, view.purchase_id, [CartLineSelection(sku_id="ANC-02", quantity=1)], view.version
    )
    view = await service.prepare_checkout(1, view.purchase_id, view.version)
    assert view.stage == "quoted"
    with pytest.raises(ConflictError):
        await service.refund(1, view.purchase_id, view.version)


async def test_timeline_resolves_order_id_and_includes_refund(shop):
    service, maker, _clock, _port = shop
    view = await paid_journey(service)
    await service.refund(1, view.purchase_id, view.version)
    order_id = await order_id_for(maker, view.purchase_id)

    # The E2E failure mode: the customer holds ONLY the provider order id.
    timeline = await build_timeline(maker, 1, order_id)
    assert timeline.found is True
    assert timeline.purchase_id == view.purchase_id
    assert timeline.stage == "refunded"
    kinds = [event.kind for event in timeline.events]
    assert "order_created" in kinds
    assert "refund_completed" in kinds

    # Foreign owners must not resolve other visitors' orders.
    assert (await build_timeline(maker, 2, order_id)).found is False


async def test_recent_orders_tool_lists_store_purchases(shop):
    service, maker, _clock, _port = shop
    view = await paid_journey(service)
    registry = ToolRegistry()
    register_recent_orders_tools(registry, maker, 1)
    tool = registry.get_tool_by_name("commerce_list_recent_orders")
    assert tool is not None
    result = await tool.handler({})
    assert result["source"] == "provider-data"
    assert "order_id" in result["recent_orders"]
    assert view.purchase_id in result["recent_orders"]

    # Owner isolation is closed over, never a model argument.
    other = ToolRegistry()
    register_recent_orders_tools(other, maker, 2)
    other_tool = other.get_tool_by_name("commerce_list_recent_orders")
    assert other_tool is not None
    empty = await other_tool.handler({})
    assert "还没有任何购买记录" in empty["message"]


async def test_timeline_tool_accepts_order_ids(shop):
    service, maker, _clock, _port = shop
    view = await paid_journey(service)
    registry = ToolRegistry()
    register_timeline_tools(registry, maker, 1)
    tool = registry.get_tool_by_name("commerce_get_order_timeline")
    assert tool is not None
    order_id = await order_id_for(maker, view.purchase_id)
    result = await tool.handler({"purchase_id": order_id})
    assert "order_timeline" in result
    assert view.purchase_id in result["order_timeline"]


def test_support_refund_tool_reroutes_store_order_ids():
    result = mock_refund({"order_id": "o_74f96d1a916345efb51d2336102a03dd", "user_id": 1})
    assert result["status"] == "reroute"
    assert "commerce_refund_order" in result["message"]
