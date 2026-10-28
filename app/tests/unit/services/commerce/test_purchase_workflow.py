"""Real domain/provider SQL transactions across confirmation, timeout and restart."""

import asyncio
import socket
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import delete, event, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.exceptions import ConflictError, NotFoundError, ValidationError
from app.models.database.base import Base
from app.models.database.commerce import (
    MockCommerceObject,
    MockCommerceReceipt,
    PurchaseOperation,
    PurchaseSession,
)
from app.models.database.session import ChatSession
from app.models.database.user import User
from app.models.schemas.commerce import ActionKind, CommerceAction, CommerceView
from app.services.commerce.mock_port import MockCommercePort
from app.services.commerce.port import ProviderUnknown
from app.services.commerce.service import PurchaseService


class Clock:
    now = datetime(2026, 10, 28, tzinfo=UTC)

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
                User(id=1, email="commerce-one@unit.invalid", hashed_password="!"),
                User(id=2, email="commerce-two@unit.invalid", hashed_password="!"),
            ]
        )
        await db.commit()
    clock = Clock()
    port = MockCommercePort(maker, clock=clock, quote_ttl_seconds=30)
    service = PurchaseService(maker, port, clock=clock, quote_ttl_seconds=30)
    yield service, port, maker, clock
    await engine.dispose()


def action(view: CommerceView, kind: ActionKind) -> CommerceAction:
    return next(a for a in view.actions if a.kind == kind)


async def quoted(service: PurchaseService) -> CommerceView:
    v = await service.start(1)
    v = await service.set_cart_line(1, v.purchase_id, "ANC-BLACK", 2, v.version)
    return await service.prepare_checkout(1, v.purchase_id, v.version)


async def counts(maker: async_sessionmaker[AsyncSession]) -> dict[str, int]:
    result: dict[str, int] = {}
    async with maker() as db:
        for kind in ("order", "payment"):
            count = await db.scalar(
                select(func.count())
                .select_from(MockCommerceObject)
                .where(MockCommerceObject.kind == kind)
            )
            assert isinstance(count, int)
            result[kind] = count
    return result


async def test_full_mock_journey_requires_two_distinct_user_actions(shop):
    s, _, m, _ = shop
    v = await quoted(s)
    commit = action(v, "commit_checkout")
    assert v.mode == "mock" and v.stage == "quoted"
    quote = next(b for b in v.blocks if b.type == "quote")
    assert quote.quote.total_minor == 59800 and quote.quote.currency == "CNY"
    assert await counts(m) == {"order": 0, "payment": 0}
    v = await s.execute_action(1, commit.action_ref, expected_version=commit.expected_version)
    assert v.stage == "awaiting_payment" and await counts(m) == {"order": 1, "payment": 1}
    again = await s.commit_checkout(1, commit.action_ref)
    assert again.stage == "awaiting_payment" and await counts(m) == {"order": 1, "payment": 1}
    pay = action(v, "mock_pay")
    paid = await s.mock_pay(1, pay.action_ref)
    assert paid.stage == "paid"
    assert (await s.mock_pay(1, pay.action_ref)).stage == "paid"
    assert await counts(m) == {"order": 1, "payment": 1}
    assert (await s.start(1)).purchase_id != v.purchase_id
    assert (await s.status(1, v.purchase_id)).stage == "paid"


async def test_active_purchase_is_unique_under_concurrent_start_and_is_owned(shop):
    s, _, m, _ = shop
    a, b = await asyncio.gather(s.start(1), s.start(1))
    other = await s.start(2)
    assert a.purchase_id == b.purchase_id != other.purchase_id
    assert (await s.active(1)).purchase_id == a.purchase_id
    async with m() as db:
        assert await db.scalar(select(func.count()).select_from(PurchaseSession)) == 2


@pytest.mark.parametrize("method", ["status", "search", "product", "action"])
async def test_foreign_owner_never_reads_or_mutates_purchase(shop, method):
    s, _, m, _ = shop
    v = await quoted(s)
    with pytest.raises(NotFoundError):
        if method == "status":
            await s.status(2, v.purchase_id)
        elif method == "search":
            await s.search(2, v.purchase_id, "耳机")
        elif method == "product":
            await s.get_product(2, v.purchase_id, "ANC-BLACK")
        else:
            await s.execute_action(2, action(v, "commit_checkout").action_ref)
    assert await counts(m) == {"order": 0, "payment": 0}


async def test_action_version_is_bound_to_original_descriptor_not_current_view(shop):
    s, _, m, _ = shop
    v = await quoted(s)
    a = action(v, "commit_checkout")
    with pytest.raises(ConflictError):
        await s.execute_action(1, a.action_ref, expected_version=a.expected_version + 1)
    assert await counts(m) == {"order": 0, "payment": 0}


async def test_material_cart_change_invalidates_old_confirmation(shop):
    s, _, m, _ = shop
    v = await quoted(s)
    old = action(v, "commit_checkout")
    fresh = await s.set_cart_line(1, v.purchase_id, "ANC-WHITE", 1, v.version)
    with pytest.raises(ConflictError):
        await s.commit_checkout(1, old.action_ref)
    assert fresh.stage == "cart" and await counts(m) == {"order": 0, "payment": 0}


async def test_expired_quote_needs_new_user_confirmation(shop):
    s, _, m, clock = shop
    v = await quoted(s)
    old = action(v, "commit_checkout")
    clock.now += timedelta(seconds=31)
    with pytest.raises(ConflictError):
        await s.commit_checkout(1, old.action_ref)
    assert await counts(m) == {"order": 0, "payment": 0}
    new = await s.prepare_checkout(1, v.purchase_id, v.version)
    assert action(new, "commit_checkout").action_ref != old.action_ref


@pytest.mark.parametrize("kind", ["order", "payment", "mock_pay"])
async def test_external_accept_then_timeout_reconciles_original_receipt_after_restart(shop, kind):
    s, p, m, clock = shop
    v = await quoted(s)
    commit = action(v, "commit_checkout")
    if kind == "mock_pay":
        v = await s.commit_checkout(1, commit.action_ref)
        target = action(v, "mock_pay")
    else:
        target = commit
    p.fail_next(kind)
    v = await s.execute_action(1, target.action_ref, expected_version=target.expected_version)
    assert v.stage == "unknown"
    async with m() as db:
        pending = await db.get(PurchaseSession, v.purchase_id)
        assert pending is not None and pending.stage == "unknown"
        saved_operation = await db.get(PurchaseOperation, pending.active_operation_id)
        assert saved_operation is not None and saved_operation.status == "unknown"
    fresh = PurchaseService(m, MockCommercePort(m, clock=clock), clock=clock)
    v = await fresh.reconcile(1, v.purchase_id)
    assert v.stage == ("paid" if kind == "mock_pay" else "awaiting_payment")
    current = await counts(m)
    assert current["order"] == 1
    if kind == "order":
        assert current["payment"] == 0  # GET recovery never creates a payment.
        v = await fresh.commit_checkout(1, commit.action_ref)
    assert (await counts(m))["payment"] == 1
    assert (await fresh.status(1, v.purchase_id)).stage == v.stage


async def test_processing_payment_never_creates_another_chargeable_attempt(shop):
    s, p, m, _ = shop
    v = await quoted(s)
    v = await s.commit_checkout(1, action(v, "commit_checkout").action_ref)
    pay = action(v, "mock_pay")
    p.fail_next("mock_pay", mode="processing")
    v = await s.mock_pay(1, pay.action_ref)
    assert v.stage == "processing" and not any(a.kind == "mock_pay" for a in v.actions)
    assert (await s.mock_pay(1, pay.action_ref)).stage == "processing"
    assert await counts(m) == {"order": 1, "payment": 1}
    payment = next(b for b in v.blocks if b.type == "payment")
    await p.settle_payment(1, payment.payment_id)
    assert (await s.status(1, v.purchase_id)).stage == "paid"


async def test_concurrent_confirmation_retries_share_one_checkout_generation(shop):
    s, p, m, _ = shop
    v = await quoted(s)
    a = action(v, "commit_checkout")
    entered, release = asyncio.Event(), asyncio.Event()
    original = p.create_order
    calls = 0

    async def held(*args, **kwargs):
        nonlocal calls
        calls += 1
        entered.set()
        await asyncio.wait_for(release.wait(), 3)
        return await original(*args, **kwargs)

    p.create_order = held
    pending = asyncio.create_task(s.commit_checkout(1, a.action_ref))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        duplicate = await s.commit_checkout(1, a.action_ref)
        assert duplicate.stage == "submitting" and calls == 1
    finally:
        release.set()
    assert (await pending).stage == "awaiting_payment"
    async with m() as db:
        assert (
            await db.scalar(
                select(func.count())
                .select_from(PurchaseOperation)
                .where(PurchaseOperation.kind == "order")
            )
            == 1
        )
    assert await counts(m) == {"order": 1, "payment": 1}


async def test_unknown_without_provider_receipt_never_resubmits_or_opens_new_purchase(shop):
    s, p, m, _ = shop
    v = await quoted(s)
    a = action(v, "commit_checkout")
    calls = 0

    async def unknown(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise ProviderUnknown("No authoritative receipt yet")

    p.create_order = unknown
    v = await s.commit_checkout(1, a.action_ref)
    for _ in range(3):
        assert (await s.status(1, v.purchase_id)).stage == "unknown"
        assert (await s.commit_checkout(1, a.action_ref)).stage == "unknown"
        assert (await s.start(1)).purchase_id == v.purchase_id
    assert calls == 1 and await counts(m) == {"order": 0, "payment": 0}


async def test_provider_rechecks_confirmed_price_at_atomic_order_creation(shop):
    s, _, m, _ = shop
    v = await quoted(s)
    async with m() as db:
        row = await db.get(MockCommerceObject, "stock:ANC-BLACK")
        row.payload = {**row.payload, "unit_price_minor": 39900}
        await db.commit()
    v = await s.commit_checkout(1, action(v, "commit_checkout").action_ref)
    assert v.stage == "failed" and await counts(m) == {"order": 0, "payment": 0}


@pytest.mark.parametrize("quantity", [True, 0, 11, "2"])
async def test_invalid_quantity_is_refused_before_external_mutation(shop, quantity):
    s, _, m, _ = shop
    v = await s.start(1)
    with pytest.raises(ValidationError):
        await s.set_cart_line(1, v.purchase_id, "ANC-BLACK", quantity, v.version)
    async with m() as db:
        assert await db.scalar(select(func.count()).select_from(PurchaseOperation)) == 0


async def test_purchase_and_external_facts_survive_chat_deletion(shop):
    s, _, m, _ = shop
    v = await quoted(s)
    v = await s.commit_checkout(1, action(v, "commit_checkout").action_ref)
    async with m() as db:
        chat = ChatSession(user_id=1, session_metadata={"access_mode": "private"})
        db.add(chat)
        await db.commit()
        await db.execute(delete(ChatSession).where(ChatSession.id == chat.id))
        await db.commit()
    assert (await s.active(1)).purchase_id == v.purchase_id
    assert await counts(m) == {"order": 1, "payment": 1}


async def test_mock_provider_idempotency_receipt_is_separate_and_parameter_bound(shop):
    s, p, m, _ = shop
    v = await quoted(s)
    a = action(v, "commit_checkout")
    v = await s.commit_checkout(1, a.action_ref)
    async with m() as db:
        op = await db.scalar(
            select(PurchaseOperation).where(
                PurchaseOperation.purchase_id == v.purchase_id, PurchaseOperation.kind == "order"
            )
        )
        receipt = await db.get(MockCommerceReceipt, op.id)
        assert receipt is not None
        request_id, quote_id = op.id, op.request["quote_id"]
    same = await p.create_order(1, quote_id, request_id)
    assert same.order_id == next(b for b in v.blocks if b.type == "payment").order_id
    from app.services.commerce.port import ProviderRejected

    with pytest.raises(ProviderRejected):
        await p.create_order(1, "other-quote", request_id)
    assert await counts(m) == {"order": 1, "payment": 1}


async def test_mock_catalog_search_is_bounded_and_does_not_invent_unknown_products(shop):
    s, p, m, _ = shop
    v = await s.start(1)
    headphones = await p.search_products(1, "我想买降噪耳机")
    assert len(headphones) == 2 and all(item.product_id == "headphones-anc" for item in headphones)
    black = await p.search_products(1, "黑色耳机")
    assert [item.sku_id for item in black] == ["ANC-BLACK"]
    assert await p.search_products(1, "不存在的冰箱商品") == []
    assert (await s.get_product(1, v.purchase_id, "KEYBOARD-87")).stage == "shopping"
    assert await counts(m) == {"order": 0, "payment": 0}


async def test_non_mock_adapter_cannot_execute_under_a_mock_payment_label(shop):
    _, p, m, _ = shop
    p.mode = "external"
    with pytest.raises(ValueError, match="mock adapters only"):
        PurchaseService(m, p)
    assert await counts(m) == {"order": 0, "payment": 0}
