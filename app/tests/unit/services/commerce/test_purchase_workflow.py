"""Real domain/provider SQL transactions across text confirmation, timeout and restart.

Target contract under TDD (red until the product changes): an order is created
only by an explicit quoted-stage confirmation, a payment only by a separate
awaiting-payment confirmation, and each confirmation binds to the version of
the purchase view the user actually saw. Views are display facts only — they no
longer publish button actions or action references.
"""

import asyncio
import socket
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError as PydanticValidationError
from sqlalchemy import delete, event, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.exceptions import ConflictError, NotFoundError, ValidationError
from app.models.database.base import Base
from app.models.database.commerce import (
    MockCommerceObject,
    MockCommerceReceipt,
    PurchaseAction,
    PurchaseOperation,
    PurchaseSession,
)
from app.models.database.session import ChatSession
from app.models.database.user import User
from app.models.schemas.commerce import (
    CartLineSelection,
    CatalogFilters,
    CatalogPage,
    CommerceView,
)
from app.services.commerce.mock_port import MockCommercePort
from app.services.commerce.port import ProviderRejected, ProviderUnknown
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


async def quoted(service: PurchaseService) -> CommerceView:
    v = await service.start(1)
    v = await service.update_cart_lines(
        1,
        v.purchase_id,
        [
            CartLineSelection(sku_id="ANC-02", quantity=2),
            CartLineSelection(sku_id="MAT-03", quantity=1),
        ],
        v.version,
    )
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


async def operations(maker: async_sessionmaker[AsyncSession], kind: str | None = None) -> int:
    async with maker() as db:
        statement = select(func.count()).select_from(PurchaseOperation)
        if kind is not None:
            statement = statement.where(PurchaseOperation.kind == kind)
        value = await db.scalar(statement)
        assert isinstance(value, int)
        return value


def quote_id(view: CommerceView) -> str:
    return next(b for b in view.blocks if b.type == "quote").quote.quote_id


async def test_full_journey_needs_separate_order_and_payment_confirmations(shop):
    s, _, m, _ = shop
    v = await quoted(s)
    assert v.mode == "mock" and v.stage == "quoted"
    quote = next(b for b in v.blocks if b.type == "quote")
    assert quote.quote.total_minor == 67100 and quote.quote.currency == "CNY"
    assert await counts(m) == {"order": 0, "payment": 0}
    confirmed = v.version
    v = await s.commit_checkout(1, v.purchase_id, expected_version=confirmed)
    assert v.stage == "awaiting_payment"
    payment = next(b for b in v.blocks if b.type == "payment")
    assert payment.status == "awaiting_payment"  # Confirming the order never pays it.
    assert await counts(m) == {"order": 1, "payment": 1}
    repeated = await s.commit_checkout(1, v.purchase_id, expected_version=confirmed)
    assert repeated.stage == "awaiting_payment" and await operations(m, "order") == 1
    paid = await s.mock_pay(1, v.purchase_id, expected_version=repeated.version)
    assert paid.stage == "paid"
    assert (await s.mock_pay(1, v.purchase_id, expected_version=repeated.version)).stage == "paid"
    assert await operations(m, "mock_pay") == 1
    assert await counts(m) == {"order": 1, "payment": 1}
    assert (await s.start(1)).purchase_id != paid.purchase_id
    assert (await s.status(1, paid.purchase_id)).stage == "paid"


async def test_views_are_facts_only_and_never_publish_button_actions(shop):
    s, _, m, _ = shop
    v = await quoted(s)
    ordered = await s.commit_checkout(1, v.purchase_id, expected_version=v.version)
    for view in (v, ordered):
        assert not hasattr(view, "actions")
        assert "actions" not in view.model_dump(mode="json")
    async with m() as db:
        assert await db.scalar(select(func.count()).select_from(PurchaseAction)) == 0


async def test_active_purchase_is_unique_under_concurrent_start_and_is_owned(shop):
    s, _, m, _ = shop
    a, b = await asyncio.gather(s.start(1), s.start(1))
    other = await s.start(2)
    assert a.purchase_id == b.purchase_id != other.purchase_id
    assert (await s.active(1)).purchase_id == a.purchase_id
    async with m() as db:
        assert await db.scalar(select(func.count()).select_from(PurchaseSession)) == 2


@pytest.mark.parametrize("method", ["status", "search", "product", "commit", "pay"])
async def test_foreign_owner_never_reads_or_mutates_purchase(shop, method):
    s, _, m, _ = shop
    v = await quoted(s)
    before = await operations(m)
    with pytest.raises(NotFoundError):
        if method == "status":
            await s.status(2, v.purchase_id)
        elif method == "search":
            await s.search(2, v.purchase_id, "耳机")
        elif method == "product":
            await s.get_product(2, v.purchase_id, "ANC-02")
        elif method == "commit":
            await s.commit_checkout(2, v.purchase_id, expected_version=v.version)
        else:
            await s.mock_pay(2, v.purchase_id, expected_version=v.version)
    assert await counts(m) == {"order": 0, "payment": 0}
    assert await operations(m) == before


async def test_order_confirmation_is_bound_to_the_quoted_view_version(shop):
    s, _, m, _ = shop
    v = await quoted(s)
    before = await operations(m)
    with pytest.raises(ConflictError):
        await s.commit_checkout(1, v.purchase_id, expected_version=v.version + 1)
    assert await counts(m) == {"order": 0, "payment": 0}
    assert await operations(m) == before


async def test_payment_confirmation_before_any_order_is_refused(shop):
    s, _, m, _ = shop
    v = await quoted(s)
    before = await operations(m)
    with pytest.raises(ConflictError):
        await s.mock_pay(1, v.purchase_id, expected_version=v.version)
    assert await counts(m) == {"order": 0, "payment": 0}
    assert await operations(m) == before


async def test_order_confirmation_without_a_quote_is_refused(shop):
    s, _, m, _ = shop
    v = await s.start(1)
    v = await s.update_cart_lines(
        1, v.purchase_id, [CartLineSelection(sku_id="ANC-02", quantity=2)], v.version
    )
    with pytest.raises(ConflictError):
        await s.commit_checkout(1, v.purchase_id, expected_version=v.version)
    assert await counts(m) == {"order": 0, "payment": 0}
    assert await operations(m, "order") == 0


async def test_material_cart_change_invalidates_the_quoted_confirmation(shop):
    s, _, m, _ = shop
    v = await quoted(s)
    confirmed = v.version
    fresh = await s.update_cart_lines(
        1, v.purchase_id, [CartLineSelection(sku_id="ANC-03", quantity=1)], v.version
    )
    assert fresh.stage == "cart"
    with pytest.raises(ConflictError):
        await s.commit_checkout(1, v.purchase_id, expected_version=confirmed)
    with pytest.raises(ConflictError):
        await s.commit_checkout(1, v.purchase_id, expected_version=fresh.version)
    assert await counts(m) == {"order": 0, "payment": 0}


async def test_expired_quote_needs_a_new_confirmation_of_a_fresh_quote(shop):
    s, _, m, clock = shop
    v = await quoted(s)
    confirmed, stale_quote = v.version, quote_id(v)
    clock.now += timedelta(seconds=31)
    with pytest.raises(ConflictError):
        await s.commit_checkout(1, v.purchase_id, expected_version=confirmed)
    assert await counts(m) == {"order": 0, "payment": 0}
    fresh = await s.prepare_checkout(1, v.purchase_id, confirmed)
    assert fresh.stage == "quoted" and quote_id(fresh) != stale_quote
    ordered = await s.commit_checkout(1, v.purchase_id, expected_version=fresh.version)
    assert ordered.stage == "awaiting_payment"
    assert (await counts(m))["order"] == 1


async def test_order_accept_timeout_recovers_via_read_and_never_hides_a_payment(shop):
    s, p, m, clock = shop
    v = await quoted(s)
    confirmed = v.version
    p.fail_next("order")
    v = await s.commit_checkout(1, v.purchase_id, expected_version=confirmed)
    assert v.stage == "unknown"
    async with m() as db:
        pending = await db.get(PurchaseSession, v.purchase_id)
        assert pending is not None and pending.stage == "unknown"
        saved = await db.get(PurchaseOperation, pending.active_operation_id)
        assert saved is not None and saved.status == "unknown"
    assert await counts(m) == {"order": 1, "payment": 0}
    fresh = PurchaseService(m, MockCommercePort(m, clock=clock), clock=clock)
    recovered = await fresh.reconcile(1, v.purchase_id)
    assert recovered.stage == "awaiting_payment"
    assert await counts(m) == {"order": 1, "payment": 0}  # A read never creates a payment.
    paid = await fresh.mock_pay(1, v.purchase_id, expected_version=recovered.version)
    assert paid.stage == "paid"
    assert await counts(m) == {"order": 1, "payment": 1}
    assert await operations(m, "order") == 1 and await operations(m, "mock_pay") == 1


async def test_payment_record_timeout_recovers_without_a_second_payment(shop):
    s, p, m, clock = shop
    v = await quoted(s)
    p.fail_next("payment")
    v = await s.commit_checkout(1, v.purchase_id, expected_version=v.version)
    # The provider accepted and persisted the payment object before its receipt timed out.
    assert v.stage == "unknown" and await counts(m) == {"order": 1, "payment": 1}
    fresh = PurchaseService(m, MockCommercePort(m, clock=clock), clock=clock)
    recovered = await fresh.reconcile(1, v.purchase_id)
    assert recovered.stage == "awaiting_payment"
    assert await counts(m) == {"order": 1, "payment": 1}
    assert await operations(m, "payment") == 1
    paid = await fresh.mock_pay(1, v.purchase_id, expected_version=recovered.version)
    assert paid.stage == "paid" and await counts(m) == {"order": 1, "payment": 1}


async def test_payment_accept_timeout_confirms_once_without_a_second_attempt(shop):
    s, p, m, clock = shop
    v = await quoted(s)
    v = await s.commit_checkout(1, v.purchase_id, expected_version=v.version)
    awaiting = v.version
    p.fail_next("mock_pay")
    v = await s.mock_pay(1, v.purchase_id, expected_version=awaiting)
    assert v.stage == "unknown"
    fresh = PurchaseService(m, MockCommercePort(m, clock=clock), clock=clock)
    recovered = await fresh.reconcile(1, v.purchase_id)
    assert recovered.stage == "paid"
    assert (await fresh.mock_pay(1, v.purchase_id, expected_version=recovered.version)).stage == (
        "paid"
    )
    assert await operations(m, "mock_pay") == 1
    assert await counts(m) == {"order": 1, "payment": 1}


async def test_processing_payment_never_creates_another_chargeable_attempt(shop):
    s, p, m, _ = shop
    v = await quoted(s)
    v = await s.commit_checkout(1, v.purchase_id, expected_version=v.version)
    awaiting = v.version
    p.fail_next("mock_pay", mode="processing")
    v = await s.mock_pay(1, v.purchase_id, expected_version=awaiting)
    assert v.stage == "processing"
    assert (await s.mock_pay(1, v.purchase_id, expected_version=awaiting)).stage == "processing"
    assert await operations(m, "mock_pay") == 1
    assert await counts(m) == {"order": 1, "payment": 1}
    payment = next(b for b in v.blocks if b.type == "payment")
    await p.settle_payment(1, payment.payment_id)
    assert (await s.status(1, v.purchase_id)).stage == "paid"


async def test_concurrent_confirmation_retries_share_one_checkout_generation(shop):
    s, p, m, _ = shop
    v = await quoted(s)
    confirmed = v.version
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
    pending = asyncio.create_task(s.commit_checkout(1, v.purchase_id, expected_version=confirmed))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        duplicate = await s.commit_checkout(1, v.purchase_id, expected_version=confirmed)
        assert duplicate.stage == "submitting" and calls == 1
    finally:
        release.set()
    assert (await pending).stage == "awaiting_payment"
    assert await operations(m, "order") == 1
    assert await counts(m) == {"order": 1, "payment": 1}


async def test_unknown_without_provider_receipt_never_resubmits_or_opens_new_purchase(shop):
    s, p, m, _ = shop
    v = await quoted(s)
    confirmed = v.version
    calls = 0

    async def unknown(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise ProviderUnknown("No authoritative receipt yet")

    p.create_order = unknown
    v = await s.commit_checkout(1, v.purchase_id, expected_version=confirmed)
    for _ in range(3):
        assert (await s.status(1, v.purchase_id)).stage == "unknown"
        assert (await s.commit_checkout(1, v.purchase_id, expected_version=confirmed)).stage == (
            "unknown"
        )
        assert (await s.start(1)).purchase_id == v.purchase_id
    assert calls == 1 and await counts(m) == {"order": 0, "payment": 0}


async def test_provider_rechecks_confirmed_price_at_atomic_order_creation(shop):
    s, _, m, _ = shop
    v = await quoted(s)
    async with m() as db:
        row = await db.get(MockCommerceObject, "stock:ANC-02")
        row.payload = {**row.payload, "unit_price_minor": 39900}
        await db.commit()
    v = await s.commit_checkout(1, v.purchase_id, expected_version=v.version)
    assert v.stage == "failed" and await counts(m) == {"order": 0, "payment": 0}


@pytest.mark.parametrize("quantity", [True, 0, -1, "2", 1.5, None])
def test_cart_line_selection_requires_a_positive_integer(quantity):
    # Typed validation is the only local quantity rule: positive integers.
    with pytest.raises(PydanticValidationError):
        CartLineSelection(sku_id="ANC-02", quantity=quantity)


@pytest.mark.parametrize(
    "lines",
    [None, [], "ANC-02", [{"sku_id": "ANC-02", "quantity": 2}], ("ANC-02",)],
)
async def test_cart_mutation_requires_a_non_empty_typed_line_list(shop, lines):
    s, _, m, _ = shop
    v = await s.start(1)
    with pytest.raises(ValidationError):
        await s.update_cart_lines(1, v.purchase_id, lines, v.version)
    async with m() as db:
        assert await db.scalar(select(func.count()).select_from(PurchaseOperation)) == 0


async def test_no_local_upper_quantity_bound_provider_limit_rejects(shop):
    s, p, m, _ = shop
    v = await s.start(1)
    # MOU-02 keeps stock 9 but the provider caps it at 7 per order; a locally
    # valid quantity 8 must be refused by the provider, never by local rules.
    v = await s.update_cart_lines(
        1, v.purchase_id, [CartLineSelection(sku_id="MOU-02", quantity=8)], v.version
    )
    assert v.stage == "failed"
    assert "限购" in next(b for b in v.blocks if b.type == "status").message
    async with m() as db:
        # The reservation exists and failed; no provider cart was committed.
        assert (
            await db.scalar(
                select(func.count())
                .select_from(PurchaseOperation)
                .where(PurchaseOperation.kind == "update_cart")
            )
            == 1
        )
        assert (
            await db.scalar(
                select(func.count())
                .select_from(MockCommerceObject)
                .where(MockCommerceObject.kind == "cart")
            )
            == 0
        )


async def test_purchase_and_external_facts_survive_chat_deletion(shop):
    s, _, m, _ = shop
    v = await quoted(s)
    v = await s.commit_checkout(1, v.purchase_id, expected_version=v.version)
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
    v = await s.commit_checkout(1, v.purchase_id, expected_version=v.version)
    async with m() as db:
        op = await db.scalar(
            select(PurchaseOperation).where(
                PurchaseOperation.purchase_id == v.purchase_id, PurchaseOperation.kind == "order"
            )
        )
        receipt = await db.get(MockCommerceReceipt, op.id)
        assert receipt is not None
        request_id, order_quote = op.id, op.request["quote_id"]
    same = await p.create_order(1, order_quote, request_id)
    assert same.order_id == next(b for b in v.blocks if b.type == "payment").order_id
    with pytest.raises(ProviderRejected):
        await p.create_order(1, "other-quote", request_id)
    assert await counts(m) == {"order": 1, "payment": 1}


async def test_catalog_search_is_structured_filtered_sorted_and_paged(shop):
    s, p, m, _ = shop
    v = await s.start(1)
    # The agent passes clean product terms; matching is lexical containment
    # over provider-owned text with no user-language phrase buckets.
    page = await p.search_products(1, "降噪耳机")
    assert [item.sku_id for item in page.items] == ["ANC-02", "ANC-03", "ANC-04", "ANC-05"]
    assert page.total_estimate == 4 and page.next_page_token is None  # ANC-01 is out of stock.
    everything = await p.search_products(1, "降噪耳机", filters=CatalogFilters(in_stock=False))
    assert [item.sku_id for item in everything.items] == [
        "ANC-01",
        "ANC-02",
        "ANC-03",
        "ANC-04",
        "ANC-05",
    ]
    mats = await p.search_products(
        1, "鼠标垫", filters=CatalogFilters(max_price_minor=7000), sort="price_desc"
    )
    assert [item.sku_id for item in mats.items] == ["MAT-04", "MAT-03", "MAT-02", "MAT-01"]
    first = await p.search_products(1, "", page_size=2)
    assert [item.sku_id for item in first.items] == ["ANC-02", "ANC-03"]
    assert first.total_estimate == 26 and first.next_page_token == "2"
    second = await p.search_products(1, "", page_size=2, page_token=first.next_page_token)
    assert [item.sku_id for item in second.items] == ["ANC-04", "ANC-05"]
    newest = await p.search_products(1, "", sort="newest", page_size=3)
    assert [item.sku_id for item in newest.items] == ["MOU-05", "MOU-04", "MOU-02"]
    assert await p.search_products(1, "不存在的冰箱商品") == CatalogPage(items=[], total_estimate=0)
    with pytest.raises(ProviderRejected):
        await p.search_products(1, "", page_size=0)
    with pytest.raises(ProviderRejected):
        await p.search_products(1, "", page_size=51)
    with pytest.raises(ProviderRejected):
        await p.search_products(1, "", page_token="not-a-cursor")
    assert (await s.get_product(1, v.purchase_id, "KEY-01")).stage == "shopping"
    assert await counts(m) == {"order": 0, "payment": 0}


async def test_multi_line_cart_preserves_provider_facts_and_replaces_whole_list(shop):
    s, _, m, _ = shop
    v = await s.start(1)
    v = await s.update_cart_lines(
        1,
        v.purchase_id,
        [
            CartLineSelection(sku_id="ANC-02", quantity=2),
            CartLineSelection(sku_id="MAT-03", quantity=1),
        ],
        v.version,
    )
    cart = next(b for b in v.blocks if b.type == "cart").cart
    lines = {line.sku_id: line for line in cart.lines}
    assert set(lines) == {"ANC-02", "MAT-03"}
    assert lines["ANC-02"].name == "轻音降噪耳机 · 白色"
    assert lines["ANC-02"].unit_price_minor == 30400 and lines["MAT-03"].unit_price_minor == 6300
    assert cart.subtotal_minor == 2 * 30400 + 6300
    # The next request carries the full authoritative line list: absent SKUs drop.
    v = await s.update_cart_lines(
        1, v.purchase_id, [CartLineSelection(sku_id="KEY-01", quantity=1)], v.version
    )
    cart = next(b for b in v.blocks if b.type == "cart").cart
    assert [line.sku_id for line in cart.lines] == ["KEY-01"]
    assert cart.cart_version == 2 and cart.subtotal_minor == 19900
    assert await counts(m) == {"order": 0, "payment": 0}


async def test_external_adapter_is_accepted_and_unknown_labels_are_refused(shop):
    _, p, m, _ = shop
    p.mode = "external"
    service = PurchaseService(m, p)
    assert service.port.mode == "external"
    p.mode = "mystery"
    with pytest.raises(ValueError, match="mock and external adapters only"):
        PurchaseService(m, p)
    assert await counts(m) == {"order": 0, "payment": 0}
