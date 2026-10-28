"""Durable core semantics: carts, quotes, orders, payments, receipts, faults."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.commerce_provider.catalog import VirtualCatalog
from app.commerce_provider.core import CommerceProviderCore
from app.commerce_provider.errors import (
    ProviderConflictError,
    ProviderNotFoundError,
    ProviderRejectedError,
    ProviderUnknownError,
)
from app.commerce_provider.faults import FaultController
from app.commerce_provider.storage import Database
from app.models.schemas.commerce import CartLineSelection

OWNER = 1


def _selections(core: CommerceProviderCore, count: int = 1) -> list[CartLineSelection]:
    page = core.catalog.search(query="显示器", page_size=50)
    usable = [item for item in page.items if item.available_quantity >= 1]
    assert len(usable) >= count
    return [CartLineSelection(sku_id=usable[index].sku_id, quantity=1) for index in range(count)]


async def _purchased_cart(
    core: CommerceProviderCore, selections: list[CartLineSelection], request_key: str
):
    return await core.update_cart_lines(OWNER, None, None, selections, request_key)


async def test_full_purchase_flow(core: CommerceProviderCore) -> None:
    selections = _selections(core, count=2)
    cart = await core.update_cart_lines(OWNER, None, None, selections, "cart-1")
    assert cart.cart_version == 1
    assert len(cart.lines) == 2
    assert cart.subtotal_minor == sum(line.subtotal_minor for line in cart.lines)

    updated = await core.update_cart_lines(OWNER, cart.cart_id, 1, selections[:1], "cart-2")
    assert updated.cart_version == 2
    assert len(updated.lines) == 1

    stored = await core.get_cart(OWNER, cart.cart_id)
    assert stored == updated

    quote = await core.prepare_checkout(OWNER, cart.cart_id, 2, "quote-1")
    assert quote.cart_id == cart.cart_id
    assert quote.lines == updated.lines
    assert quote.total_minor == updated.subtotal_minor + quote.shipping_minor

    order = await core.create_order(OWNER, quote.quote_id, "order-1")
    assert order.status == "awaiting_payment"
    assert order.total_minor == quote.total_minor

    with pytest.raises(ProviderRejectedError, match="已被使用"):
        await core.create_order(OWNER, quote.quote_id, "order-2")
    with pytest.raises(ProviderRejectedError, match="已下单"):
        await core.update_cart_lines(OWNER, cart.cart_id, 3, selections[:1], "cart-3")

    payment = await core.create_payment(OWNER, order.order_id, "pay-1")
    assert payment.status == "awaiting_payment"
    assert payment.total_minor == order.total_minor
    with pytest.raises(ProviderRejectedError, match="已有付款记录"):
        await core.create_payment(OWNER, order.order_id, "pay-again")

    paid = await core.mock_pay(OWNER, payment.payment_id, "pay-2")
    assert paid.status == "paid"
    assert (await core.get_order(OWNER, order.order_id)).status == "paid"
    assert (await core.get_status(OWNER, payment.payment_id)).status == "paid"


async def test_update_cart_enforces_versions_and_clears_lines(
    core: CommerceProviderCore,
) -> None:
    selections = _selections(core, count=1)
    cart = await core.update_cart_lines(OWNER, None, None, selections, "cart-1")
    with pytest.raises(ProviderRejectedError, match="版本变化"):
        await core.update_cart_lines(OWNER, cart.cart_id, 99, selections, "cart-2")
    with pytest.raises(ProviderRejectedError, match="版本变化"):
        await core.update_cart_lines(OWNER, cart.cart_id, None, selections, "cart-3")

    cleared = await core.update_cart_lines(OWNER, cart.cart_id, 1, [], "cart-4")
    assert cleared.lines == []
    assert cleared.subtotal_minor == 0
    assert cleared.cart_version == 2
    with pytest.raises(ProviderRejectedError, match="为空"):
        await core.prepare_checkout(OWNER, cart.cart_id, 2, "quote-1")


async def test_update_cart_rejects_duplicates_and_unknown_skus(
    core: CommerceProviderCore,
) -> None:
    selection = _selections(core, count=1)[0]
    with pytest.raises(ProviderRejectedError, match="重复"):
        await core.update_cart_lines(OWNER, None, None, [selection, selection], "cart-1")
    with pytest.raises(ProviderNotFoundError, match="商品不存在"):
        await core.update_cart_lines(
            OWNER,
            None,
            None,
            [CartLineSelection(sku_id="SKU-does-not-exist", quantity=1)],
            "cart-2",
        )


async def test_purchase_caps_and_stock_are_enforced(core: CommerceProviderCore) -> None:
    item = None
    token = None
    for _ in range(5):
        page = core.catalog.search(query="显示器", page_size=50, page_token=token)
        item = next(
            (
                entry
                for entry in page.items
                if entry.max_order_quantity
                <= entry.available_quantity
                < 2 * entry.max_order_quantity
            ),
            None,
        )
        if item is not None:
            break
        token = page.next_page_token
        if token is None:
            break
    assert item is not None, "expected a monitor whose stock fits exactly one order lot"
    with pytest.raises(ProviderRejectedError, match="限购"):
        await core.update_cart_lines(
            OWNER,
            None,
            None,
            [CartLineSelection(sku_id=item.sku_id, quantity=item.max_order_quantity + 1)],
            "cart-1",
        )

    cart = await core.update_cart_lines(
        OWNER,
        None,
        None,
        [CartLineSelection(sku_id=item.sku_id, quantity=item.max_order_quantity)],
        "cart-2",
    )
    quote = await core.prepare_checkout(OWNER, cart.cart_id, 1, "quote-1")
    order = await core.create_order(OWNER, quote.quote_id, "order-1")

    remaining = await core.get_product(OWNER, item.sku_id)
    assert remaining.available_quantity == item.available_quantity - item.max_order_quantity

    # Exhaust the stock: the same lot can no longer be bought again.
    with pytest.raises(ProviderRejectedError, match="库存不足"):
        second_cart = await core.update_cart_lines(
            OWNER,
            None,
            None,
            [CartLineSelection(sku_id=item.sku_id, quantity=item.max_order_quantity)],
            "cart-3",
        )
        second_quote = await core.prepare_checkout(OWNER, second_cart.cart_id, 1, "quote-2")
        await core.create_order(OWNER, second_quote.quote_id, "order-2")
    assert (await core.get_order(OWNER, order.order_id)).status == "awaiting_payment"


async def test_quote_expiry_and_uniqueness(core: CommerceProviderCore, clock) -> None:
    selections = _selections(core, count=1)
    cart = await core.update_cart_lines(OWNER, None, None, selections, "cart-1")
    quote = await core.prepare_checkout(OWNER, cart.cart_id, 1, "quote-1")
    replay = await core.prepare_checkout(OWNER, cart.cart_id, 1, "quote-2")
    assert replay.model_dump() == quote.model_dump()

    clock.advance(301)
    with pytest.raises(ProviderRejectedError, match="已过期"):
        await core.create_order(OWNER, quote.quote_id, "order-1")

    # After expiry a fresh quote is allowed for the still-current cart version.
    fresh = await core.prepare_checkout(OWNER, cart.cart_id, 1, "quote-3")
    assert fresh.quote_id != quote.quote_id


async def test_price_drift_invalidates_stale_carts(core: CommerceProviderCore) -> None:
    """Carts whose stored price no longer matches cannot be quoted."""

    selections = _selections(core, count=1)
    cart = await core.update_cart_lines(OWNER, None, None, selections, "cart-1")

    def tamper(connection) -> None:
        connection.execute(
            "UPDATE cart_lines SET unit_price_minor = 1 WHERE cart_id = ?", (cart.cart_id,)
        )

    await core.database.write(tamper)
    with pytest.raises(ProviderRejectedError, match="价格变化"):
        await core.prepare_checkout(OWNER, cart.cart_id, 1, "quote-1")


async def test_idempotency_replay_and_conflict(core: CommerceProviderCore) -> None:
    selections = _selections(core, count=1)
    first = await core.update_cart_lines(OWNER, None, None, selections, "cart-1")
    replay = await core.update_cart_lines(OWNER, None, None, selections, "cart-1")
    assert replay == first

    with pytest.raises(ProviderConflictError):
        await core.update_cart_lines(
            OWNER,
            None,
            None,
            [CartLineSelection(sku_id=selections[0].sku_id, quantity=2)],
            "cart-1",
        )

    receipt = await core.lookup_operation(OWNER, "cart-1")
    assert receipt is not None
    assert receipt["kind"] == "update_cart_lines"
    assert receipt["status"] == "completed"
    assert receipt["result"]["cart_id"] == first.cart_id

    assert await core.lookup_operation(OWNER, "missing-key") is None
    assert await core.lookup_operation(2, "cart-1") is None


async def test_owner_isolation_on_reads(core: CommerceProviderCore) -> None:
    selections = _selections(core, count=1)
    cart = await core.update_cart_lines(OWNER, None, None, selections, "cart-1")
    with pytest.raises(ProviderNotFoundError):
        await core.get_cart(2, cart.cart_id)
    with pytest.raises(ProviderRejectedError):
        await core.search_products(0, "显示器")
    with pytest.raises(ProviderRejectedError):
        await core.get_product(-1, selections[0].sku_id)


async def test_state_survives_reopen(tmp_path: Path, clock, core: CommerceProviderCore) -> None:
    selections = _selections(core, count=1)
    cart = await core.update_cart_lines(OWNER, None, None, selections, "cart-1")
    quote = await core.prepare_checkout(OWNER, cart.cart_id, 1, "quote-1")
    order = await core.create_order(OWNER, quote.quote_id, "order-1")
    payment = await core.create_payment(OWNER, order.order_id, "pay-1")
    await core.mock_pay(OWNER, payment.payment_id, "pay-2")

    database = Database(str(tmp_path / "provider.sqlite3"))
    await database.startup()
    reopened = CommerceProviderCore(
        database, VirtualCatalog(), clock=clock, faults=FaultController(database, enabled=True)
    )

    assert (await reopened.get_order(OWNER, order.order_id)).status == "paid"
    assert (await reopened.get_status(OWNER, payment.payment_id)).status == "paid"
    reopened_cart = await reopened.get_cart(OWNER, cart.cart_id)
    assert reopened_cart.cart_version == 2  # order creation bumped it
    receipt = await reopened.lookup_operation(OWNER, "order-1")
    assert receipt is not None
    assert receipt["result"]["order_id"] == order.order_id
    with pytest.raises(ProviderRejectedError, match="已被使用"):
        await reopened.create_order(OWNER, quote.quote_id, "order-late")


async def test_fault_reject_is_not_committed(core: CommerceProviderCore) -> None:
    assert core.faults is not None
    await core.faults.plan(kind="create_order", mode="reject")
    selections = _selections(core, count=1)
    cart = await core.update_cart_lines(OWNER, None, None, selections, "cart-1")
    quote = await core.prepare_checkout(OWNER, cart.cart_id, 1, "quote-1")
    with pytest.raises(ProviderRejectedError, match="明确拒绝"):
        await core.create_order(OWNER, quote.quote_id, "order-1")
    assert await core.lookup_operation(OWNER, "order-1") is None
    # The fault was consumed; retrying the same key succeeds.
    order = await core.create_order(OWNER, quote.quote_id, "order-1")
    assert order.status == "awaiting_payment"


async def test_fault_unknown_after_commit_leaves_reconcilable_receipt(
    core: CommerceProviderCore,
) -> None:
    selections = _selections(core, count=1)
    cart = await core.update_cart_lines(OWNER, None, None, selections, "cart-1")
    quote = await core.prepare_checkout(OWNER, cart.cart_id, 1, "quote-1")
    assert core.faults is not None
    original_stock = (await core.get_product(OWNER, selections[0].sku_id)).available_quantity
    await core.faults.plan(kind="create_order", mode="unknown_after_commit")
    with pytest.raises(ProviderUnknownError):
        await core.create_order(OWNER, quote.quote_id, "order-1")

    assert (
        await core.get_product(OWNER, selections[0].sku_id)
    ).available_quantity == original_stock - 1

    receipt = await core.lookup_operation(OWNER, "order-1")
    assert receipt is not None
    assert receipt["status"] == "completed"
    order_id = receipt["result"]["order_id"]
    assert (await core.get_order(OWNER, order_id)).status == "awaiting_payment"

    # Replaying the same request key returns the committed receipt.
    replay = await core.create_order(OWNER, quote.quote_id, "order-1")
    assert replay.order_id == order_id


async def test_fault_processing_payment_and_settlement(
    core: CommerceProviderCore,
) -> None:
    selections = _selections(core, count=1)
    cart = await core.update_cart_lines(OWNER, None, None, selections, "cart-1")
    quote = await core.prepare_checkout(OWNER, cart.cart_id, 1, "quote-1")
    order = await core.create_order(OWNER, quote.quote_id, "order-1")
    payment = await core.create_payment(OWNER, order.order_id, "pay-1")

    assert core.faults is not None
    await core.faults.plan(kind="mock_pay", mode="processing_payment", target=payment.payment_id)
    processing = await core.mock_pay(OWNER, payment.payment_id, "pay-2")
    assert processing.status == "processing"
    assert (await core.get_order(OWNER, order.order_id)).status == "processing"
    with pytest.raises(ProviderRejectedError, match="处理中"):
        await core.mock_pay(OWNER, payment.payment_id, "pay-3")

    settled = await core.settle_payment(OWNER, payment.payment_id, paid=True)
    assert settled.status == "paid"
    assert (await core.get_order(OWNER, order.order_id)).status == "paid"
    with pytest.raises(ProviderRejectedError, match="结算"):
        await core.settle_payment(OWNER, payment.payment_id, paid=False)


async def test_settle_failed_payment_marks_order_failed(
    core: CommerceProviderCore,
) -> None:
    selections = _selections(core, count=1)
    cart = await core.update_cart_lines(OWNER, None, None, selections, "cart-1")
    quote = await core.prepare_checkout(OWNER, cart.cart_id, 1, "quote-1")
    order = await core.create_order(OWNER, quote.quote_id, "order-1")
    payment = await core.create_payment(OWNER, order.order_id, "pay-1")
    assert core.faults is not None
    await core.faults.plan(kind="mock_pay", mode="processing_payment")
    assert (await core.mock_pay(OWNER, payment.payment_id, "pay-2")).status == "processing"

    failed = await core.settle_payment(OWNER, payment.payment_id, paid=False)
    assert failed.status == "failed"
    assert (await core.get_order(OWNER, order.order_id)).status == "failed"


async def test_disabled_fault_controller_is_inert(tmp_path: Path) -> None:
    database = Database(str(tmp_path / "provider.sqlite3"))
    await database.startup()
    controller = FaultController(database, enabled=False)
    with pytest.raises(ProviderRejectedError):
        await controller.plan(kind="create_order", mode="reject")
    assert await controller.consume("create_order", None) is None
