"""Order tracking timeline rebuilt from authoritative rows only.

After-sales reads ("我的订单怎么样了 / 我买过什么") deserve the same citable
facts as pre-sales ones. ``build_timeline`` assembles one purchase's full
event chain — purchase opening, every cart version with contents, quote with
totals/TTL/expiry, order creation, payment creation, payment completion, and
every idempotency ledger receipt — strictly from the durable commerce rows
(``commerce_purchases`` / ``commerce_operations`` / ``commerce_confirmations``).
There is no LLM involvement in assembly: the model only narrates over the
returned structure, and every amount is a minor-unit integer copied from a
row, never composed by narration.

Determinism: events sort by (timestamp, kind-phase rank, generation,
request_key). The rank precedes generation because operation generations are
not monotonic across kinds — cart/quote operations reserve at ``version + 1``
while order/payment/mock_pay reserve at the purchase ``generation`` — so on
timestamp ties (same-second SQLite writes) only the phase rank keeps the
chain causal. Owner scoping matches every other commerce read: the purchase
row is selected by ``(id, owner_user_id)`` and a foreign or unknown id
returns a clean ``found=False`` timeline, never an exception.
"""

from collections.abc import Callable
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.exceptions import ValidationError
from app.models.database.commerce import (
    PurchaseConfirmation,
    PurchaseOperation,
    PurchaseSession,
)
from app.models.schemas.commerce import (
    CartSnapshot,
    OrderSnapshot,
    PaymentSnapshot,
    QuoteSnapshot,
    RefundSnapshot,
)
from app.services.commerce.mock_port import as_utc, utc_now
from app.services.dialogue.tools import ToolDefinition, ToolRegistry

TimelineEventKind = Literal[
    "purchase_opened",
    "cart_updated",
    "cart_update_failed",
    "quote_prepared",
    "quote_failed",
    "order_created",
    "order_failed",
    "payment_created",
    "payment_failed",
    "payment_completed",
    "payment_completion_failed",
    "refund_completed",
]

# Phase rank for timestamp ties: encodes the purchase state machine order
# (shopping → cart → quoted → awaiting_payment → paid → refunded). Higher op
# generations must not jump an earlier phase when clocks collide.
_EVENT_RANK: dict[str, int] = {
    "purchase_opened": 0,
    "cart_updated": 1,
    "cart_update_failed": 1,
    "quote_prepared": 2,
    "quote_failed": 2,
    "order_created": 3,
    "order_failed": 3,
    "payment_created": 4,
    "payment_failed": 4,
    "payment_completed": 5,
    "payment_completion_failed": 5,
    "refund_completed": 6,
}
_RANK_UNKNOWN = len(_EVENT_RANK)

# Provider-derived payloads inside the timeline are framed with their own
# delimiters, same spotlighting mechanism as the catalog/graph data blocks.
ORDER_TIMELINE_TAGS = ("<order_timeline>", "</order_timeline>")
RECENT_ORDERS_TAGS = ("<recent_orders>", "</recent_orders>")
RECENT_ORDERS_CAP = 10


class TimelineLine(BaseModel):
    """One authoritative cart/quote line copied from a durable snapshot."""

    sku_id: str
    name: str
    quantity: int
    unit_price_minor: int
    subtotal_minor: int


class TimelineEvent(BaseModel):
    """One chain event; amounts are minor-unit integers from rows or absent."""

    kind: TimelineEventKind
    timestamp: datetime
    title: str
    summary: str
    amount_minor: int | None = None
    currency: str | None = None
    status: str | None = None
    generation: int | None = None
    request_key: str | None = None
    cart_version: int | None = None
    subtotal_minor: int | None = None
    quote_id: str | None = None
    order_id: str | None = None
    payment_id: str | None = None
    expires_at: datetime | None = None
    ttl_seconds: int | None = None
    expired: bool | None = None
    lines: list[TimelineLine] | None = None


class OrderTimeline(BaseModel):
    """Read-side projection of one purchase's authoritative event chain."""

    purchase_id: str
    owner_user_id: int
    found: bool
    stage: str | None = None
    generated_at: datetime
    events: list[TimelineEvent] = []


def _yuan(minor: int) -> str:
    return f"{minor / 100:.2f}"


def _sort_key(event: TimelineEvent) -> tuple[datetime, int, int, str]:
    return (
        event.timestamp,
        _EVENT_RANK.get(event.kind, _RANK_UNKNOWN),
        event.generation if event.generation is not None else -1,
        event.request_key or "",
    )


def _requested_line_count(op: PurchaseOperation) -> int:
    lines = op.request.get("lines")
    return len(lines) if isinstance(lines, list) else 0


def _attempt_summary(op: PurchaseOperation, subject: str) -> str:
    detail = f"：{op.error}" if op.error else ""
    return f"{subject}请求未生效（{op.status}）{detail}"[:200]


def _snapshot(op: PurchaseOperation, model: type[BaseModel]) -> Any | None:
    """Parse a completed operation's durable receipt, degrading to None."""

    if op.result is None:
        return None
    try:
        return model.model_validate(op.result)
    except PydanticValidationError:
        return None


def _opened_event(purchase: PurchaseSession) -> TimelineEvent:
    return TimelineEvent(
        kind="purchase_opened",
        timestamp=as_utc(purchase.created_at),
        title="开始选购",
        summary=f"开启模拟购买（商户 {purchase.merchant_id}，{purchase.currency}）",
    )


def _cart_event(op: PurchaseOperation) -> TimelineEvent:
    cart = _snapshot(op, CartSnapshot) if op.status == "completed" else None
    if cart is not None:
        return TimelineEvent(
            kind="cart_updated",
            timestamp=as_utc(op.created_at),
            title=f"购物车更新（v{cart.cart_version}）",
            summary=f"共 {len(cart.lines)} 行，小计 {_yuan(cart.subtotal_minor)} 元",
            amount_minor=cart.subtotal_minor,
            currency=cart.currency,
            status=op.status,
            generation=op.generation,
            request_key=op.id,
            cart_version=cart.cart_version,
            lines=[
                TimelineLine(
                    sku_id=line.sku_id,
                    name=line.name,
                    quantity=line.quantity,
                    unit_price_minor=line.unit_price_minor,
                    subtotal_minor=line.subtotal_minor,
                )
                for line in cart.lines
            ],
        )
    return TimelineEvent(
        kind="cart_update_failed",
        timestamp=as_utc(op.created_at),
        title="购物车更新未生效",
        summary=_attempt_summary(op, f"{_requested_line_count(op)} 行"),
        status=op.status,
        generation=op.generation,
        request_key=op.id,
    )


def _quote_event(
    op: PurchaseOperation, confirmation: PurchaseConfirmation | None, now: datetime
) -> TimelineEvent:
    quote = _snapshot(op, QuoteSnapshot) if op.status == "completed" else None
    if quote is not None:
        expires = (
            as_utc(confirmation.expires_at)
            if confirmation is not None
            else as_utc(quote.expires_at)
        )
        ttl = (
            int(round((expires - as_utc(op.created_at)).total_seconds()))
            if confirmation is not None
            else None
        )
        return TimelineEvent(
            kind="quote_prepared",
            timestamp=as_utc(op.created_at),
            title="报价生成",
            summary=(
                f"合计 {_yuan(quote.total_minor)} 元，报价 {ttl} 秒内有效"
                if ttl is not None
                else f"合计 {_yuan(quote.total_minor)} 元"
            ),
            amount_minor=quote.total_minor,
            subtotal_minor=quote.subtotal_minor,
            currency=quote.currency,
            status=op.status,
            generation=op.generation,
            request_key=op.id,
            cart_version=quote.cart_version,
            quote_id=quote.quote_id,
            expires_at=expires,
            ttl_seconds=ttl,
            expired=expires <= now,
            lines=[
                TimelineLine(
                    sku_id=line.sku_id,
                    name=line.name,
                    quantity=line.quantity,
                    unit_price_minor=line.unit_price_minor,
                    subtotal_minor=line.subtotal_minor,
                )
                for line in quote.lines
            ],
        )
    return TimelineEvent(
        kind="quote_failed",
        timestamp=as_utc(op.created_at),
        title="报价未生效",
        summary=_attempt_summary(op, "报价"),
        status=op.status,
        generation=op.generation,
        request_key=op.id,
    )


def _order_event(op: PurchaseOperation) -> TimelineEvent:
    order = _snapshot(op, OrderSnapshot) if op.status == "completed" else None
    if order is not None:
        return TimelineEvent(
            kind="order_created",
            timestamp=as_utc(op.created_at),
            title="订单已创建",
            summary=f"订单 {order.order_id}，应付 {_yuan(order.total_minor)} 元",
            amount_minor=order.total_minor,
            currency=order.currency,
            status=order.status,
            generation=op.generation,
            request_key=op.id,
            order_id=order.order_id,
        )
    return TimelineEvent(
        kind="order_failed",
        timestamp=as_utc(op.created_at),
        title="下单未生效",
        summary=_attempt_summary(op, "下单"),
        status=op.status,
        generation=op.generation,
        request_key=op.id,
    )


def _payment_event(op: PurchaseOperation) -> TimelineEvent:
    payment = _snapshot(op, PaymentSnapshot) if op.status == "completed" else None
    if op.kind == "payment":
        if payment is not None:
            return TimelineEvent(
                kind="payment_created",
                timestamp=as_utc(op.created_at),
                title="付款单已创建",
                summary=f"付款单 {payment.payment_id}，待支付 {_yuan(payment.total_minor)} 元",
                amount_minor=payment.total_minor,
                currency=payment.currency,
                status=payment.status,
                generation=op.generation,
                request_key=op.id,
                payment_id=payment.payment_id,
            )
        return TimelineEvent(
            kind="payment_failed",
            timestamp=as_utc(op.created_at),
            title="付款单创建未生效",
            summary=_attempt_summary(op, "付款单创建"),
            status=op.status,
            generation=op.generation,
            request_key=op.id,
        )
    # mock_pay: the completion attempt on an existing payment.
    if payment is not None and payment.status == "paid":
        return TimelineEvent(
            kind="payment_completed",
            timestamp=as_utc(op.created_at),
            title="模拟付款完成",
            summary=(f"{payment.payment_id} 已支付 {_yuan(payment.total_minor)} 元（无真实扣款）"),
            amount_minor=payment.total_minor,
            currency=payment.currency,
            status=payment.status,
            generation=op.generation,
            request_key=op.id,
            payment_id=payment.payment_id,
        )
    if payment is not None and payment.status == "processing":
        return TimelineEvent(
            kind="payment_completed",
            timestamp=as_utc(op.created_at),
            title="模拟付款处理中",
            summary=f"{payment.payment_id} 已受理，处理中待状态核实",
            amount_minor=payment.total_minor,
            currency=payment.currency,
            status=payment.status,
            generation=op.generation,
            request_key=op.id,
            payment_id=payment.payment_id,
        )
    if payment is not None:
        return TimelineEvent(
            kind="payment_completion_failed",
            timestamp=as_utc(op.created_at),
            title="付款未完成",
            summary=f"{payment.payment_id} 未完成（{payment.status}）",
            amount_minor=payment.total_minor,
            currency=payment.currency,
            status=payment.status,
            generation=op.generation,
            request_key=op.id,
            payment_id=payment.payment_id,
        )
    return TimelineEvent(
        kind="payment_completion_failed",
        timestamp=as_utc(op.created_at),
        title="付款未完成",
        summary=_attempt_summary(op, "付款"),
        status=op.status,
        generation=op.generation,
        request_key=op.id,
    )


def _refund_event(op: PurchaseOperation) -> TimelineEvent:
    refund = _snapshot(op, RefundSnapshot) if op.status == "completed" else None
    if refund is not None:
        return TimelineEvent(
            kind="refund_completed",
            timestamp=as_utc(op.created_at),
            title="退款完成",
            summary=(
                f"订单 {refund.order_id} 已退款 {_yuan(refund.total_minor)} 元，"
                "款项原路退回（模拟，无真实扣款）"
            ),
            amount_minor=refund.total_minor,
            currency=refund.currency,
            status=refund.status,
            generation=op.generation,
            request_key=op.id,
            order_id=refund.order_id,
            payment_id=refund.payment_id,
        )
    return TimelineEvent(
        kind="refund_completed",
        timestamp=as_utc(op.created_at),
        title="退款未生效",
        summary=_attempt_summary(op, "退款"),
        status=op.status,
        generation=op.generation,
        request_key=op.id,
    )


def _refresh_settlement_event(purchase: PurchaseSession) -> TimelineEvent | None:
    """Terminal payment state learned by status refresh, not a new receipt.

    ``_refresh`` in the purchase service settles ``payment_status`` from
    provider polling without minting a new operation; the purchase row itself
    is the authoritative witness, so a terminal state with no matching
    completed ``mock_pay`` receipt becomes one chain event.
    """

    if purchase.payment_status not in {"paid", "failed"}:
        return None
    quote_total = (
        purchase.quote_snapshot.get("total_minor")
        if isinstance(purchase.quote_snapshot, dict)
        else None
    )
    paid = purchase.payment_status == "paid"
    return TimelineEvent(
        kind="payment_completed" if paid else "payment_completion_failed",
        timestamp=as_utc(purchase.updated_at),
        title="模拟付款完成" if paid else "付款未完成",
        summary=(
            f"{purchase.payment_id} 经状态查询核实为已支付（无真实扣款）"
            if paid
            else f"{purchase.payment_id} 经状态查询核实为失败"
        ),
        amount_minor=quote_total if isinstance(quote_total, int) else None,
        currency=(
            purchase.quote_snapshot.get("currency")
            if isinstance(purchase.quote_snapshot, dict)
            else None
        ),
        status=purchase.payment_status,
        payment_id=purchase.payment_id,
    )


async def build_timeline(
    sessions: async_sessionmaker[AsyncSession],
    owner_user_id: int,
    purchase_id: str,
    *,
    clock: Callable[[], datetime] = utc_now,
) -> OrderTimeline:
    """Rebuild one purchase's event chain from authoritative rows.

    Owner-scoped like every other commerce read; an unknown or foreign
    ``purchase_id`` yields ``found=False`` with no events and no exception,
    which callers can distinguish from any error. ``clock`` is injectable for
    deterministic quote-expiry evaluation.
    """

    now = as_utc(clock())
    async with sessions() as db:
        purchase = await db.scalar(
            select(PurchaseSession).where(
                PurchaseSession.id == purchase_id,
                PurchaseSession.owner_user_id == owner_user_id,
            )
        )
        if purchase is None:
            # The customer only ever holds the provider's ORDER id (o_… /
            # mo_…) from receipts — never the internal purchase id. Resolve
            # it through the owner's completed order operations so "这单到
            # 哪了" works with the id the user actually has. Owner-scoped by
            # construction: foreign orders can never be matched.
            order_ops = (
                await db.scalars(
                    select(PurchaseOperation)
                    .join(
                        PurchaseSession,
                        PurchaseOperation.purchase_id == PurchaseSession.id,
                    )
                    .where(
                        PurchaseSession.owner_user_id == owner_user_id,
                        PurchaseOperation.kind == "order",
                        PurchaseOperation.status == "completed",
                    )
                )
            ).all()
            for op in order_ops:
                if isinstance(op.result, dict) and op.result.get("order_id") == purchase_id:
                    purchase = await db.scalar(
                        select(PurchaseSession).where(
                            PurchaseSession.id == op.purchase_id,
                            PurchaseSession.owner_user_id == owner_user_id,
                        )
                    )
                    break
        if purchase is None:
            return OrderTimeline(
                purchase_id=purchase_id,
                owner_user_id=owner_user_id,
                found=False,
                generated_at=now,
            )
        operations = (
            await db.scalars(
                select(PurchaseOperation).where(PurchaseOperation.purchase_id == purchase.id)
            )
        ).all()
        confirmations = (
            await db.scalars(
                select(PurchaseConfirmation).where(PurchaseConfirmation.purchase_id == purchase.id)
            )
        ).all()
        # A confirmation row's operation_id names the *order* operation that
        # consumed it, so quote events join their confirmation by the durable
        # quote_id inside the authoritative snapshot instead.
        confirmation_by_quote = {
            confirmation.snapshot.get("quote_id"): confirmation
            for confirmation in confirmations
            if isinstance(confirmation.snapshot.get("quote_id"), str)
        }

        events: list[TimelineEvent] = [_opened_event(purchase)]
        seen: set[str] = set()
        # A completed mock_pay receipt whose recorded status already equals the
        # purchase's terminal payment_status makes a refresh-derived event a
        # duplicate fact, not a new one.
        refresh_covered = any(
            op.kind == "mock_pay"
            and op.status == "completed"
            and isinstance(op.result, dict)
            and op.result.get("status") == purchase.payment_status
            for op in operations
        )
        for op in operations:
            if op.id in seen:  # idempotent replays share one request key
                continue
            seen.add(op.id)
            if op.kind == "update_cart":
                events.append(_cart_event(op))
            elif op.kind == "quote":
                quote_id = op.result.get("quote_id") if op.result is not None else None
                confirmation = (
                    confirmation_by_quote.get(quote_id) if isinstance(quote_id, str) else None
                )
                events.append(_quote_event(op, confirmation, now))
            elif op.kind == "order":
                events.append(_order_event(op))
            elif op.kind == "refund":
                events.append(_refund_event(op))
            else:  # payment / mock_pay
                events.append(_payment_event(op))
        if not refresh_covered:
            refreshed = _refresh_settlement_event(purchase)
            if refreshed is not None:
                events.append(refreshed)
        events.sort(key=_sort_key)
        return OrderTimeline(
            purchase_id=purchase.id,
            owner_user_id=owner_user_id,
            found=True,
            stage=purchase.stage,
            generated_at=now,
            events=events,
        )


def register_timeline_tools(
    registry: ToolRegistry,
    sessions: async_sessionmaker[AsyncSession],
    owner: int,
    *,
    clock: Callable[[], datetime] = utc_now,
) -> None:
    """Register the read-only order timeline tool bound to one verified owner.

    Read-tool conventions: answers directly (``requires_confirmation`` stays
    False), the owner is closed over — never a model argument — and the
    canonical timeline travels inside the established delimited provider-data
    framing so narration cannot be injected through it.

    Suggested wiring (integrator, in ``create_commerce_tools`` in
    ``app/services/commerce/tools.py``, after the read-tool loop)::

        register_timeline_tools(registry, service.sessions, owner, clock=service.clock)

    The framing helpers are imported lazily because tools.py imports this
    helper at its own module top when wired; a top-level import would cycle.
    """

    from app.services.commerce.tools import PROVIDER_DATA_NOTICE, delimited

    if type(owner) is not int or owner <= 0:
        raise ValidationError("A verified visitor identity is required")

    async def order_timeline(args: dict[str, Any]) -> dict[str, Any]:
        identifier = args.get("purchase_id")
        if not isinstance(identifier, str) or not identifier:
            raise ValidationError("A purchase identifier string is required")
        timeline = await build_timeline(sessions, owner, identifier, clock=clock)
        return {
            "source": "provider-data",
            "notice": PROVIDER_DATA_NOTICE,
            "order_timeline": delimited(ORDER_TIMELINE_TAGS, timeline.model_dump(mode="json")),
            "mode": "mock",
        }

    registry.register(
        ToolDefinition(
            name="commerce_get_order_timeline",
            intent="commerce:get_order_timeline",
            description=(
                "Read the customer's authoritative order tracking timeline for one "
                "purchase: purchase opening, every cart version with contents, quote "
                "with totals, TTL and expiry, order creation, payment creation, "
                "payment completion and refunds — each an auditable receipt event "
                "whose minor-unit amounts and timestamps come from durable rows, "
                "oldest first. Read-only: answers directly without confirmation. "
                "Use it for “我的订单怎么样了 / 我买过什么” questions; the identifier "
                "is the ORDER id printed on order/payment receipts (o_… or mo_…) — "
                "internal purchase ids also work. Never invent events or amounts "
                "that are not present in the returned timeline."
            ),
            required_slots=["purchase_id"],
            handler=order_timeline,
            parameters_schema={
                "type": "object",
                "properties": {
                    "purchase_id": {
                        "type": "string",
                        "description": (
                            "Order id from the receipts (o_…/mo_…); a purchase id also works"
                        ),
                    },
                },
                "required": ["purchase_id"],
                "additionalProperties": False,
            },
        )
    )


def register_recent_orders_tools(
    registry: ToolRegistry,
    sessions: async_sessionmaker[AsyncSession],
    owner: int,
) -> None:
    """Register the read-only "my orders" listing bound to one verified owner.

    Answers “我买过什么 / 最近订单” from the same durable purchase rows the
    timeline reads — the store's REAL orders — so the listing can never drift
    from the authoritative records. Read-tool conventions as above: owner
    closed over, answers directly, provider-data framing for the payload.
    """

    from app.services.commerce.tools import PROVIDER_DATA_NOTICE, delimited

    if type(owner) is not int or owner <= 0:
        raise ValidationError("A verified visitor identity is required")

    async def recent_orders(args: dict[str, Any]) -> dict[str, Any]:
        del args
        async with sessions() as db:
            purchases = (
                (
                    await db.scalars(
                        select(PurchaseSession)
                        .where(PurchaseSession.owner_user_id == owner)
                        .order_by(PurchaseSession.created_at.desc())
                        .limit(RECENT_ORDERS_CAP)
                    )
                )
                .unique()
                .all()
            )
        rows: list[dict[str, Any]] = []
        for purchase in purchases:
            quote_total = (
                purchase.quote_snapshot.get("total_minor")
                if isinstance(purchase.quote_snapshot, dict)
                else None
            )
            rows.append(
                {
                    "purchase_id": purchase.id,
                    "order_id": purchase.order_id,
                    "payment_id": purchase.payment_id,
                    "stage": purchase.stage,
                    "payment_status": purchase.payment_status,
                    "total_minor": quote_total,
                    "currency": "CNY" if quote_total is not None else None,
                    "created_at": purchase.created_at.isoformat(),
                }
            )
        return {
            "source": "provider-data",
            "notice": PROVIDER_DATA_NOTICE,
            "recent_orders": delimited(
                RECENT_ORDERS_TAGS,
                {"owner_user_id": owner, "count": len(rows), "orders": rows},
            ),
            "message": (
                f"共 {len(rows)} 条购买记录（最新在前，最多 {RECENT_ORDERS_CAP} 条）。"
                if rows
                else "当前访客在商城还没有任何购买记录。"
            ),
        }

    registry.register(
        ToolDefinition(
            name="commerce_list_recent_orders",
            intent="commerce:list_recent_orders",
            description=(
                "List the customer's REAL store purchases from authoritative "
                "durable rows: purchase id, order id, payment status, stage and "
                "total for each, newest first. Read-only: answers directly "
                "without confirmation. Use it for “我买过什么 / 我的订单列表”; "
                "never invent orders that are not in the returned list."
            ),
            required_slots=[],
            handler=recent_orders,
            parameters_schema={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        )
    )


__all__ = [
    "ORDER_TIMELINE_TAGS",
    "RECENT_ORDERS_CAP",
    "RECENT_ORDERS_TAGS",
    "OrderTimeline",
    "TimelineEvent",
    "TimelineEventKind",
    "TimelineLine",
    "build_timeline",
    "register_recent_orders_tools",
    "register_timeline_tools",
]
