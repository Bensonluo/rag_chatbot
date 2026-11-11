"""Durable controlled purchase workflow. No external call holds a domain transaction."""

import asyncio
import json
import logging
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any, Literal, cast
from uuid import uuid4

from pydantic import ValidationError as PydanticValidationError
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.exceptions import ConflictError, NotFoundError, ValidationError
from app.models.database.commerce import (
    PurchaseConfirmation,
    PurchaseOperation,
    PurchaseSession,
)
from app.models.database.session import ChatSession
from app.models.database.ticket import PRIORITY_HIGH, HandoffTicket
from app.models.schemas.commerce import (
    ApprovalCard,
    ApprovalLine,
    CartBlock,
    CartLineSelection,
    CartSnapshot,
    CatalogBlock,
    CatalogFilters,
    CatalogPage,
    CatalogTaxonomy,
    CommerceStage,
    CommerceView,
    MintCategoryOutcome,
    OrderSnapshot,
    PaymentBlock,
    PaymentSnapshot,
    QuoteBlock,
    QuoteSnapshot,
    RefundSnapshot,
    SpendLimitBlock,
    StatusBlock,
)
from app.repositories.ticket_repository import TICKET_STATUS_RESOLVED, TicketRepository
from app.services.commerce.mock_port import MockCommercePort, as_utc, request_hash, utc_now
from app.services.commerce.port import (
    CommercePort,
    CommerceSort,
    ProviderRejected,
    ProviderUnknown,
)

logger = logging.getLogger(__name__)

# Handoff reason for the scoped-delegation spend guard. Kept as a local
# constant because the shared handoff module normalizes unknown reasons
# down to "explicit"; the ticket row itself stores this exact string.
REASON_SPEND_LIMIT = "spend_limit"


class PurchaseService:
    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        port: CommercePort | None = None,
        *,
        quote_ttl_seconds: int = 300,
        clock: Callable[[], datetime] = utc_now,
        spend_limit_single_minor: int | None = None,
        spend_limit_daily_minor: int | None = None,
    ) -> None:
        if not 1 <= quote_ttl_seconds <= 3600:
            raise ValueError("Quote TTL must be within1..3600 seconds")
        self.sessions = sessions
        self.clock = clock
        self.port = port or MockCommercePort(
            sessions, clock=clock, quote_ttl_seconds=quote_ttl_seconds
        )
        # None = follow settings at check time; explicit values (tests,
        # programmatic wiring) win. 0 disables a limit.
        self._spend_limit_single_minor = spend_limit_single_minor
        self._spend_limit_daily_minor = spend_limit_daily_minor
        if self.port.mode not in {"mock", "external"}:
            raise ValueError("This purchase framework accepts mock and external adapters only")

    async def _load(self, owner: int, purchase_id: str) -> PurchaseSession:
        async with self.sessions() as db:
            row = await db.scalar(
                select(PurchaseSession).where(
                    PurchaseSession.id == purchase_id, PurchaseSession.owner_user_id == owner
                )
            )
            if row is None:
                raise NotFoundError("Purchase")
            return row

    async def active(self, owner_user_id: int) -> CommerceView | None:
        async with self.sessions() as db:
            row = await db.scalar(
                select(PurchaseSession).where(
                    PurchaseSession.active_slot == f"{owner_user_id}:demo-store",
                    PurchaseSession.owner_user_id == owner_user_id,
                )
            )
        return None if row is None else await self.status(owner_user_id, row.id)

    async def start(self, owner_user_id: int) -> CommerceView:
        current = await self.active(owner_user_id)
        if current is not None and current.stage != "paid":
            return current
        identifier = str(uuid4())
        async with self.sessions() as db:
            db.add(
                PurchaseSession(
                    id=identifier,
                    owner_user_id=owner_user_id,
                    active_slot=f"{owner_user_id}:demo-store",
                    mode="mock",
                    merchant_id="demo-store",
                    currency="CNY",
                    stage="shopping",
                    version=0,
                    generation=0,
                )
            )
            try:
                await db.commit()
            except IntegrityError as exc:
                await db.rollback()
                current = await self.active(owner_user_id)
                if current is not None:
                    return current
                raise ConflictError("购买流程暂不可用") from exc
        return await self.status(owner_user_id, identifier)

    async def search(
        self,
        owner_user_id: int,
        purchase_id: str,
        query: str,
        *,
        filters: CatalogFilters | None = None,
        sort: CommerceSort = "relevance",
        page_size: int = 10,
        page_token: str | None = None,
        expected_version: int | None = None,
    ) -> CommerceView:
        row = await self._load(owner_user_id, purchase_id)
        if expected_version is not None:
            self._version(row, expected_version)
        page = await self.port.search_products(
            owner_user_id,
            query[:500],
            filters=filters,
            sort=sort,
            page_size=page_size,
            page_token=page_token,
        )
        return await self._view(row, page)

    async def discover_catalog(
        self,
        owner_user_id: int,
        *,
        category_ids: set[str] | None = None,
    ) -> CatalogTaxonomy:
        """Read the provider's authoritative categories; purchase state is untouched."""

        try:
            return await self.port.catalog_categories(owner_user_id, category_ids=category_ids)
        except ProviderRejected as exc:
            raise ValidationError(str(exc)) from exc

    async def propose_category(
        self,
        owner_user_id: int,
        *,
        name: str,
        description: str,
        domain: str,
        attributes: list[tuple[str, list[str]]],
        variant_key: str,
        variant_values: list[str],
        base_price_minor: int,
        source_query: str = "",
        brand_styles: list[dict[str, str]] | None = None,
    ) -> MintCategoryOutcome:
        """Mint a dynamic category (or dedupe to its existing twin).

        Catalog facts only: no purchase session is reserved and no cart,
        quote, order or payment state changes. A fresh idempotency key per
        call keeps the graceful path — a lost response self-heals on the next
        attempt because the provider dedupes by normalized name.
        """

        try:
            return await self.port.propose_category(
                owner_user_id,
                request_key=f"mint:{uuid4()}",
                name=name,
                description=description,
                domain=domain,
                attributes=attributes,
                variant_key=variant_key,
                variant_values=variant_values,
                base_price_minor=base_price_minor,
                source_query=source_query,
                brand_styles=brand_styles,
            )
        except ProviderRejected as exc:
            raise ValidationError(str(exc)) from exc

    async def get_product(self, owner_user_id: int, purchase_id: str, sku_id: str) -> CommerceView:
        row = await self._load(owner_user_id, purchase_id)
        try:
            item = await self.port.get_product(owner_user_id, sku_id)
        except ProviderRejected as exc:
            raise ValidationError(str(exc)) from exc
        return await self._view(row, CatalogPage(items=[item], total_estimate=1))

    @staticmethod
    def _version(row: PurchaseSession, expected: int) -> None:
        if type(expected) is not int or expected != row.version:
            raise ConflictError("购买内容已变化，请刷新后确认")

    async def _reserve(
        self,
        owner: int,
        purchase_id: str,
        expected: int,
        kind: str,
        request: dict[str, Any],
        *,
        generation: int | None = None,
        confirmation_id: str | None = None,
    ) -> PurchaseOperation:
        async with self.sessions() as db:
            row = await db.scalar(
                select(PurchaseSession).where(
                    PurchaseSession.id == purchase_id, PurchaseSession.owner_user_id == owner
                )
            )
            if row is None:
                raise NotFoundError("Purchase")
            self._version(row, expected)
            if row.active_operation_id is not None:
                raise ConflictError("原操作尚未核实，请查询原购买流程")
            if kind in {"update_cart", "quote"} and row.order_id is not None:
                raise ConflictError("已有订单，不能改成另一笔购买")
            identifier = str(uuid4())
            operation = PurchaseOperation(
                id=identifier,
                purchase_id=purchase_id,
                kind=kind,
                generation=generation if generation is not None else row.version + 1,
                status="reserved",
                request_hash=request_hash(
                    {"owner": owner, "purchase_id": purchase_id, "kind": kind, "request": request}
                ),
                request=request,
            )
            changed = await db.execute(
                update(PurchaseSession)
                .where(
                    PurchaseSession.id == purchase_id,
                    PurchaseSession.owner_user_id == owner,
                    PurchaseSession.version == expected,
                    PurchaseSession.active_operation_id.is_(None),
                )
                .values(
                    stage="submitting",
                    version=expected + 1,
                    active_operation_id=identifier,
                    error=None,
                )
                .returning(PurchaseSession.id)
            )
            if changed.scalar_one_or_none() is None:
                raise ConflictError("购买流程正在变化，请先核实")
            if confirmation_id is not None:
                confirmation = await db.get(PurchaseConfirmation, confirmation_id)
                if confirmation is None or confirmation.operation_id is not None:
                    raise ConflictError("结算确认已变化")
                confirmation.operation_id = identifier
            db.add(operation)
            try:
                await db.commit()
            except SQLAlchemyError as exc:
                await db.rollback()
                raise ConflictError("购买操作尚未确认，请查询后继续") from exc
            return operation

    async def _mark(
        self, owner: int, operation: PurchaseOperation, *, failed: bool, error: str
    ) -> None:
        async with self.sessions() as db, db.begin():
            row = await db.get(PurchaseSession, operation.purchase_id)
            if row is None or row.owner_user_id != owner or row.active_operation_id != operation.id:
                return
            await db.execute(
                update(PurchaseOperation)
                .where(
                    PurchaseOperation.id == operation.id,
                    PurchaseOperation.status.in_(["reserved", "unknown"]),
                )
                .values(status="failed" if failed else "unknown", error=error)
            )
            await db.execute(
                update(PurchaseSession)
                .where(
                    PurchaseSession.id == row.id,
                    PurchaseSession.active_operation_id == operation.id,
                )
                .values(
                    stage="failed" if failed else "unknown",
                    active_operation_id=None if failed else operation.id,
                    version=PurchaseSession.version + 1,
                    error=error,
                )
            )

    async def _finish(self, owner: int, operation: PurchaseOperation, data: dict[str, Any]) -> None:
        async with self.sessions() as db, db.begin():
            row = await db.get(PurchaseSession, operation.purchase_id)
            if row is None or row.owner_user_id != owner:
                raise NotFoundError("Purchase")
            if row.active_operation_id != operation.id:
                return  # Another reconciliation already committed this receipt.
            values: dict[str, Any] = {
                "active_operation_id": None,
                "version": row.version + 1,
                "error": None,
            }
            if operation.kind == "update_cart":
                cart = CartSnapshot.model_validate(data)
                requested = sorted(
                    (line["sku_id"], line["quantity"]) for line in operation.request["lines"]
                )
                returned = sorted((line.sku_id, line.quantity) for line in cart.lines)
                if requested != returned or (
                    operation.request["cart_id"] is not None
                    and cart.cart_id != operation.request["cart_id"]
                ):
                    raise ProviderUnknown("模拟购物车回执与原请求不符")
                values.update(
                    stage="cart",
                    cart_id=cart.cart_id,
                    cart_version=cart.cart_version,
                    cart_snapshot=cart.model_dump(mode="json"),
                    generation=row.generation + 1,
                    quote_snapshot=None,
                    confirmation_id=None,
                )
            elif operation.kind == "quote":
                quote = QuoteSnapshot.model_validate(data)
                if quote.cart_id != row.cart_id or quote.cart_version != row.cart_version:
                    raise ProviderUnknown("模拟报价与购物车版本不符")
                confirmation_id = str(uuid4())
                snapshot = quote.model_dump(mode="json")
                db.add(
                    PurchaseConfirmation(
                        id=confirmation_id,
                        purchase_id=row.id,
                        purchase_version=row.version + 1,
                        generation=row.generation,
                        snapshot=snapshot,
                        snapshot_hash=request_hash(snapshot),
                        expires_at=quote.expires_at,
                    )
                )
                values.update(
                    stage="quoted", quote_snapshot=snapshot, confirmation_id=confirmation_id
                )
            elif operation.kind == "order":
                order = OrderSnapshot.model_validate(data)
                quote = QuoteSnapshot.model_validate(row.quote_snapshot)
                if (
                    order.quote_id != operation.request["quote_id"]
                    or order.total_minor != quote.total_minor
                    or order.currency != quote.currency
                ):
                    raise ProviderUnknown("模拟订单回执与已确认报价不符")
                values.update(
                    stage=order.status, order_id=order.order_id, payment_status=order.status
                )
            elif operation.kind == "refund":
                refund = RefundSnapshot.model_validate(data)
                quote = QuoteSnapshot.model_validate(row.quote_snapshot)
                if (
                    refund.order_id != row.order_id
                    or refund.payment_id != row.payment_id
                    or refund.total_minor != quote.total_minor
                    or refund.currency != quote.currency
                ):
                    raise ProviderUnknown("模拟退款回执与原订单不符")
                values.update(stage="refunded", payment_status="refunded")
            else:
                payment = PaymentSnapshot.model_validate(data)
                quote = QuoteSnapshot.model_validate(row.quote_snapshot)
                if (
                    payment.order_id != row.order_id
                    or payment.total_minor != quote.total_minor
                    or payment.currency != quote.currency
                ):
                    raise ProviderUnknown("模拟付款回执与已确认订单不符")
                values.update(
                    stage=payment.status,
                    payment_id=payment.payment_id,
                    payment_status=payment.status,
                )
                if payment.status == "paid":
                    values["active_slot"] = None
            changed = await db.execute(
                update(PurchaseSession)
                .where(
                    PurchaseSession.id == row.id,
                    PurchaseSession.active_operation_id == operation.id,
                )
                .values(**values)
                .returning(PurchaseSession.id)
            )
            if changed.scalar_one_or_none() is None:
                await db.rollback()
                return
            await db.execute(
                update(PurchaseOperation)
                .where(PurchaseOperation.id == operation.id)
                .values(status="completed", result=data, error=None)
            )

    async def _dispatch(self, owner: int, operation: PurchaseOperation) -> None:
        args = operation.request
        if operation.request_hash != request_hash(
            {
                "owner": owner,
                "purchase_id": operation.purchase_id,
                "kind": operation.kind,
                "request": args,
            }
        ):
            await self._mark(
                owner, operation, failed=True, error="原购买操作参数已损坏，未调用外部接口"
            )
            return
        result: CartSnapshot | QuoteSnapshot | OrderSnapshot | PaymentSnapshot | RefundSnapshot
        try:
            if operation.kind == "update_cart":
                result = await self.port.update_cart_lines(
                    owner,
                    args["cart_id"],
                    args["expected_cart_version"],
                    [CartLineSelection.model_validate(line) for line in args["lines"]],
                    operation.id,
                )
            elif operation.kind == "quote":
                result = await self.port.prepare_checkout(
                    owner, args["cart_id"], args["cart_version"], operation.id
                )
            elif operation.kind == "order":
                result = await self.port.create_order(owner, args["quote_id"], operation.id)
            elif operation.kind == "payment":
                result = await self.port.create_payment(owner, args["order_id"], operation.id)
            elif operation.kind == "refund":
                result = await self.port.refund_order(owner, args["order_id"], operation.id)
            else:
                result = await self.port.mock_pay(owner, args["payment_id"], operation.id)
            await self._finish(owner, operation, result.model_dump(mode="json"))
        except ProviderRejected as exc:
            await self._mark(owner, operation, failed=True, error=str(exc)[:255])
        except asyncio.CancelledError:
            await asyncio.shield(
                self._mark(owner, operation, failed=False, error="模拟业务调用中断，请核实原操作")
            )
            raise
        except Exception:
            await self._mark(
                owner, operation, failed=False, error="模拟业务结果未确认，请查询原操作"
            )

    async def update_cart_lines(
        self,
        owner_user_id: int,
        purchase_id: str,
        lines: list[CartLineSelection],
        expected_version: int,
    ) -> CommerceView:
        if (
            not isinstance(lines, list)
            or not lines
            or len(lines) > 50
            or not all(isinstance(line, CartLineSelection) for line in lines)
        ):
            raise ValidationError("购物车行须为非空且不超过 50 行的类型化规格与数量列表")
        # Canonical IDs make the provider's echoed receipt match the request;
        # the 50-line cap is a transport bound, not a local commerce rule.
        lines = [
            CartLineSelection(sku_id=line.sku_id.strip().upper(), quantity=line.quantity)
            for line in lines
        ]
        row = await self._load(owner_user_id, purchase_id)
        self._version(row, expected_version)
        if row.stage in {"submitting", "unknown", "processing", "paid"}:
            raise ConflictError("当前流程不能修改购物车")
        operation = await self._reserve(
            owner_user_id,
            purchase_id,
            expected_version,
            "update_cart",
            {
                "cart_id": row.cart_id,
                "expected_cart_version": row.cart_version,
                "lines": [{"sku_id": line.sku_id, "quantity": line.quantity} for line in lines],
            },
        )
        await self._dispatch(owner_user_id, operation)
        return await self._view(await self._load(owner_user_id, purchase_id))

    async def prepare_checkout(
        self, owner_user_id: int, purchase_id: str, expected_version: int
    ) -> CommerceView:
        row = await self._load(owner_user_id, purchase_id)
        self._version(row, expected_version)
        if row.stage not in {"cart", "quoted"} or row.cart_id is None or row.cart_version is None:
            raise ConflictError("请先选择商品和数量")
        operation = await self._reserve(
            owner_user_id,
            purchase_id,
            expected_version,
            "quote",
            {"cart_id": row.cart_id, "cart_version": row.cart_version},
        )
        await self._dispatch(owner_user_id, operation)
        return await self._view(await self._load(owner_user_id, purchase_id))

    async def _confirmation(self, row: PurchaseSession) -> PurchaseConfirmation:
        if row.confirmation_id is None:
            raise ConflictError("请先完成报价并确认当前内容")
        async with self.sessions() as db:
            confirmation = await db.get(PurchaseConfirmation, row.confirmation_id)
        if confirmation is None or confirmation.purchase_id != row.id:
            raise ConflictError("结算确认不可用")
        if request_hash(confirmation.snapshot) != confirmation.snapshot_hash:
            raise ConflictError("结算确认已损坏，不能下单")
        return confirmation

    def _effective_limits(self) -> tuple[int, int]:
        """(single, daily) minor-unit limits; 0 disables each independently."""
        from app.config.settings import get_settings

        settings = get_settings()
        single = (
            self._spend_limit_single_minor
            if self._spend_limit_single_minor is not None
            else settings.COMMERCE_SPEND_LIMIT_SINGLE_CENTS
        )
        daily = (
            self._spend_limit_daily_minor
            if self._spend_limit_daily_minor is not None
            else settings.COMMERCE_SPEND_LIMIT_DAILY_CENTS
        )
        return max(0, single), max(0, daily)

    async def _paid_window_total(self, owner: int) -> int:
        """Paid order totals inside the rolling 24h window (owner scope).

        Newest-first with a hard row cap keeps the scan bounded; precise
        timezone normalization happens in Python (sqlite stores naive UTC,
        postgres returns aware) instead of trusting dialect bind coercion.
        """

        cutoff = as_utc(self.clock()) - timedelta(hours=24)
        async with self.sessions() as db:
            stamps = (
                await db.execute(
                    select(PurchaseSession.updated_at, PurchaseSession.quote_snapshot)
                    .where(PurchaseSession.owner_user_id == owner, PurchaseSession.stage == "paid")
                    .order_by(PurchaseSession.updated_at.desc())
                    .limit(1000)
                )
            ).all()
        total = 0
        for updated_at, snapshot in stamps:
            if not isinstance(updated_at, datetime):
                continue
            if as_utc(updated_at) < cutoff:
                continue
            if isinstance(snapshot, dict) and isinstance(snapshot.get("total_minor"), int):
                total += snapshot["total_minor"]
        return total

    async def _spend_limit_delegated(self, owner: int, purchase_id: str) -> bool:
        """True when a human resolved a spend_limit ticket for this purchase.

        Scoped delegation: only the purchase named on the resolved ticket is
        authorized; a different over-limit purchase escalates on its own.
        """

        async with self.sessions() as db:
            tickets = (
                await db.scalars(
                    select(HandoffTicket)
                    .where(
                        HandoffTicket.reason == REASON_SPEND_LIMIT,
                        HandoffTicket.status == TICKET_STATUS_RESOLVED,
                        HandoffTicket.user_id == owner,
                    )
                    .order_by(HandoffTicket.id.desc())
                    .limit(5)
                )
            ).all()
        for ticket in tickets:
            try:
                summary = json.loads(ticket.summary)
            except (TypeError, ValueError):
                continue
            if isinstance(summary, dict) and summary.get("purchase_id") == purchase_id:
                return True
        return False

    async def _spend_limit_decision(self, owner: int, total_minor: int) -> dict[str, Any] | None:
        """Return a refusal fact dict when a spend limit blocks this order."""
        single, daily = self._effective_limits()
        if single <= 0 and daily <= 0:
            return None
        if total_minor > single > 0:
            return {
                "limit_kind": "single",
                "limit_minor": single,
                "attempted_minor": total_minor,
                "window_paid_minor": None,
            }
        if daily > 0:
            paid = await self._paid_window_total(owner)
            if paid + total_minor > daily:
                return {
                    "limit_kind": "daily",
                    "limit_minor": daily,
                    "attempted_minor": total_minor,
                    "window_paid_minor": paid,
                }
        return None

    async def _escalate_spend_limit(
        self, owner: int, row: PurchaseSession, refusal: dict[str, Any]
    ) -> int | None:
        """Create (or reuse) the high-priority human ticket for an over-limit order.

        Same creation pattern as the chat-path handoff: best effort, never
        raises — the structured refusal itself is the durable user answer
        even when the ticket cannot be written.
        """

        context = {
            "source": "commerce_spend_limit",
            "purchase_id": row.id,
            **refusal,
        }
        try:
            async with self.sessions() as db:
                session_id = await db.scalar(
                    select(ChatSession.id)
                    .where(ChatSession.user_id == owner)
                    .order_by(ChatSession.id.desc())
                    .limit(1)
                )
            if session_id is None:
                logger.warning(
                    "Spend-limit escalation skipped: no chat session for owner %s", owner
                )
                return None
            async with self.sessions() as db:
                ticket, _ = await TicketRepository(db).create_or_reuse_active(
                    session_id=session_id,
                    user_id=owner,
                    reason=REASON_SPEND_LIMIT,
                    priority=PRIORITY_HIGH,
                    context=context,
                )
                return ticket.id
        except Exception:  # noqa: BLE001 - availability over durability
            logger.warning("Spend-limit ticket creation failed (owner=%s)", owner, exc_info=True)
            return None

    async def commit_checkout(
        self, owner_user_id: int, purchase_id: str, expected_version: int
    ) -> CommerceView:
        row = await self._load(owner_user_id, purchase_id)
        confirmation = await self._confirmation(row)
        if confirmation.operation_id is not None:
            # Repeating a consumed confirmation replays current business state;
            # the old display version must not become a second-order error.
            if (
                row.order_id is not None
                and row.payment_id is None
                and row.active_operation_id is None
            ):
                await self._payment(owner_user_id, row)
            return await self.status(owner_user_id, row.id)
        self._version(row, expected_version)
        if as_utc(confirmation.expires_at) <= as_utc(self.clock()):
            raise ConflictError("报价已过期，请重新试算并确认")
        if (
            row.stage != "quoted"
            or row.confirmation_id != confirmation.id
            or confirmation.purchase_version != row.version
            or confirmation.generation != row.generation
        ):
            raise ConflictError("结算内容已变化，请重新确认")
        # Scoped-delegation guard: an over-limit order is never created. A
        # human-resolved spend_limit ticket delegates exactly this purchase.
        quote_total = int(confirmation.snapshot.get("total_minor", 0))
        refusal = await self._spend_limit_decision(owner_user_id, quote_total)
        if refusal is not None and not await self._spend_limit_delegated(owner_user_id, row.id):
            ticket_id = await self._escalate_spend_limit(owner_user_id, row, refusal)
            if refusal["limit_kind"] == "single":
                message = (
                    f"本单金额 {refusal['attempted_minor'] / 100:.2f} 元超过单笔限额 "
                    f"{refusal['limit_minor'] / 100:.2f} 元，未创建订单，已转人工核实。"
                )
            else:
                message = (
                    f"近 24 小时已支付 {refusal['window_paid_minor'] / 100:.2f} 元，"
                    f"加上本单 {refusal['attempted_minor'] / 100:.2f} 元将超过日限额 "
                    f"{refusal['limit_minor'] / 100:.2f} 元，未创建订单，已转人工核实。"
                )
            view = await self._view(row)
            return view.model_copy(
                update={
                    "blocks": [
                        *view.blocks,
                        SpendLimitBlock(
                            limit_kind=refusal["limit_kind"],
                            limit_minor=refusal["limit_minor"],
                            attempted_minor=refusal["attempted_minor"],
                            window_paid_minor=refusal["window_paid_minor"],
                            ticket_id=ticket_id,
                            message=message,
                        ),
                    ]
                }
            )
        try:
            operation = await self._reserve(
                owner_user_id,
                row.id,
                row.version,
                "order",
                {"quote_id": confirmation.snapshot["quote_id"]},
                generation=row.generation,
                confirmation_id=confirmation.id,
            )
        except ConflictError:
            async with self.sessions() as db:
                fresh = await db.get(PurchaseConfirmation, confirmation.id)
                if fresh is None or fresh.operation_id is None:
                    raise
            return await self.status(owner_user_id, row.id)
        await self._dispatch(owner_user_id, operation)
        row = await self._load(owner_user_id, row.id)
        if row.order_id is not None and row.active_operation_id is None:
            await self._payment(owner_user_id, row)
        # Return the just-dispatched state without reconciling it: a transport
        # result must stay unknown until the user explicitly queries it.
        return await self._view(await self._load(owner_user_id, row.id))

    async def _payment(self, owner: int, row: PurchaseSession) -> None:
        async with self.sessions() as db:
            existing = await db.scalar(
                select(PurchaseOperation).where(
                    PurchaseOperation.purchase_id == row.id,
                    PurchaseOperation.kind == "payment",
                    PurchaseOperation.generation == row.generation,
                )
            )
        if existing is not None:
            return  # Unknown/failed attempts are never turned into new chargeable attempts.
        operation = await self._reserve(
            owner,
            row.id,
            row.version,
            "payment",
            {"order_id": row.order_id},
            generation=row.generation,
        )
        await self._dispatch(owner, operation)

    async def mock_pay(
        self, owner_user_id: int, purchase_id: str, expected_version: int
    ) -> CommerceView:
        row = await self._load(owner_user_id, purchase_id)
        async with self.sessions() as db:
            existing = await db.scalar(
                select(PurchaseOperation).where(
                    PurchaseOperation.purchase_id == row.id,
                    PurchaseOperation.kind == "mock_pay",
                    PurchaseOperation.generation == row.generation,
                )
            )
        if existing is not None:
            return await self.status(owner_user_id, row.id)
        self._version(row, expected_version)
        if row.active_operation_id is not None or row.stage != "awaiting_payment":
            raise ConflictError("当前模拟付款不可执行，请核实状态")
        if row.payment_id is None:
            if row.order_id is None:
                raise ConflictError("当前模拟付款不可执行，请核实状态")
            await self._payment(owner_user_id, row)
            row = await self._load(owner_user_id, row.id)
        if row.payment_id is None or row.active_operation_id is not None:
            return await self.status(owner_user_id, row.id)
        operation = await self._reserve(
            owner_user_id,
            row.id,
            row.version,
            "mock_pay",
            {"payment_id": row.payment_id},
            generation=row.generation,
        )
        await self._dispatch(owner_user_id, operation)
        return await self._view(await self._load(owner_user_id, row.id))

    async def refund(
        self, owner_user_id: int, purchase_id: str, expected_version: int
    ) -> CommerceView:
        """Refund the paid order of one purchase; exactly-once like every op.

        Mirrors ``mock_pay``: an existing refund operation replays the
        durable state instead of dispatching again, a version mismatch is a
        conflict, and only a ``paid`` purchase with no active operation may
        reserve the refund. The provider replays the same receipt across
        request keys, so a repeated confirmation cannot refund twice.
        """
        row = await self._load(owner_user_id, purchase_id)
        async with self.sessions() as db:
            existing = await db.scalar(
                select(PurchaseOperation).where(
                    PurchaseOperation.purchase_id == row.id,
                    PurchaseOperation.kind == "refund",
                    PurchaseOperation.generation == row.generation,
                )
            )
        if existing is not None:
            return await self.status(owner_user_id, purchase_id)
        self._version(row, expected_version)
        if row.active_operation_id is not None or row.stage != "paid":
            raise ConflictError("只有已完成支付的订单可以退款")
        if row.order_id is None or row.payment_id is None:
            raise ConflictError("当前订单缺少可退款的支付记录")
        operation = await self._reserve(
            owner_user_id,
            row.id,
            row.version,
            "refund",
            {"order_id": row.order_id},
            generation=row.generation,
        )
        await self._dispatch(owner_user_id, operation)
        return await self._view(await self._load(owner_user_id, row.id))

    async def _refresh(self, owner: int, purchase_id: str) -> PurchaseSession:
        row = await self._load(owner, purchase_id)
        if row.active_operation_id is not None:
            async with self.sessions() as db:
                operation = await db.get(PurchaseOperation, row.active_operation_id)
            if operation is not None:
                try:
                    receipt = await self.port.lookup_operation(owner, operation.id)
                    data = receipt.get("data") if receipt is not None else None
                    if receipt is not None and receipt.get("status") == "completed":
                        if not isinstance(data, dict):
                            logger.error("Invalid provider receipt for operation %s", operation.id)
                        else:
                            await self._finish(owner, operation, data)
                except Exception:
                    # The original operation stays pending; never dispatch anew.
                    logger.warning(
                        "Operation reconciliation failed (%s)", operation.id, exc_info=True
                    )
            row = await self._load(owner, purchase_id)
        if row.payment_id is not None and row.active_operation_id is None and row.stage != "paid":
            try:
                payment = await self.port.get_status(owner, row.payment_id)
                if (
                    payment.order_id != row.order_id
                    or payment.total_minor
                    != QuoteSnapshot.model_validate(row.quote_snapshot).total_minor
                ):
                    raise ProviderUnknown("模拟付款状态不符")
                if row.payment_status == "processing" and payment.status == "awaiting_payment":
                    return row
                if row.payment_status != payment.status:
                    async with self.sessions() as db, db.begin():
                        await db.execute(
                            update(PurchaseSession)
                            .where(
                                PurchaseSession.id == row.id,
                                PurchaseSession.owner_user_id == owner,
                                PurchaseSession.version == row.version,
                                PurchaseSession.active_operation_id.is_(None),
                            )
                            .values(
                                stage=payment.status,
                                payment_status=payment.status,
                                version=row.version + 1,
                                active_slot=None if payment.status == "paid" else row.active_slot,
                            )
                        )
            except Exception:
                logger.warning("Payment status refresh failed (%s)", row.payment_id, exc_info=True)
        return await self._load(owner, purchase_id)

    async def status(self, owner_user_id: int, purchase_id: str) -> CommerceView:
        return await self._view(await self._refresh(owner_user_id, purchase_id))

    async def reconcile(self, owner_user_id: int, purchase_id: str) -> CommerceView:
        return await self.status(owner_user_id, purchase_id)

    async def _approval_card(self, row: PurchaseSession) -> ApprovalCard | None:
        """Structured confirmation card, or None outside a confirmation window.

        Every field is copied from the durable PurchaseConfirmation and the
        purchase row — the card is a server fact, never model prose. A
        quoted card lives exactly as long as its unexpired, unconsumed
        confirmation; an awaiting-payment card describes the already-created
        order, so the consumed confirmation is the expected source there.
        """

        if row.stage not in {"quoted", "awaiting_payment"} or row.confirmation_id is None:
            return None
        async with self.sessions() as db:
            confirmation = await db.get(PurchaseConfirmation, row.confirmation_id)
        if (
            confirmation is None
            or confirmation.purchase_id != row.id
            or request_hash(confirmation.snapshot) != confirmation.snapshot_hash
        ):
            return None
        if row.stage == "quoted" and (
            confirmation.operation_id is not None
            or as_utc(confirmation.expires_at) <= as_utc(self.clock())
        ):
            return None
        try:
            quote = QuoteSnapshot.model_validate(confirmation.snapshot)
        except PydanticValidationError:
            return None
        return ApprovalCard(
            purchase_id=row.id,
            stage=cast(Literal["quoted", "awaiting_payment"], row.stage),
            action="confirm_order" if row.stage == "quoted" else "confirm_payment",
            action_label="确认下单" if row.stage == "quoted" else "确认支付",
            lines=[
                ApprovalLine(
                    sku_id=line.sku_id,
                    name=line.name,
                    quantity=line.quantity,
                    unit_price_minor=line.unit_price_minor,
                    subtotal_minor=line.subtotal_minor,
                )
                for line in quote.lines
            ],
            total_minor=quote.total_minor,
            currency=quote.currency,
            quote_hash=confirmation.snapshot_hash,
            expires_at=as_utc(quote.expires_at),
            purchase_version=row.version,
            generation=row.generation,
        )

    async def _view(self, row: PurchaseSession, page: CatalogPage | None = None) -> CommerceView:
        blocks: list[Any] = []
        # The candidate page is a selection-state fact: it attaches only
        # when an explicit page was produced (search results / product
        # detail — the state where the customer picks and adds to cart).
        # Other reads (status, cart, quote, order) must not fabricate a
        # default catalog page: that is how order/logistics answers ended
        # up pitching products nobody asked about.
        if row.order_id is None and page is not None:
            blocks.append(
                CatalogBlock(
                    items=page.items,
                    next_page_token=page.next_page_token,
                    total_estimate=page.total_estimate,
                )
            )
        if row.cart_snapshot is not None:
            blocks.append(CartBlock(cart=CartSnapshot.model_validate(row.cart_snapshot)))
        if row.quote_snapshot is not None and row.confirmation_id is not None:
            blocks.append(
                QuoteBlock(
                    confirmation_id=row.confirmation_id,
                    quote=QuoteSnapshot.model_validate(row.quote_snapshot),
                )
            )
        if row.order_id is not None:
            quote = QuoteSnapshot.model_validate(row.quote_snapshot)
            blocks.append(
                PaymentBlock(
                    order_id=row.order_id,
                    payment_id=row.payment_id,
                    status=row.payment_status or "awaiting_payment",
                    total_minor=quote.total_minor,
                )
            )
        messages = {
            "shopping": "请从上方展示商品中选择规格和整数数量；如有含糊请先确认。",
            "cart": "虚拟商城购物车已更新，尚未下单。",
            "quoted": "请核对虚拟商城商品、数量和总价后明确确认。",
            "submitting": "原模拟操作正在提交，请查询状态，不会自动重复执行。",
            "unknown": "原模拟操作结果未确认，请按原请求核实。",
            "awaiting_payment": "模拟订单已创建，尚未付款。",
            "processing": "模拟付款处理中，请查询原付款，不会创建新尝试。",
            "paid": "模拟付款已核实成功；没有真实扣款。",
            "refunded": "退款已完成，款项已原路退回（模拟）；没有真实扣款。",
            "failed": "模拟业务未完成，请查看说明后再操作。",
        }
        blocks.append(
            StatusBlock(
                stage=cast(CommerceStage, row.stage),
                message=row.error or messages.get(row.stage, "模拟购买状态待核实"),
                operation_id=row.active_operation_id,
            )
        )
        return CommerceView(
            purchase_id=row.id,
            version=row.version,
            stage=cast(CommerceStage, row.stage),
            snapshot_at=as_utc(self.clock()),
            blocks=blocks,
            approval=await self._approval_card(row),
        )
