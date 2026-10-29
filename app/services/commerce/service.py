"""Durable controlled purchase workflow. No external call holds a domain transaction."""

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any, cast
from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.exceptions import ConflictError, NotFoundError, ValidationError
from app.models.database.commerce import (
    PurchaseConfirmation,
    PurchaseOperation,
    PurchaseSession,
)
from app.models.schemas.commerce import (
    CartBlock,
    CartLineSelection,
    CartSnapshot,
    CatalogBlock,
    CatalogFilters,
    CatalogPage,
    CatalogTaxonomy,
    CommerceStage,
    CommerceView,
    OrderSnapshot,
    PaymentBlock,
    PaymentSnapshot,
    QuoteBlock,
    QuoteSnapshot,
    StatusBlock,
)
from app.services.commerce.mock_port import MockCommercePort, as_utc, request_hash, utc_now
from app.services.commerce.port import (
    CommercePort,
    CommerceSort,
    ProviderRejected,
    ProviderUnknown,
)

logger = logging.getLogger(__name__)


class PurchaseService:
    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        port: CommercePort | None = None,
        *,
        quote_ttl_seconds: int = 300,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        if not 1 <= quote_ttl_seconds <= 3600:
            raise ValueError("Quote TTL must be within1..3600 seconds")
        self.sessions = sessions
        self.clock = clock
        self.port = port or MockCommercePort(
            sessions, clock=clock, quote_ttl_seconds=quote_ttl_seconds
        )
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

    async def discover_catalog(self, owner_user_id: int) -> CatalogTaxonomy:
        """Read the provider's authoritative categories; purchase state is untouched."""

        try:
            return await self.port.catalog_categories(owner_user_id)
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
        result: CartSnapshot | QuoteSnapshot | OrderSnapshot | PaymentSnapshot
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

    async def _view(self, row: PurchaseSession, page: CatalogPage | None = None) -> CommerceView:
        blocks: list[Any] = []
        if row.order_id is None and row.stage in {"shopping", "cart", "quoted", "failed"}:
            shown = (
                page if page is not None else await self.port.search_products(row.owner_user_id, "")
            )
            blocks.append(
                CatalogBlock(
                    items=shown.items,
                    next_page_token=shown.next_page_token,
                    total_estimate=shown.total_estimate,
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
        )
