"""Durable controlled purchase workflow. No external call holds a domain transaction."""

import asyncio
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any, cast
from uuid import UUID, uuid4, uuid5

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.exceptions import ConflictError, NotFoundError, ValidationError
from app.models.database.commerce import (
    PurchaseAction,
    PurchaseConfirmation,
    PurchaseOperation,
    PurchaseSession,
)
from app.models.schemas.commerce import (
    ActionKind,
    CartBlock,
    CartSnapshot,
    CatalogBlock,
    CatalogItem,
    CommerceAction,
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
from app.services.commerce.port import CommercePort, ProviderRejected, ProviderUnknown


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
        if self.port.mode != "mock":
            raise ValueError("This purchase framework accepts mock adapters only")

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
                    quantity=0,
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
        self, owner_user_id: int, purchase_id: str, query: str, expected_version: int | None = None
    ) -> CommerceView:
        row = await self._load(owner_user_id, purchase_id)
        if expected_version is not None:
            self._version(row, expected_version)
        return await self._view(row, await self.port.search_products(owner_user_id, query[:500]))

    async def get_product(self, owner_user_id: int, purchase_id: str, sku_id: str) -> CommerceView:
        row = await self._load(owner_user_id, purchase_id)
        try:
            item = await self.port.get_product(owner_user_id, sku_id)
        except ProviderRejected as exc:
            raise ValidationError(str(exc)) from exc
        return await self._view(row, [item])

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
            if kind in {"set_cart", "quote"} and row.order_id is not None:
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
            if operation.kind == "set_cart":
                cart = CartSnapshot.model_validate(data)
                if (
                    cart.cart_id != operation.request["cart_id"]
                    or cart.lines[0].sku_id != operation.request["sku_id"]
                    or cart.lines[0].quantity != operation.request["quantity"]
                ):
                    raise ProviderUnknown("模拟购物车回执与原请求不符")
                values.update(
                    stage="cart",
                    cart_id=cart.cart_id,
                    cart_version=cart.cart_version,
                    cart_snapshot=cart.model_dump(mode="json"),
                    sku_id=cart.lines[0].sku_id,
                    quantity=cart.lines[0].quantity,
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
            if operation.kind == "set_cart":
                result = await self.port.set_cart(
                    owner,
                    args["cart_id"],
                    args["sku_id"],
                    args["quantity"],
                    args["expected_cart_version"],
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

    async def set_cart_line(
        self,
        owner_user_id: int,
        purchase_id: str,
        sku_id: str,
        quantity: int,
        expected_version: int,
    ) -> CommerceView:
        if type(quantity) is not int or not 1 <= quantity <= 10:
            raise ValidationError("数量须为1到10的整数")
        row = await self._load(owner_user_id, purchase_id)
        self._version(row, expected_version)
        if row.stage in {"submitting", "unknown", "processing", "paid"}:
            raise ConflictError("当前流程不能修改购物车")
        operation = await self._reserve(
            owner_user_id,
            purchase_id,
            expected_version,
            "set_cart",
            {
                "cart_id": row.cart_id or str(uuid4()),
                "sku_id": sku_id,
                "quantity": quantity,
                "expected_cart_version": row.cart_version,
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

    async def _read_action(
        self,
        owner: int,
        action_ref: str,
        kind: str | None = None,
        expected_version: int | None = None,
    ) -> tuple[PurchaseAction, PurchaseSession]:
        async with self.sessions() as db:
            action = await db.get(PurchaseAction, action_ref)
            if action is None:
                raise NotFoundError("Purchase action")
            row = await db.scalar(
                select(PurchaseSession).where(
                    PurchaseSession.id == action.purchase_id, PurchaseSession.owner_user_id == owner
                )
            )
            if row is None:
                raise NotFoundError("Purchase action")
            if kind is not None and action.kind != kind:
                raise ConflictError("购买动作类型不符")
            if expected_version is not None and (
                type(expected_version) is not int or action.expected_version != expected_version
            ):
                raise ConflictError("购买动作版本不符")
            return action, row

    async def commit_checkout(self, owner_user_id: int, action_ref: str) -> CommerceView:
        action, row = await self._read_action(owner_user_id, action_ref, "commit_checkout")
        async with self.sessions() as db:
            confirmation = await db.get(
                PurchaseConfirmation, str(action.payload.get("confirmation_id", ""))
            )
            if confirmation is None or confirmation.purchase_id != row.id:
                raise ConflictError("结算确认不可用")
            if request_hash(confirmation.snapshot) != confirmation.snapshot_hash:
                raise ConflictError("结算确认已损坏，不能下单")
            consumed = confirmation.operation_id is not None
        if consumed:
            if (
                row.order_id is not None
                and row.payment_id is None
                and row.active_operation_id is None
            ):
                await self._payment(owner_user_id, row)
            return await self._view(await self._load(owner_user_id, row.id))
        self._version(row, action.expected_version)
        if as_utc(action.expires_at) <= as_utc(self.clock()) or as_utc(
            confirmation.expires_at
        ) <= as_utc(self.clock()):
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

    async def mock_pay(self, owner_user_id: int, action_ref: str) -> CommerceView:
        action, row = await self._read_action(owner_user_id, action_ref, "mock_pay")
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
        self._version(row, action.expected_version)
        if (
            as_utc(action.expires_at) <= as_utc(self.clock())
            or row.payment_id is None
            or row.stage != "awaiting_payment"
        ):
            raise ConflictError("当前模拟付款不可执行，请核实状态")
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
                    if receipt is not None and receipt.get("status") == "completed":
                        await self._finish(owner, operation, receipt["data"])
                except Exception:
                    pass  # Unknown remains pending. Never dispatch from a GET/reconciliation.
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
                pass
        return await self._load(owner, purchase_id)

    async def status(self, owner_user_id: int, purchase_id: str) -> CommerceView:
        return await self._view(await self._refresh(owner_user_id, purchase_id))

    async def reconcile(self, owner_user_id: int, purchase_id: str) -> CommerceView:
        return await self.status(owner_user_id, purchase_id)

    async def execute_action(
        self,
        owner_user_id: int,
        action_ref: str,
        *,
        quantity: int | None = None,
        expected_version: int | None = None,
    ) -> CommerceView:
        action, row = await self._read_action(
            owner_user_id, action_ref, expected_version=expected_version
        )
        if quantity is not None and action.kind != "select_sku":
            raise ValidationError("此动作不接受数量修改")
        if action.kind == "commit_checkout":
            return await self.commit_checkout(owner_user_id, action_ref)
        if action.kind == "mock_pay":
            return await self.mock_pay(owner_user_id, action_ref)
        if action.kind == "refresh":
            return await self.status(owner_user_id, row.id)
        if as_utc(action.expires_at) <= as_utc(self.clock()):
            raise ConflictError("购买动作已过期，请刷新")
        if action.kind == "select_sku":
            return await self.set_cart_line(
                owner_user_id,
                row.id,
                str(action.payload["sku_id"]),
                1 if quantity is None else quantity,
                action.expected_version,
            )
        return await self.prepare_checkout(owner_user_id, row.id, action.expected_version)

    async def _action(
        self,
        row: PurchaseSession,
        kind: ActionKind,
        label: str,
        payload: dict[str, Any],
        *,
        expires_at: datetime | None = None,
    ) -> CommerceAction:
        now = as_utc(self.clock())
        expiry = as_utc(expires_at) if expires_at else now + timedelta(minutes=5)
        bucket = 0 if expires_at else int(now.timestamp() // 300)
        identifier = str(
            uuid5(UUID(row.id), f"{row.version}:{kind}:{bucket}:{request_hash(payload)}")
        )
        async with self.sessions() as db:
            existing = await db.get(PurchaseAction, identifier)
            if existing is None:
                db.add(
                    PurchaseAction(
                        id=identifier,
                        purchase_id=row.id,
                        kind=kind,
                        expected_version=row.version,
                        payload=payload,
                        expires_at=expiry,
                    )
                )
                try:
                    await db.commit()
                except IntegrityError:
                    await db.rollback()
        return CommerceAction(
            action_ref=identifier,
            kind=kind,
            label=label,
            expected_version=row.version,
            sku_id=payload.get("sku_id"),
        )

    async def _view(
        self, row: PurchaseSession, items: list[CatalogItem] | None = None
    ) -> CommerceView:
        blocks: list[Any] = []
        actions: list[CommerceAction] = []
        if row.order_id is None and row.stage in {"shopping", "cart", "quoted", "failed"}:
            products = (
                items
                if items is not None
                else await self.port.search_products(row.owner_user_id, "")
            )
            blocks.append(CatalogBlock(items=products))
            for item in products:
                if item.available_quantity:
                    actions.append(
                        await self._action(
                            row, "select_sku", "选择模拟商品", {"sku_id": item.sku_id}
                        )
                    )
        if row.cart_snapshot is not None:
            blocks.append(CartBlock(cart=CartSnapshot.model_validate(row.cart_snapshot)))
        if row.stage in {"cart", "quoted"}:
            actions.append(
                await self._action(
                    row,
                    "prepare_checkout",
                    "重新试算" if row.stage == "quoted" else "查看模拟结算",
                    {},
                )
            )
        if row.quote_snapshot is not None and row.confirmation_id is not None:
            quote = QuoteSnapshot.model_validate(row.quote_snapshot)
            blocks.append(QuoteBlock(confirmation_id=row.confirmation_id, quote=quote))
            if row.stage == "quoted" and as_utc(quote.expires_at) > as_utc(self.clock()):
                actions.append(
                    await self._action(
                        row,
                        "commit_checkout",
                        "确认模拟下单（不扣款）",
                        {"confirmation_id": row.confirmation_id},
                        expires_at=quote.expires_at,
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
            if row.stage == "awaiting_payment" and row.active_operation_id is None:
                if row.payment_id is not None:
                    actions.append(
                        await self._action(
                            row,
                            "mock_pay",
                            "确认模拟付款（不会真实扣款）",
                            {"payment_id": row.payment_id},
                        )
                    )
                elif row.confirmation_id is not None:
                    actions.append(
                        await self._action(
                            row,
                            "commit_checkout",
                            "继续生成模拟付款入口",
                            {"confirmation_id": row.confirmation_id},
                        )
                    )
        messages = {
            "shopping": "请选择模拟商品；本流程不会产生真实订单或扣款。",
            "cart": "模拟购物车已更新，尚未下单。",
            "quoted": "请核对模拟商品、数量和总价后明确确认。",
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
        actions.append(await self._action(row, "refresh", "核实模拟购买状态", {}))
        return CommerceView(
            purchase_id=row.id,
            version=row.version,
            stage=cast(CommerceStage, row.stage),
            snapshot_at=as_utc(self.clock()),
            blocks=blocks,
            actions=actions,
        )
