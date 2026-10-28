"""A stateful external-system emulator using independent SQL transactions.

Domain reservations commit before this adapter is called. Provider objects and
idempotency receipts deliberately have no domain or chat foreign keys.
"""

import hashlib
import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.database.commerce import MockCommerceObject, MockCommerceReceipt
from app.models.schemas.commerce import (
    CartLine,
    CartSnapshot,
    CatalogItem,
    OrderSnapshot,
    PaymentSnapshot,
    QuoteSnapshot,
)
from app.services.commerce.port import ProviderRejected, ProviderUnknown

CATALOG = (
    {
        "product_id": "headphones-anc",
        "sku_id": "ANC-BLACK",
        "name": "轻音降噪耳机 · 黑色",
        "description": "模拟商品：无线连接、主动降噪；用于演示购物流程。",
        "attributes": {"color": "黑色"},
        "unit_price_minor": 29900,
        "available_quantity": 12,
    },
    {
        "product_id": "headphones-anc",
        "sku_id": "ANC-WHITE",
        "name": "轻音降噪耳机 · 白色",
        "description": "模拟商品：无线连接、主动降噪；用于演示购物流程。",
        "attributes": {"color": "白色"},
        "unit_price_minor": 29900,
        "available_quantity": 8,
    },
    {
        "product_id": "keyboard-87",
        "sku_id": "KEYBOARD-87",
        "name": "87键机械键盘",
        "description": "模拟商品：有线键盘、紧凑布局；用于演示购物流程。",
        "attributes": {"layout": "87键", "color": "灰白"},
        "unit_price_minor": 19900,
        "available_quantity": 20,
    },
)


def utc_now() -> datetime:
    return datetime.now(UTC)


def as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def request_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


class MockCommercePort:
    mode: Literal["mock"] = "mock"

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        *,
        clock: Callable[[], datetime] = utc_now,
        quote_ttl_seconds: int = 300,
    ) -> None:
        self.sessions = sessions
        self.clock = clock
        self.quote_ttl_seconds = quote_ttl_seconds
        self._faults: dict[str, str] = {}

    def fail_next(self, kind: str, *, mode: str = "accepted_timeout") -> None:
        """Test/sandbox control, never a model tool or public payment API."""
        if mode not in {"accepted_timeout", "processing", "reject"}:
            raise ValueError("Unsupported mock fault")
        self._faults[kind] = mode

    async def _seed(self) -> None:
        async with self.sessions() as db:
            if await db.get(MockCommerceObject, "stock:ANC-BLACK") is not None:
                return
            db.add_all(
                MockCommerceObject(
                    id="stock:" + str(item["sku_id"]),
                    kind="stock",
                    owner_user_id=0,
                    version=0,
                    payload=dict(item),
                )
                for item in CATALOG
            )
            try:
                await db.commit()
            except IntegrityError:
                await db.rollback()  # A concurrent seeder commits the same fixed catalogue.

    async def _object(
        self, db: AsyncSession, identifier: str, owner: int, kind: str
    ) -> MockCommerceObject:
        row = await db.get(MockCommerceObject, identifier)
        if row is None or row.owner_user_id != owner or row.kind != kind:
            raise ProviderRejected("模拟业务对象不可用")
        return row

    async def search_products(self, owner_user_id: int, query: str) -> list[CatalogItem]:
        if type(owner_user_id) is not int or owner_user_id <= 0:
            raise ProviderRejected("模拟访问身份无效")
        await self._seed()
        async with self.sessions() as db:
            rows = (
                await db.scalars(
                    select(MockCommerceObject)
                    .where(MockCommerceObject.kind == "stock")
                    .order_by(MockCommerceObject.id)
                )
            ).all()
            items = [CatalogItem.model_validate(row.payload) for row in rows]
        terms = query.lower().strip()
        if not terms:
            return items[:10]
        found = [
            item
            for item in items
            if terms in (item.name + item.description + item.sku_id + item.product_id).lower()
        ]
        if not found:
            if any(term in terms for term in ("耳机", "降噪", "headphone", "earphone", "anc")):
                found = [item for item in items if item.product_id == "headphones-anc"]
            elif any(term in terms for term in ("键盘", "keyboard")):
                found = [item for item in items if item.product_id == "keyboard-87"]
        if any(term in terms for term in ("黑色", "black")):
            found = [item for item in found if item.attributes.get("color") == "黑色"]
        elif any(term in terms for term in ("白色", "white")):
            found = [item for item in found if "白" in item.attributes.get("color", "")]
        return found[:10]

    async def get_product(self, owner_user_id: int, sku_id: str) -> CatalogItem:
        if type(owner_user_id) is not int or owner_user_id <= 0:
            raise ProviderRejected("模拟访问身份无效")
        await self._seed()
        async with self.sessions() as db:
            row = await self._object(db, "stock:" + sku_id, 0, "stock")
            return CatalogItem.model_validate(row.payload)

    async def _run(
        self,
        owner: int,
        kind: str,
        key: str,
        arguments: dict[str, Any],
        work: Callable[[AsyncSession], Awaitable[dict[str, Any]]],
    ) -> dict[str, Any]:
        digest = request_hash({"kind": kind, "arguments": arguments})

        async def existing() -> dict[str, Any] | None:
            async with self.sessions() as db:
                receipt = await db.get(MockCommerceReceipt, key)
                if receipt is None:
                    return None
                if (
                    receipt.owner_user_id != owner
                    or receipt.kind != kind
                    or receipt.request_hash != digest
                ):
                    raise ProviderRejected("模拟请求身份或参数不匹配")
                if receipt.status == "failed":
                    raise ProviderRejected(str(receipt.result.get("error", "模拟业务拒绝")))
                return dict(receipt.result)

        result = await existing()
        if result is not None:
            return result
        if self._faults.get(kind) == "reject":
            self._faults.pop(kind)
            raise ProviderRejected("模拟业务明确拒绝，未提交")
        try:
            async with self.sessions() as db, db.begin():
                result = await work(db)
                db.add(
                    MockCommerceReceipt(
                        request_key=key,
                        owner_user_id=owner,
                        kind=kind,
                        request_hash=digest,
                        status="completed",
                        result=result,
                    )
                )
        except IntegrityError as exc:
            result = await existing()
            if result is None:
                raise ProviderUnknown("模拟并发结果需查询") from exc
        except ProviderRejected:
            raise
        except SQLAlchemyError as exc:
            raise ProviderUnknown("模拟业务结果未确认") from exc
        if self._faults.pop(kind, None) == "accepted_timeout":
            raise ProviderUnknown("模拟外部已受理，但回执超时；请按原请求核实")
        return result

    async def set_cart(
        self,
        owner_user_id: int,
        cart_id: str,
        sku_id: str,
        quantity: int,
        expected_cart_version: int | None,
        request_key: str,
    ) -> CartSnapshot:
        await self._seed()
        if type(quantity) is not int or not 1 <= quantity <= 10:
            raise ProviderRejected("数量须为1到10的整数")

        async def work(db: AsyncSession) -> dict[str, Any]:
            stock = await self._object(db, "stock:" + sku_id, 0, "stock")
            item = CatalogItem.model_validate(stock.payload)
            if quantity > item.available_quantity:
                raise ProviderRejected("模拟库存不足")
            old = await db.get(MockCommerceObject, cart_id)
            if old is not None and (old.kind != "cart" or old.owner_user_id != owner_user_id):
                raise ProviderRejected("模拟购物车不可用")
            if old is not None and old.payload.get("ordered"):
                raise ProviderRejected("此购物车已下单")
            if (old.version if old is not None else None) != expected_cart_version:
                raise ProviderRejected("模拟购物车版本变化")
            version = (old.version if old else 0) + 1
            line = CartLine(
                sku_id=item.sku_id,
                name=item.name,
                attributes=item.attributes,
                quantity=quantity,
                unit_price_minor=item.unit_price_minor,
                subtotal_minor=quantity * item.unit_price_minor,
            )
            result = CartSnapshot(
                cart_id=cart_id,
                cart_version=version,
                lines=[line],
                subtotal_minor=line.subtotal_minor,
            ).model_dump(mode="json")
            if old is None:
                db.add(
                    MockCommerceObject(
                        id=cart_id,
                        kind="cart",
                        owner_user_id=owner_user_id,
                        version=version,
                        payload=result,
                    )
                )
            else:
                changed = await db.execute(
                    update(MockCommerceObject)
                    .where(
                        MockCommerceObject.id == cart_id, MockCommerceObject.version == old.version
                    )
                    .values(version=version, payload=result)
                    .returning(MockCommerceObject.id)
                )
                if changed.scalar_one_or_none() is None:
                    raise ProviderRejected("模拟购物车并发变化")
            return result

        result = await self._run(
            owner_user_id,
            "set_cart",
            request_key,
            {
                "cart_id": cart_id,
                "sku_id": sku_id,
                "quantity": quantity,
                "version": expected_cart_version,
            },
            work,
        )
        return CartSnapshot.model_validate(result)

    async def get_cart(self, owner_user_id: int, cart_id: str) -> CartSnapshot:
        async with self.sessions() as db:
            row = await self._object(db, cart_id, owner_user_id, "cart")
            return CartSnapshot.model_validate(
                {
                    **{k: v for k, v in row.payload.items() if k != "ordered"},
                    "cart_version": row.version,
                }
            )

    async def prepare_checkout(
        self, owner_user_id: int, cart_id: str, expected_cart_version: int, request_key: str
    ) -> QuoteSnapshot:
        async def work(db: AsyncSession) -> dict[str, Any]:
            cart = await self._object(db, cart_id, owner_user_id, "cart")
            if cart.version != expected_cart_version or cart.payload.get("ordered"):
                raise ProviderRejected("模拟购物车版本变化")
            data = CartSnapshot.model_validate(cart.payload)
            stock = await self._object(db, "stock:" + data.lines[0].sku_id, 0, "stock")
            item = CatalogItem.model_validate(stock.payload)
            line = data.lines[0]
            if (
                line.quantity > item.available_quantity
                or line.unit_price_minor != item.unit_price_minor
            ):
                raise ProviderRejected("模拟库存或价格变化，请重新选择")
            quote = QuoteSnapshot(
                quote_id="mq_" + str(uuid4()),
                cart_id=cart_id,
                cart_version=cart.version,
                lines=data.lines,
                subtotal_minor=data.subtotal_minor,
                shipping_minor=0,
                total_minor=data.subtotal_minor,
                expires_at=as_utc(self.clock()) + timedelta(seconds=self.quote_ttl_seconds),
            )
            result = quote.model_dump(mode="json")
            db.add(
                MockCommerceObject(
                    id=quote.quote_id,
                    kind="quote",
                    owner_user_id=owner_user_id,
                    version=0,
                    payload=result,
                )
            )
            return result

        result = await self._run(
            owner_user_id,
            "quote",
            request_key,
            {"cart_id": cart_id, "version": expected_cart_version},
            work,
        )
        return QuoteSnapshot.model_validate(result)

    async def create_order(
        self, owner_user_id: int, quote_id: str, request_key: str
    ) -> OrderSnapshot:
        async def work(db: AsyncSession) -> dict[str, Any]:
            quote_row = await self._object(db, quote_id, owner_user_id, "quote")
            quote = QuoteSnapshot.model_validate(quote_row.payload)
            if as_utc(quote.expires_at) <= as_utc(self.clock()):
                raise ProviderRejected("模拟报价已过期")
            cart = await self._object(db, quote.cart_id, owner_user_id, "cart")
            if cart.version != quote.cart_version or cart.payload.get("ordered"):
                raise ProviderRejected("模拟报价对应的购物车已变化")
            line = quote.lines[0]
            stock = await self._object(db, "stock:" + line.sku_id, 0, "stock")
            item = CatalogItem.model_validate(stock.payload)
            if (
                item.available_quantity < line.quantity
                or item.unit_price_minor != line.unit_price_minor
            ):
                raise ProviderRejected("模拟库存或价格变化，需要新确认")
            order = OrderSnapshot(
                order_id="mo_" + str(uuid4()),
                quote_id=quote_id,
                total_minor=quote.total_minor,
                status="awaiting_payment",
            )
            changed = await db.execute(
                update(MockCommerceObject)
                .where(
                    MockCommerceObject.id == cart.id,
                    MockCommerceObject.version == quote.cart_version,
                )
                .values(
                    version=cart.version + 1, payload={**cart.payload, "ordered": order.order_id}
                )
                .returning(MockCommerceObject.id)
            )
            if changed.scalar_one_or_none() is None:
                raise ProviderRejected("模拟购物车并发变化")
            changed = await db.execute(
                update(MockCommerceObject)
                .where(
                    MockCommerceObject.id == stock.id, MockCommerceObject.version == stock.version
                )
                .values(
                    version=stock.version + 1,
                    payload={
                        **stock.payload,
                        "available_quantity": item.available_quantity - line.quantity,
                    },
                )
                .returning(MockCommerceObject.id)
            )
            if changed.scalar_one_or_none() is None:
                raise ProviderRejected("模拟库存并发变化")
            result = order.model_dump(mode="json")
            db.add(
                MockCommerceObject(
                    id=order.order_id,
                    kind="order",
                    owner_user_id=owner_user_id,
                    parent_id=quote_id,
                    version=0,
                    payload=result,
                )
            )
            return result

        result = await self._run(owner_user_id, "order", request_key, {"quote_id": quote_id}, work)
        return OrderSnapshot.model_validate(result)

    async def create_payment(
        self, owner_user_id: int, order_id: str, request_key: str
    ) -> PaymentSnapshot:
        async def work(db: AsyncSession) -> dict[str, Any]:
            order = OrderSnapshot.model_validate(
                (await self._object(db, order_id, owner_user_id, "order")).payload
            )
            payment = PaymentSnapshot(
                payment_id="mp_" + str(uuid4()),
                order_id=order_id,
                total_minor=order.total_minor,
                status=order.status,
            )
            result = payment.model_dump(mode="json")
            db.add(
                MockCommerceObject(
                    id=payment.payment_id,
                    kind="payment",
                    owner_user_id=owner_user_id,
                    parent_id=order_id,
                    version=0,
                    payload=result,
                )
            )
            return result

        result = await self._run(
            owner_user_id, "payment", request_key, {"order_id": order_id}, work
        )
        return PaymentSnapshot.model_validate(result)

    async def mock_pay(
        self, owner_user_id: int, payment_id: str, request_key: str
    ) -> PaymentSnapshot:
        async def work(db: AsyncSession) -> dict[str, Any]:
            row = await self._object(db, payment_id, owner_user_id, "payment")
            payment = PaymentSnapshot.model_validate(row.payload)
            if payment.status not in {"awaiting_payment", "paid"}:
                raise ProviderRejected("原模拟付款仍处理中或已终止，不能再次付款")
            status = "processing" if self._faults.get("mock_pay") == "processing" else "paid"
            result = payment.model_copy(update={"status": status}).model_dump(mode="json")
            row.payload = result
            row.version += 1
            order = await self._object(db, payment.order_id, owner_user_id, "order")
            order.payload = {**order.payload, "status": status}
            order.version += 1
            return result

        result = await self._run(
            owner_user_id, "mock_pay", request_key, {"payment_id": payment_id}, work
        )
        return PaymentSnapshot.model_validate(result)

    async def get_status(self, owner_user_id: int, payment_id: str) -> PaymentSnapshot:
        async with self.sessions() as db:
            return PaymentSnapshot.model_validate(
                (await self._object(db, payment_id, owner_user_id, "payment")).payload
            )

    async def lookup_operation(self, owner_user_id: int, request_key: str) -> dict[str, Any] | None:
        async with self.sessions() as db:
            receipt = await db.get(MockCommerceReceipt, request_key)
            if receipt is None or receipt.owner_user_id != owner_user_id:
                return None
            return {"kind": receipt.kind, "status": receipt.status, "data": dict(receipt.result)}

    async def settle_payment(
        self, owner_user_id: int, payment_id: str, *, paid: bool = True
    ) -> None:
        """Simulate a later provider settlement in tests; never exposed as an Agent tool."""
        async with self.sessions() as db, db.begin():
            row = await self._object(db, payment_id, owner_user_id, "payment")
            if row.payload["status"] != "processing":
                raise ProviderRejected("只有处理中模拟付款可异步结算")
            status = "paid" if paid else "failed"
            row.payload = {**row.payload, "status": status}
            order = await self._object(db, str(row.payload["order_id"]), owner_user_id, "order")
            order.payload = {**order.payload, "status": status}
