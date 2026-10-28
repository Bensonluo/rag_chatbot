"""A stateful external-system emulator using independent SQL transactions.

Domain reservations commit before this adapter is called. Provider objects and
idempotency receipts deliberately have no domain or chat foreign keys.

The catalogue is generated deterministically from a compact family spec instead
of a hand-written trio of SKUs, so the fake scales to real paging/filtering
tests. Search is the provider's own lexical engine over provider-owned text
(token containment); user-language semantics stay in the LLM Agent, which
constructs the typed query, filters, sort and page cursor.
"""

import hashlib
import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.database.commerce import MockCommerceObject, MockCommerceReceipt
from app.models.schemas.commerce import (
    CartLine,
    CartLineSelection,
    CartSnapshot,
    CatalogFilters,
    CatalogItem,
    CatalogPage,
    OrderSnapshot,
    PaymentSnapshot,
    QuoteSnapshot,
)
from app.services.commerce.port import (
    CommerceSort,
    ProviderOperationReceipt,
    ProviderRejected,
    ProviderUnknown,
)

PAGE_SIZE_MAX = 50

# Compact family spec: the provider expands families × variants into the full
# SKU list with derived prices, stock and per-SKU order limits.
_FAMILY_SPECS: tuple[dict[str, Any], ...] = (
    {
        "product_id": "headphones-anc",
        "code": "ANC",
        "family": "轻音降噪耳机",
        "description": "模拟商品：无线连接、主动降噪；用于演示购物流程。",
        "attribute": "color",
        "variants": ("黑色", "白色", "银色", "藏青", "米白"),
        "base_price_minor": 29900,
        "price_step_minor": 500,
    },
    {
        "product_id": "keyboard-mechanical",
        "code": "KEY",
        "family": "机械键盘",
        "description": "模拟商品：有线连接、可热插拔轴体；用于演示购物流程。",
        "attribute": "layout",
        "variants": ("68键", "87键", "98键", "104键", "双模87键"),
        "base_price_minor": 19900,
        "price_step_minor": 1500,
    },
    {
        "product_id": "mouse-ergonomic",
        "code": "MOU",
        "family": "人体工学鼠标",
        "description": "模拟商品：垂直握持、可调 dpi；用于演示购物流程。",
        "attribute": "grip",
        "variants": ("右手型", "左手型", "中型", "大型", "紧凑型"),
        "base_price_minor": 12900,
        "price_step_minor": 800,
    },
    {
        "product_id": "monitor-4k",
        "code": "MON",
        "family": "4K显示器",
        "description": "模拟商品：4K面板、可升降支架；用于演示购物流程。",
        "attribute": "size",
        "variants": ("24英寸", "27英寸", "32英寸", "34英寸", "27英寸Type-C"),
        "base_price_minor": 129900,
        "price_step_minor": 20000,
    },
    {
        "product_id": "desk-mat",
        "code": "MAT",
        "family": "桌面鼠标垫",
        "description": "模拟商品：锁边工艺、防滑底面；用于演示购物流程。",
        "attribute": "size",
        "variants": ("小号", "中号", "大号", "加宽", "超大号"),
        "base_price_minor": 4900,
        "price_step_minor": 700,
    },
    {
        "product_id": "usb-hub",
        "code": "HUB",
        "family": "桌面拓展坞",
        "description": "模拟商品：多接口拓展、独立供电；用于演示购物流程。",
        "attribute": "ports",
        "variants": ("4口", "6口", "8口", "10口", "12口"),
        "base_price_minor": 9900,
        "price_step_minor": 1200,
    },
)


def generate_catalog() -> tuple[dict[str, Any], ...]:
    """Deterministically expand the family spec into the provider catalogue."""
    items: list[dict[str, Any]] = []
    for family_index, spec in enumerate(_FAMILY_SPECS):
        for variant_index, variant in enumerate(spec["variants"]):
            items.append(
                {
                    "product_id": spec["product_id"],
                    "sku_id": f"{spec['code']}-{variant_index + 1:02d}",
                    "name": f"{spec['family']} · {variant}",
                    "description": spec["description"],
                    "attributes": {spec["attribute"]: variant},
                    "unit_price_minor": spec["base_price_minor"]
                    + variant_index * spec["price_step_minor"],
                    "available_quantity": max(0, (3 * family_index + 4 * variant_index) % 13 - 1),
                    "max_order_quantity": 3 + (family_index + 2 * variant_index) % 6,
                }
            )
    return tuple(items)


CATALOG = generate_catalog()


def utc_now() -> datetime:
    return datetime.now(UTC)


def as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def request_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


class MockCommercePort:
    mode: str = "mock"

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
            if await db.get(MockCommerceObject, "stock:" + str(CATALOG[0]["sku_id"])) is not None:
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

    @staticmethod
    def _matches(item: CatalogItem, terms: list[str]) -> bool:
        if not terms:
            return True
        text = " ".join(
            (item.name, item.description, item.sku_id, item.product_id, *item.attributes.values())
        ).lower()
        return all(term in text for term in terms)

    @staticmethod
    def _passes(item: CatalogItem, filters: CatalogFilters) -> bool:
        if filters.in_stock and item.available_quantity <= 0:
            return False
        return not (
            (
                filters.min_price_minor is not None
                and item.unit_price_minor < filters.min_price_minor
            )
            or (
                filters.max_price_minor is not None
                and item.unit_price_minor > filters.max_price_minor
            )
        )

    async def search_products(
        self,
        owner_user_id: int,
        query: str,
        *,
        filters: CatalogFilters | None = None,
        sort: CommerceSort = "relevance",
        page_size: int = 10,
        page_token: str | None = None,
    ) -> CatalogPage:
        if type(owner_user_id) is not int or owner_user_id <= 0:
            raise ProviderRejected("模拟访问身份无效")
        if type(page_size) is not int or not 1 <= page_size <= PAGE_SIZE_MAX:
            raise ProviderRejected(f"模拟分页每页数量须为1到{PAGE_SIZE_MAX}")
        offset = 0
        if page_token is not None:
            if not page_token.isdigit():
                raise ProviderRejected("模拟分页游标无效")
            offset = int(page_token)
        await self._seed()
        async with self.sessions() as db:
            rows = (
                await db.scalars(
                    select(MockCommerceObject).where(MockCommerceObject.kind == "stock")
                )
            ).all()
        # Legacy three-SKU rows predate provider-owned per-SKU limits and are
        # retained as data, but they are not part of this catalog generation.
        items = [
            CatalogItem.model_validate(row.payload)
            for row in rows
            if "max_order_quantity" in row.payload
        ]
        terms = [term for term in query.lower().split() if term]
        matched = [
            item
            for item in items
            if self._matches(item, terms) and self._passes(item, filters or CatalogFilters())
        ]
        if sort == "price_asc":
            matched.sort(key=lambda item: (item.unit_price_minor, item.sku_id))
        elif sort == "price_desc":
            matched.sort(key=lambda item: (-item.unit_price_minor, item.sku_id))
        elif sort == "newest":
            matched.sort(key=lambda item: item.sku_id, reverse=True)
        else:
            matched.sort(key=lambda item: item.sku_id)
        window = matched[offset : offset + page_size]
        return CatalogPage(
            items=window,
            next_page_token=(
                str(offset + page_size) if offset + page_size < len(matched) else None
            ),
            total_estimate=len(matched),
        )

    async def get_product(self, owner_user_id: int, sku_id: str) -> CatalogItem:
        if type(owner_user_id) is not int or owner_user_id <= 0:
            raise ProviderRejected("模拟访问身份无效")
        await self._seed()
        async with self.sessions() as db:
            row = await self._object(db, "stock:" + sku_id, 0, "stock")
            if "max_order_quantity" not in row.payload:
                raise ProviderRejected("模拟商品不可用")
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

    async def update_cart_lines(
        self,
        owner_user_id: int,
        cart_id: str | None,
        expected_cart_version: int | None,
        lines: list[CartLineSelection],
        request_key: str,
    ) -> CartSnapshot:
        await self._seed()
        if not isinstance(lines, list) or not lines:
            raise ProviderRejected("模拟购物车至少需要一行商品")
        if len({line.sku_id for line in lines}) != len(lines):
            raise ProviderRejected("同一规格不能拆成多行，请合并数量")

        async def work(db: AsyncSession) -> dict[str, Any]:
            resolved: list[tuple[CatalogItem, int]] = []
            for line in lines:
                stock = await self._object(db, "stock:" + line.sku_id, 0, "stock")
                item = CatalogItem.model_validate(stock.payload)
                if line.quantity > item.max_order_quantity:
                    raise ProviderRejected(
                        f"模拟限购：{item.name} 每单最多 {item.max_order_quantity} 件"
                    )
                if line.quantity > item.available_quantity:
                    raise ProviderRejected(f"模拟库存不足：{item.name}")
                resolved.append((item, line.quantity))
            identifier = cart_id or "mc_" + str(uuid4())
            old = await db.get(MockCommerceObject, identifier)
            if old is not None and (old.kind != "cart" or old.owner_user_id != owner_user_id):
                raise ProviderRejected("模拟购物车不可用")
            if old is not None and old.payload.get("ordered"):
                raise ProviderRejected("此购物车已下单")
            if (old.version if old is not None else None) != expected_cart_version:
                raise ProviderRejected("模拟购物车版本变化")
            version = (old.version if old else 0) + 1
            cart_lines = [
                CartLine(
                    sku_id=item.sku_id,
                    name=item.name,
                    attributes=item.attributes,
                    quantity=quantity,
                    unit_price_minor=item.unit_price_minor,
                    subtotal_minor=quantity * item.unit_price_minor,
                )
                for item, quantity in resolved
            ]
            result = CartSnapshot(
                cart_id=identifier,
                cart_version=version,
                lines=cart_lines,
                subtotal_minor=sum(line.subtotal_minor for line in cart_lines),
            ).model_dump(mode="json")
            if old is None:
                db.add(
                    MockCommerceObject(
                        id=identifier,
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
                        MockCommerceObject.id == identifier,
                        MockCommerceObject.version == old.version,
                    )
                    .values(version=version, payload=result)
                    .returning(MockCommerceObject.id)
                )
                if changed.scalar_one_or_none() is None:
                    raise ProviderRejected("模拟购物车并发变化")
            return result

        result = await self._run(
            owner_user_id,
            "update_cart",
            request_key,
            {
                "cart_id": cart_id,
                "expected_cart_version": expected_cart_version,
                "lines": [{"sku_id": line.sku_id, "quantity": line.quantity} for line in lines],
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
            for line in data.lines:
                stock = await self._object(db, "stock:" + line.sku_id, 0, "stock")
                item = CatalogItem.model_validate(stock.payload)
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
            stock_rows: dict[str, MockCommerceObject] = {}
            for line in quote.lines:
                stock = await self._object(db, "stock:" + line.sku_id, 0, "stock")
                item = CatalogItem.model_validate(stock.payload)
                if (
                    item.available_quantity < line.quantity
                    or item.unit_price_minor != line.unit_price_minor
                ):
                    raise ProviderRejected("模拟库存或价格变化，需要新确认")
                stock_rows[line.sku_id] = stock
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
            for line in quote.lines:
                stock = stock_rows[line.sku_id]
                item = CatalogItem.model_validate(stock.payload)
                changed = await db.execute(
                    update(MockCommerceObject)
                    .where(
                        MockCommerceObject.id == stock.id,
                        MockCommerceObject.version == stock.version,
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

    async def get_order(self, owner_user_id: int, order_id: str) -> OrderSnapshot:
        async with self.sessions() as db:
            return OrderSnapshot.model_validate(
                (await self._object(db, order_id, owner_user_id, "order")).payload
            )

    async def get_status(self, owner_user_id: int, payment_id: str) -> PaymentSnapshot:
        async with self.sessions() as db:
            return PaymentSnapshot.model_validate(
                (await self._object(db, payment_id, owner_user_id, "payment")).payload
            )

    async def lookup_operation(
        self, owner_user_id: int, request_key: str
    ) -> ProviderOperationReceipt | None:
        async with self.sessions() as db:
            receipt = await db.get(MockCommerceReceipt, request_key)
            if receipt is None or receipt.owner_user_id != owner_user_id:
                return None
            return cast(
                ProviderOperationReceipt,
                {"kind": receipt.kind, "status": receipt.status, "data": dict(receipt.result)},
            )

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
