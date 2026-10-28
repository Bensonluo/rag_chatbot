"""Durable commerce core: carts, quotes, orders, payments, stock, receipts.

Everything runs in explicit SQLite transactions (see ``storage``) with
provider-side business rules only: stock and price validation, per-SKU
purchase caps, optimistic cart versions, quote expiry/uniqueness, quote
consumption on order creation, and a separate payment lifecycle. Every
mutation is recorded as an idempotency receipt keyed by the caller's
``request_key`` so replays and unknown outcomes reconcile deterministically.

The core is deliberately reusable without HTTP: the FastAPI layer in
``api`` is a thin typed facade over these methods.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from uuid import uuid4

from app.commerce_provider.catalog import CommerceSort, VirtualCatalog
from app.commerce_provider.errors import (
    ProviderConflictError,
    ProviderNotFoundError,
    ProviderRejectedError,
    ProviderUnknownError,
)
from app.commerce_provider.faults import FaultController
from app.commerce_provider.storage import Database
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

MAX_CART_LINES = 50


def utc_now() -> datetime:
    return datetime.now(UTC)


def to_iso(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat()


def from_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def request_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class _Receipt:
    owner_user_id: int
    kind: str
    request_hash: str
    status: str
    result: dict[str, Any]


class CommerceProviderCore:
    """Owns every commerce fact; no user, chat, or domain foreign keys."""

    def __init__(
        self,
        database: Database,
        catalog: VirtualCatalog,
        *,
        clock: Callable[[], datetime] = utc_now,
        quote_ttl_seconds: int = 300,
        faults: FaultController | None = None,
    ) -> None:
        self.database = database
        self.catalog = catalog
        self.clock = clock
        self.quote_ttl_seconds = quote_ttl_seconds
        self.faults = faults
        self._stock_delta_cache: dict[str, int] | None = None

    # --- shared plumbing ----------------------------------------------------
    @staticmethod
    def _require_owner(owner_user_id: int) -> None:
        if type(owner_user_id) is not int or owner_user_id <= 0:
            raise ProviderRejectedError("模拟访问身份无效")

    @staticmethod
    def _receipt_from_row(row: sqlite3.Row) -> _Receipt:
        result: dict[str, Any] = json.loads(str(row["result_json"]))
        return _Receipt(
            owner_user_id=int(row["owner_user_id"]),
            kind=str(row["kind"]),
            request_hash=str(row["request_hash"]),
            status=str(row["status"]),
            result=result,
        )

    @staticmethod
    def _replay_or_conflict(
        receipt: _Receipt, owner: int, kind: str, digest: str
    ) -> dict[str, Any] | None:
        if receipt.owner_user_id != owner or receipt.kind != kind:
            raise ProviderConflictError("请求键已被其它请求占用")
        if receipt.request_hash != digest:
            raise ProviderConflictError("同一请求键对应不同的请求参数")
        return dict(receipt.result)

    @staticmethod
    def _fetch_receipt(connection: sqlite3.Connection, request_key: str) -> _Receipt | None:
        row = connection.execute(
            "SELECT owner_user_id, kind, request_hash, status, result_json "
            "FROM operations WHERE request_key = ?",
            (request_key,),
        ).fetchone()
        return None if row is None else CommerceProviderCore._receipt_from_row(row)

    async def _consume_fault(
        self, kind: str, target: str | None, *, modes: tuple[str, ...]
    ) -> str | None:
        if self.faults is None:
            return None
        return await self.faults.consume(kind, target, modes=modes)

    async def _mutate(
        self,
        owner: int,
        kind: str,
        request_key: str,
        arguments: dict[str, Any],
        target: str | None,
        work: Callable[[sqlite3.Connection, bool], dict[str, Any]],
    ) -> dict[str, Any]:
        self._require_owner(owner)
        digest = request_hash({"kind": kind, "arguments": arguments})

        def read_receipt(connection: sqlite3.Connection) -> _Receipt | None:
            return self._fetch_receipt(connection, request_key)

        receipt = await self.database.read(read_receipt)
        if receipt is not None:
            replay = self._replay_or_conflict(receipt, owner, kind, digest)
            if replay is not None:
                return replay

        pre_fault = await self._consume_fault(kind, target, modes=("reject", "processing_payment"))
        if pre_fault == "reject":
            raise ProviderRejectedError("模拟业务明确拒绝，未提交")
        processing = pre_fault == "processing_payment"

        def transaction(connection: sqlite3.Connection) -> dict[str, Any]:
            concurrent = self._fetch_receipt(connection, request_key)
            if concurrent is not None:
                replay = self._replay_or_conflict(concurrent, owner, kind, digest)
                if replay is not None:
                    return replay
            result = work(connection, processing)
            connection.execute(
                "INSERT INTO operations "
                "(request_key, owner_user_id, kind, request_hash, status, result_json, created_at) "
                "VALUES (?, ?, ?, ?, 'completed', ?, ?)",
                (
                    request_key,
                    owner,
                    kind,
                    digest,
                    json.dumps(result, ensure_ascii=False),
                    to_iso(self.clock()),
                ),
            )
            return result

        try:
            result = await self.database.write(transaction)
            # Do this before any post-commit fault can raise: the provider has
            # already committed, so displayed stock must not stay stale.
            self._invalidate_stock_cache()
        except sqlite3.IntegrityError as exc:
            # Another request with the same key committed concurrently.
            receipt = await self.database.read(read_receipt)
            if receipt is None:
                raise ProviderUnknownError("模拟并发结果需查询") from exc
            replay = self._replay_or_conflict(receipt, owner, kind, digest)
            if replay is None:  # pragma: no cover - defensive
                raise ProviderUnknownError("模拟并发结果需查询") from exc
            return replay
        except sqlite3.OperationalError as exc:
            if "locked" in str(exc) or "busy" in str(exc):
                raise ProviderUnknownError("模拟业务结果未确认") from exc
            raise

        post_fault = await self._consume_fault(kind, target, modes=("unknown_after_commit",))
        if post_fault is not None:
            raise ProviderUnknownError("模拟外部已受理，但回执超时；请按原请求核实")
        return result

    # --- catalog ------------------------------------------------------------
    async def _stock_deltas(self) -> dict[str, int]:
        if self._stock_delta_cache is None:

            def work(connection: sqlite3.Connection) -> list[sqlite3.Row]:
                return connection.execute(
                    "SELECT sku_id, COALESCE(SUM(delta), 0) AS total "
                    "FROM stock_deltas GROUP BY sku_id"
                ).fetchall()

            rows = await self.database.read(work)
            self._stock_delta_cache = {str(row["sku_id"]): int(row["total"]) for row in rows}
        return self._stock_delta_cache

    def _invalidate_stock_cache(self) -> None:
        self._stock_delta_cache = None

    def _deltas_for(self, connection: sqlite3.Connection, sku_ids: list[str]) -> dict[str, int]:
        if not sku_ids:
            return {}
        placeholders = ",".join("?" for _ in sku_ids)
        rows = connection.execute(
            "SELECT sku_id, COALESCE(SUM(delta), 0) AS total FROM stock_deltas "
            f"WHERE sku_id IN ({placeholders}) GROUP BY sku_id",
            sku_ids,
        ).fetchall()
        return {str(row["sku_id"]): int(row["total"]) for row in rows}

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
        self._require_owner(owner_user_id)
        deltas = await self._stock_deltas()
        return await asyncio.to_thread(
            self.catalog.search,
            query=query,
            filters=filters,
            sort=sort,
            page_size=page_size,
            page_token=page_token,
            stock_deltas=deltas,
        )

    async def get_product(self, owner_user_id: int, sku_id: str) -> CatalogItem:
        self._require_owner(owner_user_id)
        n = self.catalog.parse_sku_id(sku_id)
        if n is None:
            raise ProviderNotFoundError("模拟商品不存在")

        def work(connection: sqlite3.Connection) -> int:
            row = connection.execute(
                "SELECT COALESCE(SUM(delta), 0) AS total FROM stock_deltas WHERE sku_id = ?",
                (sku_id,),
            ).fetchone()
            return int(row["total"]) if row is not None else 0

        delta = await self.database.read(work)
        return await asyncio.to_thread(self.catalog.materialize, n, delta)

    # --- carts --------------------------------------------------------------
    def _validated_lines(
        self, connection: sqlite3.Connection, selections: list[CartLineSelection]
    ) -> list[CartLine]:
        deltas = self._deltas_for(connection, [line.sku_id for line in selections])
        validated: list[CartLine] = []
        seen: set[str] = set()
        for selection in selections:
            n = self.catalog.parse_sku_id(selection.sku_id)
            if n is None:
                raise ProviderNotFoundError("模拟商品不存在")
            if selection.sku_id in seen:
                raise ProviderRejectedError("购物车存在重复商品行")
            seen.add(selection.sku_id)
            item = self.catalog.materialize(n, deltas.get(selection.sku_id, 0))
            if selection.quantity > item.max_order_quantity:
                raise ProviderRejectedError(
                    f"模拟商品 {item.sku_id} 单笔限购 {item.max_order_quantity} 件"
                )
            if selection.quantity > item.available_quantity:
                raise ProviderRejectedError("模拟库存不足")
            validated.append(
                CartLine(
                    sku_id=item.sku_id,
                    name=item.name,
                    attributes=item.attributes,
                    quantity=selection.quantity,
                    unit_price_minor=item.unit_price_minor,
                    subtotal_minor=selection.quantity * item.unit_price_minor,
                )
            )
        return validated

    def _fetch_cart(self, connection: sqlite3.Connection, owner: int, cart_id: str) -> sqlite3.Row:
        row = connection.execute(
            "SELECT cart_id, owner_user_id, version, ordered_order_id, subtotal_minor "
            "FROM carts WHERE cart_id = ?",
            (cart_id,),
        ).fetchone()
        if row is None or int(row["owner_user_id"]) != owner:
            raise ProviderNotFoundError("模拟购物车不可用")
        return cast(sqlite3.Row, row)

    def _cart_lines(self, connection: sqlite3.Connection, cart_id: str) -> list[CartLine]:
        rows = connection.execute(
            "SELECT sku_id, name, attributes_json, quantity, unit_price_minor, subtotal_minor "
            "FROM cart_lines WHERE cart_id = ? ORDER BY line_no",
            (cart_id,),
        ).fetchall()
        return [
            CartLine(
                sku_id=str(row["sku_id"]),
                name=str(row["name"]),
                attributes=json.loads(str(row["attributes_json"])),
                quantity=int(row["quantity"]),
                unit_price_minor=int(row["unit_price_minor"]),
                subtotal_minor=int(row["subtotal_minor"]),
            )
            for row in rows
        ]

    async def update_cart_lines(
        self,
        owner_user_id: int,
        cart_id: str | None,
        expected_cart_version: int | None,
        lines: list[CartLineSelection],
        request_key: str,
    ) -> CartSnapshot:
        if len(lines) > MAX_CART_LINES:
            raise ProviderRejectedError(f"购物车最多 {MAX_CART_LINES} 行")

        arguments: dict[str, Any] = {
            "cart_id": cart_id,
            "expected_cart_version": expected_cart_version,
            "lines": [{"sku_id": line.sku_id, "quantity": line.quantity} for line in lines],
        }

        def work(connection: sqlite3.Connection, _processing: bool) -> dict[str, Any]:
            validated = self._validated_lines(connection, lines)
            subtotal = sum(line.subtotal_minor for line in validated)
            now = to_iso(self.clock())
            if cart_id is None:
                if expected_cart_version is not None:
                    raise ProviderRejectedError("模拟购物车版本变化")
                new_cart_id = "c_" + uuid4().hex
                version = 1
                connection.execute(
                    "INSERT INTO carts (cart_id, owner_user_id, version, ordered_order_id, "
                    "subtotal_minor, updated_at) VALUES (?, ?, 1, NULL, ?, ?)",
                    (new_cart_id, owner_user_id, subtotal, now),
                )
                effective_cart_id = new_cart_id
            else:
                row = self._fetch_cart(connection, owner_user_id, cart_id)
                if row["ordered_order_id"] is not None:
                    raise ProviderRejectedError("此购物车已下单")
                if expected_cart_version is None or int(row["version"]) != expected_cart_version:
                    raise ProviderRejectedError("模拟购物车版本变化")
                version = expected_cart_version + 1
                changed = connection.execute(
                    "UPDATE carts SET version = ?, subtotal_minor = ?, updated_at = ? "
                    "WHERE cart_id = ? AND version = ?",
                    (version, subtotal, now, cart_id, expected_cart_version),
                )
                if changed.rowcount != 1:
                    raise ProviderRejectedError("模拟购物车并发变化")
                effective_cart_id = cart_id
            connection.execute("DELETE FROM cart_lines WHERE cart_id = ?", (effective_cart_id,))
            connection.executemany(
                "INSERT INTO cart_lines (cart_id, line_no, sku_id, name, attributes_json, "
                "quantity, unit_price_minor, subtotal_minor) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        effective_cart_id,
                        line_no,
                        line.sku_id,
                        line.name,
                        json.dumps(line.attributes, ensure_ascii=False),
                        line.quantity,
                        line.unit_price_minor,
                        line.subtotal_minor,
                    )
                    for line_no, line in enumerate(validated)
                ],
            )
            return CartSnapshot(
                cart_id=effective_cart_id,
                cart_version=version,
                lines=validated,
                subtotal_minor=subtotal,
            ).model_dump(mode="json")

        result = await self._mutate(
            owner_user_id, "update_cart_lines", request_key, arguments, cart_id, work
        )
        return CartSnapshot.model_validate(result)

    async def get_cart(self, owner_user_id: int, cart_id: str) -> CartSnapshot:
        self._require_owner(owner_user_id)

        def work(connection: sqlite3.Connection) -> dict[str, Any]:
            row = self._fetch_cart(connection, owner_user_id, cart_id)
            lines = self._cart_lines(connection, cart_id)
            return CartSnapshot(
                cart_id=cart_id,
                cart_version=int(row["version"]),
                lines=lines,
                subtotal_minor=int(row["subtotal_minor"]),
            ).model_dump(mode="json")

        return CartSnapshot.model_validate(await self.database.read(work))

    # --- quotes -------------------------------------------------------------
    async def prepare_checkout(
        self,
        owner_user_id: int,
        cart_id: str,
        expected_cart_version: int,
        request_key: str,
    ) -> QuoteSnapshot:
        arguments: dict[str, Any] = {
            "cart_id": cart_id,
            "expected_cart_version": expected_cart_version,
        }

        def work(connection: sqlite3.Connection, _processing: bool) -> dict[str, Any]:
            row = self._fetch_cart(connection, owner_user_id, cart_id)
            if row["ordered_order_id"] is not None:
                raise ProviderRejectedError("此购物车已下单")
            if int(row["version"]) != expected_cart_version:
                raise ProviderRejectedError("模拟购物车版本变化")
            stored_lines = self._cart_lines(connection, cart_id)
            if not stored_lines:
                raise ProviderRejectedError("模拟购物车为空")
            selections = [
                CartLineSelection(sku_id=line.sku_id, quantity=line.quantity)
                for line in stored_lines
            ]
            current_lines = self._validated_lines(connection, selections)
            for stored, current in zip(stored_lines, current_lines, strict=True):
                if stored.unit_price_minor != current.unit_price_minor:
                    raise ProviderRejectedError("模拟价格变化，请重新确认购物车")
            now = self.clock()
            open_quote = connection.execute(
                "SELECT quote_id, cart_id, cart_version, lines_json, subtotal_minor, "
                "shipping_minor, total_minor, currency, expires_at FROM quotes "
                "WHERE cart_id = ? AND cart_version = ? AND owner_user_id = ? "
                "AND status = 'open' AND expires_at > ?",
                (cart_id, expected_cart_version, owner_user_id, to_iso(now)),
            ).fetchone()
            if open_quote is not None:
                # Asking for checkout again for an unchanged live cart returns
                # the provider's current authoritative quote; it is not a new
                # business offer and cannot invalidate the old confirmation.
                existing = QuoteSnapshot.model_validate(
                    {
                        "quote_id": str(open_quote["quote_id"]),
                        "cart_id": str(open_quote["cart_id"]),
                        "cart_version": int(open_quote["cart_version"]),
                        "lines": json.loads(str(open_quote["lines_json"])),
                        "subtotal_minor": int(open_quote["subtotal_minor"]),
                        "shipping_minor": int(open_quote["shipping_minor"]),
                        "total_minor": int(open_quote["total_minor"]),
                        "currency": str(open_quote["currency"]),
                        "expires_at": from_iso(str(open_quote["expires_at"])),
                    }
                )
                return existing.model_dump(mode="json")
            subtotal = sum(line.subtotal_minor for line in current_lines)
            quote = QuoteSnapshot(
                quote_id="q_" + uuid4().hex,
                cart_id=cart_id,
                cart_version=expected_cart_version,
                lines=current_lines,
                subtotal_minor=subtotal,
                shipping_minor=0,
                total_minor=subtotal,
                expires_at=now + timedelta(seconds=self.quote_ttl_seconds),
            )
            connection.execute(
                "INSERT INTO quotes (quote_id, owner_user_id, cart_id, cart_version, lines_json, "
                "subtotal_minor, shipping_minor, total_minor, currency, expires_at, status, "
                "order_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', NULL, ?)",
                (
                    quote.quote_id,
                    owner_user_id,
                    cart_id,
                    expected_cart_version,
                    json.dumps(
                        [line.model_dump(mode="json") for line in current_lines],
                        ensure_ascii=False,
                    ),
                    quote.subtotal_minor,
                    quote.shipping_minor,
                    quote.total_minor,
                    "CNY",
                    to_iso(quote.expires_at),
                    to_iso(now),
                ),
            )
            return quote.model_dump(mode="json")

        result = await self._mutate(
            owner_user_id, "prepare_checkout", request_key, arguments, cart_id, work
        )
        return QuoteSnapshot.model_validate(result)

    # --- orders -------------------------------------------------------------
    async def create_order(
        self, owner_user_id: int, quote_id: str, request_key: str
    ) -> OrderSnapshot:
        arguments: dict[str, Any] = {"quote_id": quote_id}

        def work(connection: sqlite3.Connection, _processing: bool) -> dict[str, Any]:
            quote_row = connection.execute(
                "SELECT quote_id, owner_user_id, cart_id, cart_version, lines_json, "
                "total_minor, expires_at, status FROM quotes WHERE quote_id = ?",
                (quote_id,),
            ).fetchone()
            if quote_row is None or int(quote_row["owner_user_id"]) != owner_user_id:
                raise ProviderNotFoundError("模拟报价不可用")
            if str(quote_row["status"]) != "open":
                raise ProviderRejectedError("模拟报价已被使用")
            if from_iso(str(quote_row["expires_at"])) <= self.clock():
                raise ProviderRejectedError("模拟报价已过期")
            cart_id = str(quote_row["cart_id"])
            cart_version = int(quote_row["cart_version"])
            cart_row = self._fetch_cart(connection, owner_user_id, cart_id)
            if int(cart_row["version"]) != cart_version or cart_row["ordered_order_id"] is not None:
                raise ProviderRejectedError("模拟报价对应的购物车已变化")
            stored_lines: list[dict[str, Any]] = json.loads(str(quote_row["lines_json"]))
            selections = [
                CartLineSelection(sku_id=str(line["sku_id"]), quantity=int(line["quantity"]))
                for line in stored_lines
            ]
            current_lines = self._validated_lines(connection, selections)
            for stored, current in zip(stored_lines, current_lines, strict=True):
                if int(stored["unit_price_minor"]) != current.unit_price_minor:
                    raise ProviderRejectedError("模拟价格变化，需要重新确认")
            order_id = "o_" + uuid4().hex
            now = to_iso(self.clock())
            total = int(quote_row["total_minor"])
            connection.execute(
                "INSERT INTO orders (order_id, owner_user_id, quote_id, total_minor, currency, "
                "status, created_at, updated_at) VALUES (?, ?, ?, ?, 'CNY', 'awaiting_payment', ?, ?)",
                (order_id, owner_user_id, quote_id, total, now, now),
            )
            consumed = connection.execute(
                "UPDATE quotes SET status = 'consumed', order_id = ? "
                "WHERE quote_id = ? AND status = 'open'",
                (order_id, quote_id),
            )
            if consumed.rowcount != 1:
                raise ProviderRejectedError("模拟报价并发使用")
            ordered = connection.execute(
                "UPDATE carts SET ordered_order_id = ?, version = version + 1, updated_at = ? "
                "WHERE cart_id = ? AND version = ? AND ordered_order_id IS NULL",
                (order_id, now, cart_id, cart_version),
            )
            if ordered.rowcount != 1:
                raise ProviderRejectedError("模拟购物车并发变化")
            connection.executemany(
                "INSERT INTO stock_deltas (sku_id, order_id, delta, created_at) "
                "VALUES (?, ?, ?, ?)",
                [(line.sku_id, order_id, -line.quantity, now) for line in current_lines],
            )
            return OrderSnapshot(
                order_id=order_id,
                quote_id=quote_id,
                total_minor=total,
                status="awaiting_payment",
            ).model_dump(mode="json")

        result = await self._mutate(
            owner_user_id, "create_order", request_key, arguments, quote_id, work
        )

        return OrderSnapshot.model_validate(result)

    async def get_order(self, owner_user_id: int, order_id: str) -> OrderSnapshot:
        self._require_owner(owner_user_id)

        def work(connection: sqlite3.Connection) -> dict[str, Any]:
            row = connection.execute(
                "SELECT order_id, owner_user_id, quote_id, total_minor, currency, status "
                "FROM orders WHERE order_id = ?",
                (order_id,),
            ).fetchone()
            if row is None or int(row["owner_user_id"]) != owner_user_id:
                raise ProviderNotFoundError("模拟订单不可用")
            return {
                "order_id": str(row["order_id"]),
                "quote_id": str(row["quote_id"]),
                "total_minor": int(row["total_minor"]),
                "currency": "CNY",
                "status": str(row["status"]),
            }

        return OrderSnapshot.model_validate(await self.database.read(work))

    # --- payments -----------------------------------------------------------
    def _fetch_payment(
        self, connection: sqlite3.Connection, owner: int, payment_id: str
    ) -> sqlite3.Row:
        row = connection.execute(
            "SELECT payment_id, owner_user_id, order_id, total_minor, status "
            "FROM payments WHERE payment_id = ?",
            (payment_id,),
        ).fetchone()
        if row is None or int(row["owner_user_id"]) != owner:
            raise ProviderNotFoundError("模拟付款不可用")
        return cast(sqlite3.Row, row)

    async def create_payment(
        self, owner_user_id: int, order_id: str, request_key: str
    ) -> PaymentSnapshot:
        arguments: dict[str, Any] = {"order_id": order_id}

        def work(connection: sqlite3.Connection, _processing: bool) -> dict[str, Any]:
            order_row = connection.execute(
                "SELECT order_id, owner_user_id, total_minor, status FROM orders WHERE order_id = ?",
                (order_id,),
            ).fetchone()
            if order_row is None or int(order_row["owner_user_id"]) != owner_user_id:
                raise ProviderNotFoundError("模拟订单不可用")
            if str(order_row["status"]) != "awaiting_payment":
                raise ProviderRejectedError("订单当前状态不能创建付款")
            existing = connection.execute(
                "SELECT payment_id FROM payments WHERE order_id = ?", (order_id,)
            ).fetchone()
            if existing is not None:
                raise ProviderRejectedError("该订单已有付款记录")
            payment = PaymentSnapshot(
                payment_id="pay_" + uuid4().hex,
                order_id=order_id,
                total_minor=int(order_row["total_minor"]),
                status="awaiting_payment",
            )
            now = to_iso(self.clock())
            connection.execute(
                "INSERT INTO payments (payment_id, owner_user_id, order_id, total_minor, "
                "currency, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, 'CNY', ?, ?, ?)",
                (
                    payment.payment_id,
                    owner_user_id,
                    order_id,
                    payment.total_minor,
                    payment.status,
                    now,
                    now,
                ),
            )
            return payment.model_dump(mode="json")

        result = await self._mutate(
            owner_user_id, "create_payment", request_key, arguments, order_id, work
        )
        return PaymentSnapshot.model_validate(result)

    async def mock_pay(
        self, owner_user_id: int, payment_id: str, request_key: str
    ) -> PaymentSnapshot:
        arguments: dict[str, Any] = {"payment_id": payment_id}

        def work(connection: sqlite3.Connection, processing: bool) -> dict[str, Any]:
            row = self._fetch_payment(connection, owner_user_id, payment_id)
            status = str(row["status"])
            if status in {"processing", "failed"}:
                raise ProviderRejectedError("原模拟付款仍处理中或已终止，不能再次付款")
            new_status = "processing" if processing else "paid"
            now = to_iso(self.clock())
            connection.execute(
                "UPDATE payments SET status = ?, updated_at = ? WHERE payment_id = ?",
                (new_status, now, payment_id),
            )
            connection.execute(
                "UPDATE orders SET status = ?, updated_at = ? WHERE order_id = ?",
                (new_status, now, str(row["order_id"])),
            )
            return PaymentSnapshot(
                payment_id=payment_id,
                order_id=str(row["order_id"]),
                total_minor=int(row["total_minor"]),
                status=new_status,
            ).model_dump(mode="json")

        result = await self._mutate(
            owner_user_id, "mock_pay", request_key, arguments, payment_id, work
        )
        return PaymentSnapshot.model_validate(result)

    async def get_status(self, owner_user_id: int, payment_id: str) -> PaymentSnapshot:
        self._require_owner(owner_user_id)

        def work(connection: sqlite3.Connection) -> dict[str, Any]:
            row = self._fetch_payment(connection, owner_user_id, payment_id)
            return {
                "payment_id": payment_id,
                "order_id": str(row["order_id"]),
                "total_minor": int(row["total_minor"]),
                "currency": "CNY",
                "status": str(row["status"]),
            }

        return PaymentSnapshot.model_validate(await self.database.read(work))

    async def settle_payment(
        self, owner_user_id: int, payment_id: str, *, paid: bool = True
    ) -> PaymentSnapshot:
        """Test-only settlement of a processing payment (async provider event)."""

        self._require_owner(owner_user_id)

        def work(connection: sqlite3.Connection) -> dict[str, Any]:
            row = self._fetch_payment(connection, owner_user_id, payment_id)
            if str(row["status"]) != "processing":
                raise ProviderRejectedError("只有处理中的模拟付款可以结算")
            new_status = "paid" if paid else "failed"
            now = to_iso(self.clock())
            connection.execute(
                "UPDATE payments SET status = ?, updated_at = ? WHERE payment_id = ?",
                (new_status, now, payment_id),
            )
            connection.execute(
                "UPDATE orders SET status = ?, updated_at = ? WHERE order_id = ?",
                (new_status, now, str(row["order_id"])),
            )
            return PaymentSnapshot(
                payment_id=payment_id,
                order_id=str(row["order_id"]),
                total_minor=int(row["total_minor"]),
                status=new_status,
            ).model_dump(mode="json")

        return PaymentSnapshot.model_validate(await self.database.write(work))

    # --- receipts -----------------------------------------------------------
    async def lookup_operation(self, owner_user_id: int, request_key: str) -> dict[str, Any] | None:
        self._require_owner(owner_user_id)

        def work(connection: sqlite3.Connection) -> _Receipt | None:
            return self._fetch_receipt(connection, request_key)

        receipt = await self.database.read(work)
        if receipt is None or receipt.owner_user_id != owner_user_id:
            return None
        return {
            "kind": receipt.kind,
            "status": receipt.status,
            "result": receipt.result,
        }


__all__ = [
    "CommerceProviderCore",
    "MAX_CART_LINES",
    "from_iso",
    "request_hash",
    "to_iso",
    "utc_now",
]
