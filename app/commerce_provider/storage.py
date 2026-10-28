"""Durable SQLite storage for the commerce provider core.

The provider owns its facts in a single SQLite database file managed with the
stdlib ``sqlite3`` module (no new dependency). Every operation opens its own
connection inside ``asyncio.to_thread``; WAL journaling plus ``BEGIN
IMMEDIATE`` serialize writers while readers stay concurrent. Explicit
transactions wrap every unit of work, and the schema is created idempotently
at startup, so state survives service restarts on the same file.
"""

from __future__ import annotations

import asyncio
import contextlib
import sqlite3
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

SCHEMA = """
CREATE TABLE IF NOT EXISTS carts (
    cart_id           TEXT PRIMARY KEY,
    owner_user_id     INTEGER NOT NULL,
    version           INTEGER NOT NULL,
    ordered_order_id  TEXT,
    subtotal_minor    INTEGER NOT NULL,
    updated_at        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cart_lines (
    cart_id          TEXT NOT NULL REFERENCES carts(cart_id) ON DELETE CASCADE,
    line_no          INTEGER NOT NULL,
    sku_id           TEXT NOT NULL,
    name             TEXT NOT NULL,
    attributes_json  TEXT NOT NULL,
    quantity         INTEGER NOT NULL,
    unit_price_minor INTEGER NOT NULL,
    subtotal_minor   INTEGER NOT NULL,
    PRIMARY KEY (cart_id, line_no)
);

CREATE TABLE IF NOT EXISTS quotes (
    quote_id       TEXT PRIMARY KEY,
    owner_user_id  INTEGER NOT NULL,
    cart_id        TEXT NOT NULL,
    cart_version   INTEGER NOT NULL,
    lines_json     TEXT NOT NULL,
    subtotal_minor INTEGER NOT NULL,
    shipping_minor INTEGER NOT NULL,
    total_minor    INTEGER NOT NULL,
    currency       TEXT NOT NULL,
    expires_at     TEXT NOT NULL,
    status         TEXT NOT NULL CHECK (status IN ('open', 'consumed')),
    order_id       TEXT,
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    order_id      TEXT PRIMARY KEY,
    owner_user_id INTEGER NOT NULL,
    quote_id      TEXT NOT NULL,
    total_minor   INTEGER NOT NULL,
    currency      TEXT NOT NULL,
    status        TEXT NOT NULL CHECK (status IN ('awaiting_payment', 'processing', 'paid', 'failed')),
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS payments (
    payment_id    TEXT PRIMARY KEY,
    owner_user_id INTEGER NOT NULL,
    order_id      TEXT NOT NULL,
    total_minor   INTEGER NOT NULL,
    currency      TEXT NOT NULL,
    status        TEXT NOT NULL CHECK (status IN ('awaiting_payment', 'processing', 'paid', 'failed')),
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS stock_deltas (
    sku_id     TEXT NOT NULL,
    order_id   TEXT NOT NULL REFERENCES orders(order_id) ON DELETE CASCADE,
    delta      INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (sku_id, order_id)
);

CREATE TABLE IF NOT EXISTS operations (
    request_key  TEXT PRIMARY KEY,
    owner_user_id INTEGER NOT NULL,
    kind         TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    status       TEXT NOT NULL,
    result_json  TEXT NOT NULL,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fault_rules (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    kind       TEXT NOT NULL,
    mode       TEXT NOT NULL,
    target     TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_carts_owner ON carts(owner_user_id);
CREATE INDEX IF NOT EXISTS idx_quotes_cart ON quotes(cart_id);
CREATE INDEX IF NOT EXISTS idx_orders_owner ON orders(owner_user_id);
CREATE INDEX IF NOT EXISTS idx_payments_order ON payments(order_id);
CREATE INDEX IF NOT EXISTS idx_stock_deltas_sku ON stock_deltas(sku_id);
CREATE INDEX IF NOT EXISTS idx_operations_owner ON operations(owner_user_id);
"""


class Database:
    """Async facade over a per-operation ``sqlite3`` connection."""

    def __init__(self, path: str, *, busy_timeout_ms: int = 5_000) -> None:
        self.path = path
        self.busy_timeout_ms = busy_timeout_ms

    def _new_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=self.busy_timeout_ms / 1000,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    def _startup(self) -> None:
        connection = self._new_connection()
        try:
            connection.executescript(SCHEMA)
        finally:
            connection.close()

    def _run_read(self, work: Callable[[sqlite3.Connection], T]) -> T:
        connection = self._new_connection()
        try:
            return work(connection)
        finally:
            connection.close()

    def _run_write(self, work: Callable[[sqlite3.Connection], T]) -> T:
        connection = self._new_connection()
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                result = work(connection)
            except BaseException:
                with contextlib.suppress(sqlite3.Error):
                    connection.execute("ROLLBACK")
                raise
            connection.execute("COMMIT")
            return result
        finally:
            connection.close()

    async def startup(self) -> None:
        """Create the schema (idempotent, safe across concurrent starts)."""

        await asyncio.to_thread(self._startup)

    async def read(self, work: Callable[[sqlite3.Connection], T]) -> T:
        """Run a read-only unit of work on a fresh connection."""

        return await asyncio.to_thread(self._run_read, work)

    async def write(self, work: Callable[[sqlite3.Connection], T]) -> T:
        """Run one explicit ``BEGIN IMMEDIATE`` transaction and commit it."""

        return await asyncio.to_thread(self._run_write, work)


__all__ = ["Database", "SCHEMA"]
