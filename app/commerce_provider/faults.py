"""Test-only fault injection for the commerce provider.

Rules are persisted in SQLite so a planned fault survives restarts — which is
exactly what "unknown after commit" reconciliation tests need. The whole
facility is disabled unless explicitly configured
(``COMMERCE_PROVIDER_FAULTS_ENABLED=true``): when disabled, planning raises
and consumption is a no-op, and the HTTP layer hides the testing routes.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import Any

from app.commerce_provider.errors import ProviderRejectedError
from app.commerce_provider.storage import Database

FAULT_KINDS: tuple[str, ...] = (
    "update_cart_lines",
    "prepare_checkout",
    "create_order",
    "create_payment",
    "mock_pay",
)

FAULT_MODES: tuple[str, ...] = (
    "reject",  # definitive refusal before any write
    "unknown_after_commit",  # commit, then answer as an unconfirmed outcome
    "processing_payment",  # sandbox-pay parks the payment in "processing"
)

_PRE_COMMIT_MODES: tuple[str, ...] = ("reject", "processing_payment")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class FaultController:
    """Durable store of planned faults; inert unless ``enabled``."""

    def __init__(self, database: Database, *, enabled: bool) -> None:
        self.database = database
        self.enabled = enabled

    def _require_enabled(self) -> None:
        if not self.enabled:
            raise ProviderRejectedError("故障注入未启用")

    async def plan(
        self, *, kind: str, mode: str, target: str | None = None, times: int = 1
    ) -> None:
        """Queue the next ``kind`` operation to fail in ``mode``."""

        self._require_enabled()
        if kind not in FAULT_KINDS:
            raise ProviderRejectedError("未知的故障注入操作类型")
        if mode not in FAULT_MODES:
            raise ProviderRejectedError("未知的故障注入模式")

        def work(connection: sqlite3.Connection) -> None:
            created_at = _now_iso()
            connection.executemany(
                "INSERT INTO fault_rules (kind, mode, target, created_at) VALUES (?, ?, ?, ?)",
                [(kind, mode, target, created_at)] * times,
            )

        await self.database.write(work)

    async def list_rules(self) -> list[dict[str, Any]]:
        self._require_enabled()

        def work(connection: sqlite3.Connection) -> list[sqlite3.Row]:
            return connection.execute(
                "SELECT id, kind, mode, target, created_at FROM fault_rules ORDER BY id"
            ).fetchall()

        rows = await self.database.read(work)
        return [
            {
                "id": int(row["id"]),
                "kind": str(row["kind"]),
                "mode": str(row["mode"]),
                "target": row["target"],
                "created_at": str(row["created_at"]),
            }
            for row in rows
        ]

    async def clear(self) -> int:
        self._require_enabled()

        def work(connection: sqlite3.Connection) -> int:
            cursor = connection.execute("DELETE FROM fault_rules")
            return int(cursor.rowcount)

        return await self.database.write(work)

    async def consume(
        self,
        kind: str,
        target: str | None,
        *,
        modes: tuple[str, ...] = _PRE_COMMIT_MODES,
    ) -> str | None:
        """Atomically take one matching rule and return its mode."""

        if not self.enabled:
            return None

        def work(connection: sqlite3.Connection) -> str | None:
            placeholders = ",".join("?" for _ in modes)
            row = connection.execute(
                "SELECT id, mode FROM fault_rules "
                f"WHERE kind = ? AND mode IN ({placeholders}) "
                "AND (target IS NULL OR target = ?) ORDER BY id LIMIT 1",
                (kind, *modes, target),
            ).fetchone()
            if row is None:
                return None
            connection.execute("DELETE FROM fault_rules WHERE id = ?", (int(row["id"]),))
            return str(row["mode"])

        try:
            return await self.database.write(work)
        except sqlite3.Error:
            # Faults must never turn a healthy business operation into an
            # unknown outcome; failing open keeps the provider available.
            return None


__all__ = ["FAULT_KINDS", "FAULT_MODES", "FaultController"]
