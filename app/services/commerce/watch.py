"""Proactive price watches: pre-authorized condition + event-driven trigger + ask.

Industry-consensus proactive pattern (Sierra/Fin-style): the visitor
explicitly pre-authorizes a watch ("SKU X at or below ¥Y → ask me"), a
trigger fires when the condition holds against a freshly read provider
price, and the agent ASKS. A watch never adds to a cart and never creates
an order or payment: the visitor's answer still travels the existing
quote → order → payment confirmation state machine.

Trigger evaluation is EVENT-DRIVEN ONLY — there is no background
scheduler. Evaluation happens at conversation time whenever price facts
for a watched SKU are (re)read: the search / get_product tool reads, the
creation read itself, and a session-start refresh for visitors with
active rules. A production deployment would additionally run the same
``evaluate`` from a cron/worker for off-session notifications.

Card discipline (same doctrine as ``ApprovalCard``): the ask card's
every field is copied from the durable rule row plus the provider price
fact that triggered it; the model never authors card data.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from typing import Literal

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.exceptions import ValidationError
from app.models.database.commerce_watch import PriceWatch
from app.models.database.guest import GuestVisitor
from app.models.schemas.commerce import WatchCard

logger = logging.getLogger(__name__)

CONDITION_AT_OR_BELOW = "at_or_below"
MAX_SKU_ID_LENGTH = 64
MAX_SKU_NAME_LENGTH = 128


class WatchEntry(BaseModel):
    """Read-side projection of one watch rule."""

    watch_id: str
    visitor_id: str
    sku_id: str
    sku_name: str
    target_price_minor: int
    condition: Literal["at_or_below"]
    status: Literal["active", "triggered", "asked", "cancelled"]
    created_at: datetime | None = None
    expires_at: datetime | None = None
    triggered_at: datetime | None = None
    triggered_price_minor: int | None = None


def utc_now() -> datetime:
    return datetime.now(UTC)


def _aware(value: datetime) -> datetime:
    """SQLite returns naive datetimes; normalize to UTC for comparisons."""

    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _as_aware(value: datetime | None) -> datetime | None:
    return None if value is None else _aware(value)


def _entry_from(row: PriceWatch) -> WatchEntry:
    return WatchEntry(
        watch_id=row.id,
        visitor_id=row.visitor_id,
        sku_id=row.sku_id,
        sku_name=row.sku_name,
        target_price_minor=row.target_price_minor,
        condition=row.condition,  # DB-checked by ck_commerce_price_watches_condition
        status=row.status,  # DB-checked by ck_commerce_price_watches_status
        created_at=_as_aware(row.created_at),
        expires_at=_as_aware(row.expires_at),
        triggered_at=_as_aware(row.triggered_at),
        triggered_price_minor=row.triggered_price_minor,
    )


def _normalize_sku(sku_id: str) -> str:
    """Canonical SKU identity, same normalization as cart lines."""

    normalized = sku_id.strip().upper()
    if not normalized or len(normalized) > MAX_SKU_ID_LENGTH:
        raise ValidationError("sku_id must be 1 to 64 non-space characters")
    return normalized


def _card_from(entry: WatchEntry) -> WatchCard:
    """Field-by-field copy from the durable rule; no model-authored input."""

    if (
        entry.triggered_at is None
        or entry.triggered_price_minor is None
        or entry.expires_at is None
    ):
        raise ValidationError("A triggered watch rule is required to build the ask card")
    return WatchCard(
        watch_id=entry.watch_id,
        sku_id=entry.sku_id,
        sku_name=entry.sku_name,
        target_price_minor=entry.target_price_minor,
        current_price_minor=entry.triggered_price_minor,
        condition=entry.condition,
        triggered_at=entry.triggered_at,
        expires_at=entry.expires_at,
    )


class PriceWatchService:
    """Per-visitor watch-rule storage plus event-driven trigger evaluation.

    ``maker`` is the shared async session factory (``app.api.database`` in
    production); ``clock`` is injectable for deterministic TTL tests.
    ``max_rules`` / ``max_age_days`` default to settings at write time;
    explicit values (tests, programmatic wiring) win.
    """

    def __init__(
        self,
        maker: async_sessionmaker[AsyncSession],
        *,
        clock: Callable[[], datetime] = utc_now,
        max_rules: int | None = None,
        max_age_days: int | None = None,
    ) -> None:
        self._maker = maker
        self._clock = clock
        self._max_rules = max_rules
        self._max_age_days = max_age_days

    def _limits(self) -> tuple[int, int]:
        """(max active rules, max rule age in days) from settings unless injected."""

        from app.config.settings import get_settings

        settings = get_settings()
        maximum = (
            self._max_rules
            if self._max_rules is not None
            else (settings.COMMERCE_WATCH_MAX_RULES_PER_VISITOR)
        )
        age = (
            self._max_age_days
            if self._max_age_days is not None
            else (settings.COMMERCE_WATCH_MAX_AGE_DAYS)
        )
        return max(1, maximum), max(1, age)

    async def visitor_id_for(self, owner: int) -> str | None:
        """Resolve the trusted guest visitor id for a verified owner.

        The owner id comes from the authenticated principal (never from
        model arguments); a missing mapping means this caller is not a
        public visitor and watch features stay unavailable.
        """

        if type(owner) is not int or owner <= 0:
            return None
        async with self._maker() as db:
            return await db.scalar(select(GuestVisitor.id).where(GuestVisitor.user_id == owner))

    async def create(
        self,
        visitor_id: str,
        sku_id: str,
        *,
        sku_name: str = "",
        target_price_minor: int,
    ) -> WatchEntry:
        """Create (or re-arm) the visitor's watch rule for one SKU.

        One row per ``(visitor_id, sku_id)``: watching an already-watched
        SKU updates the target in place, so re-arming never consumes the
        active-rule cap and never accumulates duplicate rules.
        """

        sku = _normalize_sku(sku_id)
        name = sku_name.strip()[:MAX_SKU_NAME_LENGTH]
        if type(target_price_minor) is not int or target_price_minor < 1:
            raise ValidationError("target_price_minor must be a positive integer of minor units")
        max_rules, max_age_days = self._limits()
        now = self._clock()
        expires_at = now + timedelta(days=max_age_days)
        async with self._maker() as db, db.begin():
            row = (
                await db.execute(
                    select(PriceWatch).where(
                        PriceWatch.visitor_id == visitor_id, PriceWatch.sku_id == sku
                    )
                )
            ).scalar_one_or_none()
            if row is None:
                active = (
                    await db.execute(
                        select(PriceWatch.id)
                        .where(
                            PriceWatch.visitor_id == visitor_id,
                            PriceWatch.status == "active",
                        )
                        .limit(max_rules + 1)
                    )
                ).scalars()
                if len(active.all()) >= max_rules:
                    raise ValidationError(
                        f"每位访客最多同时关注 {max_rules} 个 SKU 的价格，请先取消不需要的提醒"
                    )
                row = PriceWatch(
                    visitor_id=visitor_id,
                    sku_id=sku,
                    sku_name=name,
                    target_price_minor=target_price_minor,
                    condition=CONDITION_AT_OR_BELOW,
                    status="active",
                    expires_at=expires_at,
                )
                db.add(row)
            else:
                # Re-arm in place: fresh target, fresh window, clean receipt.
                row.sku_name = name
                row.target_price_minor = target_price_minor
                row.status = "active"
                row.triggered_at = None
                row.triggered_price_minor = None
                row.expires_at = expires_at
            await db.flush()
            await db.refresh(row)
            return _entry_from(row)

    async def _rows(self, visitor_id: str, *, statuses: set[str]) -> list[PriceWatch]:
        now = self._clock()
        async with self._maker() as db:
            rows = (
                (
                    await db.execute(
                        select(PriceWatch)
                        .where(PriceWatch.visitor_id == visitor_id)
                        .order_by(PriceWatch.created_at, PriceWatch.id)
                    )
                )
                .scalars()
                .all()
            )
        return [row for row in rows if row.status in statuses and _aware(row.expires_at) > now]

    async def active_rules(self, visitor_id: str) -> list[WatchEntry]:
        """Armed rules (not yet triggered, cancelled or expired)."""

        return [_entry_from(row) for row in await self._rows(visitor_id, statuses={"active"})]

    async def list_rules(self, visitor_id: str) -> list[WatchEntry]:
        """All non-expired rules, including triggered/asked receipts."""

        return [
            _entry_from(row)
            for row in await self._rows(visitor_id, statuses={"active", "triggered", "asked"})
        ]

    async def cancel(self, visitor_id: str, sku_id: str) -> bool:
        """Disarm one rule. Returns False if it did not exist or was already
        disarmed — a repeat cancel is a no-op, never a fresh receipt."""

        sku = _normalize_sku(sku_id)
        async with self._maker() as db, db.begin():
            row = (
                await db.execute(
                    select(PriceWatch).where(
                        PriceWatch.visitor_id == visitor_id,
                        PriceWatch.sku_id == sku,
                        PriceWatch.status != "cancelled",
                    )
                )
            ).scalar_one_or_none()
            if row is None:
                return False
            row.status = "cancelled"
            return True

    async def evaluate(
        self,
        visitor_id: str,
        price_facts: Mapping[str, tuple[int, str]],
    ) -> list[WatchEntry]:
        """Flip armed rules to ``triggered`` when freshly read prices satisfy them.

        ``price_facts`` maps sku_id → (unit price minor, provider name) for
        SKUs whose price was JUST read from the provider — never cached or
        model-narrated values. Expired and non-active rules never trigger.
        """

        if not price_facts:
            return []
        now = self._clock()
        facts = {_normalize_sku(sku): (price, name) for sku, (price, name) in price_facts.items()}
        triggered: list[WatchEntry] = []
        async with self._maker() as db, db.begin():
            rows = (
                (
                    await db.execute(
                        select(PriceWatch).where(
                            PriceWatch.visitor_id == visitor_id,
                            PriceWatch.sku_id.in_(facts.keys()),
                            PriceWatch.status == "active",
                        )
                    )
                )
                .scalars()
                .all()
            )
            for row in rows:
                if _aware(row.expires_at) <= now:
                    continue
                price, name = facts[row.sku_id]
                if row.condition == CONDITION_AT_OR_BELOW and price <= row.target_price_minor:
                    row.status = "triggered"
                    row.triggered_at = now
                    row.triggered_price_minor = price
                    if name.strip():
                        row.sku_name = name.strip()[:MAX_SKU_NAME_LENGTH]
                    triggered.append(_entry_from(row))
        return triggered

    async def claim_ask(self, visitor_id: str) -> WatchCard | None:
        """Claim the pending ask exactly once and build its structured card.

        The oldest un-delivered trigger transitions to ``asked`` and its
        card is returned; a second claim yields None. The ask rides the
        next completion as a ``WatchCard`` — asking only, never ordering.
        """

        async with self._maker() as db, db.begin():
            row = (
                await db.execute(
                    select(PriceWatch)
                    .where(
                        PriceWatch.visitor_id == visitor_id,
                        PriceWatch.status == "triggered",
                    )
                    .order_by(PriceWatch.triggered_at, PriceWatch.id)
                    .limit(1)
                )
            ).scalar_one_or_none()
            if row is None:
                return None
            if _aware(row.expires_at) <= self._clock():
                row.status = "cancelled"
                return None
            row.status = "asked"
            return _card_from(_entry_from(row))


__all__ = [
    "CONDITION_AT_OR_BELOW",
    "PriceWatchService",
    "WatchEntry",
    "utc_now",
]
