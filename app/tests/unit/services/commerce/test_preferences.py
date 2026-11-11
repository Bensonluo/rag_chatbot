"""Visitor preference memory service (service layer only, no conversation wiring).

TDD contract:
- ``save`` upserts one row per ``(visitor_id, pref_key)`` and appends an audit
  row for every write (old -> new).
- ``load_active`` filters out rows whose ``expires_at`` has passed.
- ``forget`` removes the row and audits the removal.
- Visitors are strictly isolated from each other.
- Quota/limit/permission-style keys are rejected outright: preferences and
  permissions never share storage (redline documented in the service module).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.models.database.base import Base
from app.models.database.guest import GuestVisitor
from app.models.database.session import ChatSession  # noqa: F401 — metadata dependency
from app.models.database.user import User
from app.models.database.visitor_preference import (
    VisitorPreference,
    VisitorPreferenceAudit,
)
from app.services.commerce.preferences import PreferenceMemoryService


class FixedClock:
    now = datetime(2026, 11, 11, 9, 0, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: int) -> None:
        self.now = self.now + timedelta(seconds=seconds)


@pytest.fixture
async def shop(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/preferences.sqlite")

    @event.listens_for(engine.sync_engine, "connect")
    def fk(connection, _):
        connection.execute("PRAGMA foreign_keys=ON")

    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    visitor_ids: list[str] = []
    async with maker() as db:
        for index in (1, 2):
            db.add(User(id=index, email=f"pref-{index}@unit.invalid", hashed_password="!"))
        # Flush first: the users->guest_visitors FK has no relationship() for
        # the unit-of-work sorter to see, so ordering must be explicit.
        await db.flush()
        for index in (1, 2):
            visitor_id = str(uuid4())
            visitor_ids.append(visitor_id)
            db.add(
                GuestVisitor(
                    id=visitor_id,
                    user_id=index,
                    token_hash=f"{'0' * 63}{index}",
                    expires_at=datetime(2026, 12, 1, tzinfo=UTC),
                )
            )
        await db.commit()

    clock = FixedClock()
    service = PreferenceMemoryService(maker, clock=clock)
    yield maker, service, visitor_ids, clock
    await engine.dispose()


async def test_save_and_load_active_roundtrip(shop) -> None:
    _, service, visitor_ids, _ = shop
    entry = await service.save(
        visitor_ids[0], "display_preference", {"size": "27英寸", "panel": "IPS"}, source="confirmed"
    )
    assert entry.pref_key == "display_preference"
    assert entry.pref_value == {"size": "27英寸", "panel": "IPS"}
    assert entry.source == "confirmed"
    assert entry.expires_at is None

    active = await service.load_active(visitor_ids[0])
    assert [item.pref_key for item in active] == ["display_preference"]
    assert active[0].pref_value == {"size": "27英寸", "panel": "IPS"}


async def test_ttl_expiry_filters_expired_rows(shop) -> None:
    _, service, visitor_ids, clock = shop
    await service.save(visitor_ids[0], "budget_hint", {"ceiling_minor": 300000}, ttl_seconds=60)
    await service.save(visitor_ids[0], "brand_preference", {"brand": "恒锐"})

    clock.advance(61)
    active = await service.load_active(visitor_ids[0])
    assert [item.pref_key for item in active] == ["brand_preference"]


async def test_upsert_reuses_the_unique_key_row(shop) -> None:
    maker, service, visitor_ids, _ = shop
    await service.save(visitor_ids[0], "display_preference", {"size": "24英寸"})
    await service.save(visitor_ids[0], "display_preference", {"size": "34英寸"}, source="confirmed")

    async with maker() as db:
        rows = (
            (
                await db.execute(
                    select(VisitorPreference).where(VisitorPreference.visitor_id == visitor_ids[0])
                )
            )
            .scalars()
            .all()
        )
        audits = (
            (
                await db.execute(
                    select(VisitorPreferenceAudit).where(
                        VisitorPreferenceAudit.visitor_id == visitor_ids[0]
                    )
                )
            )
            .scalars()
            .all()
        )
    assert len(rows) == 1
    assert rows[0].pref_value == {"size": "34英寸"}
    assert rows[0].source == "confirmed"
    assert len(audits) == 2  # one audit row per write, not per row


async def test_audit_rows_capture_old_and_new_values(shop) -> None:
    maker, service, visitor_ids, _ = shop
    await service.save(visitor_ids[0], "display_preference", {"size": "24英寸"})
    await service.save(visitor_ids[0], "display_preference", {"size": "34英寸"})
    await service.forget(visitor_ids[0], "display_preference")

    async with maker() as db:
        audits = (
            (
                await db.execute(
                    select(VisitorPreferenceAudit)
                    .where(VisitorPreferenceAudit.visitor_id == visitor_ids[0])
                    .order_by(VisitorPreferenceAudit.id)
                )
            )
            .scalars()
            .all()
        )
    assert [row.action for row in audits] == ["save", "save", "forget"]
    assert audits[0].old_value is None
    assert audits[0].new_value == {"size": "24英寸"}
    assert audits[1].old_value == {"size": "24英寸"}
    assert audits[1].new_value == {"size": "34英寸"}
    assert audits[2].old_value == {"size": "34英寸"}
    assert audits[2].new_value is None


async def test_forget_removes_preference_and_reports_existence(shop) -> None:
    _, service, visitor_ids, _ = shop
    await service.save(visitor_ids[0], "display_preference", {"size": "27英寸"})

    assert await service.forget(visitor_ids[0], "display_preference") is True
    assert await service.load_active(visitor_ids[0]) == []
    assert await service.forget(visitor_ids[0], "display_preference") is False


async def test_visitors_never_see_each_others_preferences(shop) -> None:
    _, service, visitor_ids, _ = shop
    await service.save(visitor_ids[0], "display_preference", {"size": "27英寸"})
    await service.save(visitor_ids[1], "display_preference", {"size": "24英寸"})

    first = {item.pref_value["size"] for item in await service.load_active(visitor_ids[0])}
    second = {item.pref_value["size"] for item in await service.load_active(visitor_ids[1])}
    assert first == {"27英寸"}
    assert second == {"24英寸"}

    assert await service.forget(visitor_ids[1], "display_preference") is True
    assert len(await service.load_active(visitor_ids[0])) == 1


@pytest.mark.parametrize(
    "key",
    [
        "quota.monthly_spend",
        "limit.cart_lines",
        "permission.admin_tools",
        "policy.refund_exception",
    ],
)
async def test_quota_and_permission_keys_are_rejected(shop, key: str) -> None:
    _, service, visitor_ids, _ = shop
    with pytest.raises(ValueError, match="not a preference"):
        await service.save(visitor_ids[0], key, {"x": 1})


async def test_save_validates_key_shape_and_source(shop) -> None:
    _, service, visitor_ids, _ = shop
    with pytest.raises(ValueError, match="pref_key"):
        await service.save(visitor_ids[0], "", {"x": 1})
    with pytest.raises(ValueError, match="pref_key"):
        await service.save(visitor_ids[0], "k" * 65, {"x": 1})
    with pytest.raises(ValueError, match="source"):
        await service.save(visitor_ids[0], "display_preference", {"x": 1}, source="guessed")
    with pytest.raises(ValueError, match="ttl"):
        await service.save(visitor_ids[0], "display_preference", {"x": 1}, ttl_seconds=0)


async def test_inferred_never_silently_overwrites_confirmed(shop) -> None:
    """Confirmed facts outrank inferences; a later inferred write must not downgrade."""

    _, service, visitor_ids, _ = shop
    await service.save(visitor_ids[0], "display_preference", {"size": "27英寸"}, source="confirmed")
    entry = await service.save(
        visitor_ids[0], "display_preference", {"size": "24英寸"}, source="inferred"
    )
    assert entry.source == "confirmed"
    assert entry.pref_value == {"size": "27英寸"}
