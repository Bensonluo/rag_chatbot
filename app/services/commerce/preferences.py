"""Visitor preference memory: save / recall / forget per-visitor preferences.

Service-layer module only — conversation integration (injecting preferences
into the commerce context and registering the tools) is deliberately out of
scope this phase; see ``docs/design/visitor-preference-memory.md`` for the
wiring points.

Storage contract (``visitor_preferences``): one row per
``(visitor_id, pref_key)``, optional per-key TTL, ``source`` marking how the
fact was learned (``inferred`` from conversation vs ``confirmed`` by the
visitor), and an append-only audit row for every write.

REDLINE — preference vs permission separation: this service stores
*preferences* only (subjective wants: display size, brand, budget hints).
Quota / limit / permission / policy configuration is NEVER accepted here:
those are enforcement settings that live in code and server configuration,
not in agent-writable storage. Keys under the reserved namespaces below are
rejected outright, and ``confirmed`` facts are never silently downgraded by
later ``inferred`` writes.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.database.visitor_preference import (
    VisitorPreference,
    VisitorPreferenceAudit,
)

PreferenceSource = Literal["inferred", "confirmed"]

RESERVED_KEY_PREFIXES = ("quota.", "limit.", "permission.", "policy.")
"""Keys that would smuggle enforcement settings into preference storage."""

MAX_PREF_KEY_LENGTH = 64
MIN_TTL_SECONDS = 1


class VisitorPreferenceEntry(BaseModel):
    """Read-side projection of one stored preference."""

    visitor_id: str
    pref_key: str
    pref_value: Any
    source: PreferenceSource
    expires_at: datetime | None = None
    updated_at: datetime | None = None


def utc_now() -> datetime:
    return datetime.now(UTC)


def _validate_key(key: str) -> None:
    if not key or len(key) > MAX_PREF_KEY_LENGTH:
        raise ValueError(f"pref_key must be 1-{MAX_PREF_KEY_LENGTH} characters")
    lowered = key.lower()
    if any(lowered.startswith(prefix) for prefix in RESERVED_KEY_PREFIXES):
        raise ValueError(
            f"{key!r} is not a preference: quota/limit/permission/policy keys are forbidden"
        )


class PreferenceMemoryService:
    """Per-visitor preference memory on the application database.

    ``maker`` is the shared async session factory (``app.api.database`` in
    production); ``clock`` is injectable for deterministic TTL tests.
    """

    def __init__(
        self,
        maker: async_sessionmaker[AsyncSession],
        *,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self._maker = maker
        self._clock = clock

    async def save(
        self,
        visitor_id: str,
        key: str,
        value: Any,
        *,
        source: PreferenceSource = "inferred",
        ttl_seconds: int | None = None,
    ) -> VisitorPreferenceEntry:
        """Upsert one preference and append an audit row, atomically.

        An ``inferred`` write against an existing ``confirmed`` fact is a
        no-op that returns the confirmed entry unchanged: confirmation is an
        upgrade, never something a later inference silently overwrites.
        """

        _validate_key(key)
        if source not in ("inferred", "confirmed"):
            raise ValueError(f"source must be 'inferred' or 'confirmed', got {source!r}")
        if ttl_seconds is not None and ttl_seconds < MIN_TTL_SECONDS:
            raise ValueError(f"ttl_seconds must be >= {MIN_TTL_SECONDS}")
        # Fail fast on non-JSON values instead of relying on driver coercion.
        json.dumps(value, ensure_ascii=False)

        now = self._clock()
        expires_at = now + timedelta(seconds=ttl_seconds) if ttl_seconds is not None else None

        async with self._maker() as session, session.begin():
            existing = (
                await session.execute(
                    select(VisitorPreference).where(
                        VisitorPreference.visitor_id == visitor_id,
                        VisitorPreference.pref_key == key,
                    )
                )
            ).scalar_one_or_none()

            if existing is not None and existing.source == "confirmed" and source == "inferred":
                # Build before the transaction closes: attribute loads must
                # stay inside the session's IO context.
                return _entry_from(existing)

            if existing is None:
                row = VisitorPreference(
                    visitor_id=visitor_id,
                    pref_key=key,
                    pref_value=value,
                    source=source,
                    expires_at=expires_at,
                )
                session.add(row)
                old_value = None
                target = row
            else:
                old_value = existing.pref_value
                existing.pref_value = value
                existing.source = source
                existing.expires_at = expires_at
                target = existing

            session.add(
                VisitorPreferenceAudit(
                    visitor_id=visitor_id,
                    pref_key=key,
                    action="save",
                    old_value=old_value,
                    new_value=value,
                )
            )
            await session.flush()
            # UPDATE flush expires the onupdate column; refresh explicitly
            # (attribute access alone would try synchronous IO).
            await session.refresh(target)
            entry = _entry_from(target)
        return entry

    async def load_active(self, visitor_id: str) -> list[VisitorPreferenceEntry]:
        """Return the visitor's preferences whose TTL (if any) has not passed."""

        now = self._clock()
        async with self._maker() as session:
            rows = (
                (
                    await session.execute(
                        select(VisitorPreference)
                        .where(
                            VisitorPreference.visitor_id == visitor_id,
                        )
                        .order_by(VisitorPreference.pref_key)
                    )
                )
                .scalars()
                .all()
            )
        return [_entry_from(row) for row in rows if _is_active(row.expires_at, now)]

    async def forget(self, visitor_id: str, key: str) -> bool:
        """Delete one preference (audited). Returns False if it did not exist."""

        _validate_key(key)
        async with self._maker() as session, session.begin():
            existing = (
                await session.execute(
                    select(VisitorPreference).where(
                        VisitorPreference.visitor_id == visitor_id,
                        VisitorPreference.pref_key == key,
                    )
                )
            ).scalar_one_or_none()
            if existing is None:
                return False
            old_value = existing.pref_value
            await session.execute(
                delete(VisitorPreference).where(
                    VisitorPreference.visitor_id == visitor_id,
                    VisitorPreference.pref_key == key,
                )
            )
            session.add(
                VisitorPreferenceAudit(
                    visitor_id=visitor_id,
                    pref_key=key,
                    action="forget",
                    old_value=old_value,
                    new_value=None,
                )
            )
            return True


def _as_aware(value: datetime | None) -> datetime | None:
    """SQLite returns naive datetimes; normalize to UTC for comparisons."""

    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _is_active(expires_at: datetime | None, now: datetime) -> bool:
    if expires_at is None:
        return True
    aware = expires_at if expires_at.tzinfo is not None else expires_at.replace(tzinfo=UTC)
    return aware > now


def _entry_from(row: VisitorPreference) -> VisitorPreferenceEntry:
    return VisitorPreferenceEntry(
        visitor_id=row.visitor_id,
        pref_key=row.pref_key,
        pref_value=row.pref_value,
        source=row.source,  # DB-checked by ck_visitor_preferences_source
        expires_at=_as_aware(row.expires_at),
        updated_at=_as_aware(row.updated_at),
    )


__all__ = [
    "PreferenceMemoryService",
    "VisitorPreferenceEntry",
    "PreferenceSource",
]
