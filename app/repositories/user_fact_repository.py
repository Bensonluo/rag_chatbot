"""UserFact repository: ADD-only cross-session user memory (Phase B).

mem0 v3 production pattern: extraction inserts facts and never
UPDATE/DELETEs existing rows on the hot path — contradictions resolve
by recency at read time plus retention pruning, not in-line conflict
resolution. Reads are newest-first so the recall side (Phase B2) gets
the freshest facts first.
"""

from __future__ import annotations

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database.user_fact import UserFactRecord


class UserFactRepository:
    """Async repository for user facts (write side of layered memory)."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def add(
        self,
        *,
        user_id: int,
        fact: str,
        category: str = "general",
        source_session_id: int | str | None = None,
    ) -> UserFactRecord:
        """Insert one user fact."""
        record = UserFactRecord(
            user_id=user_id,
            fact=fact,
            category=category,
            source_session_id=str(source_session_id) if source_session_id is not None else "",
        )
        self._db.add(record)
        await self._db.commit()
        await self._db.refresh(record)
        return record

    async def recent_for_user(self, *, user_id: int, limit: int = 20) -> list[UserFactRecord]:
        """Newest facts first; id tiebreaks same-second inserts."""
        stmt = (
            select(UserFactRecord)
            .where(UserFactRecord.user_id == user_id)
            .order_by(UserFactRecord.created_at.desc(), UserFactRecord.id.desc())
            .limit(limit)
        )
        result = await self._db.execute(stmt)
        return list(result.scalars().all())

    async def existing_texts(self, *, user_id: int) -> set[str]:
        """All fact texts for a user — the exact-dedup key set."""
        stmt = select(UserFactRecord.fact).where(UserFactRecord.user_id == user_id)
        result = await self._db.execute(stmt)
        return {row[0] for row in result.all()}

    async def prune_for_user(self, *, user_id: int, keep: int) -> int:
        """Delete all but the newest ``keep`` facts; returns rows removed."""
        keep_ids = (
            select(UserFactRecord.id)
            .where(UserFactRecord.user_id == user_id)
            .order_by(UserFactRecord.created_at.desc(), UserFactRecord.id.desc())
            .limit(keep)
        )
        stmt = delete(UserFactRecord).where(
            UserFactRecord.user_id == user_id,
            UserFactRecord.id.not_in(keep_ids),
        )
        result = await self._db.execute(stmt)
        await self._db.commit()
        # Async execution types DELETE as generic Result; the driver
        # always returns a CursorResult with rowcount at runtime.
        return int(getattr(result, "rowcount", 0) or 0)
