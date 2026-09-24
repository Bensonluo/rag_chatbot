"""Knowledge-gap repository: best-effort writes + aggregated reads."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import exists, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from app.models.database.knowledge_gap import KnowledgeGapRecord
from app.models.database.knowledge_gap_resolution import KnowledgeGapResolution

_WHITESPACE_RE = re.compile(r"\s+")

# Bound the aggregation key so pathological long queries stay indexable.
_MAX_NORMALIZED_LEN = 200


def normalize_query(query: str) -> str:
    """Collapse whitespace and lowercase — the aggregation key."""
    return _WHITESPACE_RE.sub(" ", query.strip().lower())[:_MAX_NORMALIZED_LEN]


class KnowledgeGapRepository:
    """Async repository for knowledge-gap telemetry."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def record(
        self,
        query: str,
        intent: str,
        session_id: str | int | None = None,
        user_id: int = 0,
        top_score: float | None = None,
    ) -> KnowledgeGapRecord:
        """Insert one gap record. Caller owns the commit."""
        record = KnowledgeGapRecord(
            session_id=str(session_id) if session_id is not None else "",
            user_id=user_id,
            query=query,
            normalized_query=normalize_query(query),
            intent=intent,
            top_score=top_score,
        )
        self._db.add(record)
        await self._db.commit()
        await self._db.refresh(record)
        return record

    async def resolve_query(
        self, normalized_query: str, resolved_by: int = 0
    ) -> KnowledgeGapResolution:
        """Mark a normalized gap query resolved (upsert on the key).

        Re-resolving updates ``resolved_at`` / ``resolved_by`` in place —
        the worklist action is idempotent, and a later gap occurrence
        still re-opens the group because ``top_gaps`` compares against
        the latest hit.
        """
        normalized = normalize_query(normalized_query)
        result = await self._db.execute(
            select(KnowledgeGapResolution).where(
                KnowledgeGapResolution.normalized_query == normalized
            )
        )
        resolution = result.scalar_one_or_none()
        if resolution is None:
            resolution = KnowledgeGapResolution(
                normalized_query=normalized, resolved_by=resolved_by
            )
        else:
            resolution.resolved_at = func.now()
            resolution.resolved_by = resolved_by
        self._db.add(resolution)
        await self._db.commit()
        await self._db.refresh(resolution)
        return resolution

    async def top_gaps(
        self,
        days: int = 7,
        limit: int = 20,
        include_resolved: bool = False,
    ) -> list[dict[str, Any]]:
        """Most frequent unanswered queries in the window, newest-hit first.

        A group counts as resolved when a resolution exists whose
        ``resolved_at`` is at or after the group's latest occurrence;
        resolved groups are hidden unless ``include_resolved`` is set
        (they still come back annotated, and a fresh occurrence after
        the resolution re-opens the group).

        Returns dicts (not ORM rows) so the API layer can serialize
        directly: ``normalized_query``, ``hits``, ``last_seen``,
        ``sample_query``, ``sample_intent``, ``resolved``.
        """
        since = datetime.now(UTC) - timedelta(days=days)
        # Correlated "latest occurrence of this group" for the resolution
        # comparison — an alias keeps the subquery independent of the
        # outer GROUP BY.
        occurrences = aliased(KnowledgeGapRecord)
        last_hit = (
            select(func.max(occurrences.created_at))
            .where(occurrences.normalized_query == KnowledgeGapRecord.normalized_query)
            .scalar_subquery()
        )
        stmt = (
            select(
                KnowledgeGapRecord.normalized_query,
                func.count().label("hits"),
                func.max(KnowledgeGapRecord.created_at).label("last_seen"),
                func.min(KnowledgeGapRecord.query).label("sample_query"),
                func.min(KnowledgeGapRecord.intent).label("sample_intent"),
            )
            .where(KnowledgeGapRecord.created_at >= since)
            .group_by(KnowledgeGapRecord.normalized_query)
            .order_by(func.count().desc())
            .limit(limit)
        )
        if not include_resolved:
            stmt = stmt.where(
                ~exists().where(
                    KnowledgeGapResolution.normalized_query == KnowledgeGapRecord.normalized_query,
                    KnowledgeGapResolution.resolved_at >= last_hit,
                )
            )
        result = await self._db.execute(stmt)
        groups = [
            {
                "normalized_query": row.normalized_query,
                "hits": row.hits,
                "last_seen": row.last_seen,
                "sample_query": row.sample_query,
                "sample_intent": row.sample_intent,
            }
            for row in result.fetchall()
        ]
        if not groups:
            return []

        # Annotate resolved status (one small IN query — resolutions are
        # curator-managed and tiny next to the telemetry table).
        resolution_rows = await self._db.execute(
            select(
                KnowledgeGapResolution.normalized_query,
                KnowledgeGapResolution.resolved_at,
            ).where(
                KnowledgeGapResolution.normalized_query.in_([g["normalized_query"] for g in groups])
            )
        )
        resolved_at_by_query: dict[str, datetime] = {}
        for query, resolved_at in resolution_rows.fetchall():
            resolved_at_by_query[query] = resolved_at

        return [
            {
                "normalized_query": g["normalized_query"],
                "hits": g["hits"],
                "last_seen": g["last_seen"].isoformat() if g["last_seen"] else None,
                "sample_query": g["sample_query"],
                "sample_intent": g["sample_intent"],
                "resolved": (
                    (r := resolved_at_by_query.get(g["normalized_query"])) is not None
                    and r >= g["last_seen"]
                ),
            }
            for g in groups
        ]
