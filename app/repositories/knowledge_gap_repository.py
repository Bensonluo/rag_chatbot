"""Knowledge-gap repository: best-effort writes + aggregated reads."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database.knowledge_gap import KnowledgeGapRecord

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

    async def top_gaps(
        self,
        days: int = 7,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Most frequent unanswered queries in the window, newest-hit first.

        Returns dicts (not ORM rows) so the API layer can serialize
        directly: ``normalized_query``, ``hits``, ``last_seen``,
        ``sample_query``, ``sample_intent``.
        """
        since = datetime.now(UTC) - timedelta(days=days)
        result = await self._db.execute(
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
        return [
            {
                "normalized_query": row.normalized_query,
                "hits": row.hits,
                "last_seen": row.last_seen.isoformat() if row.last_seen else None,
                "sample_query": row.sample_query,
                "sample_intent": row.sample_intent,
            }
            for row in result.fetchall()
        ]
