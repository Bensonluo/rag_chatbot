"""Analytics API endpoints for knowledge-gap reporting.

Read side of knowledge-gap telemetry: the queries the bot could not
ground, aggregated by normalized query. This is the operational signal
KB curators work from — every mainstream CS platform (Zendesk,
Intercom, 阿里小蜜) ships an equivalent "unanswered questions" report.
"""

from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_active_user
from app.models.database.user import User
from app.repositories.knowledge_gap_repository import KnowledgeGapRepository

router = APIRouter(prefix="/analytics", tags=["analytics"])


async def _get_db() -> AsyncGenerator[AsyncSession, None]:
    from app.api.database import async_session_maker

    async with async_session_maker() as session:
        yield session


@router.get("/knowledge-gaps")
async def get_knowledge_gaps(
    days: int = Query(default=7, ge=1, le=90, description="Lookback window in days"),
    limit: int = Query(default=20, ge=1, le=100, description="Max gap groups to return"),
    db: AsyncSession = Depends(_get_db),
    current_user: User = Depends(get_current_active_user),  # noqa: ARG001  # FastAPI DI: enforces auth; value unused
) -> dict[str, Any]:
    """Most frequent unanswered knowledge-intent queries in the window."""
    repo = KnowledgeGapRepository(db)
    gaps = await repo.top_gaps(days=days, limit=limit)
    return {"days": days, "total": len(gaps), "gaps": gaps}
