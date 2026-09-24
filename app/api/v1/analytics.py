"""Analytics API endpoints for knowledge-gap reporting.

Read side of knowledge-gap telemetry: the queries the bot could not
ground, aggregated by normalized query. This is the operational signal
KB curators work from — every mainstream CS platform (Zendesk,
Intercom, 阿里小蜜) ships an equivalent "unanswered questions" report.
"""

from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_active_user
from app.models.database.user import User
from app.repositories.knowledge_gap_repository import KnowledgeGapRepository

router = APIRouter(prefix="/analytics", tags=["analytics"])


async def _get_db() -> AsyncGenerator[AsyncSession, None]:
    from app.api.database import async_session_maker

    async with async_session_maker() as session:
        yield session


class ResolveGapRequest(BaseModel):
    """Body for POST /knowledge-gaps/resolve."""

    normalized_query: str = Field(..., min_length=1)

    @field_validator("normalized_query")
    @classmethod
    def _strip_and_reject_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("normalized_query must not be blank")
        return stripped


@router.get("/knowledge-gaps")
async def get_knowledge_gaps(
    days: int = Query(default=7, ge=1, le=90, description="Lookback window in days"),
    limit: int = Query(default=20, ge=1, le=100, description="Max gap groups to return"),
    include_resolved: bool = Query(
        default=False, description="Also return resolved groups (annotated)"
    ),
    db: AsyncSession = Depends(_get_db),
    current_user: User = Depends(get_current_active_user),  # noqa: ARG001  # FastAPI DI: enforces auth; value unused
) -> dict[str, Any]:
    """Most frequent unanswered knowledge-intent queries in the window."""
    repo = KnowledgeGapRepository(db)
    gaps = await repo.top_gaps(days=days, limit=limit, include_resolved=include_resolved)
    return {"days": days, "total": len(gaps), "gaps": gaps}


@router.post("/knowledge-gaps/resolve")
async def resolve_knowledge_gap(
    request: ResolveGapRequest,
    db: AsyncSession = Depends(_get_db),
    current_user: User = Depends(get_current_active_user),
) -> dict[str, Any]:
    """Mark a gap group handled; a later occurrence re-opens it automatically."""
    repo = KnowledgeGapRepository(db)
    resolution = await repo.resolve_query(request.normalized_query, resolved_by=current_user.id)
    return {
        "normalized_query": resolution.normalized_query,
        "resolved_by": resolution.resolved_by,
        "resolved_at": resolution.resolved_at.isoformat() if resolution.resolved_at else None,
    }
