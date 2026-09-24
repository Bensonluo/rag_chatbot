"""Feedback API endpoints for message rating."""

import logging
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_active_user
from app.models.database.user import User
from app.models.schemas.feedback import FeedbackCreate, FeedbackResponse
from app.repositories.feedback_repository import FeedbackRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/feedback", tags=["feedback"])


async def _get_db() -> AsyncGenerator[AsyncSession, None]:
    from app.api.database import async_session_maker

    async with async_session_maker() as session:
        yield session


@router.post("", response_model=FeedbackResponse, status_code=status.HTTP_200_OK)
async def submit_feedback(
    feedback: FeedbackCreate,
    db: AsyncSession = Depends(_get_db),
    current_user: User = Depends(get_current_active_user),  # noqa: ARG001  # FastAPI DI: enforces auth; value unused
) -> FeedbackResponse:
    """Submit feedback (thumbs up/down) for an assistant message."""
    repo = FeedbackRepository(db)
    message = await repo.submit_feedback(
        message_id=feedback.message_id,
        rating=feedback.rating,
        text=feedback.text,
    )
    if message is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Message {feedback.message_id} not found",
        )
    # User-verified quality signal: a downvote is the operational
    # closest proxy to "this answer was not a resolution" (north star).
    from app.services.chat.metrics import FEEDBACK_RATINGS

    FEEDBACK_RATINGS.labels(rating=str(feedback.rating)).inc()
    if feedback.rating < 0:
        # North-star cleanup: stop the pyramid replaying a user-rejected
        # answer (L0 exact + L1 semantic eviction; fail-open inside, so
        # a cache outage can never fail the feedback submission itself).
        from app.services.chat.cache_eviction import evict_downvoted_response

        evicted = await evict_downvoted_response(message.content)
        if evicted:
            logger.info("Downvote evicted %d cache entries for message %s", evicted, message.id)
    return FeedbackResponse(
        success=True,
        message_id=message.id,
        rating=message.user_rating or 0,
    )


@router.get("/stats")
async def get_feedback_stats(
    db: AsyncSession = Depends(_get_db),
    current_user: User = Depends(get_current_active_user),  # noqa: ARG001  # FastAPI DI: enforces auth; value unused
) -> dict[str, Any]:
    """Get aggregate feedback statistics."""
    repo = FeedbackRepository(db)
    return await repo.get_feedback_stats()
