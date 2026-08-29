"""Feedback API endpoints for message rating."""
import logging

from fastapi import APIRouter, HTTPException, status, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schemas.feedback import FeedbackCreate, FeedbackResponse
from app.repositories.feedback_repository import FeedbackRepository
from app.api.deps import get_current_active_user
from app.models.database.user import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/feedback", tags=["feedback"])


async def _get_db():
    from app.api.database import async_session_maker
    async with async_session_maker() as session:
        yield session


@router.post("", response_model=FeedbackResponse, status_code=status.HTTP_200_OK)
async def submit_feedback(
    feedback: FeedbackCreate,
    db: AsyncSession = Depends(_get_db),
    current_user: User = Depends(get_current_active_user),
):
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
    return FeedbackResponse(
        success=True,
        message_id=message.id,
        rating=message.user_rating or 0,
    )


@router.get("/stats")
async def get_feedback_stats(
    db: AsyncSession = Depends(_get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Get aggregate feedback statistics."""
    repo = FeedbackRepository(db)
    return await repo.get_feedback_stats()
