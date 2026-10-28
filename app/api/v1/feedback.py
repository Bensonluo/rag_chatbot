"""Feedback API endpoints for message rating."""

import logging
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_active_user
from app.api.deps.private_response import PrivateAPIRoute
from app.config.settings import settings
from app.models.database.user import User
from app.models.schemas.feedback import FeedbackCreate, FeedbackResponse
from app.repositories.feedback_repository import FeedbackRepository
from app.services.chat.coordination import (
    ConversationBusyError,
    CoordinationUnavailableError,
    SessionCoordinator,
    get_session_coordinator,
)
from app.services.chat.session_identity import ConversationChangedError, require_session_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/feedback", tags=["feedback"], route_class=PrivateAPIRoute)


async def _get_db() -> AsyncGenerator[AsyncSession, None]:
    from app.api.database import async_session_maker

    async with async_session_maker() as session:
        yield session


@router.post("", response_model=FeedbackResponse, status_code=status.HTTP_200_OK)
async def submit_feedback(
    feedback: FeedbackCreate,
    db: AsyncSession = Depends(_get_db),
    current_user: User = Depends(get_current_active_user),
    coordinator: SessionCoordinator = Depends(get_session_coordinator),
) -> FeedbackResponse:
    """Submit feedback (thumbs up/down) for an assistant message."""
    return await submit_scoped_feedback(
        feedback, db, current_user, coordinator, allow_shared=settings.DEMO_MODE
    )


async def submit_scoped_feedback(
    feedback: FeedbackCreate,
    db: AsyncSession,
    current_user: User,
    coordinator: SessionCoordinator,
    *,
    allow_shared: bool,
) -> FeedbackResponse:
    """The visitor boundary always disables legacy shared-demo authorization."""
    repo = FeedbackRepository(db)
    # Public Demo rows retain shared feedback. Private sessions always
    # require their authenticated owner, including when DEMO_MODE is on.
    target = await repo.get_feedback_target(
        feedback.message_id, current_user.id, allow_shared=allow_shared
    )
    if target is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Message {feedback.message_id} not found",
        )
    # End discovery's read transaction before waiting for the gate: SQLite
    # snapshots and ORM identity maps must not authorize the subsequent write.
    await db.rollback()
    try:
        async with coordinator.hold(target.session_id):
            current = await repo.get_feedback_target(
                feedback.message_id,
                current_user.id,
                allow_shared=allow_shared,
                session_id=target.session_id,
            )
            if current is None:
                raise HTTPException(404, f"Message {feedback.message_id} not found")
            # A keyless shared request also stays bound to its first lookup.
            require_session_key(current.session_key, target.session_key)
            require_session_key(
                current.session_key,
                feedback.expected_session_key,
                required=current.is_private,
            )
            message = await repo.submit_feedback(
                message_id=feedback.message_id,
                rating=feedback.rating,
                text=feedback.text,
                user_id=current_user.id,
                session_id=current.session_id,
                session_key=current.session_key,
                allow_shared=allow_shared,
            )
            if message is None:
                raise HTTPException(404, f"Message {feedback.message_id} not found")
    except (ConversationChangedError, ConversationBusyError) as exc:
        raise HTTPException(409, "Conversation changed; reopen it before rating") from exc
    except CoordinationUnavailableError as exc:
        raise HTTPException(503, "Conversation coordination is unavailable") from exc

    # The committed receipt is immutable. Cache cleanup must not hold the
    # conversation gate or reread a replacement message after a reset.
    from app.services.chat.metrics import FEEDBACK_RATINGS

    FEEDBACK_RATINGS.labels(rating=str(feedback.rating)).inc()
    if feedback.rating < 0:
        from app.services.chat.cache_eviction import evict_downvoted_response

        evicted = await evict_downvoted_response(message.content)
        if evicted:
            logger.info("Downvote evicted %d cache entries for message %s", evicted, message.id)
    return FeedbackResponse(
        success=True,
        message_id=message.id,
        rating=message.user_rating,
        session_id=message.session_id,
        session_key=message.session_key,
    )


@router.get("/stats")
async def get_feedback_stats(
    db: AsyncSession = Depends(_get_db),
    current_user: User = Depends(get_current_active_user),  # noqa: ARG001  # FastAPI DI: enforces auth; value unused
) -> dict[str, Any]:
    """Get aggregate feedback statistics."""
    repo = FeedbackRepository(db)
    return await repo.get_feedback_stats()
