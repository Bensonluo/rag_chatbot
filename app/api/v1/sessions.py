"""
Session API endpoints.

Provides endpoints for managing chat sessions.
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    get_current_active_user,
    get_db,
    get_session_repository,
)
from app.models.database.session import ChatSession
from app.models.database.user import User
from app.models.schemas.session import (
    SessionCreate,
    SessionListResponse,
    SessionResponse,
    SessionUpdate,
)
from app.repositories.session_repository import SessionRepository
from app.repositories.ticket_repository import TicketRepository
from app.services.session_service import SessionService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("/one-shot-stats")
async def get_one_shot_stats(
    window_days: Annotated[int, Query(ge=1, le=90)] = 7,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),  # noqa: ARG001  # FastAPI DI: enforces auth; value unused
) -> dict[str, Any]:
    """Session-level one-shot-resolution rate (north-star KPI).

    Denominator: sessions the bot served (≥1 assistant message) in
    the window. Numerator: served sessions with no handoff ticket —
    any ticket status counts as "a human was pulled in", so the rate
    is the honest complement of the handoff share at session
    granularity. Registered before the ``/{session_id}`` route:
    a path-param route would otherwise swallow this literal path.
    """
    since = datetime.now(UTC) - timedelta(days=window_days)
    repo = TicketRepository(db)
    return await repo.get_one_shot_stats(since=since)


async def get_session_service(
    session_repo: SessionRepository = Depends(get_session_repository),
) -> SessionService:
    """
    Dependency to get session service.

    Args:
        session_repo: Session repository

    Returns:
        SessionService: Session service instance
    """
    return SessionService(session_repo)


@router.post(
    "",
    response_model=SessionResponse,
    status_code=status.HTTP_200_OK,
    summary="Create a new chat session",
)
async def create_session(
    session_data: SessionCreate,
    current_user: Annotated[User, Depends(get_current_active_user)],
    session_service: Annotated[SessionService, Depends(get_session_service)],
) -> ChatSession:
    """
    Create a new chat session for the authenticated user.

    - **title**: Optional session title
    - **memory_type**: Memory strategy (sliding_window, summarization, or hybrid)
    - **context_window**: Number of messages to keep in context (1-100)

    Returns the created session.
    """
    try:
        session = await session_service.create_session(
            user_id=current_user.id,
            title=session_data.title,
            memory_type=session_data.memory_type,
            context_window=session_data.context_window,
        )
        return session
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create session: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create session",
        ) from e


@router.get(
    "",
    response_model=SessionListResponse,
    summary="List user's chat sessions",
)
async def list_sessions(
    current_user: Annotated[User, Depends(get_current_active_user)],
    session_service: Annotated[SessionService, Depends(get_session_service)],
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
) -> dict[str, Any]:
    """
    List all chat sessions for the authenticated user with pagination.

    Returns a paginated list of sessions ordered by most recently updated.
    """
    try:
        skip = (page - 1) * page_size
        sessions = await session_service.get_user_sessions(
            user_id=current_user.id,
            skip=skip,
            limit=page_size,
        )

        # Get total count
        total = await session_service.count_user_sessions(current_user.id)

        return {
            "items": sessions,
            "total": total,
            "page": page,
            "page_size": page_size,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list sessions",
        ) from e


@router.get(
    "/{session_id}",
    response_model=SessionResponse,
    summary="Get a specific chat session",
)
async def get_session(
    session_id: int,
    current_user: Annotated[User, Depends(get_current_active_user)],
    session_service: Annotated[SessionService, Depends(get_session_service)],
) -> ChatSession:
    """
    Get a specific chat session by ID.

    Only returns sessions that belong to the authenticated user.
    """
    try:
        session = await session_service.get_session_by_id(session_id, current_user.id)
        return session
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Session lookup failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        ) from e


@router.put(
    "/{session_id}",
    response_model=SessionResponse,
    summary="Update a chat session",
)
async def update_session(
    session_id: int,
    session_data: SessionUpdate,
    current_user: Annotated[User, Depends(get_current_active_user)],
    session_service: Annotated[SessionService, Depends(get_session_service)],
) -> ChatSession:
    """
    Update a chat session.

    Only updates sessions that belong to the authenticated user.
    All fields are optional - only provided fields will be updated.
    """
    try:
        session = await session_service.update_session(
            session_id=session_id,
            user_id=current_user.id,
            title=session_data.title,
            memory_type=session_data.memory_type,
            context_window=session_data.context_window,
        )
        return session
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update session: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update session",
        ) from e


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a chat session",
)
async def delete_session(
    session_id: int,
    current_user: Annotated[User, Depends(get_current_active_user)],
    session_service: Annotated[SessionService, Depends(get_session_service)],
) -> None:
    """
    Delete a chat session.

    Only deletes sessions that belong to the authenticated user.
    All messages associated with the session will also be deleted.
    """
    try:
        await session_service.delete_session(session_id, current_user.id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete session: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete session",
        ) from e
