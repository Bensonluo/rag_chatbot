"""
Session API endpoints.

Provides endpoints for managing chat sessions.
"""
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status, Query

from app.models.schemas.session import (
    SessionCreate,
    SessionUpdate,
    SessionResponse,
    SessionListResponse,
)
from app.api.deps import (
    get_session_repository,
    get_current_user,
)
from app.services.session_service import SessionService
from app.repositories.session_repository import SessionRepository
from app.models.database.user import User

router = APIRouter(prefix="/sessions", tags=["sessions"])


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
    current_user: Annotated[User, Depends(get_current_user)],
    session_service: Annotated[SessionService, Depends(get_session_service)],
) -> dict:
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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create session: {str(e)}",
        ) from e


@router.get(
    "",
    response_model=SessionListResponse,
    summary="List user's chat sessions",
)
async def list_sessions(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    current_user: Annotated[User, Depends(get_current_user)],
    session_service: Annotated[SessionService, Depends(get_session_service)],
) -> dict:
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
            detail=f"Failed to list sessions: {str(e)}",
        ) from e


@router.get(
    "/{session_id}",
    response_model=SessionResponse,
    summary="Get a specific chat session",
)
async def get_session(
    session_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    session_service: Annotated[SessionService, Depends(get_session_service)],
) -> dict:
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
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {str(e)}",
        ) from e


@router.put(
    "/{session_id}",
    response_model=SessionResponse,
    summary="Update a chat session",
)
async def update_session(
    session_id: int,
    session_data: SessionUpdate,
    current_user: Annotated[User, Depends(get_current_user)],
    session_service: Annotated[SessionService, Depends(get_session_service)],
) -> dict:
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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update session: {str(e)}",
        ) from e


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a chat session",
)
async def delete_session(
    session_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete session: {str(e)}",
        ) from e
