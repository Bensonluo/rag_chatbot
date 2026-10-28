"""
Session API endpoints.

Provides endpoints for managing chat sessions.
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    get_current_active_user,
    get_db,
    get_session_repository,
)
from app.api.deps.authorization import require_qa_reviewer
from app.api.deps.private_response import PrivateAPIRoute
from app.core.exceptions import NotFoundError
from app.models.database.session import ChatSession
from app.models.database.user import User
from app.models.schemas.session import (
    SessionCreate,
    SessionListResponse,
    SessionResponse,
    SessionReviewVerdict,
    SessionUpdate,
)
from app.models.schemas.session_key import SessionKey
from app.repositories.session_repository import ReviewChangedError, SessionRepository
from app.repositories.ticket_repository import TicketRepository
from app.services.chat.chat_service import ChatService
from app.services.chat.coordination import (
    ConversationBusyError,
    CoordinationUnavailableError,
    HumanHandlingError,
    SessionCoordinator,
    get_session_coordinator,
)
from app.services.chat.session_identity import ConversationChangedError, require_session_key
from app.services.session_service import SessionService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sessions", tags=["sessions"], route_class=PrivateAPIRoute)


def get_session_chat_service() -> ChatService:
    # Lazy import avoids the chat router's session-service dependency cycle.
    from app.api.v1.chat import get_chat_service

    return get_chat_service()


@router.get("/containment-stats")
async def get_containment_stats(
    window_days: Annotated[int, Query(ge=1, le=90)] = 7,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),  # noqa: ARG001  # FastAPI DI: enforces auth; value unused
) -> dict[str, Any]:
    """Session-level containment rate (north-star KPI).

    Denominator: sessions the bot served (≥1 assistant message) in
    the window. Numerator: served sessions with no handoff ticket and
    no downvoted assistant reply in the window —
    any ticket status counts as "a human was pulled in", so the rate
    excludes both escalation and explicit downvote at session
    granularity. Containment is an operational proxy, not a
    verified problem-resolution rate (review 2026-09-26, #11).
    Additive counts expose sessions with FAILED/COMPLETED assistant
    replies and those whose latest in-window reply is FAILED. The two
    populations can overlap and neither implies confirmed resolution.
    Registered before the ``/{session_id}`` route:
    a path-param route would otherwise swallow this literal path.
    """
    since = datetime.now(UTC) - timedelta(days=window_days)
    repo = TicketRepository(db)
    return await repo.get_containment_stats(since=since)


@router.get("/review-queue")
async def get_review_queue(
    window_days: Annotated[int, Query(ge=1, le=90)] = 7,
    limit: Annotated[int, Query(ge=1, le=20)] = 10,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    coordinator: SessionCoordinator = Depends(get_session_coordinator),
) -> dict[str, Any]:
    """QA review sample: contained sessions with no resolution evidence.

    Containment counts "no ticket, no downvote" as success, but a user
    who gave up silently also has neither (review 2026-09-26, #11). The
    queue samples that inflation-risk population — served in window,
    contained, resolved_at IS NULL, not yet reviewed — with a preview
    of the last user message so a reviewer can triage before opening
    the transcript. Registered before ``/{session_id}`` like
    containment-stats: the path-param route would swallow this literal.
    """
    require_qa_reviewer(current_user)
    since = datetime.now(UTC) - timedelta(days=window_days)
    repo = SessionRepository(db)
    sampled = await repo.sample_review_queue(since=since, limit=limit)
    identifiers = [chat.id for chat in sampled]
    await db.rollback()  # End the discovery transaction before taking conversation gates.
    items = []
    for session_id in identifiers:
        try:
            async with coordinator.hold(session_id):
                snapshot = await repo.get_review_snapshot(session_id, since=since)
                if snapshot is not None and snapshot.eligible:
                    preview = await repo.last_user_message(session_id)
                    items.append(snapshot.response(preview))
        except ConversationBusyError:
            continue  # Sampling is best effort; never offer an inconsistent proposal.
        except CoordinationUnavailableError as exc:
            raise HTTPException(503, "Review coordination is unavailable") from exc
        finally:
            await db.rollback()
    return {"window_start": since.isoformat(), "items": items}


@router.get("/{session_id}/review")
async def get_review_context(
    session_id: Annotated[int, Path(ge=1, le=2_147_483_647)],
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    coordinator: SessionCoordinator = Depends(get_session_coordinator),
) -> dict[str, Any]:
    """An active reviewer explicitly refreshes a proposal, including corrections."""
    require_qa_reviewer(current_user)
    await db.rollback()
    repo = SessionRepository(db)
    try:
        async with coordinator.hold(session_id):
            snapshot = await repo.get_review_snapshot(session_id)
            if snapshot is None:
                raise HTTPException(404, "Session not found")
            preview = await repo.last_user_message(session_id)
            return snapshot.response(preview)
    except ConversationBusyError as exc:
        raise HTTPException(409, "A conversation action is still in progress") from exc
    except CoordinationUnavailableError as exc:
        raise HTTPException(503, "Review coordination is unavailable") from exc


@router.post("/{session_id}/review")
async def review_session(
    session_id: Annotated[int, Path(ge=1, le=2_147_483_647)],
    verdict: SessionReviewVerdict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    coordinator: SessionCoordinator = Depends(get_session_coordinator),
) -> dict[str, Any]:
    """Record a QA verdict on a sampled session.

    Any verdict removes the session from the review queue; a resolved
    verdict stamps resolved_at, while an unresolved verdict clears it.
    The latest verdict determines the session's contribution to
    sessions_resolved without counting a review correction as a user reopen.
    """
    require_qa_reviewer(current_user)
    repo = SessionRepository(db)
    await db.rollback()
    try:
        async with coordinator.hold(session_id):
            row = await repo.record_review_verdict(
                session_id,
                resolved=verdict.resolved,
                note=verdict.note,
                expected_session_key=verdict.expected_session_key,
                expected_review_version=verdict.expected_review_version,
            )
            if row is None:
                raise HTTPException(404, "Session not found")
            snapshot = await repo.get_review_snapshot(session_id)
            if snapshot is None:
                raise ReviewChangedError("Review conversation changed")
            return {
                "session_id": row.id,
                "session_key": row.session_key,
                "review_version": snapshot.version,
                "last_message_id": snapshot.last_message_id,
                "last_human_message_id": snapshot.last_human_message_id,
                "resolved": verdict.resolved,
                "resolved_at": row.resolved_at.isoformat() if row.resolved_at else None,
            }
    except ReviewChangedError as exc:
        raise HTTPException(409, "Review evidence changed; refresh the review") from exc
    except ConversationBusyError as exc:
        raise HTTPException(409, "A conversation action is still in progress") from exc
    except CoordinationUnavailableError as exc:
        raise HTTPException(503, "Review coordination is unavailable") from exc
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to record review verdict: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record verdict",
        ) from e


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
            expected_session_key=session_data.expected_session_key,
            require_key=True,
        )
        return session
    except ConversationChangedError as e:
        raise HTTPException(409, "Conversation changed; reopen it before continuing") from e
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
    coordinator: Annotated[SessionCoordinator, Depends(get_session_coordinator)],
    chat_service: Annotated[ChatService, Depends(get_session_chat_service)],
    expected_session_key: SessionKey | None = Query(None),
) -> None:
    """
    Delete a chat session.

    Only deletes sessions that belong to the authenticated user.
    All messages associated with the session will also be deleted.
    """
    try:
        async with coordinator.hold(session_id):
            current = await session_service.get_session_by_id(session_id, current_user.id)
            state = await coordinator.state(session_id)
            require_session_key(
                state.session_key, expected_session_key, required=current.is_private
            )
            if state.claimed_ticket_id is not None:
                raise HumanHandlingError("Human support is handling this conversation")
            await chat_service.discard_session_state(session_id, state.epoch)
            await session_service.delete_session(session_id, current_user.id)
    except ConversationChangedError as e:
        raise HTTPException(409, "Conversation changed; reopen it before continuing") from e
    except ConversationBusyError as e:
        raise HTTPException(409, "A conversation action is still in progress") from e
    except HumanHandlingError as e:
        raise HTTPException(423, "Human support is handling this conversation") from e
    except CoordinationUnavailableError as e:
        raise HTTPException(503, "Conversation coordination is unavailable") from e
    except NotFoundError as e:
        raise HTTPException(404, "Session not found") from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete session: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete session",
        ) from e
