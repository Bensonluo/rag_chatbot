"""
Handoff API endpoints for the human-agent workspace.

Agents monitor the open queue, claim a ticket, read its paginated
conversation through the assigned-ticket history route, and resolve it.

Staff endpoints require an active admin account. Customer message
endpoints require the active owner of a private session in both modes.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Response, status

from app.api.deps import get_current_active_user, require_admin
from app.api.deps.private_response import PrivateAPIRoute
from app.core.exceptions import ConflictError
from app.models.database.user import User
from app.models.schemas.handoff import (
    HandoffConversationPage,
    HandoffHistoryResponse,
    HandoffMessageAcknowledgement,
    HandoffMessageCreate,
    HandoffStaffMessageCreate,
)
from app.services.chat.coordination import ConversationBusyError, CoordinationUnavailableError
from app.services.handoff import HandoffService, create_handoff_service
from app.services.handoff.service import RequiresPrivateSessionError

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/handoff", tags=["handoff"], route_class=PrivateAPIRoute)

_handoff_service: HandoffService | None = None


def set_handoff_service(service: HandoffService | None) -> None:
    """Set (or clear, with None) the handoff service — startup / tests."""
    global _handoff_service
    _handoff_service = service


def get_handoff_service() -> HandoffService:
    """Return the handoff service, constructing the default on first use."""
    if _handoff_service is None:
        set_handoff_service(create_handoff_service())
    service = _handoff_service
    assert service is not None  # set_handoff_service just assigned it
    return service


# --- Endpoints ---


def _human_error(exc: Exception) -> HTTPException:
    headers = {"Cache-Control": "private, no-store"}
    if isinstance(exc, LookupError):
        return HTTPException(404, "Conversation unavailable", headers=headers)
    if isinstance(exc, RequiresPrivateSessionError):
        return HTTPException(409, "requires_private_session", headers=headers)
    if isinstance(exc, (ConflictError, ConversationBusyError)):
        return HTTPException(
            409, "Conversation command conflicts with its current state", headers=headers
        )
    if isinstance(exc, CoordinationUnavailableError):
        return HTTPException(503, "Conversation coordination is unavailable", headers=headers)
    if isinstance(exc, ValueError):
        return HTTPException(422, "Invalid conversation request", headers=headers)
    logger.error("Human conversation operation failed", exc_info=exc)
    return HTTPException(503, "Human conversation could not be confirmed", headers=headers)


@router.get("/sessions/{session_id}/messages", response_model=HandoffConversationPage)
async def get_customer_messages(
    response: Response,
    session_id: int = Path(..., gt=0, le=2_147_483_647),
    after_id: int = Query(0, ge=0, le=2_147_483_647),
    limit: int = Query(100, ge=1, le=100),
    customer: User = Depends(get_current_active_user),
    service: HandoffService = Depends(get_handoff_service),
) -> HandoffConversationPage:
    """The authenticated owner receives the private human channel in both modes."""
    response.headers["Cache-Control"] = "private, no-store"
    try:
        return await service.get_customer_messages(session_id, customer.id, after_id, limit)
    except Exception as exc:
        raise _human_error(exc) from exc


@router.post("/sessions/{session_id}/messages", response_model=HandoffMessageAcknowledgement)
async def send_customer_message(
    request: HandoffMessageCreate,
    response: Response,
    session_id: int = Path(..., gt=0, le=2_147_483_647),
    customer: User = Depends(get_current_active_user),
    service: HandoffService = Depends(get_handoff_service),
) -> HandoffMessageAcknowledgement:
    response.headers["Cache-Control"] = "private, no-store"
    try:
        return await service.send_customer_message(
            session_id, customer.id, request.client_message_id, request.content
        )
    except Exception as exc:
        raise _human_error(exc) from exc


@router.get("/tickets/{ticket_id}/messages", response_model=HandoffConversationPage)
async def get_staff_messages(
    response: Response,
    ticket_id: int = Path(..., gt=0, le=2_147_483_647),
    after_id: int = Query(0, ge=0, le=2_147_483_647),
    limit: int = Query(100, ge=1, le=100),
    admin: User = Depends(require_admin),
    service: HandoffService = Depends(get_handoff_service),
) -> HandoffConversationPage:
    response.headers["Cache-Control"] = "private, no-store"
    try:
        return await service.get_staff_messages(ticket_id, admin.id, after_id, limit)
    except Exception as exc:
        raise _human_error(exc) from exc


@router.post("/tickets/{ticket_id}/messages", response_model=HandoffMessageAcknowledgement)
async def send_staff_message(
    request: HandoffStaffMessageCreate,
    response: Response,
    ticket_id: int = Path(..., gt=0, le=2_147_483_647),
    admin: User = Depends(require_admin),
    service: HandoffService = Depends(get_handoff_service),
) -> HandoffMessageAcknowledgement:
    response.headers["Cache-Control"] = "private, no-store"
    try:
        return await service.send_staff_message(
            ticket_id, admin.id, request.client_message_id, request.content, resolve=request.resolve
        )
    except Exception as exc:
        raise _human_error(exc) from exc


@router.get("/tickets")
async def list_tickets(
    response: Response,
    ticket_status: str = Query("open", pattern="^(open|claimed|resolved)$", alias="status"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    _admin: User = Depends(require_admin),
    service: HandoffService = Depends(get_handoff_service),
) -> dict[str, Any]:
    """List handoff tickets in FIFO order (agent queue view)."""
    response.headers["Cache-Control"] = "private, no-store"
    tickets = await service.list_tickets(ticket_status, skip=skip, limit=limit)
    return {"tickets": tickets, "count": len(tickets)}


@router.get("/stats")
async def queue_stats(
    response: Response,
    _admin: User = Depends(require_admin),
    service: HandoffService = Depends(get_handoff_service),
) -> dict[str, int | float | None]:
    """Queue depth per status plus wait-time SLA dimensions (agent dashboard).

    ``oldest_open_wait_seconds`` is the worst live wait (None = empty
    queue); ``open_sla_breaches`` counts open tickets past
    ``HANDOFF_SLA_WAIT_SECONDS`` (GB/T 47746—2026 转人工时效).
    """
    response.headers["Cache-Control"] = "private, no-store"
    return await service.queue_stats()


@router.get("/tickets/{ticket_id}/history", response_model=HandoffHistoryResponse)
async def get_ticket_history(
    response: Response,
    ticket_id: int = Path(..., gt=0, le=2_147_483_647),
    skip: int = Query(0, ge=0, le=2_147_483_647),
    limit: int = Query(50, ge=1, le=100),
    admin: User = Depends(require_admin),
    service: HandoffService = Depends(get_handoff_service),
) -> HandoffHistoryResponse:
    """Read persisted customer/assistant messages for the caller's active claim.

    The ticket determines the session; caller-supplied session IDs cannot
    grant access. Missing, unclaimed, closed and other-agent tickets all
    return the same 404 in demo and strict postures. Ordinary chat-history
    ownership remains independent of this internal workspace route.
    """
    cache_headers = {"Cache-Control": "private, no-store"}
    try:
        history = await service.get_ticket_history(ticket_id, admin.id, skip=skip, limit=limit)
    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Ticket history unavailable",
            headers=cache_headers,
        ) from exc
    response.headers.update(cache_headers)
    return history


@router.post("/tickets/{ticket_id}/claim")
async def claim_ticket(
    response: Response,
    ticket_id: int,
    admin: User = Depends(require_admin),
    service: HandoffService = Depends(get_handoff_service),
) -> dict[str, Any]:
    """Claim an open ticket for the requesting agent."""
    response.headers["Cache-Control"] = "private, no-store"
    try:
        return await service.claim_ticket(ticket_id, admin.id)
    except (ConversationBusyError, CoordinationUnavailableError) as exc:
        raise _human_error(exc) from exc
    except LookupError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Ticket {ticket_id} is not claimable (missing or already claimed)",
            headers={"Cache-Control": "private, no-store"},
        ) from None


@router.post("/tickets/{ticket_id}/resolve")
async def resolve_ticket(
    response: Response,
    ticket_id: int,
    admin: User = Depends(require_admin),
    service: HandoffService = Depends(get_handoff_service),
) -> dict[str, Any]:
    """Resolve a ticket (claiming agent or any admin)."""
    response.headers["Cache-Control"] = "private, no-store"
    try:
        return await service.resolve_ticket(ticket_id, admin.id)
    except (ConversationBusyError, CoordinationUnavailableError) as exc:
        raise _human_error(exc) from exc
    except LookupError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Ticket {ticket_id} is not resolvable (missing or already resolved)",
            headers={"Cache-Control": "private, no-store"},
        ) from None
