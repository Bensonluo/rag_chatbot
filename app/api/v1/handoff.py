"""
Handoff API endpoints for the human-agent workspace.

Agents monitor the open queue, claim a ticket, read its paginated
conversation through the assigned-ticket history route, and resolve it.

Authorization: every endpoint requires an admin-role account — this
is the internal workspace, not a customer-facing surface. Anonymous
requests are rejected with 401 by ``require_admin``.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Response, status

from app.api.deps import require_admin
from app.models.database.user import User
from app.models.schemas.handoff import HandoffHistoryResponse
from app.services.handoff import HandoffService, create_handoff_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/handoff", tags=["handoff"])

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


@router.get("/tickets")
async def list_tickets(
    ticket_status: str = Query("open", pattern="^(open|claimed|resolved)$", alias="status"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    _admin: User = Depends(require_admin),
    service: HandoffService = Depends(get_handoff_service),
) -> dict[str, Any]:
    """List handoff tickets in FIFO order (agent queue view)."""
    tickets = await service.list_tickets(ticket_status, skip=skip, limit=limit)
    return {"tickets": tickets, "count": len(tickets)}


@router.get("/stats")
async def queue_stats(
    _admin: User = Depends(require_admin),
    service: HandoffService = Depends(get_handoff_service),
) -> dict[str, int | float | None]:
    """Queue depth per status plus wait-time SLA dimensions (agent dashboard).

    ``oldest_open_wait_seconds`` is the worst live wait (None = empty
    queue); ``open_sla_breaches`` counts open tickets past
    ``HANDOFF_SLA_WAIT_SECONDS`` (GB/T 47746—2026 转人工时效).
    """
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
    ticket_id: int,
    admin: User = Depends(require_admin),
    service: HandoffService = Depends(get_handoff_service),
) -> dict[str, Any]:
    """Claim an open ticket for the requesting agent."""
    try:
        return await service.claim_ticket(ticket_id, admin.id)
    except LookupError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Ticket {ticket_id} is not claimable (missing or already claimed)",
        ) from None


@router.post("/tickets/{ticket_id}/resolve")
async def resolve_ticket(
    ticket_id: int,
    admin: User = Depends(require_admin),
    service: HandoffService = Depends(get_handoff_service),
) -> dict[str, Any]:
    """Resolve a ticket (claiming agent or any admin)."""
    try:
        return await service.resolve_ticket(ticket_id, admin.id)
    except LookupError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Ticket {ticket_id} is not resolvable (missing or already resolved)",
        ) from None
