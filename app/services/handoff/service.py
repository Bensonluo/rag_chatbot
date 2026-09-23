"""
Human handoff service.

Creates and manages escalation tickets when a user asks for a human
agent (explicit), when negative-emotion escalation fires, or when an
irreversible action exceeds its auto-processing threshold.

The chat-path method (``create_ticket_for_session``) never raises —
same availability-over-durability contract as the chat persister: a
database hiccup must not fail an in-flight chat turn. Agent-workspace
methods (list / claim / resolve) do raise ``LookupError`` so the API
layer can map them to 404/409 semantics.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from app.models.database.ticket import PRIORITY_HIGH, PRIORITY_NORMAL, HandoffTicket
from app.repositories.ticket_repository import (
    TICKET_STATUS_CLAIMED,
    TICKET_STATUS_OPEN,
    TICKET_STATUS_RESOLVED,
    TicketRepository,
)

logger = logging.getLogger(__name__)

# Why the handoff was created.
REASON_EXPLICIT = "explicit"  # user asked for a human
REASON_EMOTION = "emotion"  # negative-emotion escalation
REASON_REFUND_THRESHOLD = "refund_threshold"  # amount over auto-process bar

VALID_REASONS = frozenset({REASON_EXPLICIT, REASON_EMOTION, REASON_REFUND_THRESHOLD})

# Reason → queue tier. Angry users and high-value refund disputes are
# churn/chargeback risks: they jump the FIFO queue ahead of explicit
# requests (industry-standard priority routing).
_PRIORITY_BY_REASON = {
    REASON_EMOTION: PRIORITY_HIGH,
    REASON_REFUND_THRESHOLD: PRIORITY_HIGH,
    REASON_EXPLICIT: PRIORITY_NORMAL,
}


def _priority_for_reason(reason: str) -> int:
    return _PRIORITY_BY_REASON.get(reason, PRIORITY_NORMAL)


class HandoffService:
    """Manages the human-agent handoff ticket queue."""

    def __init__(self, session_maker: Callable[[], Any]) -> None:
        """
        Args:
            session_maker: Factory producing context-managed async sessions
                (e.g. ``app.api.database.async_session_maker``).
        """
        self._session_maker = session_maker

    # ── Chat path (never raises) ─────────────────────────────────────────────

    async def create_ticket_for_session(
        self,
        session_id: int,
        user_id: int | None,
        reason: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Create (or reuse) a handoff ticket for a chat session.

        Re-requesting a human while a ticket is already open for the
        session returns the existing ticket — users say 转人工 more than
        once while waiting, and duplicate tickets would inflate the
        agent queue.

        Args:
            session_id: Chat session the handoff originated from
            user_id: Requesting user (None → 0 for anonymous)
            reason: One of explicit / emotion / refund_threshold
            context: Dialogue context payload for the human agent

        Returns:
            Dict with ticket_id, queue_position, and reused flag.
            queue_position is the 1-based position in the priority-ordered
            open queue. ticket_id is None when persistence failed (the
            chat turn still succeeds; the response just omits the number).
        """
        if reason not in VALID_REASONS:
            reason = REASON_EXPLICIT
        try:
            async with self._session_maker() as session:
                repo = TicketRepository(session)
                existing = await repo.get_open_ticket_for_session(session_id)
                if existing is not None:
                    return {
                        "ticket_id": existing.id,
                        "queue_position": await repo.position_in_queue(existing.id),
                        "reused": True,
                    }

                ticket = HandoffTicket(
                    session_id=session_id,
                    user_id=user_id or 0,
                    reason=reason,
                    priority=_priority_for_reason(reason),
                    summary=json.dumps(context, ensure_ascii=False, default=str),
                )
                created = await repo.create(ticket)
                return {
                    "ticket_id": created.id,
                    "queue_position": await repo.position_in_queue(created.id),
                    "reused": False,
                }
        except Exception as exc:  # noqa: BLE001 - availability over durability
            logger.warning(
                "Failed to create handoff ticket (session=%s): %s",
                session_id,
                exc,
            )
            return {"ticket_id": None, "queue_position": None, "reused": False}

    # ── Agent workspace (raises on conflict) ─────────────────────────────────

    async def list_tickets(
        self,
        status: str = TICKET_STATUS_OPEN,
        skip: int = 0,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List tickets by status in serving order (agent queue view)."""
        async with self._session_maker() as session:
            repo = TicketRepository(session)
            tickets = await repo.list_by_status(status, skip=skip, limit=limit)
            return [_ticket_to_dict(t) for t in tickets]

    async def claim_ticket(self, ticket_id: int, agent_id: int) -> dict[str, Any]:
        """
        Claim an open ticket for an agent.

        Raises:
            LookupError: Ticket not found or already claimed.
        """
        async with self._session_maker() as session:
            repo = TicketRepository(session)
            ticket = await repo.claim(ticket_id, agent_id)
            if ticket is None:
                raise LookupError(f"Ticket {ticket_id} is not claimable")
            return _ticket_to_dict(ticket)

    async def resolve_ticket(self, ticket_id: int, agent_id: int) -> dict[str, Any]:
        """
        Resolve a claimed or open ticket.

        Raises:
            LookupError: Ticket not found or already resolved.
        """
        async with self._session_maker() as session:
            repo = TicketRepository(session)
            ticket = await repo.resolve(ticket_id, agent_id)
            if ticket is None:
                raise LookupError(f"Ticket {ticket_id} is not resolvable")
            return _ticket_to_dict(ticket)

    async def queue_stats(self) -> dict[str, int]:
        """Counts per status for the agent dashboard."""
        async with self._session_maker() as session:
            repo = TicketRepository(session)
            counts = {}
            for status_value in (TICKET_STATUS_OPEN, TICKET_STATUS_CLAIMED, TICKET_STATUS_RESOLVED):
                counts[status_value] = len(await repo.list_by_status(status_value, limit=10000))
            return counts


def _ticket_to_dict(ticket: HandoffTicket) -> dict[str, Any]:
    """Serialize a ticket row for API responses."""
    return {
        "id": ticket.id,
        "session_id": ticket.session_id,
        "user_id": ticket.user_id,
        "reason": ticket.reason,
        "priority": ticket.priority,
        "summary": ticket.summary,
        "status": ticket.status,
        "assigned_to": ticket.assigned_to,
        "created_at": ticket.created_at.isoformat() if ticket.created_at else None,
        "updated_at": ticket.updated_at.isoformat() if ticket.updated_at else None,
    }


def create_handoff_service() -> HandoffService:
    """Build the default handoff service bound to the app's session maker."""
    from app.api.database import async_session_maker

    return HandoffService(session_maker=async_session_maker)
