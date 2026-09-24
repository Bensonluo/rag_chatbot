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

import contextlib
import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from app.models.database.ticket import PRIORITY_HIGH, PRIORITY_NORMAL, HandoffTicket
from app.repositories.ticket_repository import (
    TICKET_STATUS_CLAIMED,
    TICKET_STATUS_OPEN,
    TICKET_STATUS_RESOLVED,
    TicketRepository,
)
from app.services.handoff.summarizer import SummaryLLM

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


async def _utcnow() -> datetime:
    """Current aware UTC time (asyncio-friendly seam for tests)."""
    return datetime.now(UTC)


def _as_utc(value: datetime) -> datetime:
    """Normalize a stored timestamp to aware UTC.

    Postgres returns aware datetimes; sqlite (unit tests) returns naive
    ones stored as UTC wall-clock — treat naive as UTC.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


class HandoffService:
    """Manages the human-agent handoff ticket queue."""

    def __init__(
        self,
        session_maker: Callable[[], Any],
        llm_service: SummaryLLM | None = None,
        summary_history_messages: int = 20,
        summary_timeout_seconds: float = 3.0,
    ) -> None:
        """
        Args:
            session_maker: Factory producing context-managed async sessions
                (e.g. ``app.api.database.async_session_maker``).
            llm_service: Optional LLM used for a best-effort conversation
                summary on the ticket. None (or any LLM failure) keeps
                the structured context only — escalation never blocks.
            summary_history_messages: Recent turns fed to the summary
            summary_timeout_seconds: Budget before the summary is dropped
        """
        self._session_maker = session_maker
        self._llm_service = llm_service
        self._summary_history_messages = summary_history_messages
        self._summary_timeout_seconds = summary_timeout_seconds

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
        # Best-effort conversation summary: runs before the ticket
        # transaction so the LLM call never holds a DB connection, and
        # any failure simply leaves the structured context as-is.
        if self._llm_service is not None:
            with contextlib.suppress(Exception):
                summary = await self._summarize_session(session_id, self._llm_service)
                if summary:
                    context = {**context, "conversation_summary": summary}
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

    async def _summarize_session(self, session_id: int, llm: SummaryLLM) -> str | None:
        """Load recent turns (own DB session) and summarize them."""
        from app.repositories.message_repository import MessageRepository
        from app.services.handoff.summarizer import summarize_for_handoff

        async with self._session_maker() as session:
            messages = await MessageRepository(session).get_recent_messages(
                session_id, limit=self._summary_history_messages
            )
        if not messages:
            return None
        # Repo rows are newest-first; the transcript must read oldest →
        # newest like a chat log.
        return await summarize_for_handoff(
            list(reversed(messages)), llm, timeout_seconds=self._summary_timeout_seconds
        )

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

    async def queue_stats(self) -> dict[str, int | float | None]:
        """Queue counts plus wait-time SLA dimensions for the agent dashboard.

        Counts come from COUNT queries (the old implementation
        materialized up to 10k rows per status to take their length —
        unworkable at scale). The SLA dimensions (GB/T 47746—2026
        转人工时效): the oldest open ticket's age is the worst wait a
        customer is experiencing, and ``open_sla_breaches`` counts open
        tickets waiting past ``HANDOFF_SLA_WAIT_SECONDS``. Gauges are
        refreshed here so Prometheus sees the same numbers the API
        returns.
        """
        from app.config.settings import get_settings
        from app.services.handoff.metrics import (
            HANDOFF_QUEUE_OLDEST_WAIT_SECONDS,
            HANDOFF_QUEUE_SLA_BREACHES,
        )

        sla_seconds = get_settings().HANDOFF_SLA_WAIT_SECONDS
        async with self._session_maker() as session:
            repo = TicketRepository(session)
            stats: dict[str, int | float | None] = {}
            for status_value in (TICKET_STATUS_OPEN, TICKET_STATUS_CLAIMED, TICKET_STATUS_RESOLVED):
                stats[status_value] = await repo.count_by_status(status_value)
            oldest = await repo.oldest_open_created_at()
            stats["oldest_open_wait_seconds"] = (
                None
                if oldest is None
                else max(0.0, (await _utcnow() - _as_utc(oldest)).total_seconds())
            )
            breaches = await repo.count_open_older_than(sla_seconds)
            stats["open_sla_breaches"] = breaches
            stats["sla_wait_seconds"] = sla_seconds

        HANDOFF_QUEUE_OLDEST_WAIT_SECONDS.set(stats["oldest_open_wait_seconds"] or 0.0)
        HANDOFF_QUEUE_SLA_BREACHES.set(float(breaches))
        return stats


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


def create_handoff_service(llm_service: SummaryLLM | None = None) -> HandoffService:
    """Build the default handoff service bound to the app's session maker."""
    from app.api.database import async_session_maker

    return HandoffService(session_maker=async_session_maker, llm_service=llm_service)
