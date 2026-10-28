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
import logging
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, nullcontext
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.core.exceptions import ConflictError
from app.models.database.handoff_message import HandoffMessage
from app.models.database.session import ChatSession
from app.models.database.ticket import PRIORITY_HIGH, PRIORITY_NORMAL, HandoffTicket
from app.models.enums.message import MessageRole
from app.models.schemas.citation import SourceCitation
from app.models.schemas.handoff import (
    HandoffConversationMessage,
    HandoffConversationPage,
    HandoffConversationTicket,
    HandoffHistoryMessage,
    HandoffHistoryResponse,
    HandoffMessageAcknowledgement,
    HandoffMessageCreate,
    HandoffStaffMessageCreate,
)
from app.repositories.ticket_repository import (
    TICKET_STATUS_CLAIMED,
    TICKET_STATUS_OPEN,
    TICKET_STATUS_RESOLVED,
    TicketRepository,
)
from app.services.chat.citations import normalize_citations
from app.services.chat.resolution import apply_resolution_transition
from app.services.handoff.summarizer import SummaryLLM

if TYPE_CHECKING:
    from app.services.chat.coordination import SessionCoordinator

logger = logging.getLogger(__name__)

# Why the handoff was created.
REASON_EXPLICIT = "explicit"  # user asked for a human
REASON_EMOTION = "emotion"  # negative-emotion escalation
REASON_REFUND_THRESHOLD = "refund_threshold"  # amount over auto-process bar
REASON_AGENT = "agent"  # agent loop judged the task unresolvable

VALID_REASONS = frozenset({REASON_EXPLICIT, REASON_EMOTION, REASON_REFUND_THRESHOLD, REASON_AGENT})


class RequiresPrivateSessionError(ConflictError):
    """Shared demo tickets cannot receive private human conversation content."""

    def __init__(self) -> None:
        super().__init__("requires_private_session")


# Reason → queue tier. Angry users and high-value refund disputes are
# churn/chargeback risks: they jump the FIFO queue ahead of explicit
# requests (industry-standard priority routing). Agent escalations sit
# at normal priority: the loop already exhausted the automated paths,
# but the user is not (yet) an escalation-risk signal.
_PRIORITY_BY_REASON = {
    REASON_EMOTION: PRIORITY_HIGH,
    REASON_REFUND_THRESHOLD: PRIORITY_HIGH,
    REASON_EXPLICIT: PRIORITY_NORMAL,
    REASON_AGENT: PRIORITY_NORMAL,
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
        coordinator: SessionCoordinator | None = None,
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
        self._coordinator = coordinator

    def _hold_session(self, session_id: int) -> AbstractAsyncContextManager[None]:
        return (
            self._coordinator.hold(session_id) if self._coordinator is not None else nullcontext()
        )

    async def _ticket_session(self, ticket_id: int) -> int:
        async with self._session_maker() as session:
            session_id = await session.scalar(
                select(HandoffTicket.session_id).where(HandoffTicket.id == ticket_id)
            )
        if not isinstance(session_id, int):
            raise LookupError("Ticket unavailable")
        return session_id

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

        Concurrent requests share one active ticket per session. Reuse
        preserves assignment and original context, retains the latest
        request, and raises priority for urgent escalation without downgrading
        it on a later routine retry. Closed tickets remain historical.

        Args:
            session_id: Chat session the handoff originated from
            user_id: Requesting user (None → 0 for anonymous)
            reason: One of explicit / emotion / refund_threshold
            context: Dialogue context payload for the human agent

        Returns:
            Dict with ticket_id, queue_position, and reused flag.
            queue_position is the 1-based position in the priority-ordered
            open queue, or None when unavailable. A queue lookup failure
            preserves the confirmed ticket_id. ticket_id is None when no
            ticket could be confirmed; callers must clearly report that
            the handoff failed instead of promising an agent response.
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
            # A first-turn escalation on a fresh demo session arrives
            # here before any persist_turn, so the parent chat_sessions
            # row may not exist yet — the insert would die on
            # handoff_tickets_session_id_fkey and fail-open would
            # swallow the loss (observed live 2026-10-03). Lazy import:
            # persistence pulls in ChatService, which reaches this
            # module again via the graph wiring.
            from app.services.chat.persistence import ensure_demo_session_row

            await ensure_demo_session_row(self._session_maker, session_id)
            async with self._session_maker() as session:
                repo = TicketRepository(session)
                ticket, reused = await repo.create_or_reuse_active(
                    session_id=session_id,
                    user_id=user_id or 0,
                    reason=reason,
                    priority=_priority_for_reason(reason),
                    context=context,
                )
                ticket_id = ticket.id
        except Exception as exc:  # noqa: BLE001 - availability over durability
            logger.warning(
                "Failed to create handoff ticket (session=%s): %s",
                session_id,
                exc,
            )
            return {"ticket_id": None, "queue_position": None, "reused": False}

        # Queue position is advisory. Its read transaction must not turn a
        # confirmed handoff into a reported failure after the ticket commits.
        queue_position = None
        try:
            async with self._session_maker() as session:
                queue_position = await TicketRepository(session).position_in_queue(ticket_id)
        except Exception as exc:  # noqa: BLE001 - retain the confirmed ticket
            logger.warning("Failed to read handoff queue position (ticket=%s): %s", ticket_id, exc)
        return {"ticket_id": ticket_id, "queue_position": queue_position, "reused": reused}

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
            privacy = {}
            if tickets:
                rows = await session.execute(
                    select(ChatSession.id, ChatSession.session_metadata).where(
                        ChatSession.id.in_({ticket.session_id for ticket in tickets})
                    )
                )
                privacy = {
                    identifier: isinstance(metadata, dict)
                    and metadata.get("access_mode") == "private"
                    for identifier, metadata in rows
                }
            return [
                {**_ticket_to_dict(ticket), "is_private": privacy.get(ticket.session_id, False)}
                for ticket in tickets
            ]

    async def claim_ticket(self, ticket_id: int, agent_id: int) -> dict[str, Any]:
        """
        Claim an open ticket for an agent.

        Raises:
            LookupError: Ticket not found or already claimed.
        """
        session_id = await self._ticket_session(ticket_id)
        async with self._hold_session(session_id), self._session_maker() as session:
            ticket = await TicketRepository(session).claim(ticket_id, agent_id)
            if ticket is None:
                raise LookupError(f"Ticket {ticket_id} is not claimable")
            return _ticket_to_dict(ticket)

    async def get_ticket_history(
        self, ticket_id: int, agent_id: int, skip: int = 0, limit: int = 50
    ) -> HandoffHistoryResponse:
        """Read persisted turns only while this agent handles the claimed ticket.

        Ticket state and assignment are checked anew for each request. The
        ticket's session ID is derived on the server and the ticket row stays
        locked through message serialization, so resolve cannot revoke access
        halfway through an authorized read.
        """
        if (
            not isinstance(skip, int)
            or isinstance(skip, bool)
            or not 0 <= skip <= 2_147_483_647
            or not isinstance(limit, int)
            or isinstance(limit, bool)
            or not 1 <= limit <= 100
        ):
            raise ValueError(
                "History pagination requires skip between 0 and 2147483647 and limit between 1 and 100"
            )
        if any(
            not isinstance(identifier, int)
            or isinstance(identifier, bool)
            or not 1 <= identifier <= 2_147_483_647
            for identifier in (ticket_id, agent_id)
        ):
            raise LookupError("Ticket history not found")
        async with self._session_maker() as session, session.begin():
            repo = TicketRepository(session)
            ticket = await repo.get_claimed_for_history(ticket_id, agent_id)
            if ticket is None:
                raise LookupError("Ticket history not found")
            total = await repo.count_history_messages(ticket.session_id)
            rows = await repo.get_history_messages(ticket.session_id, skip=skip, limit=limit)
            messages = []
            for row in rows:
                metadata = row.message_metadata
                metadata = metadata if isinstance(metadata, dict) else {}
                assistant = row.role == MessageRole.ASSISTANT
                citations = normalize_citations(metadata.get("citations")) if assistant else []
                messages.append(
                    HandoffHistoryMessage(
                        id=row.id,
                        role="assistant" if assistant else "user",
                        content=row.content,
                        timestamp=_as_utc(row.created_at).isoformat(),
                        sources=_history_sources(metadata.get("sources")) if assistant else None,
                        citations=[
                            SourceCitation.model_validate(citation) for citation in citations
                        ],
                    )
                )
            return HandoffHistoryResponse(
                ticket_id=ticket.id,
                session_id=ticket.session_id,
                skip=skip,
                limit=limit,
                total=total,
                messages=messages,
            )

    async def resolve_ticket(self, ticket_id: int, agent_id: int) -> dict[str, Any]:
        """
        Resolve a claimed or open ticket.

        Raises:
            LookupError: Ticket not found or already resolved.
        """
        session_id = await self._ticket_session(ticket_id)
        async with (
            self._hold_session(session_id),
            self._session_maker() as session,
            session.begin(),
        ):
            ticket = await session.scalar(
                select(HandoffTicket)
                .where(HandoffTicket.id == ticket_id)
                .with_for_update()
                .execution_options(populate_existing=True)
            )
            if ticket is None or ticket.status not in {TICKET_STATUS_OPEN, TICKET_STATUS_CLAIMED}:
                raise LookupError(f"Ticket {ticket_id} is not resolvable")
            await self._resolve_in_transaction(session, ticket, agent_id)
            await session.flush()
            await session.refresh(ticket)
            return _ticket_to_dict(ticket)

    @staticmethod
    async def _resolve_in_transaction(session: Any, ticket: HandoffTicket, agent_id: int) -> None:
        chat = await session.get(ChatSession, ticket.session_id, with_for_update=True)
        if chat is None:
            raise LookupError("Conversation unavailable")
        ticket.status = TICKET_STATUS_RESOLVED
        if ticket.assigned_to is None:
            ticket.assigned_to = agent_id
        chat.automation_epoch += 1
        # Ticket closure is not evidence that the customer's problem is solved.

    @staticmethod
    async def _owned_session(session: Any, session_id: int, user_id: int) -> ChatSession:
        chat = await session.scalar(
            select(ChatSession).where(ChatSession.id == session_id, ChatSession.user_id == user_id)
        )
        if not isinstance(chat, ChatSession) or not chat.is_private:
            raise LookupError("Conversation unavailable")
        return chat

    async def get_customer_messages(
        self, session_id: int, user_id: int, after_id: int = 0, limit: int = 100
    ) -> HandoffConversationPage:
        _validate_channel_page(session_id, user_id, after_id, limit)
        async with self._session_maker() as session:
            await self._owned_session(session, session_id, user_id)
            ticket = await session.scalar(
                select(HandoffTicket)
                .where(HandoffTicket.session_id == session_id)
                .order_by(HandoffTicket.id.desc())
                .limit(1)
            )
            return await _conversation_page(session, session_id, ticket, after_id, limit)

    async def get_staff_messages(
        self, ticket_id: int, agent_id: int, after_id: int = 0, limit: int = 100
    ) -> HandoffConversationPage:
        _validate_channel_page(ticket_id, agent_id, after_id, limit)
        async with self._session_maker() as session, session.begin():
            ticket = await TicketRepository(session).get_claimed_for_history(ticket_id, agent_id)
            if ticket is None:
                raise LookupError("Conversation unavailable")
            chat = await session.get(ChatSession, ticket.session_id)
            if chat is None or not chat.is_private:
                raise LookupError("Conversation unavailable")
            return await _conversation_page(session, ticket.session_id, ticket, after_id, limit)

    async def send_customer_message(
        self, session_id: int, user_id: int, client_message_id: str, content: str
    ) -> HandoffMessageAcknowledgement:
        request = HandoffMessageCreate(client_message_id=client_message_id, content=content)
        _validate_channel_page(session_id, user_id, 0, 1)
        # Do not reveal whether a foreign session currently holds a busy lock.
        async with self._session_maker() as session:
            await self._owned_session(session, session_id, user_id)
        async with self._hold_session(session_id):
            return await self._send_channel_message(
                session_id, user_id, "customer", request.client_message_id, request.content
            )

    async def send_staff_message(
        self,
        ticket_id: int,
        agent_id: int,
        client_message_id: str,
        content: str,
        resolve: bool = False,
    ) -> HandoffMessageAcknowledgement:
        request = HandoffStaffMessageCreate(
            client_message_id=client_message_id, content=content, resolve=resolve
        )
        _validate_channel_page(ticket_id, agent_id, 0, 1)
        async with self._session_maker() as session:
            ticket = await session.scalar(
                select(HandoffTicket).where(
                    HandoffTicket.id == ticket_id,
                    HandoffTicket.assigned_to == agent_id,
                    HandoffTicket.claimed_at.is_not(None),
                    HandoffTicket.status.in_((TICKET_STATUS_CLAIMED, TICKET_STATUS_RESOLVED)),
                )
            )
            if ticket is None:
                raise LookupError("Conversation unavailable")
            session_id = ticket.session_id
            chat = await session.get(ChatSession, session_id)
            if chat is None:
                raise LookupError("Conversation unavailable")
            if not chat.is_private:
                raise RequiresPrivateSessionError()
        async with self._hold_session(session_id):
            return await self._send_channel_message(
                session_id,
                agent_id,
                "staff",
                request.client_message_id,
                request.content,
                ticket_id=ticket_id,
                resolve=request.resolve,
            )

    async def _send_channel_message(
        self,
        session_id: int,
        author_id: int,
        kind: str,
        client_id: str,
        content: str,
        *,
        ticket_id: int | None = None,
        resolve: bool = False,
    ) -> HandoffMessageAcknowledgement:
        # The unique key remains the final arbitration point for callers
        # using this service without the production session coordinator.
        for attempt in range(2):
            try:
                async with self._session_maker() as session, session.begin():
                    if kind == "customer":
                        await self._owned_session(session, session_id, author_id)
                        ticket = await session.scalar(
                            select(HandoffTicket)
                            .where(HandoffTicket.session_id == session_id)
                            .order_by(HandoffTicket.id.desc())
                            .limit(1)
                            .with_for_update()
                            .execution_options(populate_existing=True)
                        )
                    else:
                        ticket = await session.scalar(
                            select(HandoffTicket)
                            .where(
                                HandoffTicket.id == ticket_id,
                                HandoffTicket.session_id == session_id,
                                HandoffTicket.assigned_to == author_id,
                                HandoffTicket.claimed_at.is_not(None),
                            )
                            .with_for_update()
                            .execution_options(populate_existing=True)
                        )
                        if ticket is None:
                            raise LookupError("Conversation unavailable")
                        chat = await session.get(ChatSession, session_id)
                        if chat is None:
                            raise LookupError("Conversation unavailable")
                        if not chat.is_private:
                            raise RequiresPrivateSessionError()
                    existing = await session.scalar(
                        select(HandoffMessage)
                        .where(
                            HandoffMessage.session_id == session_id,
                            HandoffMessage.client_message_id == client_id,
                        )
                        .order_by(HandoffMessage.id)
                        .limit(1)
                    )
                    if existing is not None:
                        if (
                            ticket is None
                            or existing.ticket_id != ticket.id
                            or existing.author_user_id != author_id
                            or existing.author_kind != kind
                            or existing.content != content
                            or existing.resolves_ticket != resolve
                        ):
                            raise ConflictError("Message identifier conflicts with a prior send")
                        return HandoffMessageAcknowledgement(
                            message=_conversation_message(existing),
                            ticket=_conversation_ticket(ticket),
                            reused=True,
                        )
                    allowed = (
                        {TICKET_STATUS_OPEN, TICKET_STATUS_CLAIMED}
                        if kind == "customer"
                        else {TICKET_STATUS_CLAIMED}
                    )
                    if ticket is None or ticket.status not in allowed:
                        if kind == "staff":
                            raise LookupError("Conversation unavailable")
                        raise ConflictError("There is no active human conversation")
                    if kind == "customer":
                        await apply_resolution_transition(session, session_id, content)
                    message = HandoffMessage(
                        session_id=session_id,
                        ticket_id=ticket.id,
                        author_user_id=author_id,
                        author_kind=kind,
                        client_message_id=client_id,
                        content=content,
                        resolves_ticket=resolve,
                    )
                    session.add(message)
                    if resolve:
                        await self._resolve_in_transaction(session, ticket, author_id)
                    await session.flush()
                    return HandoffMessageAcknowledgement(
                        message=_conversation_message(message),
                        ticket=_conversation_ticket(ticket),
                        reused=False,
                    )
            except IntegrityError:
                if attempt:
                    raise
        raise RuntimeError("Unreachable human message retry")

    async def get_recent_context(self, session_id: int, limit: int = 20) -> list[dict[str, str]]:
        if (
            type(session_id) is not int
            or not 1 <= session_id <= 2_147_483_647
            or type(limit) is not int
            or not 1 <= limit <= 100
        ):
            raise ValueError("Invalid human context scope")
        async with self._session_maker() as session:
            chat = await session.get(ChatSession, session_id)
            if chat is None or not chat.is_private:
                return []
            rows = (
                await session.scalars(
                    select(HandoffMessage)
                    .where(HandoffMessage.session_id == session_id)
                    .order_by(HandoffMessage.id.desc())
                    .limit(limit)
                )
            ).all()
            return [
                {"author_kind": row.author_kind, "content": row.content} for row in reversed(rows)
            ]

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
            # AHT dimensions (contact-center canonical KPI): recent-window
            # averages, None when no ticket has reached the stage yet.
            stats["avg_pickup_seconds"] = await repo.avg_pickup_seconds()
            stats["avg_handle_seconds"] = await repo.avg_handle_seconds()

        HANDOFF_QUEUE_OLDEST_WAIT_SECONDS.set(stats["oldest_open_wait_seconds"] or 0.0)
        HANDOFF_QUEUE_SLA_BREACHES.set(float(breaches))
        from app.services.handoff.metrics import (
            HANDOFF_HANDLE_AVG_SECONDS,
            HANDOFF_PICKUP_AVG_SECONDS,
        )

        HANDOFF_PICKUP_AVG_SECONDS.set(stats["avg_pickup_seconds"] or 0.0)
        HANDOFF_HANDLE_AVG_SECONDS.set(stats["avg_handle_seconds"] or 0.0)
        return stats


def _history_sources(value: object) -> list[str] | None:
    """Keep only bounded legacy source identifiers from persisted metadata."""
    if not isinstance(value, list):
        return None
    sources: list[str] = []
    for source in value:
        if (
            isinstance(source, str)
            and source.strip()
            and len(source) <= 512
            and source not in sources
        ):
            sources.append(source)
            if len(sources) == 50:
                break
    return sources or None


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
        "claimed_at": ticket.claimed_at.isoformat() if ticket.claimed_at else None,
        "updated_at": ticket.updated_at.isoformat() if ticket.updated_at else None,
    }


def create_handoff_service(llm_service: SummaryLLM | None = None) -> HandoffService:
    """Build the default handoff service bound to the app's session maker."""
    from app.api.database import async_session_maker
    from app.services.chat.coordination import get_session_coordinator

    return HandoffService(
        session_maker=async_session_maker,
        llm_service=llm_service,
        coordinator=get_session_coordinator(),
    )


def _validate_channel_page(scope: int, actor: int, after_id: int, limit: int) -> None:
    if any(type(value) is not int or not 1 <= value <= 2_147_483_647 for value in (scope, actor)):
        raise LookupError("Conversation unavailable")
    if (
        type(after_id) is not int
        or not 0 <= after_id <= 2_147_483_647
        or type(limit) is not int
        or not 1 <= limit <= 100
    ):
        raise ValueError("Invalid conversation pagination")


def _conversation_ticket(ticket: HandoffTicket) -> HandoffConversationTicket:
    return HandoffConversationTicket.model_validate(
        {"id": ticket.id, "status": ticket.status, "assigned_to": ticket.assigned_to}
    )


def _conversation_message(message: HandoffMessage) -> HandoffConversationMessage:
    return HandoffConversationMessage.model_validate(
        {
            "id": message.id,
            "session_id": message.session_id,
            "ticket_id": message.ticket_id,
            "author_kind": message.author_kind,
            "author_user_id": message.author_user_id,
            "client_message_id": message.client_message_id,
            "content": message.content,
            "timestamp": _as_utc(message.created_at).isoformat(),
        }
    )


async def _conversation_page(
    session: Any, session_id: int, ticket: HandoffTicket | None, after_id: int, limit: int
) -> HandoffConversationPage:
    rows = list(
        (
            await session.scalars(
                select(HandoffMessage)
                .where(HandoffMessage.session_id == session_id, HandoffMessage.id > after_id)
                .order_by(HandoffMessage.id)
                .limit(limit + 1)
            )
        ).all()
    )
    page = rows[:limit]
    return HandoffConversationPage(
        session_id=session_id,
        ticket=_conversation_ticket(ticket) if ticket is not None else None,
        messages=[_conversation_message(row) for row in page],
        next_after_id=page[-1].id if page else after_id,
        has_more=len(rows) > limit,
    )
