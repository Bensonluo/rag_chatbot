"""
Ticket repository for human handoff ticket data access.

Provides database operations specific to the HandoffTicket model:
queue listings for the agent workspace, claim/resolve transitions,
and open-queue depth for position estimates.
"""

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database.message import Message
from app.models.database.ticket import HandoffTicket
from app.models.enums.message import MessageRole
from app.repositories.base import BaseRepository

# Ticket lifecycle: open (waiting in queue) → claimed (an agent is on it)
# → resolved. Kept as plain strings on the model to stay migration-light;
# tighten to an enum if a fourth state ever appears.
TICKET_STATUS_OPEN = "open"
TICKET_STATUS_CLAIMED = "claimed"
TICKET_STATUS_RESOLVED = "resolved"


def _duration_seconds(start: datetime, end: datetime) -> float:
    """end - start in seconds, tolerating naive stored timestamps.

    Postgres returns aware datetimes; sqlite (unit tests) returns naive
    ones stored as UTC wall-clock — treat naive as UTC.
    """
    return (_as_utc(end) - _as_utc(start)).total_seconds()


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


class TicketRepository(BaseRepository[HandoffTicket]):
    """
    Repository for HandoffTicket entity operations.

    Extends BaseRepository with ticket-queue-specific queries.
    """

    def __init__(self, session: AsyncSession) -> None:
        """
        Initialize the ticket repository.

        Args:
            session: Async database session
        """
        super().__init__(session)

    async def get_by_id(
        self,
        id: int,
        model: type[HandoffTicket] = HandoffTicket,
    ) -> HandoffTicket | None:
        """Get a ticket by ID without requiring callers to repeat the model."""
        return await super().get_by_id(id, model)

    async def list_by_status(
        self,
        status: str,
        skip: int = 0,
        limit: int = 100,
    ) -> list[HandoffTicket]:
        """
        List tickets in a given status in queue-serving order.

        Priority tier first (lower value = served first), then FIFO by
        creation time within each tier — high-priority escalations jump
        the queue, everything else stays strictly oldest-first.

        Args:
            status: Ticket status filter (open / claimed / resolved)
            skip: Pagination offset
            limit: Maximum tickets to return

        Returns:
            List[HandoffTicket]: Tickets in serving order
        """
        stmt = (
            select(HandoffTicket)
            .where(HandoffTicket.status == status)
            .order_by(
                HandoffTicket.priority.asc(),
                HandoffTicket.created_at.asc(),
                HandoffTicket.id.asc(),
            )
            .offset(skip)
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def position_in_queue(self, ticket_id: int) -> int | None:
        """1-based serving position of an open ticket; None otherwise.

        Computed over the priority-ordered open queue, so a
        high-priority ticket created last can still be first.
        """
        stmt = (
            select(HandoffTicket.id)
            .where(HandoffTicket.status == TICKET_STATUS_OPEN)
            .order_by(
                HandoffTicket.priority.asc(),
                HandoffTicket.created_at.asc(),
                HandoffTicket.id.asc(),
            )
        )
        result = await self.session.execute(stmt)
        ids = [row[0] for row in result.fetchall()]
        try:
            return ids.index(ticket_id) + 1
        except ValueError:
            return None

    async def count_open(self) -> int:
        """Count tickets waiting in the queue (status=open)."""
        stmt = (
            select(func.count())
            .select_from(HandoffTicket)
            .where(HandoffTicket.status == TICKET_STATUS_OPEN)
        )
        result = await self.session.execute(stmt)
        return result.scalar() or 0

    async def count_by_status(self, status: str) -> int:
        """Count tickets in a status via COUNT — never materialized rows.

        Ops dashboards poll queue_stats at scale; listing up to 10k
        tickets per status to take their length is unworkable at
        800K-1M daily requests.
        """
        stmt = select(func.count()).select_from(HandoffTicket).where(HandoffTicket.status == status)
        result = await self.session.execute(stmt)
        return result.scalar() or 0

    async def oldest_open_created_at(self) -> datetime | None:
        """Creation time of the oldest open ticket; None when queue is empty.

        The queue-wait SLA clock (GB/T 47746—2026 转人工时效): the oldest
        ticket's age is the worst wait a customer is experiencing right
        now.
        """
        stmt = select(func.min(HandoffTicket.created_at)).where(
            HandoffTicket.status == TICKET_STATUS_OPEN
        )
        result = await self.session.execute(stmt)
        return result.scalar()

    async def count_open_older_than(self, seconds: float) -> int:
        """Count open tickets whose created_at predates the cutoff.

        The caller passes an aware-UTC datetime; the naive storage
        convention (sqlite tests) stores UTC wall-clock, so the cutoff is
        normalized to naive UTC for comparison.
        """
        cutoff = datetime.now(UTC) - timedelta(seconds=seconds)
        naive_cutoff = cutoff.replace(tzinfo=None)
        stmt = (
            select(func.count())
            .select_from(HandoffTicket)
            .where(
                HandoffTicket.status == TICKET_STATUS_OPEN,
                HandoffTicket.created_at < naive_cutoff,
            )
        )
        result = await self.session.execute(stmt)
        return result.scalar() or 0

    async def avg_pickup_seconds(self, limit: int = 100) -> float | None:
        """Average created→claimed duration over the most recent claims.

        None when no ticket has ever been claimed. Bounded to ``limit``
        samples — dashboards poll this on every stats pull, so the query
        must stay O(limit), not O(table).
        """
        stmt = (
            select(HandoffTicket.created_at, HandoffTicket.claimed_at)
            .where(HandoffTicket.claimed_at.is_not(None))
            .order_by(HandoffTicket.id.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        samples = [
            _duration_seconds(created, claimed)
            for created, claimed in result.fetchall()
            if created is not None and claimed is not None
        ]
        if not samples:
            return None
        return sum(samples) / len(samples)

    async def avg_handle_seconds(self, limit: int = 100) -> float | None:
        """Average claimed→resolved duration over recent resolved tickets.

        Resolve stamps ``updated_at`` (onupdate), so handle time is
        ``updated_at - claimed_at``. Tickets resolved without ever being
        claimed have no claimed_at and no meaningful handle time — they
        are excluded from the sample.
        """
        stmt = (
            select(HandoffTicket.updated_at, HandoffTicket.claimed_at)
            .where(
                HandoffTicket.status == TICKET_STATUS_RESOLVED,
                HandoffTicket.claimed_at.is_not(None),
            )
            .order_by(HandoffTicket.id.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        samples = [
            _duration_seconds(claimed, updated)
            for updated, claimed in result.fetchall()
            if updated is not None and claimed is not None
        ]
        if not samples:
            return None
        return sum(samples) / len(samples)

    async def get_one_shot_stats(self, since: datetime) -> dict[str, Any]:
        """Session-level one-shot-resolution stats for a window.

        The north star's highest-weighted metric, at its honest
        granularity: the denominator is sessions the bot actually
        served (≥1 assistant message) in the window, and the numerator
        is served sessions with NO handoff ticket (any status — a
        resolved ticket still means a human was pulled in) and NO
        downvoted assistant answer in-window (a thumbs-down is the
        operational proxy for "not a resolution" — the same doctrine
        the cache-eviction loop applies). Turn-level handoff share
        understates failure: five turns plus one handoff is 20% there,
        0% here. Tickets whose session was never served in-window count
        nowhere (queue noise, not a resolution outcome). A session
        carrying both failure signals is subtracted once: the failed
        set is the UNION of ticketed and downvoted sessions, while
        sessions_downvoted reports the signal on its own.
        """
        in_window = (
            Message.role == MessageRole.ASSISTANT,
            Message.created_at >= since.replace(tzinfo=None),
        )
        served_sessions = select(func.distinct(Message.session_id)).where(*in_window)
        ticketed = select(func.distinct(HandoffTicket.session_id).label("session_id")).where(
            HandoffTicket.session_id.in_(served_sessions)
        )
        downvoted = select(func.distinct(Message.session_id).label("session_id")).where(
            *in_window,
            Message.user_rating < 0,
        )
        failed = ticketed.union(downvoted).subquery()

        served = select(func.count()).select_from(served_sessions.subquery())
        escalated = select(func.count()).select_from(ticketed.subquery())
        rejected = select(func.count()).select_from(downvoted.subquery())
        failed_count = select(func.count()).select_from(failed)

        result = await self.session.execute(served)
        total = result.scalar() or 0
        result = await self.session.execute(escalated)
        handed_off = result.scalar() or 0
        result = await self.session.execute(rejected)
        rejected_n = result.scalar() or 0
        result = await self.session.execute(failed_count)
        failed_n = result.scalar() or 0
        return {
            "window_start": since.isoformat(),
            "sessions_served": total,
            "sessions_escalated": handed_off,
            "sessions_downvoted": rejected_n,
            "one_shot_rate": round((total - failed_n) / total, 4) if total else None,
        }

    async def get_open_ticket_for_session(
        self,
        session_id: int,
    ) -> HandoffTicket | None:
        """Return the session's waiting ticket, if any (dedupe re-requests)."""
        stmt = (
            select(HandoffTicket)
            .where(
                HandoffTicket.session_id == session_id,
                HandoffTicket.status != TICKET_STATUS_RESOLVED,
            )
            .order_by(HandoffTicket.id.desc())
            .limit(1)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def claim(
        self,
        ticket_id: int,
        agent_id: int,
    ) -> HandoffTicket | None:
        """
        Atomically claim a ticket for an agent.

        Returns None when the ticket does not exist or has already been
        claimed by someone else (optimistic concurrency: only an open
        ticket can transition to claimed).

        Args:
            ticket_id: Ticket to claim
            agent_id: Claiming agent's user id

        Returns:
            HandoffTicket | None: The claimed ticket, or None on conflict
        """
        stmt = select(HandoffTicket).where(
            HandoffTicket.id == ticket_id,
            HandoffTicket.status == TICKET_STATUS_OPEN,
        )
        result = await self.session.execute(stmt)
        ticket = result.scalar_one_or_none()
        if ticket is None:
            return None

        ticket.status = TICKET_STATUS_CLAIMED
        ticket.assigned_to = agent_id
        # AHT boundary: queue wait ends, agent handling begins (GB/T 47746
        # 时效). Stamped here — not defaulted — so the open→claimed
        # transition is the only writer.
        ticket.claimed_at = datetime.now(UTC)
        await self.session.commit()
        await self.session.refresh(ticket)
        return ticket

    async def resolve(
        self,
        ticket_id: int,
        agent_id: int,
    ) -> HandoffTicket | None:
        """
        Resolve a ticket.

        The claiming agent or any admin may resolve; only claimed or open
        tickets can transition. Returns None when not found or already
        resolved.

        Args:
            ticket_id: Ticket to resolve
            agent_id: Resolving agent's user id (recorded on assignment
                when the ticket was still open)

        Returns:
            HandoffTicket | None: The resolved ticket, or None on conflict
        """
        stmt = select(HandoffTicket).where(
            HandoffTicket.id == ticket_id,
            HandoffTicket.status != TICKET_STATUS_RESOLVED,
        )
        result = await self.session.execute(stmt)
        ticket = result.scalar_one_or_none()
        if ticket is None:
            return None

        ticket.status = TICKET_STATUS_RESOLVED
        if ticket.assigned_to is None:
            ticket.assigned_to = agent_id
        await self.session.commit()
        await self.session.refresh(ticket)
        return ticket
