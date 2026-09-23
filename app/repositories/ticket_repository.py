"""
Ticket repository for human handoff ticket data access.

Provides database operations specific to the HandoffTicket model:
queue listings for the agent workspace, claim/resolve transitions,
and open-queue depth for position estimates.
"""

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database.ticket import HandoffTicket
from app.repositories.base import BaseRepository

# Ticket lifecycle: open (waiting in queue) → claimed (an agent is on it)
# → resolved. Kept as plain strings on the model to stay migration-light;
# tighten to an enum if a fourth state ever appears.
TICKET_STATUS_OPEN = "open"
TICKET_STATUS_CLAIMED = "claimed"
TICKET_STATUS_RESOLVED = "resolved"


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
