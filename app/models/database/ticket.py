"""Human handoff ticket database model."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.database.base import Base, TimestampMixin

# Queue priority tiers (lower value = served first). Emotion and
# high-value-refund escalations jump ahead of explicit requests: at
# 800K-1M daily requests an angry churn-risk customer must not queue
# behind idle chitchat escalations (industry-standard priority routing).
PRIORITY_HIGH = 1
PRIORITY_NORMAL = 2


class HandoffTicket(Base, TimestampMixin):
    """A request to escalate a chat session to a human agent.

    Created by the dialogue graph when the user explicitly asks for a
    human, when negative-emotion escalation fires, or when an
    irreversible tool exceeds its auto-processing threshold. The human
    agent workspace claims and resolves tickets through the handoff API.

    Attributes:
        id: Primary key
        session_id: Chat session the handoff originated from
        user_id: User who requested the handoff (0 for anonymous)
        reason: Why the handoff fired: explicit | emotion | refund_threshold
        summary: Structured context payload for the human agent
        status: open | claimed | resolved
        priority: Queue tier — PRIORITY_HIGH jumps the FIFO queue
        assigned_to: User id of the claiming agent (null while open)
        claimed_at: When an agent claimed the ticket (null while open) —
            the boundary between queue wait and agent handling; pickup
            (created→claimed) and handle (claimed→resolved) durations
            derive from it (GB/T 47746 时效 / AHT)
    """

    __tablename__ = "handoff_tickets"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[int] = mapped_column(nullable=False, default=0)
    reason: Mapped[str] = mapped_column(String(50), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="open", index=True)
    priority: Mapped[int] = mapped_column(nullable=False, default=PRIORITY_NORMAL)
    assigned_to: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)
    claimed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )

    def __repr__(self) -> str:
        return (
            f"<HandoffTicket(id={self.id}, session_id={self.session_id}, "
            f"reason={self.reason}, status={self.status})>"
        )
