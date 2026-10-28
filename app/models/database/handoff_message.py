"""Human-channel messages, separate from automated reply and quality records."""

from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    false,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.models.database.base import Base


class HandoffMessage(Base):
    __tablename__ = "handoff_messages"
    __table_args__ = (
        UniqueConstraint(
            "session_id", "author_user_id", "client_message_id", name="uq_handoff_message_client"
        ),
        CheckConstraint(
            "author_kind IN ('customer', 'staff')", name="ck_handoff_message_author_kind"
        ),
        Index("ix_handoff_messages_session_id_id", "session_id", "id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions.id", ondelete="CASCADE"))
    ticket_id: Mapped[int] = mapped_column(ForeignKey("handoff_tickets.id", ondelete="CASCADE"))
    author_user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    author_kind: Mapped[str] = mapped_column(String(16))
    client_message_id: Mapped[str] = mapped_column(String(36))
    content: Mapped[str] = mapped_column(Text)
    resolves_ticket: Mapped[bool] = mapped_column(Boolean, default=False, server_default=false())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
