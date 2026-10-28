"""Chat session database model"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from app.models.database.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.database.message import Message
    from app.models.database.user import User


def _new_session_key() -> str:
    return str(uuid4())


class ChatSession(Base, TimestampMixin):
    """
    Chat session model representing conversation sessions.

    Attributes:
        id: Primary key
        user_id: Foreign key to user
        title: Session title
        memory_type: Type of memory strategy to use
        context_window: Number of messages to keep in context
        metadata: Additional metadata as JSON
        user: Relationship to User
        messages: Relationship to Message objects
    """

    __tablename__ = "chat_sessions"
    __table_args__ = (UniqueConstraint("session_key", name="uq_chat_sessions_session_key"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    session_key: Mapped[str] = mapped_column(String(36), nullable=False, default=_new_session_key)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(
        String(255),
        default="New Chat",
        nullable=False,
    )
    memory_type: Mapped[str] = mapped_column(
        String(50),
        default="sliding_window",
        nullable=False,
    )
    context_window: Mapped[int] = mapped_column(
        Integer,
        default=10,
        nullable=False,
    )
    session_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    resolved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        default=None,
        nullable=True,
        comment="Last user-confirmed resolution (review #11); None = open/reopened",
    )
    reopened_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        server_default="0",
        nullable=False,
        comment="Times a resolved session was reopened by a follow-up message",
    )
    automation_epoch: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0", nullable=False
    )

    @validates("session_key")
    def _validate_session_key(self, _name: str, value: str) -> str:
        if type(value) is not str:
            raise ValueError("Session key must be a canonical UUID string")
        try:
            canonical = str(UUID(value))
        except ValueError as exc:
            raise ValueError("Session key must be a canonical UUID string") from exc
        if canonical != value:
            raise ValueError("Session key must be a canonical UUID string")
        return value

    @property
    def is_private(self) -> bool:
        """Server-owned access scope; unmarked legacy demo rows remain shared."""
        return bool(
            isinstance(self.session_metadata, dict)
            and self.session_metadata.get("access_mode") == "private"
        )

    # Relationships
    user: Mapped[User] = relationship(
        back_populates="sessions",
    )
    messages: Mapped[list[Message]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )

    def __repr__(self) -> str:
        return f"<ChatSession(id={self.id}, title={self.title}, user_id={self.user_id})>"
