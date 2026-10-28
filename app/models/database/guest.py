"""Opaque browser credentials for anonymous, independently owned visitors."""

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, validates

from app.models.database.base import Base, TimestampMixin


class GuestVisitor(Base, TimestampMixin):
    __tablename__ = "guest_visitors"
    __table_args__ = (
        UniqueConstraint("user_id", name="uq_guest_visitors_user_id"),
        UniqueConstraint("token_hash", name="uq_guest_visitors_token_hash"),
        CheckConstraint("length(token_hash) = 64", name="ck_guest_visitors_token_hash_length"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    current_session_id: Mapped[int | None] = mapped_column(
        ForeignKey("chat_sessions.id", ondelete="SET NULL"), nullable=True
    )

    @validates("id")
    def _canonical_id(self, _name: str, value: str) -> str:
        if type(value) is not str or str(UUID(value)) != value:
            raise ValueError("Guest identity must be a canonical UUID")
        return value
