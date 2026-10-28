"""User database model"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import Boolean, CheckConstraint, String, false
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.database.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.database.session import ChatSession


class User(Base, TimestampMixin):
    """
    User model representing application users.

    Attributes:
        id: Primary key
        email: User email (must be unique)
        hashed_password: Bcrypt hashed password
        full_name: User's full name (optional)
        is_active: Whether the user account is active
        is_admin: Whether the user has admin privileges
        sessions: Relationship to user's chat sessions
    """

    __tablename__ = "users"
    __table_args__ = (
        CheckConstraint(
            "(is_guest = true AND email IS NULL AND hashed_password IS NULL AND is_admin = false) "
            "OR (is_guest = false AND email IS NOT NULL AND hashed_password IS NOT NULL)",
            name="ck_users_identity_kind",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str | None] = mapped_column(
        String(255),
        unique=True,
        nullable=True,
        index=True,
    )
    hashed_password: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    full_name: Mapped[str | None] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_guest: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default=false(), nullable=False
    )

    # Relationships
    sessions: Mapped[list[ChatSession]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, email={self.email})>"
