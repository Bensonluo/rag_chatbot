"""Durable tool execution facts survive conversation history deletion."""

from typing import Any

from sqlalchemy import JSON, CheckConstraint, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.database.base import Base, TimestampMixin


class ActionExecution(Base, TimestampMixin):
    __tablename__ = "action_executions"
    __table_args__ = (
        CheckConstraint(
            "status IN ('in_progress', 'completed', 'unknown')",
            name="ck_action_execution_status",
        ),
        CheckConstraint(
            "scope_session_id > 0 AND scope_session_id <= 2147483647",
            name="ck_action_execution_session_scope",
        ),
        CheckConstraint("user_id >= 0", name="ck_action_execution_user_scope"),
        CheckConstraint(
            "(status = 'completed' AND result IS NOT NULL) OR "
            "(status != 'completed' AND result IS NULL)",
            name="ck_action_execution_result",
        ),
    )

    action_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    scope_session_id: Mapped[int] = mapped_column(Integer, index=True)
    # The immutable scope remains after deletion; a NULL live reference
    # prevents replay if a local database later reuses the numeric ID.
    live_session_id: Mapped[int | None] = mapped_column(
        ForeignKey("chat_sessions.id", ondelete="SET NULL"), nullable=True
    )
    # Effective tool identity includes anonymous Demo IDs. This is a
    # snapshot, not a claim that the actor has an authenticated User row.
    user_id: Mapped[int] = mapped_column(Integer)
    scope: Mapped[str] = mapped_column(String(96))
    tool_name: Mapped[str] = mapped_column(String(255))
    arguments_hash: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16))
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON(none_as_null=True), nullable=True)
