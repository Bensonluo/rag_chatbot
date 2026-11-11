"""Visitor preference memory: per-visitor commerce preferences with audit trail.

Scope redline (state here, enforce in the service): this table stores
*preferences* — display size, brand affinity, budget hints the visitor
expressed. It must NEVER store quota/limit/permission-style configuration
(monthly spend caps, tool permissions, policy overrides): those are
enforcement concerns that live in code and settings, not in a table an
agent can sweet-talk a write into. ``PreferenceMemoryService`` rejects such
keys at the boundary.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    ForeignKey,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.database.base import Base, TimestampMixin

# Postgres stores JSONB (queryable, task requirement); SQLite (tests, local)
# falls back to the generic JSON type.
JsonType = JSON(none_as_null=True).with_variant(JSONB(none_as_null=True), "postgresql")


class VisitorPreference(Base, TimestampMixin):
    """One durable preference for one guest visitor.

    Exactly one row per ``(visitor_id, pref_key)``: re-saving a key updates
    the row in place (upsert), and every write lands in
    ``visitor_preference_audit``. ``expires_at`` is optional per-key TTL;
    ``load_active`` filters expired rows without deleting them.
    """

    __tablename__ = "visitor_preferences"
    __table_args__ = (
        UniqueConstraint("visitor_id", "pref_key", name="uq_visitor_preferences_visitor_key"),
        CheckConstraint(
            "source IN ('inferred', 'confirmed')", name="ck_visitor_preferences_source"
        ),
        CheckConstraint("length(pref_key) >= 1", name="ck_visitor_preferences_key_nonempty"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    visitor_id: Mapped[str] = mapped_column(
        ForeignKey("guest_visitors.id", ondelete="CASCADE"), nullable=False, index=True
    )
    pref_key: Mapped[str] = mapped_column(String(64), nullable=False)
    pref_value: Mapped[dict[str, Any]] = mapped_column(JsonType, nullable=False)
    source: Mapped[str] = mapped_column(String(16), nullable=False, default="inferred")
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    def __repr__(self) -> str:
        return f"<VisitorPreference(visitor={self.visitor_id[:8]}, key={self.pref_key!r})>"


class VisitorPreferenceAudit(Base, TimestampMixin):
    """Append-only audit log of every preference write.

    Telemetry-style on purpose: no FK to ``guest_visitors`` so the audit
    trail survives visitor-row purges (same convention as ``user_facts``),
    while the live table cascades away with its owner.
    """

    __tablename__ = "visitor_preference_audit"
    __table_args__ = (
        CheckConstraint("action IN ('save', 'forget')", name="ck_visitor_preference_audit_action"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    visitor_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    pref_key: Mapped[str] = mapped_column(String(64), nullable=False)
    action: Mapped[str] = mapped_column(String(16), nullable=False)
    old_value: Mapped[dict[str, Any] | None] = mapped_column(JsonType, nullable=True)
    new_value: Mapped[dict[str, Any] | None] = mapped_column(JsonType, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<VisitorPreferenceAudit(id={self.id}, key={self.pref_key!r}, action={self.action})>"
        )
