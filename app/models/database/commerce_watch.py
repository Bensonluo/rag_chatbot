"""Proactive price watches: one pre-authorized "SKU at or below ¥Y, ask me" rule.

A watch rule is the visitor's explicit pre-authorization for the agent to
ASK when a price condition holds — never to buy. The row is the durable
record of that authorization: target price, lifecycle status and the
receipt of the moment the condition was met. Exactly one row per
``(visitor_id, sku_id)``: re-watching a SKU re-arms the same row.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import uuid4

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.models.database.base import Base, TimestampMixin

WatchStatus = Literal["active", "triggered", "asked", "cancelled"]


class PriceWatch(Base, TimestampMixin):
    """One price-watch rule owned by one guest visitor."""

    __tablename__ = "commerce_price_watches"
    __table_args__ = (
        UniqueConstraint("visitor_id", "sku_id", name="uq_commerce_price_watches_visitor_sku"),
        CheckConstraint(
            "status IN ('active', 'triggered', 'asked', 'cancelled')",
            name="ck_commerce_price_watches_status",
        ),
        CheckConstraint(
            "condition IN ('at_or_below')",
            name="ck_commerce_price_watches_condition",
        ),
        CheckConstraint("target_price_minor >= 1", name="ck_commerce_price_watches_target"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    visitor_id: Mapped[str] = mapped_column(
        ForeignKey("guest_visitors.id", ondelete="CASCADE"), nullable=False, index=True
    )
    sku_id: Mapped[str] = mapped_column(String(64), nullable=False)
    # Provider-name snapshot copied at creation / trigger time so the ask
    # card never depends on model prose for display strings.
    sku_name: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    target_price_minor: Mapped[int] = mapped_column(Integer, nullable=False)
    condition: Mapped[str] = mapped_column(String(16), nullable=False, default="at_or_below")
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    triggered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    triggered_price_minor: Mapped[int | None] = mapped_column(Integer, nullable=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    def __repr__(self) -> str:
        return f"<PriceWatch(visitor={self.visitor_id[:8]}, sku={self.sku_id}, {self.status})>"
