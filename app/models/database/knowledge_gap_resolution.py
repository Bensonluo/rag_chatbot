"""Knowledge-gap resolution ledger.

Closing the loop on gap telemetry: when a curator ships an FAQ for a
top gap, the query is marked resolved here. ``top_gaps`` hides groups
whose latest occurrence predates the resolution — a later occurrence
re-opens the group automatically, so a regression never stays silently
dismissed (the worklist semantics Zendesk / Intercom unanswered-question
reports use).
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.database.base import Base, TimestampMixin


class KnowledgeGapResolution(Base, TimestampMixin):
    """A curator action marking one normalized gap query as handled.

    Attributes:
        id: Primary key
        normalized_query: Aggregation key this resolution covers (unique)
        resolved_at: When it was resolved — a gap occurring after this
            re-opens the group
        resolved_by: User id of the curator (0 = system)
    """

    __tablename__ = "knowledge_gap_resolutions"

    id: Mapped[int] = mapped_column(primary_key=True)
    normalized_query: Mapped[str] = mapped_column(
        String(200), nullable=False, unique=True, index=True
    )
    resolved_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    resolved_by: Mapped[int] = mapped_column(nullable=False, default=0)

    def __repr__(self) -> str:
        return (
            f"<KnowledgeGapResolution(id={self.id}, "
            f"normalized_query={self.normalized_query!r}, resolved_by={self.resolved_by})>"
        )
