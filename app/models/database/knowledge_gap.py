"""Knowledge-gap telemetry database model.

Industry baseline (Zendesk knowledge-gap reporting, 阿里小蜜 知识缺口挖掘):
a customer-service bot must surface the queries it could not answer —
that list is the input for KB curation. Every mainstream platform
treats "no answer found" as a first-class operational signal, not a
log line to grep.
"""

from __future__ import annotations

from sqlalchemy import Float, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.database.base import Base, TimestampMixin


class KnowledgeGapRecord(Base, TimestampMixin):
    """One knowledge-intent turn whose retrieval produced no usable context.

    Written best-effort from the chat pipeline; telemetry only — never
    a dependency of the response path.

    Attributes:
        id: Primary key
        session_id: Chat session the gap occurred in (string, no FK:
            gaps are telemetry and must not fail on session cleanup)
        user_id: User id (0 for anonymous)
        query: The user's original message
        normalized_query: Lowercased, whitespace-collapsed query used for
            aggregation (indexed — the analytics query groups on it)
        intent: Detected intent
        top_score: Best retrieval score when results existed but scored
            below the usability threshold; None when nothing was returned
    """

    __tablename__ = "knowledge_gaps"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    user_id: Mapped[int] = mapped_column(nullable=False, default=0)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_query: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    intent: Mapped[str] = mapped_column(String(50), nullable=False, default="unknown")
    top_score: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)

    def __repr__(self) -> str:
        return (
            f"<KnowledgeGapRecord(id={self.id}, normalized_query={self.normalized_query!r}, "
            f"intent={self.intent}, top_score={self.top_score})>"
        )
