"""User-fact database model: cross-session user memory (Phase B).

Industry baseline (mem0 v3 production pattern, 阿里小蜜用户画像, AWS
Bedrock event-sourced memory): long-lived user facts — membership
tier, stable preferences, stated constraints — extracted from
conversations and recalled in later sessions. The table is
telemetry-style like knowledge_gaps: deliberately no FK to users, so
memory writes never couple chat availability to user-row lifecycle,
and bounded per user by retention pruning in the extractor.
"""

from __future__ import annotations

from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.database.base import Base, TimestampMixin


class UserFactRecord(Base, TimestampMixin):
    """One durable fact about a user, extracted from a conversation.

    Attributes:
        id: Primary key
        user_id: Owning user (no FK — telemetry-style; indexed because
            every read filters on it)
        fact: The fact text itself, third-person normalized ("用户是
            PLUS 会员"), already PII-filtered at extraction time
        category: Coarse bucket (general/preference/constraint) used
            for recall-side grouping
        source_session_id: Session the fact was extracted from —
            provenance for diagnostics and future re-extraction
            (string to match knowledge_gaps' convention)
    """

    __tablename__ = "user_facts"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(nullable=False, default=0, index=True)
    fact: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(32), nullable=False, default="general")
    source_session_id: Mapped[str] = mapped_column(String(64), nullable=False, default="")

    def __repr__(self) -> str:
        return f"<UserFactRecord(id={self.id}, user_id={self.user_id}, fact={self.fact[:20]!r})>"
