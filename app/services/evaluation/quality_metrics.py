"""Containment and CSAT quality metrics (the KPIs the business runs on).

Industry baseline: every mainstream platform reports containment (share
of conversations the bot resolves without a human — Intercom Fin
"resolution rate", Sierra containment) and CSAT (thumbs up/down ratio).
Both are computable from data this system already records: handoff
tickets mark the sessions that left the bot, message ratings capture
user satisfaction. This module turns that raw data into the two KPIs,
never raising into monitoring paths.
"""

import logging

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.database.message import Message
from app.models.database.session import ChatSession
from app.models.database.ticket import HandoffTicket

logger = logging.getLogger(__name__)


def containment_rate(total_sessions: int, sessions_with_handoff: int) -> float | None:
    """Share of sessions resolved without human handoff.

    Returns None when there is no traffic — a fresh deployment has no
    containment story to tell, and 0.0 would misreport it.
    """
    if total_sessions <= 0:
        return None
    handoff = min(sessions_with_handoff, total_sessions)
    return 1.0 - handoff / total_sessions


def csat_score(positive: int, negative: int) -> float | None:
    """Thumbs-up share of rated responses (None when nothing is rated)."""
    total = positive + negative
    if total <= 0:
        return None
    return positive / total


class QualityMetricsService:
    """Snapshot containment/CSAT/queue depth from the app's own tables.

    One read-only pass over three cheap aggregate queries; safe to call
    from monitoring dashboards on a timer. Never raises — on database
    trouble the snapshot carries the error and None metrics.
    """

    def __init__(self, session_maker: async_sessionmaker[AsyncSession]) -> None:
        self._session_maker = session_maker

    async def snapshot(self) -> dict:
        try:
            async with self._session_maker() as session:
                total_sessions = (
                    await session.execute(select(func.count(ChatSession.id)))
                ).scalar() or 0
                handoff_sessions = (
                    await session.execute(
                        select(func.count(func.distinct(HandoffTicket.session_id)))
                    )
                ).scalar() or 0
                open_tickets = (
                    await session.execute(
                        select(func.count(HandoffTicket.id)).where(HandoffTicket.status == "open")
                    )
                ).scalar() or 0
                rating_rows = (
                    await session.execute(
                        select(Message.user_rating).where(Message.user_rating.isnot(None))
                    )
                ).fetchall()
        except Exception as exc:  # noqa: BLE001 - monitoring must not raise
            logger.error("Quality metrics query failed: %s", exc)
            return {"error": str(exc.__class__.__name__), "containment_rate": None, "csat": None}

        ratings = [value for (value,) in rating_rows]
        positive = sum(1 for value in ratings if value > 0)
        negative = sum(1 for value in ratings if value < 0)

        return {
            "containment_rate": containment_rate(total_sessions, handoff_sessions),
            "csat": csat_score(positive, negative),
            "total_sessions": total_sessions,
            "sessions_with_handoff": handoff_sessions,
            "open_tickets": open_tickets,
            "feedback_positive": positive,
            "feedback_negative": negative,
        }
