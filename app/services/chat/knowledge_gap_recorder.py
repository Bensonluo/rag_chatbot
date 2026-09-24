"""Knowledge-gap recording for unanswered knowledge-intent turns.

Industry baseline (Zendesk knowledge-gap reporting, 阿里小蜜 知识缺口
挖掘): the bot surfaces queries it could not ground, and that list
drives KB curation. The recorder is the write side — the analytics API
is the read side.

Design mirrors ``persistence.py``: request-scoped sessions, failures
logged but never propagated (chat availability over telemetry), and an
optional dependency on ChatService so the graph stays DB-free.
Sampling keeps the write volume bounded on hot paths; at 800K-1M
requests/day a 100% gap rate on a cold KB would otherwise be a
self-inflicted write storm.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable
from typing import Any

from app.models.enums.intent import GRAPH_INTENTS, RAG_INTENTS
from app.repositories.knowledge_gap_repository import KnowledgeGapRepository

logger = logging.getLogger(__name__)

# Intent buckets whose empty retrieval is a *knowledge* gap. Task/
# chitchat/greeting turns legitimately have no documents.
GAP_INTENTS = frozenset(RAG_INTENTS) | frozenset(GRAPH_INTENTS)


class KnowledgeGapRecorder:
    """Records knowledge gaps with sampling; never raises."""

    def __init__(
        self,
        session_maker: Callable[[], Any],
        sample_rate: float = 1.0,
        rng: random.Random | None = None,
    ) -> None:
        """
        Args:
            session_maker: Factory producing context-managed async sessions.
            sample_rate: Fraction of gaps to record, 0.0-1.0 (volume valve
                for hot paths; aggregation still shows relative frequency).
            rng: Injectable RNG so tests are deterministic.
        """
        self._session_maker = session_maker
        self._sample_rate = sample_rate
        self._rng = rng or random.Random()

    async def record_if_gap(
        self,
        query: str,
        intent: str | None,
        retrieved_docs: list[Any] | None,
        session_id: str | int | None = None,
        user_id: int = 0,
    ) -> bool:
        """Record a gap when the turn qualifies; returns whether it wrote.

        A gap is: a knowledge-intent turn whose retrieval produced no
        usable documents. Failures are logged and swallowed — telemetry
        must never fail a chat response.
        """
        if not query or intent not in GAP_INTENTS:
            return False
        if retrieved_docs:
            return False
        # Metric first, sampling second: the counter is the unsampled
        # truth (dashboards divide it by RAG traffic; a sampled
        # numerator would misstate KB coverage), while the DB write
        # below keeps its sampling valve for write-volume control.
        from app.services.chat.metrics import KNOWLEDGE_GAPS

        KNOWLEDGE_GAPS.inc()
        if self._sample_rate <= 0.0 or self._rng.random() >= self._sample_rate:
            return False

        try:
            async with self._session_maker() as session:
                repo = KnowledgeGapRepository(session)
                await repo.record(
                    query=query,
                    intent=intent,
                    session_id=session_id,
                    user_id=user_id,
                    top_score=None,
                )
            return True
        except Exception as exc:  # noqa: BLE001 - availability over telemetry
            logger.warning("Failed to record knowledge gap (session=%s): %s", session_id, exc)
            return False


def create_knowledge_gap_recorder() -> KnowledgeGapRecorder:
    """Build the default recorder bound to the app's session maker."""
    from app.api.database import async_session_maker
    from app.config.settings import get_settings

    settings = get_settings()
    return KnowledgeGapRecorder(
        session_maker=async_session_maker,
        sample_rate=settings.KNOWLEDGE_GAP_SAMPLE_RATE,
    )
