"""KnowledgeGapRecorder: gap gating, sampling, and failure isolation.

The recorder is telemetry: it must record only true knowledge gaps
(knowledge intent + empty retrieval), respect the sampling valve, and
never let a database failure touch the chat path.
"""

from collections.abc import Callable
from typing import Any
from unittest.mock import Mock

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.models.database.base import Base
from app.models.database.knowledge_gap import KnowledgeGapRecord
from app.services.chat.knowledge_gap_recorder import KnowledgeGapRecorder


@pytest.fixture
async def session_maker():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    yield maker
    await engine.dispose()


def _rng(value: float) -> Mock:
    rng = Mock()
    rng.random.return_value = value
    return rng


def _recorder(
    session_maker: Callable[[], Any], sample_rate: float = 1.0, rng_value: float = 0.0
) -> KnowledgeGapRecorder:
    return KnowledgeGapRecorder(
        session_maker=session_maker,
        sample_rate=sample_rate,
        rng=_rng(rng_value),
    )


class TestGapGating:
    async def test_records_unanswered_faq_intent(self, session_maker):
        recorder = _recorder(session_maker)
        recorded = await recorder.record_if_gap(
            query="发票怎么开", intent="faq", retrieved_docs=[], session_id=7, user_id=42
        )
        assert recorded is True
        async with session_maker() as session:
            rows = (await session.execute(select(KnowledgeGapRecord))).fetchall()
        assert len(rows) == 1
        assert rows[0][0].query == "发票怎么开"
        assert rows[0][0].session_id == "7"
        assert rows[0][0].user_id == 42

    async def test_no_record_when_docs_present(self, session_maker):
        recorder = _recorder(session_maker)
        recorded = await recorder.record_if_gap(
            query="退货政策", intent="faq", retrieved_docs=[{"content": "7天无理由"}]
        )
        assert recorded is False
        async with session_maker() as session:
            rows = (await session.execute(select(KnowledgeGapRecord))).fetchall()
        assert rows == []

    @pytest.mark.parametrize("intent", ["chitchat", "greeting", "refund", "handoff"])
    async def test_no_record_for_non_knowledge_intents(self, session_maker, intent):
        recorder = _recorder(session_maker)
        recorded = await recorder.record_if_gap(query="你好", intent=intent, retrieved_docs=[])
        assert recorded is False

    async def test_no_record_for_empty_query(self, session_maker):
        recorder = _recorder(session_maker)
        recorded = await recorder.record_if_gap(query="", intent="faq", retrieved_docs=[])
        assert recorded is False

    async def test_none_intent_is_not_a_gap(self, session_maker):
        recorder = _recorder(session_maker)
        recorded = await recorder.record_if_gap(query="你好", intent=None, retrieved_docs=[])
        assert recorded is False


class TestSampling:
    async def test_sample_rate_zero_never_records(self, session_maker):
        recorder = _recorder(session_maker, sample_rate=0.0, rng_value=0.0)
        recorded = await recorder.record_if_gap(query="发票怎么开", intent="faq", retrieved_docs=[])
        assert recorded is False

    async def test_draw_above_rate_is_skipped(self, session_maker):
        recorder = _recorder(session_maker, sample_rate=0.5, rng_value=0.999)
        recorded = await recorder.record_if_gap(query="发票怎么开", intent="faq", retrieved_docs=[])
        assert recorded is False

    async def test_draw_below_rate_records(self, session_maker):
        recorder = _recorder(session_maker, sample_rate=0.5, rng_value=0.1)
        recorded = await recorder.record_if_gap(query="发票怎么开", intent="faq", retrieved_docs=[])
        assert recorded is True


class TestFailureIsolation:
    async def test_exception_is_swallowed(self):
        def exploding_maker():
            raise RuntimeError("db down")

        recorder = KnowledgeGapRecorder(session_maker=exploding_maker, rng=_rng(0.0))
        recorded = await recorder.record_if_gap(query="发票怎么开", intent="faq", retrieved_docs=[])
        assert recorded is False


class TestGapMetric:
    """The counter is unsampled truth; only the DB write is sampled."""

    def _gaps(self) -> float:
        from prometheus_client import REGISTRY

        value = REGISTRY.get_sample_value("knowledge_gaps_total")
        return value if value is not None else 0.0

    async def test_gap_counts_even_when_sampling_skips_write(self, session_maker):
        # sample_rate=0 suppresses every DB write, but the metric must
        # still reflect the true gap rate — dashboards divide it by RAG
        # traffic, and a sampled numerator would silently misstate the
        # KB's coverage.
        recorder = _recorder(session_maker, sample_rate=0.0, rng_value=0.0)
        before = self._gaps()
        recorded = await recorder.record_if_gap(query="发票怎么开", intent="faq", retrieved_docs=[])
        assert recorded is False
        assert self._gaps() == before + 1.0

    async def test_non_gap_does_not_count(self, session_maker):
        recorder = _recorder(session_maker)
        before = self._gaps()
        await recorder.record_if_gap(query="退货政策", intent="faq", retrieved_docs=[{"id": "d1"}])
        await recorder.record_if_gap(query="你好", intent="chitchat", retrieved_docs=[])
        assert self._gaps() == before
