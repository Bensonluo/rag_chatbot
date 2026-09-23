"""Quality metrics: containment/CSAT math and the DB snapshot service."""

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.models.database.base import Base
from app.models.database.message import Message, MessageRole  # noqa: F401 (table registration)
from app.models.database.session import ChatSession  # noqa: F401 (table registration)
from app.models.database.ticket import HandoffTicket
from app.services.evaluation import QualityMetricsService, containment_rate, csat_score


@pytest.fixture
async def session_maker():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    yield maker
    await engine.dispose()


class TestContainmentRate:
    def test_full_containment_when_no_handoffs(self):
        assert containment_rate(100, 0) == 1.0

    def test_ratio(self):
        assert containment_rate(100, 30) == pytest.approx(0.7)

    def test_no_traffic_is_none_not_zero(self):
        # A fresh deployment has no containment story; 0.0 would misreport.
        assert containment_rate(0, 0) is None

    def test_handoffs_clamped_to_total(self):
        assert containment_rate(10, 25) == 0.0


class TestCsatScore:
    def test_ratio(self):
        assert csat_score(80, 20) == pytest.approx(0.8)

    def test_unrated_is_none(self):
        assert csat_score(0, 0) is None

    def test_all_negative(self):
        assert csat_score(0, 5) == 0.0


class TestQualityMetricsSnapshot:
    async def test_snapshot_computes_kpis_from_tables(self, session_maker):
        async with session_maker() as session:
            for index in range(4):
                chat = ChatSession(user_id=1, title=f"s{index}")
                session.add(chat)
                await session.flush()
                if index < 3:
                    session.add(
                        Message(
                            session_id=chat.id,
                            role=MessageRole.ASSISTANT,
                            content="回答",
                            user_rating=1 if index < 2 else -1,
                        )
                    )
            # Sessions 0 and 1 escalated; session 1 twice (deduplicated).
            session.add(HandoffTicket(session_id=1, user_id=1, reason="explicit", status="open"))
            session.add(HandoffTicket(session_id=1, user_id=1, reason="emotion", status="open"))
            session.add(
                HandoffTicket(session_id=2, user_id=1, reason="explicit", status="resolved")
            )
            await session.commit()

        snapshot = await QualityMetricsService(session_maker).snapshot()

        assert snapshot["total_sessions"] == 4
        assert snapshot["sessions_with_handoff"] == 2  # distinct session ids
        assert snapshot["containment_rate"] == pytest.approx(0.5)
        assert snapshot["csat"] == pytest.approx(2 / 3)
        assert snapshot["feedback_positive"] == 2
        assert snapshot["feedback_negative"] == 1
        assert snapshot["open_tickets"] == 2

    async def test_empty_db_returns_none_metrics_not_crash(self, session_maker):
        snapshot = await QualityMetricsService(session_maker).snapshot()

        assert snapshot["total_sessions"] == 0
        assert snapshot["containment_rate"] is None
        assert snapshot["csat"] is None
        assert snapshot["open_tickets"] == 0

    async def test_database_failure_degrades_to_error_snapshot(self):
        def broken_maker():
            raise RuntimeError("db down")

        snapshot = await QualityMetricsService(broken_maker).snapshot()

        assert snapshot["containment_rate"] is None
        assert snapshot["csat"] is None
        assert snapshot["error"] == "RuntimeError"
