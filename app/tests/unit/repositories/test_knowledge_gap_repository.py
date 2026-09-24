"""KnowledgeGapRepository: normalization, persistence, and aggregation."""

from datetime import datetime

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.models.database.base import Base
from app.models.database.knowledge_gap import KnowledgeGapRecord
from app.repositories.knowledge_gap_repository import (
    KnowledgeGapRepository,
    normalize_query,
)


@pytest.fixture
async def session_maker():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    yield maker
    await engine.dispose()


class TestNormalizeQuery:
    def test_collapses_whitespace_and_lowercases(self):
        assert normalize_query("  退货   政策 是什么 ") == "退货 政策 是什么"

    def test_truncates_to_200_chars(self):
        assert len(normalize_query("a" * 500)) == 200

    def test_empty_query_stays_empty(self):
        assert normalize_query("   ") == ""


class TestRecord:
    async def test_inserts_with_normalized_query(self, session_maker):
        async with session_maker() as session:
            repo = KnowledgeGapRepository(session)
            record = await repo.record(query="  退货政策 ", intent="policy")
        assert record.normalized_query == "退货政策"
        assert record.intent == "policy"
        assert record.id is not None

    async def test_defaults_for_anonymous_turns(self, session_maker):
        async with session_maker() as session:
            repo = KnowledgeGapRepository(session)
            record = await repo.record(query="q", intent="faq")
        assert record.session_id == ""
        assert record.user_id == 0
        assert record.top_score is None


class TestTopGaps:
    async def test_aggregates_by_normalized_query(self, session_maker):
        # Same question asked three ways (extra surrounding whitespace,
        # different casing) → one gap group with 3 hits.
        async with session_maker() as session:
            repo = KnowledgeGapRepository(session)
            await repo.record(query="发票怎么开", intent="faq", session_id=1)
            await repo.record(query="  发票怎么开 ", intent="faq", session_id=2)
            await repo.record(query="发票怎么开  ", intent="faq", session_id=3)
            await repo.record(query="怎么退货", intent="policy", session_id=4)
            gaps = await repo.top_gaps(days=7)
        assert len(gaps) == 2
        top = gaps[0]
        assert top["normalized_query"] == "发票怎么开"
        assert top["hits"] == 3
        assert top["sample_intent"] == "faq"
        assert top["last_seen"] is not None

    async def test_orders_by_hit_count_desc(self, session_maker):
        async with session_maker() as session:
            repo = KnowledgeGapRepository(session)
            await repo.record(query="b", intent="faq")
            await repo.record(query="a", intent="faq")
            await repo.record(query="a", intent="faq")
            gaps = await repo.top_gaps(days=7)
        assert [g["normalized_query"] for g in gaps] == ["a", "b"]

    async def test_respects_limit(self, session_maker):
        async with session_maker() as session:
            repo = KnowledgeGapRepository(session)
            for i in range(5):
                await repo.record(query=f"q{i}", intent="faq")
            gaps = await repo.top_gaps(days=7, limit=3)
        assert len(gaps) == 3

    async def test_window_excludes_old_rows(self, session_maker):
        async with session_maker() as session:
            old = KnowledgeGapRecord(
                session_id="1",
                user_id=0,
                query="old question",
                normalized_query="old question",
                intent="faq",
                top_score=None,
                created_at=datetime(2020, 1, 1),
                updated_at=datetime(2020, 1, 1),
            )
            session.add(old)
            await session.commit()
        async with session_maker() as session:
            repo = KnowledgeGapRepository(session)
            gaps = await repo.top_gaps(days=7)
        assert gaps == []


class TestResolveQuery:
    async def test_resolve_upserts_and_returns_row(self, session_maker):
        async with session_maker() as session:
            repo = KnowledgeGapRepository(session)
            resolution = await repo.resolve_query("发票怎么开", resolved_by=7)
        assert resolution.normalized_query == "发票怎么开"
        assert resolution.resolved_by == 7
        assert resolution.resolved_at is not None

    async def test_re_resolve_updates_same_row(self, session_maker):
        async with session_maker() as session:
            repo = KnowledgeGapRepository(session)
            first = await repo.resolve_query("q")
            second = await repo.resolve_query("q", resolved_by=9)
        assert second.id == first.id  # updated in place, not duplicated
        assert second.resolved_by == 9


class TestTopGapsLifecycle:
    async def test_resolved_group_hidden_by_default(self, session_maker):
        async with session_maker() as session:
            repo = KnowledgeGapRepository(session)
            await repo.record(query="发票怎么开", intent="faq")
            await repo.record(query="另一个问题", intent="faq")
            await repo.resolve_query("发票怎么开")

            gaps = await repo.top_gaps()

        assert [g["normalized_query"] for g in gaps] == ["另一个问题"]

    async def test_new_occurrence_reopens_resolved_group(self, session_maker):
        from datetime import timedelta

        async with session_maker() as session:
            repo = KnowledgeGapRepository(session)
            await repo.record(query="发票怎么开", intent="faq")
            await repo.resolve_query("发票怎么开")
            reopened = await repo.record(query="发票怎么开", intent="faq")
            # SQLite CURRENT_TIMESTAMP is second-granular: force the new
            # occurrence to land strictly after the resolution.
            reopened.created_at = reopened.created_at + timedelta(seconds=5)
            session.add(reopened)
            await session.commit()

            gaps = await repo.top_gaps()

        assert [g["normalized_query"] for g in gaps] == ["发票怎么开"]
        assert gaps[0]["resolved"] is False

    async def test_include_resolved_annotates_resolved_groups(self, session_maker):
        async with session_maker() as session:
            repo = KnowledgeGapRepository(session)
            await repo.record(query="发票怎么开", intent="faq")
            await repo.resolve_query("发票怎么开")

            gaps = await repo.top_gaps(include_resolved=True)

        assert len(gaps) == 1
        assert gaps[0]["resolved"] is True
