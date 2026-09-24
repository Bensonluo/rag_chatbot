"""UserFact repository: cross-session user memory writes and reads.

Phase B write side. User facts are long-lived preferences and
constraints ("PLUS 会员", "偏好上午配送") extracted from conversations
and recalled in later sessions (mem0-style layered memory; industry
baseline: 阿里小蜜用户画像, AWS Bedrock event-sourced memory). The table
is telemetry-style like knowledge_gaps — no FK to users — so memory
writes never couple chat availability to user-row integrity.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.database.base import Base
from app.models.database.user_fact import UserFactRecord
from app.repositories.user_fact_repository import UserFactRepository


@pytest.fixture
async def session_maker() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    yield maker
    await engine.dispose()


async def _add(
    maker: async_sessionmaker[AsyncSession],
    user_id: int,
    fact: str,
    category: str = "general",
) -> None:
    async with maker() as session:
        repo = UserFactRepository(session)
        await repo.add(user_id=user_id, fact=fact, category=category)


class TestAddAndRead:
    async def test_add_and_recent_returns_newest_first(
        self, session_maker: async_sessionmaker[AsyncSession]
    ) -> None:
        """Rows created within the same second must order by id as
        tiebreaker — same contract as message_repository."""
        for i in range(3):
            await _add(session_maker, user_id=1, fact=f"事实 {i}")

        async with session_maker() as session:
            repo = UserFactRepository(session)
            recent = await repo.recent_for_user(user_id=1, limit=10)

        assert [r.fact for r in recent] == ["事实 2", "事实 1", "事实 0"]
        assert all(r.user_id == 1 for r in recent)

    async def test_recent_respects_limit(
        self, session_maker: async_sessionmaker[AsyncSession]
    ) -> None:
        for i in range(5):
            await _add(session_maker, user_id=1, fact=f"事实 {i}")

        async with session_maker() as session:
            repo = UserFactRepository(session)
            recent = await repo.recent_for_user(user_id=1, limit=2)

        assert [r.fact for r in recent] == ["事实 4", "事实 3"]

    async def test_facts_are_isolated_per_user(
        self, session_maker: async_sessionmaker[AsyncSession]
    ) -> None:
        await _add(session_maker, user_id=1, fact="用户一的事实")
        await _add(session_maker, user_id=2, fact="用户二的事实")

        async with session_maker() as session:
            repo = UserFactRepository(session)
            recent = await repo.recent_for_user(user_id=1, limit=10)

        assert [r.fact for r in recent] == ["用户一的事实"]

    async def test_category_and_source_session_persisted(
        self, session_maker: async_sessionmaker[AsyncSession]
    ) -> None:
        async with session_maker() as session:
            repo = UserFactRepository(session)
            record = await repo.add(
                user_id=7, fact="偏好上午配送", category="preference", source_session_id=42
            )

        assert record.category == "preference"
        assert record.source_session_id == "42"

    async def test_existing_texts_returns_stripped_set(
        self, session_maker: async_sessionmaker[AsyncSession]
    ) -> None:
        await _add(session_maker, user_id=1, fact="PLUS 会员")

        async with session_maker() as session:
            repo = UserFactRepository(session)
            texts = await repo.existing_texts(user_id=1)
            unknown = await repo.existing_texts(user_id=999)

        assert texts == {"PLUS 会员"}
        assert unknown == set()


class TestPrune:
    async def test_prune_keeps_newest_n(
        self, session_maker: async_sessionmaker[AsyncSession]
    ) -> None:
        for i in range(5):
            await _add(session_maker, user_id=1, fact=f"事实 {i}")

        async with session_maker() as session:
            repo = UserFactRepository(session)
            removed = await repo.prune_for_user(user_id=1, keep=3)
            remaining = await repo.recent_for_user(user_id=1, limit=10)

        assert removed == 2
        assert [r.fact for r in remaining] == ["事实 4", "事实 3", "事实 2"]

    async def test_prune_below_keep_is_noop(
        self, session_maker: async_sessionmaker[AsyncSession]
    ) -> None:
        await _add(session_maker, user_id=1, fact="唯一事实")

        async with session_maker() as session:
            repo = UserFactRepository(session)
            removed = await repo.prune_for_user(user_id=1, keep=10)

        assert removed == 0

    async def test_prune_never_touches_other_users(
        self, session_maker: async_sessionmaker[AsyncSession]
    ) -> None:
        for i in range(5):
            await _add(session_maker, user_id=1, fact=f"事实 {i}")
        await _add(session_maker, user_id=2, fact="用户二的事实")

        async with session_maker() as session:
            repo = UserFactRepository(session)
            await repo.prune_for_user(user_id=1, keep=2)
            user2 = await repo.recent_for_user(user_id=2, limit=10)

        assert [r.fact for r in user2] == ["用户二的事实"]


class TestModelShape:
    def test_table_is_fk_free_telemetry_style(self) -> None:
        """No FK to users: memory writes must not fail on user-row
        lifecycle (cascade deletes), same rationale as knowledge_gaps."""
        columns = UserFactRecord.__table__.columns
        assert "user_id" in columns
        assert not any(col.foreign_keys for col in columns)
        assert columns["user_id"].index is True
