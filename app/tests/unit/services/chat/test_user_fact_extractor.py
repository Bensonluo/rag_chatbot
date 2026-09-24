"""UserFactExtractor: background cross-session fact extraction (Phase B).

mem0 v3 production pattern distilled from 2026 sources (mem0 docs,
arXiv Mem0 paper, 百度千帆记忆后台更新, AWS Bedrock event-sourced
memory):

1. **ADD-only single pass** — extraction inserts new facts and never
   UPDATE/DELETEs existing ones on the hot path; contradictions are
   handled by recency at read time plus retention pruning, not by
   in-line conflict resolution.
2. **Post-durability background trigger** — extraction runs after the
   turn is committed (fire-and-forget), so chat latency never pays for
   memory writes.
3. **The LLM proposes, code verifies** — extracted facts pass
   deterministic post-filters (PII drop, exact dedup, length cap,
   per-run cap) before touching the table.

Maybe-extract never raises: memory extraction is an enhancement, never
a dependency of the chat path.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.database.base import Base
from app.models.database.message import Message
from app.models.database.session import ChatSession
from app.models.database.user import User
from app.models.enums.message import MessageRole
from app.repositories.user_fact_repository import UserFactRepository
from app.services.chat.user_fact_extractor import UserFactExtractor


@pytest.fixture
async def session_maker() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    yield maker
    await engine.dispose()


async def _seed_dialogue(
    maker: async_sessionmaker[AsyncSession],
    *,
    session_id: int = 1,
    user_id: int = 1,
    turns: int,
) -> None:
    """Seed a session with ``turns`` user+assistant exchanges."""
    async with maker() as session:
        session.add(User(id=user_id, email=f"u{user_id}@test.local", hashed_password="x"))
        session.add(ChatSession(id=session_id, user_id=user_id, title="t"))
        await session.flush()
        for i in range(turns):
            session.add(
                Message(
                    session_id=session_id,
                    role=MessageRole.USER,
                    content=f"我是 PLUS 会员，第 {i} 轮",
                )
            )
            session.add(
                Message(
                    session_id=session_id,
                    role=MessageRole.ASSISTANT,
                    content=f"好的，已记录第 {i} 轮",
                )
            )
        await session.commit()


def _llm_returning(payload: Any) -> Mock:
    """Light-LLM stub returning the given payload as JSON text."""
    llm = Mock()
    llm.generate = AsyncMock(return_value=Mock(content=json.dumps(payload, ensure_ascii=False)))
    return llm


def _extractor(
    maker: async_sessionmaker[AsyncSession], llm: Any, **overrides: Any
) -> UserFactExtractor:
    # threshold counts messages (user+assistant), same unit as the
    # compressor's CHAT_SUMMARY_THRESHOLD — one exchange seeds 2 rows.
    params: dict[str, Any] = {
        "session_maker": maker,
        "llm_service": llm,
        "threshold": 12,
        "interval": 6,
        "timeout_seconds": 1.0,
        "max_facts_per_run": 5,
        "retention": 50,
    }
    params.update(overrides)
    return UserFactExtractor(**params)


async def _facts_of(maker: async_sessionmaker[AsyncSession], user_id: int = 1) -> list[str]:
    async with maker() as session:
        repo = UserFactRepository(session)
        rows = await repo.recent_for_user(user_id=user_id, limit=100)
    return [r.fact for r in rows]


class TestTrigger:
    async def test_below_threshold_skips_llm(
        self, session_maker: async_sessionmaker[AsyncSession]
    ) -> None:
        await _seed_dialogue(session_maker, turns=5)
        llm = _llm_returning(["用户是 PLUS 会员"])
        extractor = _extractor(session_maker, llm)

        assert await extractor.maybe_extract(session_id=1) is False
        llm.generate.assert_not_awaited()
        assert await _facts_of(session_maker) == []

    async def test_between_triggers_skips_llm(
        self, session_maker: async_sessionmaker[AsyncSession]
    ) -> None:
        """8 turns with threshold=6/interval=6: past the first trigger,
        not yet at the second — no extraction."""
        await _seed_dialogue(session_maker, turns=8)
        llm = _llm_returning(["用户是 PLUS 会员"])
        extractor = _extractor(session_maker, llm)

        assert await extractor.maybe_extract(session_id=1) is False
        llm.generate.assert_not_awaited()

    async def test_at_threshold_extracts_and_persists(
        self, session_maker: async_sessionmaker[AsyncSession]
    ) -> None:
        await _seed_dialogue(session_maker, turns=6)
        llm = _llm_returning(["用户是 PLUS 会员", "偏好上午配送"])
        extractor = _extractor(session_maker, llm)

        assert await extractor.maybe_extract(session_id=1) is True
        llm.generate.assert_awaited_once()

        facts = await _facts_of(session_maker)
        assert set(facts) == {"用户是 PLUS 会员", "偏好上午配送"}

    async def test_interval_retrigger_extracts_again(
        self, session_maker: async_sessionmaker[AsyncSession]
    ) -> None:
        # 24 messages: past the first trigger, exactly at the second
        # ((24 - 12) % 6 == 0) — proves the interval cadence, not just
        # the first crossing.
        await _seed_dialogue(session_maker, turns=12)
        llm = _llm_returning(["用户是 PLUS 会员"])
        extractor = _extractor(session_maker, llm)

        assert await extractor.maybe_extract(session_id=1) is True

    async def test_prompt_carries_transcript(
        self, session_maker: async_sessionmaker[AsyncSession]
    ) -> None:
        await _seed_dialogue(session_maker, turns=6)
        llm = _llm_returning([])
        extractor = _extractor(session_maker, llm)

        await extractor.maybe_extract(session_id=1)

        prompt = llm.generate.await_args.kwargs["messages"][0].content
        assert "我是 PLUS 会员，第 0 轮" in prompt
        assert "好的，已记录第 5 轮" in prompt

    async def test_facts_attributed_to_session_user(
        self, session_maker: async_sessionmaker[AsyncSession]
    ) -> None:
        """user_id comes from the session row, not the caller — the
        extractor never trusts a client-supplied identity."""
        await _seed_dialogue(session_maker, session_id=1, user_id=7, turns=6)
        llm = _llm_returning(["用户是 PLUS 会员"])
        extractor = _extractor(session_maker, llm)

        await extractor.maybe_extract(session_id=1)

        assert await _facts_of(session_maker, user_id=7) == ["用户是 PLUS 会员"]
        assert await _facts_of(session_maker, user_id=1) == []

    async def test_missing_session_returns_false(
        self, session_maker: async_sessionmaker[AsyncSession]
    ) -> None:
        llm = _llm_returning(["用户是 PLUS 会员"])
        extractor = _extractor(session_maker, llm)

        assert await extractor.maybe_extract(session_id=999) is False
        llm.generate.assert_not_awaited()


class TestDeterministicFilters:
    async def test_pii_facts_are_dropped(
        self, session_maker: async_sessionmaker[AsyncSession]
    ) -> None:
        """Phone / national-ID / email must never land in the memory
        table — PII in a cross-session store is a compliance incident,
        not a recall feature."""
        await _seed_dialogue(session_maker, turns=6)
        llm = _llm_returning(
            [
                "用户手机号 13800001234",
                "身份证号 110101199003077758",
                "用户是 PLUS 会员",
                "邮箱 a@b.com",
            ]
        )
        extractor = _extractor(session_maker, llm)

        assert await extractor.maybe_extract(session_id=1) is True
        assert await _facts_of(session_maker) == ["用户是 PLUS 会员"]

    async def test_exact_duplicates_not_reinserted(
        self, session_maker: async_sessionmaker[AsyncSession]
    ) -> None:
        await _seed_dialogue(session_maker, turns=6)
        async with session_maker() as session:
            await UserFactRepository(session).add(user_id=1, fact="用户是 PLUS 会员")

        llm = _llm_returning(["用户是 PLUS 会员", "偏好上午配送"])
        extractor = _extractor(session_maker, llm)

        await extractor.maybe_extract(session_id=1)

        facts = await _facts_of(session_maker)
        assert facts.count("用户是 PLUS 会员") == 1
        assert set(facts) == {"用户是 PLUS 会员", "偏好上午配送"}

    async def test_overlong_facts_dropped(
        self, session_maker: async_sessionmaker[AsyncSession]
    ) -> None:
        await _seed_dialogue(session_maker, turns=6)
        llm = _llm_returning(["长" * 200, "偏好上午配送"])
        extractor = _extractor(session_maker, llm)

        await extractor.maybe_extract(session_id=1)

        assert await _facts_of(session_maker) == ["偏好上午配送"]

    async def test_per_run_cap_limits_inserts(
        self, session_maker: async_sessionmaker[AsyncSession]
    ) -> None:
        await _seed_dialogue(session_maker, turns=6)
        llm = _llm_returning([f"事实 {i}" for i in range(8)])
        extractor = _extractor(session_maker, llm, max_facts_per_run=3)

        await extractor.maybe_extract(session_id=1)

        assert len(await _facts_of(session_maker)) == 3

    async def test_dict_payloads_with_category_supported(
        self, session_maker: async_sessionmaker[AsyncSession]
    ) -> None:
        await _seed_dialogue(session_maker, turns=6)
        llm = _llm_returning([{"fact": "偏好上午配送", "category": "preference"}])
        extractor = _extractor(session_maker, llm)

        await extractor.maybe_extract(session_id=1)

        async with session_maker() as session:
            repo = UserFactRepository(session)
            rows = await repo.recent_for_user(user_id=1, limit=10)
        assert rows[0].category == "preference"


class TestRetention:
    async def test_retention_prunes_to_newest(
        self, session_maker: async_sessionmaker[AsyncSession]
    ) -> None:
        """ADD-only memory still needs a bounded table: after insert,
        prune keeps the newest ``retention`` rows per user."""
        await _seed_dialogue(session_maker, turns=6)
        async with session_maker() as session:
            repo = UserFactRepository(session)
            for i in range(3):
                await repo.add(user_id=1, fact=f"旧事实 {i}")

        llm = _llm_returning(["新事实 A", "新事实 B"])
        extractor = _extractor(session_maker, llm, retention=3)

        await extractor.maybe_extract(session_id=1)

        facts = await _facts_of(session_maker)
        assert len(facts) == 3
        assert "新事实 A" in facts and "新事实 B" in facts


class TestResilience:
    async def test_llm_none_disables_extraction(
        self, session_maker: async_sessionmaker[AsyncSession]
    ) -> None:
        await _seed_dialogue(session_maker, turns=6)
        extractor = _extractor(session_maker, llm=None)

        assert await extractor.maybe_extract(session_id=1) is False
        assert await _facts_of(session_maker) == []

    async def test_llm_error_returns_false(
        self, session_maker: async_sessionmaker[AsyncSession]
    ) -> None:
        await _seed_dialogue(session_maker, turns=6)
        llm = Mock()
        llm.generate = AsyncMock(side_effect=RuntimeError("llm down"))
        extractor = _extractor(session_maker, llm)

        assert await extractor.maybe_extract(session_id=1) is False
        assert await _facts_of(session_maker) == []

    async def test_malformed_llm_output_writes_nothing(
        self, session_maker: async_sessionmaker[AsyncSession]
    ) -> None:
        await _seed_dialogue(session_maker, turns=6)
        llm = Mock()
        llm.generate = AsyncMock(return_value=Mock(content="这不是 JSON"))
        extractor = _extractor(session_maker, llm)

        assert await extractor.maybe_extract(session_id=1) is False
        assert await _facts_of(session_maker) == []

    async def test_fenced_json_is_parsed(
        self, session_maker: async_sessionmaker[AsyncSession]
    ) -> None:
        """LLMs wrap JSON in ```json fences — strip before parsing."""
        await _seed_dialogue(session_maker, turns=6)
        llm = Mock()
        llm.generate = AsyncMock(return_value=Mock(content='```json\n["偏好上午配送"]\n```'))
        extractor = _extractor(session_maker, llm)

        assert await extractor.maybe_extract(session_id=1) is True
        assert await _facts_of(session_maker) == ["偏好上午配送"]

    async def test_non_list_payload_writes_nothing(
        self, session_maker: async_sessionmaker[AsyncSession]
    ) -> None:
        await _seed_dialogue(session_maker, turns=6)
        llm = _llm_returning({"fact": "不是列表"})
        extractor = _extractor(session_maker, llm)

        assert await extractor.maybe_extract(session_id=1) is False
        assert await _facts_of(session_maker) == []
