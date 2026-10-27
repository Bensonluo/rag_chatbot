"""Shared answer-cache reads must not bypass durable customer context.

Real SQLite persistence, facts, BM25, cache service and compiled graph are
used. Only Redis/epoch and the LLM provider are local deterministic seams.
"""

import json
import socket
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from langgraph.checkpoint.memory import MemorySaver
from sqlalchemy import event, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config.settings import settings
from app.models.database.base import Base
from app.models.database.message import Message
from app.models.database.session import ChatSession
from app.models.database.user import User
from app.models.database.user_fact import UserFactRecord
from app.models.enums.message import MessageRole
from app.services.chat import answer_cache as answer_cache_module
from app.services.chat.answer_cache import AnswerCacheService
from app.services.chat.chat_service import HEARTBEAT, STREAM_ERROR, ChatService, CitationEvent
from app.services.chat.persistence import ChatMessagePersister
from app.services.dialogue.graph import build_dialogue_graph
from app.services.dialogue.tools import ToolRegistry
from app.services.guardrails.factory import GuardrailFactory
from app.services.intent.rule_based import RuleBasedIntentDetector
from app.services.llm.base import LLMMessage, LLMResponse, LLMServiceBase
from app.services.retrieval.hybrid_search import KeywordSearch

_QUERY = "What is the warranty coverage?"
_PREMIUM_FACT = "Premium plan customer"
_STANDARD = "The standard plan covers manufacturing defects."
_PREMIUM = "Your Premium plan also covers accidental damage."
_DOCUMENT = (
    "Warranty coverage: the standard plan covers manufacturing defects; "
    "the Premium plan also covers accidental damage."
)


class _MemoryRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.hashes: dict[str, dict[str, str]] = {}
        self.reads: list[str] = []
        self.writes: list[str] = []

    async def get(self, key: str) -> str | None:
        self.reads.append(key)
        return self.store.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> bool:
        self.writes.append(key)
        self.store[key] = value
        return True

    async def hget(self, name: str, key: str) -> str | None:
        return self.hashes.get(name, {}).get(key)

    async def hset(self, name: str, key: str, value: str) -> int:
        self.hashes.setdefault(name, {})[key] = value
        return 1

    async def expire(self, name: str, ttl: int) -> bool:
        return True


class _ContextLLM(LLMServiceBase):
    def __init__(self) -> None:
        super().__init__(api_key="offline", model="context-cache-fixture")
        self.prompts: list[list[LLMMessage]] = []

    def _answer(self, messages: list[LLMMessage]) -> str:
        self.prompts.append([LLMMessage(role=m.role, content=m.content) for m in messages])
        context = "\n".join(message.content for message in messages[:-1])
        return _PREMIUM if _PREMIUM_FACT in context else _STANDARD

    async def generate(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        return LLMResponse(content=self._answer(messages), model=self.model)

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        answer = self._answer(messages)
        yield answer[:15]
        yield answer[15:]

    def estimate_tokens(self, text: str) -> int:
        return len(text)

    async def count_tokens(self, messages: list[LLMMessage]) -> int:
        return sum(len(message.content) for message in messages)


@dataclass
class _Harness:
    maker: async_sessionmaker[AsyncSession]
    persister: ChatMessagePersister
    cache: AnswerCacheService
    redis: _MemoryRedis
    llm: _ContextLLM
    search: KeywordSearch
    cached: ChatService | None = None
    uncached: ChatService | None = None
    history_reads: list[int] = field(default_factory=list)
    fact_reads: list[int] = field(default_factory=list)
    history_errors: list[tuple[int, str]] = field(default_factory=list)
    provider_failures: set[int] = field(default_factory=set)
    database_failures: set[int] = field(default_factory=set)
    injecting_history_failure: bool = False

    async def history(self, session_id: int) -> list[LLMMessage]:
        self.history_reads.append(session_id)
        try:
            if session_id in self.provider_failures:
                raise RuntimeError("isolated history-provider outage")
            self.injecting_history_failure = session_id in self.database_failures
            rows = await self.persister.get_history(
                session_id, limit=100, include_summary=True, raise_on_error=True
            )
            return [LLMMessage(role=row.role, content=row.content) for row in rows]
        except Exception as exc:
            self.history_errors.append((session_id, type(exc).__name__))
            raise
        finally:
            self.injecting_history_failure = False

    async def facts(self, user_id: int) -> list[str]:
        self.fact_reads.append(user_id)
        return await self.persister.get_user_facts(user_id=user_id)

    def service(self, use_cache: bool) -> ChatService:
        return ChatService(
            graph=build_dialogue_graph(
                intent_detector=RuleBasedIntentDetector(),
                slot_filler=None,
                tool_registry=ToolRegistry(),
                retrieval_pipeline={"hybrid_search": self.search},
                llm_service=self.llm,
                guardrail_service=GuardrailFactory.create(),
                checkpointer=MemorySaver(),
                history_provider=self.history,
                user_facts_provider=self.facts,
                answer_cache=self.cache if use_cache else None,
            ),
            persister=self.persister,
            answer_cache=self.cache if use_cache else None,
        )


@pytest.fixture
async def harness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncGenerator[_Harness, None]:
    monkeypatch.setattr(settings, "DEMO_MODE", False)
    monkeypatch.setattr(settings, "CHAT_TRACE_STREAM_ENABLED", False)
    # This gate measures generation calls independently of query rewriting.
    monkeypatch.setattr(settings, "QUERY_REWRITE_ENABLED", False)
    monkeypatch.setattr(
        answer_cache_module, "get_kb_epoch", AsyncMock(return_value="context-cache-epoch")
    )
    attempts: list[str] = []

    def forbid_network(_socket: Any, address: Any, *args: Any, **kwargs: Any) -> None:
        attempts.append(str(address))
        raise AssertionError("context-cache evaluation must not connect to external services")

    monkeypatch.setattr(socket.socket, "connect", forbid_network)
    monkeypatch.setattr(socket.socket, "connect_ex", forbid_network)
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/context-cache.sqlite3")

    @event.listens_for(engine.sync_engine, "connect")
    def foreign_keys(connection: Any, _record: Any) -> None:
        connection.execute("PRAGMA foreign_keys=ON")

    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(
            lambda conn: Base.metadata.create_all(
                conn,
                tables=[
                    Base.metadata.tables[name]
                    for name in ("users", "chat_sessions", "messages", "user_facts")
                ],
            )
        )
    async with maker() as db:
        db.add_all(
            [
                User(id=1, email="anonymous@evaluation.invalid", hashed_password="!"),
                User(id=42, email="premium@evaluation.invalid", hashed_password="!"),
            ]
        )
        await db.flush()
        db.add_all(
            [
                ChatSession(id=sid, user_id=42 if sid in {42, 43} else 1)
                for sid in (1, 2, 30, 31, 42, 43)
            ]
        )
        db.add(UserFactRecord(user_id=42, fact=_PREMIUM_FACT, source_session_id="42"))
        await db.commit()
    redis = _MemoryRedis()
    cache = AnswerCacheService(redis_url="redis://offline.invalid/0", ttl_seconds=60)
    cache._redis = redis  # type: ignore[assignment]
    search = KeywordSearch()
    await search.add_documents(
        [
            {
                "id": "warranty-chunk-1",
                "content": _DOCUMENT,
                "metadata": {
                    "document_id": "warranty-policy",
                    "version_id": "warranty-v1",
                    "chunk_id": "warranty-chunk-1",
                    "title": "Warranty coverage",
                    "page_start": 2,
                    "page_end": 2,
                },
            }
        ]
    )
    case = _Harness(maker, ChatMessagePersister(maker), cache, redis, _ContextLLM(), search)

    @event.listens_for(engine.sync_engine, "before_cursor_execute")
    def fail_history_read(
        connection: Any, cursor: Any, statement: str, parameters: Any, context: Any, many: bool
    ) -> None:
        if (
            case.injecting_history_failure
            and statement.lstrip().lower().startswith("select")
            and "messages" in statement.lower()
        ):
            raise OperationalError(
                statement, parameters, RuntimeError("isolated history DB outage")
            )

    case.cached = case.service(True)
    case.uncached = case.service(False)
    try:
        yield case
    finally:
        await case.persister.drain()
        await engine.dispose()
        assert attempts == []


@dataclass
class _Turn:
    content: str
    cache_route: str
    citations: list[dict[str, Any]]
    prompts: list[list[LLMMessage]]
    cache_reads: list[str]
    history_reads: list[int]
    fact_reads: list[int]
    cache_writes: list[str]


async def _turn(
    case: _Harness, service: ChatService, session_id: int, user_id: int, transport: str
) -> _Turn:
    before = (
        len(case.llm.prompts),
        len(case.redis.reads),
        len(case.history_reads),
        len(case.fact_reads),
        len(case.redis.writes),
    )
    if transport == "normal":
        response = await service.process_message(session_id, _QUERY, user_id)
        content, citations = response.content, response.citations
    else:
        items = [
            item
            async for item in service.process_message_stream(
                session_id, _QUERY, user_id, include_citations=True
            )
        ]
        assert STREAM_ERROR not in items
        content = "".join(item for item in items if isinstance(item, str) and item != HEARTBEAT)
        references = [item.citations for item in items if isinstance(item, CitationEvent)]
        assert len(references) == 1
        citations = references[0]
    state = (
        await service.graph.aget_state({"configurable": {"thread_id": str(session_id)}})
    ).values
    async with case.maker() as db:
        saved = await db.scalar(
            select(Message)
            .where(Message.session_id == session_id, Message.role == MessageRole.ASSISTANT)
            .order_by(Message.id.desc())
            .limit(1)
        )
    assert saved is not None and saved.content == content
    assert (saved.message_metadata or {}).get("citations") == citations
    assert len(citations) == 1
    assert citations[0]["document_id"] == "warranty-policy"
    assert citations[0]["version_id"] == "warranty-v1"
    assert citations[0]["chunk_id"] == "warranty-chunk-1"
    assert citations[0]["excerpt"] == _DOCUMENT
    assert citations[0]["page_start"] == 2
    return _Turn(
        content,
        state["route_after_cache"],
        citations,
        case.llm.prompts[before[0] :],
        case.redis.reads[before[1] :],
        case.history_reads[before[2] :],
        case.fact_reads[before[3] :],
        case.redis.writes[before[4] :],
    )


async def _seed_cache(case: _Harness, transport: str) -> _Turn:
    assert case.cached is not None
    cold = await _turn(case, case.cached, 1, 0, transport)
    assert cold.content == _STANDARD
    assert cold.cache_route == "miss"
    assert len(cold.prompts) == 1
    assert cold.history_reads == [1]
    assert len(case.redis.store) == 1
    return cold


@pytest.mark.parametrize("transport", ["normal", "stream"])
async def test_context_free_anonymous_turn_retains_a_versioned_cache_hit(
    harness: _Harness, transport: str
) -> None:
    cold = await _seed_cache(harness, transport)
    assert harness.cached is not None
    warm = await _turn(harness, harness.cached, 2, 0, transport)
    assert warm.cache_route == "hit"
    assert warm.content == cold.content
    assert warm.citations == cold.citations
    assert warm.prompts == []
    assert len(warm.cache_reads) == 1
    assert warm.history_reads == [2]
    assert warm.fact_reads == []


@pytest.mark.parametrize("transport", ["normal", "stream"])
@pytest.mark.parametrize(
    "context_kind", ["facts", "history", "summary", "provider_error", "db_error"]
)
async def test_context_or_unavailable_history_cannot_replay_a_generic_answer(
    harness: _Harness, transport: str, context_kind: str
) -> None:
    await _seed_cache(harness, transport)
    if context_kind == "facts":
        session_id, control_id, user_id = 42, 43, 42
    else:
        session_id, control_id, user_id = 30, 31, 0
        async with harness.maker() as db:
            for sid in (session_id, control_id):
                db.add(
                    Message(
                        session_id=sid,
                        role=MessageRole.SYSTEM if context_kind == "summary" else MessageRole.USER,
                        content=f"I am a {_PREMIUM_FACT}. Answer only for my Premium plan.",
                        created_at=datetime.now(UTC).replace(tzinfo=None) - timedelta(minutes=5),
                    )
                )
            await db.commit()
        if context_kind == "provider_error":
            harness.provider_failures.update({session_id, control_id})
        elif context_kind == "db_error":
            harness.database_failures.update({session_id, control_id})
    assert harness.cached is not None and harness.uncached is not None
    cached = await _turn(harness, harness.cached, session_id, user_id, transport)
    control = await _turn(harness, harness.uncached, control_id, user_id, transport)
    # The provider invocation and the original prompt are independent
    # evidence of fresh generation; matching response prose is not enough.
    assert cached.cache_route == "miss"
    assert cached.cache_reads == []
    assert cached.cache_writes == []
    assert len(cached.prompts) == len(control.prompts) == 1
    assert cached.content == control.content
    assert cached.citations == control.citations
    if context_kind in {"facts", "history", "summary"}:
        assert _PREMIUM_FACT in "\n".join(m.content for m in cached.prompts[0][:-1])
        assert cached.content == _PREMIUM
        assert cached.history_reads == [session_id]
        if context_kind == "facts":
            assert cached.fact_reads == [42]
        else:
            assert cached.fact_reads == []
        if context_kind == "summary":
            assert any(
                m.role == "system" and m.content.startswith("Previous conversation:")
                for m in cached.prompts[0]
            )
    else:
        assert cached.history_reads
        error_type = "OperationalError" if context_kind == "db_error" else "RuntimeError"
        assert (session_id, error_type) in harness.history_errors
        assert (control_id, error_type) in harness.history_errors
    # Personalized generation must not replace the shared standard answer.
    assert len(harness.redis.store) == 1
    assert json.loads(next(iter(harness.redis.store.values())))["response"] == _STANDARD
