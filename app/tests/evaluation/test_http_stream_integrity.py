"""Real authenticated HTTP streams preserve guarded text and durable citations."""

import json
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from langgraph.checkpoint.memory import MemorySaver
from redis.exceptions import ConnectionError as RedisConnectionError
from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.api.deps import get_db
from app.api.v1 import chat as chat_api
from app.config.settings import settings
from app.core.security import create_access_token
from app.middleware.rate_limiter_redis import EndpointRateLimiter, _RedisClientHolder
from app.models.database.base import Base
from app.models.database.message import Message
from app.models.database.session import ChatSession
from app.models.database.user import User
from app.services.chat.chat_service import ChatService
from app.services.chat.persistence import ChatMessagePersister
from app.services.dialogue.graph import build_dialogue_graph
from app.services.dialogue.tools import ToolRegistry
from app.services.guardrails.factory import GuardrailFactory
from app.services.intent.rule_based import RuleBasedIntentDetector
from app.services.llm.base import LLMMessage, LLMResponse, LLMServiceBase
from app.services.retrieval.hybrid_search import KeywordSearch


class _LocalLLM(LLMServiceBase):
    """Deterministic chunks at the provider boundary; no external model calls."""

    def __init__(self, chunks: list[str]) -> None:
        super().__init__(api_key="isolated-fixture", model="isolated-fixture")
        self.chunks = chunks

    async def generate(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        return LLMResponse(content="".join(self.chunks), model=self.model)

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        for chunk in self.chunks:
            yield chunk

    def estimate_tokens(self, text: str) -> int:
        return len(text)

    async def count_tokens(self, messages: list[LLMMessage]) -> int:
        return sum(len(message.content) for message in messages)


def _decode_events(body: str) -> list[tuple[str, str]]:
    """Decode SSE data fields as a consumer, preserving payload whitespace."""
    events = []
    for frame in body.split("\n\n"):
        event_type = "message"
        data = []
        for line in frame.split("\n"):
            if line.startswith("event:"):
                event_type = line[6:].removeprefix(" ")
            elif line.startswith("data:"):
                data.append(line[5:].removeprefix(" "))
        if data:
            events.append((event_type, "\n".join(data)))
    return events


@pytest.mark.parametrize(
    "message,chunks,expected,cited",
    [
        ("你好", ["Hello", " there."], "Hello there.", False),
        ("你好", ["第一句。", "没有句末标点"], "第一句。没有句末标点", False),
        ("你好", ["Contact sample@", "example.invalid"], "Contact [REDACTED]", False),
        (
            "你好",
            ["Line one\n", "\n", "Line two\n", "Final."],
            "Line one\n\nLine two\nFinal.",
            False,
        ),
        (
            "What warranty coverage is included?",
            ["Coverage includes ", "parts and labor."],
            "Coverage includes parts and labor.",
            True,
        ),
    ],
    ids=["english-period", "chinese-tail", "redacted-tail", "newlines", "rag-citation"],
)
async def test_http_stream_matches_persisted_guarded_answer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    message: str,
    chunks: list[str],
    expected: str,
    cited: bool,
) -> None:
    monkeypatch.setattr(settings, "DEMO_MODE", False)
    monkeypatch.setattr(settings, "CHAT_TRACE_STREAM_ENABLED", False)
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(settings, "SECRET_KEY", "isolated-http-stream-integrity-key-32bytes")

    def unavailable_redis() -> None:
        raise RedisConnectionError("isolated HTTP stream test uses local fallback")

    monkeypatch.setattr(_RedisClientHolder, "get", unavailable_redis)
    monkeypatch.setattr(
        chat_api,
        "_chat_rate_limiter",
        EndpointRateLimiter(scope="http-stream-integrity", requests_per_minute=10),
    )
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/stream.sqlite3")

    @event.listens_for(engine.sync_engine, "connect")
    def enable_foreign_keys(connection: Any, _record: Any) -> None:
        connection.execute("PRAGMA foreign_keys=ON")

    maker = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(
                lambda conn: Base.metadata.create_all(
                    conn,
                    tables=[
                        Base.metadata.tables[name]
                        for name in ("users", "chat_sessions", "messages", "handoff_tickets")
                    ],
                )
            )
        async with maker() as session:
            session.add(User(id=1, email="owner@stream.invalid", hashed_password="!"))
            await session.flush()
            session.add(ChatSession(id=1, user_id=1))
            await session.commit()

        search = KeywordSearch()
        await search.add_documents(
            [
                {
                    "id": "warranty-doc",
                    "content": "Warranty coverage includes parts and labor.",
                    "metadata": {
                        "version_id": "warranty-v1",
                        "chunk_id": "warranty-chunk-1",
                        "title": "Warranty coverage",
                    },
                }
            ]
        )
        service = ChatService(
            graph=build_dialogue_graph(
                intent_detector=RuleBasedIntentDetector(),
                slot_filler=None,
                tool_registry=ToolRegistry(),
                retrieval_pipeline={"hybrid_search": search},
                llm_service=_LocalLLM(chunks),
                guardrail_service=GuardrailFactory.create(),
                checkpointer=MemorySaver(),
            ),
            persister=ChatMessagePersister(maker),
        )

        async def isolated_db() -> AsyncGenerator[AsyncSession, None]:
            async with maker() as session:
                yield session

        app = FastAPI()
        app.include_router(chat_api.router, prefix="/api/v1")
        app.dependency_overrides[get_db] = isolated_db
        app.dependency_overrides[chat_api.get_chat_service] = lambda: service
        token = create_access_token({"sub": "owner@stream.invalid", "user_id": 1})
        async with AsyncClient(
            transport=ASGITransport(app), base_url="http://isolated-stream.invalid"
        ) as client:
            response = await client.post(
                "/api/v1/chat/stream",
                headers={"Authorization": f"Bearer {token}"},
                json={"session_id": 1, "message": message, "include_citations": True},
            )
        assert response.status_code == 200, response.text
        events = _decode_events(response.text)
        assert events[-1] == ("message", "[DONE]")
        assert not any(payload == "[STREAM_ERROR]" for _, payload in events)
        text = "".join(
            payload for kind, payload in events if kind == "message" and payload != "[DONE]"
        )
        assert text == expected
        assert "sample@example.invalid" not in response.text
        citations = [
            json.loads(payload)["citations"] for kind, payload in events if kind == "citations"
        ]
        assert len(citations) == 1
        assert bool(citations[0]) is cited
        if cited:
            assert citations[0][0]["document_id"] == "warranty-doc"
            assert citations[0][0]["version_id"] == "warranty-v1"

        async with maker() as session:
            rows = (await session.scalars(select(Message).order_by(Message.id))).all()
        assert len(rows) == 2
        assert rows[0].content == message
        assert rows[1].content == text
        assert rows[1].status.value == "completed"
        assert (rows[1].message_metadata or {}).get("citations", []) == citations[0]
        history = await service.get_chat_history(1)
        assert history[-1].content == text
        assert history[-1].citations == citations[0]
    finally:
        await engine.dispose()
