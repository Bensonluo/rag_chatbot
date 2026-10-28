"""Durable reply identities across real HTTP, graph, SQL and feedback paths.

Only the model provider and cache-eviction IO boundary are local substitutes.
SQLite foreign keys, JWT verification, compiled dialogue graph, persistence,
history, feedback repository and Prometheus counter are real.
"""

import asyncio
import json
import socket
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient, Response
from langgraph.checkpoint.memory import MemorySaver
from sqlalchemy import event, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from starlette.types import Message as ASGIMessage
from starlette.types import Receive, Scope, Send

from app.api.deps import get_db
from app.api.v1 import chat as chat_api
from app.api.v1 import feedback as feedback_api
from app.config.settings import settings
from app.core.security import create_access_token
from app.models.database.base import Base
from app.models.database.message import Message
from app.models.database.session import ChatSession
from app.models.database.user import User
from app.models.enums.message import MessageRole, MessageStatus
from app.services.chat import cache_eviction
from app.services.chat.chat_service import ChatService
from app.services.chat.metrics import FEEDBACK_RATINGS
from app.services.chat.persistence import ChatMessagePersister
from app.services.dialogue.graph import build_dialogue_graph
from app.services.dialogue.tools import ToolRegistry
from app.services.intent.rule_based import RuleBasedIntentDetector
from app.services.llm.base import LLMMessage, LLMResponse, LLMServiceBase

_ANSWER = "Identical local reply.\nA second line."


class _LocalProvider(LLMServiceBase):
    def __init__(self) -> None:
        super().__init__(api_key="offline", model="deterministic-feedback-evaluation")
        self.calls = 0
        self.fail = False
        self.overlap = False
        self.arrivals = asyncio.Event()

    async def _enter(self) -> None:
        self.calls += 1
        if self.overlap:
            if self.calls >= 2:
                self.arrivals.set()
            await asyncio.wait_for(self.arrivals.wait(), timeout=2)

    async def generate(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        await self._enter()
        return LLMResponse(content=_ANSWER, model=self.model)

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        await self._enter()
        yield "Identical local reply."
        if self.fail:
            # Exercise a cancelled producer through the compiled graph;
            # unlike a handled provider exception, this is a failed stream.
            raise asyncio.CancelledError("isolated provider cancellation")
        yield "\nA second line."

    def estimate_tokens(self, value: str) -> int:
        return len(value)

    async def count_tokens(self, messages: list[LLMMessage]) -> int:
        return sum(len(message.content) for message in messages)


@dataclass
class _Case:
    maker: async_sessionmaker[AsyncSession]
    client: AsyncClient
    provider: _LocalProvider
    evicted: list[str]
    headers: dict[int, dict[str, str]]
    delivered: list[dict[str, Any]]


@pytest.fixture
async def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncGenerator[_Case, None]:
    monkeypatch.setattr(settings, "DEMO_MODE", False)
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", False)
    monkeypatch.setattr(settings, "CHAT_TRACE_STREAM_ENABLED", False)
    monkeypatch.setattr(settings, "SECRET_KEY", "isolated-feedback-identity-signing-key-32bytes")
    attempts: list[str] = []

    def no_network(*args: Any, **kwargs: Any) -> None:
        attempts.append(repr(args[1:]))
        raise AssertionError("feedback identity evaluation must not open network connections")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", no_network)
    evicted: list[str] = []

    async def capture_eviction(content: str) -> int:
        evicted.append(content)
        return 1

    monkeypatch.setattr(cache_eviction, "evict_downvoted_response", capture_eviction)
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/feedback.sqlite3")

    @event.listens_for(engine.sync_engine, "connect")
    def foreign_keys(connection: Any, _record: Any) -> None:
        connection.execute("PRAGMA foreign_keys=ON")

    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(
            lambda conn: Base.metadata.create_all(
                conn,
                tables=[
                    Base.metadata.tables[name] for name in ("users", "chat_sessions", "messages")
                ],
            )
        )
    async with maker() as db:
        db.add_all(
            [
                User(id=1, email="owner@feedback.example.com", hashed_password="!"),
                User(id=2, email="other@feedback.example.com", hashed_password="!"),
                User(id=3, email="staff@feedback.example.com", hashed_password="!", is_admin=True),
            ]
        )
        await db.flush()
        db.add_all([ChatSession(id=1, user_id=1), ChatSession(id=2, user_id=1)])
        await db.commit()
    provider = _LocalProvider()
    service = ChatService(
        graph=build_dialogue_graph(
            intent_detector=RuleBasedIntentDetector(),
            slot_filler=None,
            tool_registry=ToolRegistry(),
            retrieval_pipeline=None,
            llm_service=provider,
            checkpointer=MemorySaver(),
        ),
        persister=ChatMessagePersister(maker),
    )
    app = FastAPI()
    app.include_router(chat_api.router, prefix="/api/v1")
    app.include_router(feedback_api.router, prefix="/api/v1")

    async def local_db() -> AsyncGenerator[AsyncSession, None]:
        async with maker() as db:
            yield db

    app.dependency_overrides[get_db] = local_db
    app.dependency_overrides[feedback_api._get_db] = local_db
    app.dependency_overrides[chat_api.get_chat_service] = lambda: service
    delivered: list[dict[str, Any]] = []

    async def observe_delivery(scope: Scope, receive: Receive, send: Send) -> None:
        async def observed_send(message: ASGIMessage) -> None:
            body = message.get("body", b"")
            if message["type"] == "http.response.body" and b"event: completion\n" in body:
                payload = json.loads(body.decode().split("data: ", 1)[1])
                # Query through a different SQL session before the client
                # receives this frame: an uncommitted flush is insufficient.
                async with maker() as db:
                    row = await db.get(Message, payload["message_id"])
                assert row is not None
                assert row.role == MessageRole.ASSISTANT
                assert row.status == MessageStatus.COMPLETED
                assert row.session_id == payload["session_id"]
                delivered.append(payload)
            await send(message)

        await app(scope, receive, observed_send)

    headers = {
        user_id: {
            "Authorization": "Bearer " + create_access_token({"sub": email, "user_id": user_id})
        }
        for user_id, email in (
            (1, "owner@feedback.example.com"),
            (2, "other@feedback.example.com"),
            (3, "staff@feedback.example.com"),
        )
    }
    try:
        async with AsyncClient(
            transport=ASGITransport(app=observe_delivery),
            base_url="http://isolated-feedback.invalid",
        ) as client:
            yield _Case(maker, client, provider, evicted, headers, delivered)
    finally:
        assert service.persister is not None
        await service.persister.drain()
        await engine.dispose()
        assert attempts == []


async def _rows(case: _Case) -> list[Message]:
    async with case.maker() as db:
        return list((await db.scalars(select(Message).order_by(Message.id))).all())


async def _chat(case: _Case, transport: str, session_id: int = 1, **options: Any) -> Response:
    return await case.client.post(
        "/api/v1/chat" + ("/stream" if transport == "stream" else ""),
        headers=case.headers[1],
        json={"session_id": session_id, "message": "hello", **options},
    )


def _events(response: Response) -> list[tuple[str, str]]:
    events = []
    for frame in response.text.replace("\r\n", "\n").split("\n\n"):
        lines = frame.split("\n")
        data = [line[5:].removeprefix(" ") for line in lines if line.startswith("data:")]
        if not data:
            continue
        name = next((line[6:].strip() for line in lines if line.startswith("event:")), "message")
        events.append((name, "\n".join(data)))
    return events


def _receipt(response: Response, transport: str) -> dict[str, Any]:
    assert response.status_code == 200
    if transport == "normal":
        payload = response.json()
        assert isinstance(payload, dict)
        return payload
    events = _events(response)
    assert events[-1] == ("message", "[DONE]")
    assert ("message", "[STREAM_ERROR]") not in events
    completions = [json.loads(value) for kind, value in events if kind == "completion"]
    assert len(completions) == 1
    assert events[-2][0] == "completion"
    result = completions[0]
    assert isinstance(result, dict)
    assert set(result) == {"message_id", "session_id", "status"}
    assert type(result["message_id"]) is int
    result["content"] = "".join(
        value for kind, value in events if kind == "message" and value != "[DONE]"
    )
    return result


async def _history(case: _Case, session_id: int = 1) -> list[dict[str, Any]]:
    response = await case.client.get(
        "/api/v1/chat/history", params={"session_id": session_id}, headers=case.headers[1]
    )
    assert response.status_code == 200
    messages = response.json()["messages"]
    assert isinstance(messages, list)
    return messages


async def _feedback(case: _Case, message_id: int, rating: int, user_id: int = 1) -> Response:
    return await case.client.post(
        "/api/v1/feedback",
        headers=case.headers[user_id],
        json={"message_id": message_id, "rating": rating, "text": "local feedback"},
    )


def _rating_counts() -> list[float]:
    return [FEEDBACK_RATINGS.labels(rating=str(value))._value.get() for value in (-1, 0, 1)]


async def test_rest_reply_identity_is_the_committed_assistant_not_a_text_match(case: _Case) -> None:
    response = await _chat(case, "normal")
    assert response.status_code == 200
    body = response.json()
    rows = await _rows(case)
    assert len(rows) == 2
    assert rows[1].role == MessageRole.ASSISTANT
    assert body.get("message_id") == rows[1].id
    assert body.get("status") == "completed"
    assert body["content"] == rows[1].content == _ANSWER
    assert case.provider.calls == 1


async def test_opted_in_stream_receipt_is_committed_before_delivery_and_matches_history(
    case: _Case,
) -> None:
    receipt = _receipt(await _chat(case, "stream", include_completion=True), "stream")
    rows = await _rows(case)
    assert receipt == {
        "message_id": rows[1].id,
        "session_id": 1,
        "status": "completed",
        "content": _ANSWER,
    }
    history = await _history(case)
    assert [entry["message_id"] for entry in history] == [row.id for row in rows]
    assert [entry["status"] for entry in history] == ["completed", "completed"]
    assert [entry["user_rating"] for entry in history] == [None, None]
    assert history[1]["content"] == rows[1].content == receipt["content"]
    assert case.delivered == [{key: value for key, value in receipt.items() if key != "content"}]


async def test_legacy_stream_without_opt_in_has_no_completion_event(case: _Case) -> None:
    response = await _chat(case, "stream")
    assert response.status_code == 200
    events = _events(response)
    assert events[-1] == ("message", "[DONE]")
    assert not any(kind == "completion" for kind, _ in events)
    assert case.delivered == []
    assert len(await _rows(case)) == 2


@pytest.mark.parametrize("transport", ["normal", "stream"])
async def test_identical_concurrent_answers_keep_distinct_durable_feedback_targets(
    case: _Case, transport: str
) -> None:
    case.provider.overlap = True
    responses = await asyncio.gather(
        _chat(case, transport, session_id=1, include_completion=True),
        _chat(case, transport, session_id=2, include_completion=True),
    )
    receipts = [_receipt(response, transport) for response in responses]
    assert case.provider.calls == 2
    assert receipts[0]["message_id"] != receipts[1]["message_id"]
    assert receipts[0]["content"] == receipts[1]["content"] == _ANSWER
    rows = await _rows(case)
    for session_id, receipt in enumerate(receipts, 1):
        saved = next(row for row in rows if row.id == receipt["message_id"])
        assert (saved.session_id, saved.role, saved.status, saved.content) == (
            session_id,
            MessageRole.ASSISTANT,
            MessageStatus.COMPLETED,
            _ANSWER,
        )
        history = await _history(case, session_id)
        assert history[-1]["message_id"] == saved.id
    accepted = await _feedback(case, receipts[0]["message_id"], -1)
    assert accepted.status_code == 200
    assert accepted.json()["message_id"] == receipts[0]["message_id"]
    rows = await _rows(case)
    assert next(row for row in rows if row.id == receipts[0]["message_id"]).user_rating == -1
    assert next(row for row in rows if row.id == receipts[1]["message_id"]).user_rating is None
    assert case.evicted == [_ANSWER]


@pytest.mark.parametrize("transport", ["normal", "stream"])
async def test_assistant_insert_rejection_has_no_receipt_or_orphan_user_turn(
    case: _Case, transport: str
) -> None:
    # Keep a previous real turn so a lookup of the latest matching answer
    # would produce a plausible, but incorrect, feedback target.
    await _chat(case, "normal")
    before = [(row.id, row.role, row.content) for row in await _rows(case)]
    async with case.maker() as db:
        await db.execute(
            text(
                "CREATE TRIGGER reject_assistant BEFORE INSERT ON messages "
                "WHEN lower(NEW.role) = 'assistant' BEGIN "
                "SELECT RAISE(ABORT, 'isolated assistant write rejected'); END"
            )
        )
        await db.commit()
    response = await _chat(case, transport, include_completion=True)
    assert response.status_code == 200
    if transport == "normal":
        assert response.json().get("message_id") is None
        assert response.json().get("status") is None
        assert response.json()["content"] == _ANSWER
    else:
        events = _events(response)
        assert events[-1] == ("message", "[DONE]")
        assert not any(kind == "completion" for kind, _ in events)
        assert "".join(value for kind, value in events[:-1] if kind == "message") == _ANSWER
        assert case.delivered == []
    assert [(row.id, row.role, row.content) for row in await _rows(case)] == before


@pytest.mark.parametrize("transport", ["normal", "stream"])
async def test_identical_overlapping_turns_in_one_session_keep_both_reply_ids(
    case: _Case, transport: str
) -> None:
    case.provider.overlap = True
    responses = await asyncio.gather(
        _chat(case, transport, include_completion=True),
        _chat(case, transport, include_completion=True),
    )
    receipts = [_receipt(response, transport) for response in responses]
    assert case.provider.calls == 2
    ids = {receipt["message_id"] for receipt in receipts}
    assert len(ids) == 2
    assistants = [row for row in await _rows(case) if row.role == MessageRole.ASSISTANT]
    assert {row.id for row in assistants} == ids
    assert [row.content for row in assistants] == [_ANSWER, _ANSWER]
    history = await _history(case)
    assert {entry["message_id"] for entry in history if entry["role"] == "assistant"} == ids
    assert (await _feedback(case, receipts[0]["message_id"], -1)).status_code == 200
    assistants = [row for row in await _rows(case) if row.role == MessageRole.ASSISTANT]
    ratings = {row.id: row.user_rating for row in assistants}
    assert ratings == {receipts[0]["message_id"]: -1, receipts[1]["message_id"]: None}


async def test_failed_partial_stream_has_no_completion_and_cannot_be_rated(case: _Case) -> None:
    case.provider.fail = True
    response = await _chat(case, "stream", include_completion=True)
    assert response.status_code == 200
    events = _events(response)
    assert ("message", "[STREAM_ERROR]") in events
    assert events[-1] == ("message", "[DONE]")
    assert not any(kind == "completion" for kind, _ in events)
    rows = await _rows(case)
    assert len(rows) == 2
    assistant = rows[1]
    assert assistant.status == MessageStatus.FAILED
    assert assistant.content == "Identical local reply."
    history = await _history(case)
    assert history[-1]["message_id"] == assistant.id
    assert history[-1]["status"] == "failed"
    before = _rating_counts()
    assert (await _feedback(case, assistant.id, -1)).status_code == 404
    assert _rating_counts() == before
    assert case.evicted == []
    assert case.delivered == []


async def test_accepted_downvote_then_zero_survives_database_and_history(case: _Case) -> None:
    receipt = _receipt(await _chat(case, "normal"), "normal")
    before = _rating_counts()
    for value in (-1, 0, 1):
        response = await _feedback(case, receipt["message_id"], value)
        assert response.status_code == 200
        assert response.json() == {
            "success": True,
            "message_id": receipt["message_id"],
            "rating": value,
        }
        saved = next(row for row in await _rows(case) if row.id == receipt["message_id"])
        assert saved.user_rating == value
        assert saved.feedback_text == "local feedback"
        assert (await _history(case))[-1]["user_rating"] == value
    assert _rating_counts() == [value + 1 for value in before]
    assert case.evicted == [_ANSWER]


@pytest.mark.parametrize(
    ("role", "status"),
    [
        (MessageRole.USER, MessageStatus.COMPLETED),
        (MessageRole.SYSTEM, MessageStatus.COMPLETED),
        (MessageRole.ASSISTANT, MessageStatus.PENDING),
        (MessageRole.ASSISTANT, MessageStatus.PROCESSING),
        (MessageRole.ASSISTANT, MessageStatus.FAILED),
    ],
)
async def test_noncompleted_or_nonassistant_rows_are_not_feedback_targets(
    case: _Case, role: MessageRole, status: MessageStatus
) -> None:
    async with case.maker() as db:
        row = Message(session_id=1, role=role, status=status, content="Ineligible local row")
        db.add(row)
        await db.commit()
        message_id = row.id
    before = _rating_counts()
    response = await _feedback(case, message_id, -1)
    assert response.status_code == 404
    assert response.json() == {"detail": f"Message {message_id} not found"}
    saved = (await _rows(case))[0]
    assert saved.user_rating is None
    assert saved.feedback_text is None
    assert _rating_counts() == before
    assert case.evicted == []


@pytest.mark.parametrize("identity", ["foreign", "nonowner_admin", "missing"])
async def test_foreign_admin_and_missing_feedback_targets_share_the_not_found_contract(
    case: _Case, identity: str
) -> None:
    receipt = _receipt(await _chat(case, "normal"), "normal")
    user_id = {"foreign": 2, "nonowner_admin": 3, "missing": 1}[identity]
    message_id = receipt["message_id"] if identity != "missing" else 999999
    before = _rating_counts()
    response = await _feedback(case, message_id, -1, user_id)
    assert response.status_code == 404
    assert response.json() == {"detail": f"Message {message_id} not found"}
    assert all(row.user_rating is None for row in await _rows(case))
    assert _rating_counts() == before
    assert case.evicted == []


@pytest.mark.parametrize("rating", [-2, 2])
async def test_out_of_range_feedback_is_rejected_without_side_effects(
    case: _Case, rating: int
) -> None:
    receipt = _receipt(await _chat(case, "normal"), "normal")
    before = _rating_counts()
    assert (await _feedback(case, receipt["message_id"], rating)).status_code == 422
    assert all(row.user_rating is None for row in await _rows(case))
    assert _rating_counts() == before
    assert case.evicted == []


async def test_feedback_requires_an_actual_authenticated_user(case: _Case) -> None:
    receipt = _receipt(await _chat(case, "normal"), "normal")
    before = _rating_counts()
    response = await case.client.post(
        "/api/v1/feedback", json={"message_id": receipt["message_id"], "rating": -1}
    )
    assert response.status_code == 401
    assert all(row.user_rating is None for row in await _rows(case))
    assert _rating_counts() == before
    assert case.evicted == []
