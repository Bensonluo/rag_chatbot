"""Authenticated human conversations remain separate from bot replies.

JWT, HTTP routes, SQLite foreign keys, graph, tools and persistence are real.
Only Redis transport and model/network boundaries are local. SQLite checks
the single-process coordinator; it cannot validate PostgreSQL process locks.
"""

import asyncio
import socket
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient, Response
from langgraph.checkpoint.memory import MemorySaver
from redis.asyncio import Redis
from sqlalchemy import event, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from starlette.types import Message as ASGIMessage
from starlette.types import Receive, Scope, Send

from app.api.deps import get_db
from app.api.v1 import chat as chat_api
from app.api.v1 import feedback as feedback_api
from app.api.v1 import handoff as handoff_api
from app.api.v1 import sessions as sessions_api
from app.config.settings import settings
from app.core.security import create_access_token
from app.models.database.base import Base
from app.models.database.handoff_message import HandoffMessage
from app.models.database.message import Message
from app.models.database.session import ChatSession
from app.models.database.ticket import HandoffTicket
from app.models.database.user import User
from app.models.enums.message import MessageRole, MessageStatus
from app.services.chat import answer_cache, cache_eviction
from app.services.chat.answer_cache import AnswerCacheService
from app.services.chat.chat_service import ChatService
from app.services.chat.coordination import SessionCoordinator, get_session_coordinator
from app.services.chat.metrics import FEEDBACK_RATINGS
from app.services.chat.persistence import ChatMessagePersister
from app.services.dialogue.graph import build_dialogue_graph
from app.services.dialogue.tools import create_default_tool_registry
from app.services.handoff.service import HandoffService
from app.services.intent.rule_based import RuleBasedIntentDetector
from app.tests.unit.services.chat.test_cache_eviction import _FakeRedis


@dataclass
class _Calls:
    graph: int = 0
    cache_epoch: int = 0
    refunds: list[dict[str, Any]] = field(default_factory=list)
    evictions: list[str] = field(default_factory=list)
    tool_users: list[int | None] = field(default_factory=list)


@dataclass
class _WriterGate:
    pause_next: bool = False
    entered: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class _StreamGate:
    pause_next: bool = False
    entered: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class _Case:
    maker: async_sessionmaker[AsyncSession]
    client: AsyncClient
    chat: ChatService
    handoff: HandoffService
    private_id: int
    headers: dict[int, dict[str, str]]
    calls: _Calls
    registry: Any
    coordinator: SessionCoordinator
    writer_gate: _WriterGate
    stream_gate: _StreamGate


@pytest.fixture
async def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncGenerator[_Case, None]:
    monkeypatch.setattr(settings, "DEMO_MODE", False)
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", False)
    monkeypatch.setattr(settings, "CHAT_TRACE_STREAM_ENABLED", False)
    monkeypatch.setattr(settings, "SECRET_KEY", "isolated-human-conversation-signing-key-32bytes")
    attempts: list[str] = []

    def no_network(*args: Any, **kwargs: Any) -> None:
        attempts.append(repr(args[1:]))
        raise AssertionError("human conversation evaluation forbids external sockets")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", no_network)
    calls = _Calls()

    async def epoch() -> str:
        calls.cache_epoch += 1
        return "local-human-conversation-epoch"

    async def eviction(content: str) -> int:
        calls.evictions.append(content)
        return 1

    monkeypatch.setattr(answer_cache, "get_kb_epoch", epoch)
    monkeypatch.setattr(cache_eviction, "evict_downvoted_response", eviction)
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/human.sqlite3")

    @event.listens_for(engine.sync_engine, "connect")
    def foreign_keys(connection: Any, _record: Any) -> None:
        connection.execute("PRAGMA foreign_keys=ON")

    writer_gate = _WriterGate()

    class ObservedSession(AsyncSession):
        async def flush(self, objects: Any = None) -> None:
            observe = writer_gate.pause_next and any(
                isinstance(item, HandoffMessage) for item in self.new
            )
            await super().flush(objects)
            if observe:
                # Timing-only barrier after real INSERT, before commit.
                writer_gate.pause_next = False
                writer_gate.entered.set()
                await asyncio.wait_for(writer_gate.release.wait(), timeout=5)

    maker: async_sessionmaker[AsyncSession] = async_sessionmaker(
        engine, class_=ObservedSession, expire_on_commit=False
    )
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    emails = {index: f"human-{index}@evaluation.example.com" for index in range(1, 5)}
    async with maker() as db:
        db.add_all(
            User(id=index, email=email, hashed_password="!", is_admin=index in (3, 4))
            for index, email in emails.items()
        )
        await db.flush()
        db.add_all([ChatSession(id=10, user_id=1), ChatSession(id=20, user_id=2)])
        await db.commit()
    registry = create_default_tool_registry()
    original_execute = registry.execute

    async def recorded_execute(*args: Any, **kwargs: Any) -> Any:
        calls.tool_users.append(kwargs.get("user_id"))
        return await original_execute(*args, **kwargs)

    monkeypatch.setattr(registry, "execute", recorded_execute)
    refund = registry.get_tool_for_intent("refund")
    assert refund is not None
    original_refund = refund.handler

    def recorded_refund(args: dict[str, Any]) -> Any:
        calls.refunds.append(dict(args))
        return original_refund(args)

    monkeypatch.setattr(refund, "handler", recorded_refund)
    coordinator = SessionCoordinator(maker)
    handoff = HandoffService(maker, coordinator=coordinator)
    cache = AnswerCacheService(redis_url="unused", ttl_seconds=60)
    cache._redis = cast("Redis[str]", _FakeRedis())
    graph = build_dialogue_graph(
        intent_detector=RuleBasedIntentDetector(),
        slot_filler=None,
        tool_registry=registry,
        retrieval_pipeline=None,
        llm_service=None,
        handoff_service=handoff,
        answer_cache=cache,
        checkpointer=MemorySaver(),
    )
    original_invoke = graph.ainvoke

    async def tracked_invoke(*args: Any, **kwargs: Any) -> Any:
        calls.graph += 1
        return await original_invoke(*args, **kwargs)

    monkeypatch.setattr(graph, "ainvoke", tracked_invoke)
    chat = ChatService(
        graph=graph,
        persister=ChatMessagePersister(maker),
        answer_cache=cache,
        coordinator=coordinator,
    )
    app = FastAPI()
    for router in (chat_api.router, handoff_api.router, sessions_api.router, feedback_api.router):
        app.include_router(router, prefix="/api/v1")

    async def local_db() -> AsyncGenerator[AsyncSession, None]:
        async with maker() as db:
            yield db

    app.dependency_overrides[get_db] = local_db
    app.dependency_overrides[feedback_api._get_db] = local_db
    app.dependency_overrides[chat_api.get_chat_service] = lambda: chat
    app.dependency_overrides[handoff_api.get_handoff_service] = lambda: handoff
    app.dependency_overrides[get_session_coordinator] = lambda: coordinator
    app.dependency_overrides[sessions_api.get_session_chat_service] = lambda: chat
    stream_gate = _StreamGate()

    async def observed_app(scope: Scope, receive: Receive, send: Send) -> None:
        async def observed_send(message: ASGIMessage) -> None:
            if (
                scope.get("path") == "/api/v1/chat/stream"
                and message["type"] == "http.response.start"
                and stream_gate.pause_next
            ):
                stream_gate.pause_next = False
                stream_gate.entered.set()
                await asyncio.wait_for(stream_gate.release.wait(), timeout=5)
            await send(message)

        await app(scope, receive, observed_send)

    headers = {
        index: {"Authorization": "Bearer " + create_access_token({"sub": email, "user_id": index})}
        for index, email in emails.items()
    }
    try:
        async with AsyncClient(
            transport=ASGITransport(app=observed_app), base_url="http://human-evaluation.invalid"
        ) as client:
            response = await client.post(
                "/api/v1/sessions", headers=headers[1], json={"title": "Private human conversation"}
            )
            assert response.status_code == 200
            yield _Case(
                maker,
                client,
                chat,
                handoff,
                response.json()["id"],
                headers,
                calls,
                registry,
                coordinator,
                writer_gate,
                stream_gate,
            )
    finally:
        writer_gate.release.set()
        stream_gate.release.set()
        assert chat.persister is not None
        await chat.persister.drain()
        await engine.dispose()
        assert attempts == []


async def test_new_session_exposes_the_private_conversation_contract(case: _Case) -> None:
    response = await _get(case, f"/api/v1/sessions/{case.private_id}", headers=case.headers[1])
    assert response.status_code == 200
    assert response.json().get("is_private") is True


async def _session_key(case: _Case, session_id: int | None = None) -> str:
    async with case.maker() as db:
        chat = await db.get(ChatSession, session_id or case.private_id)
        return chat.session_key if chat is not None else "00000000-0000-4000-8000-000000000000"


async def _url_session_key(case: _Case, url: str) -> str:
    import re

    match = re.search(r"/handoff/(sessions|tickets)/(\d+)", url)
    session_id = case.private_id
    if match is not None:
        identifier = int(match[2])
        if match[1] == "sessions":
            session_id = identifier
        else:
            async with case.maker() as db:
                ticket = await db.get(HandoffTicket, identifier)
                if ticket is not None:
                    session_id = ticket.session_id
    return await _session_key(case, session_id)


async def _get(case: _Case, url: str, **kwargs: Any) -> Response:
    # Existing positive commands explicitly discover the current object; negative
    # incarnation tests issue raw HTTP without this convenience helper.
    from urllib.parse import parse_qsl, urlsplit

    params = dict(parse_qsl(urlsplit(url).query))
    params.update(kwargs.pop("params", {}))
    if "expected_session_key" not in params:
        session_id = params.get("session_id")
        params["expected_session_key"] = (
            await _session_key(case, int(session_id))
            if session_id is not None
            else await _url_session_key(case, url)
        )
    return await case.client.get(url, params=params, **kwargs)


async def _delete(case: _Case, url: str, **kwargs: Any) -> Response:
    separator = "&" if "?" in url else "?"
    return await case.client.delete(
        url + separator + "expected_session_key=" + await _url_session_key(case, url), **kwargs
    )


def _customer_url(case: _Case) -> str:
    return f"/api/v1/handoff/sessions/{case.private_id}/messages"


def _staff_url(ticket_id: int) -> str:
    return f"/api/v1/handoff/tickets/{ticket_id}/messages"


def _no_store(response: Response) -> None:
    assert "no-store" in response.headers.get("cache-control", "")


async def _human_rows(case: _Case) -> list[HandoffMessage]:
    async with case.maker() as db:
        return list((await db.scalars(select(HandoffMessage).order_by(HandoffMessage.id))).all())


async def _bot_rows(case: _Case) -> list[Message]:
    async with case.maker() as db:
        return list((await db.scalars(select(Message).order_by(Message.id))).all())


async def _ticket(case: _Case, *, claim: bool = True) -> int:
    created = await case.handoff.create_ticket_for_session(
        case.private_id, 1, "explicit", {"user_message": "human agent"}
    )
    ticket_id = created["ticket_id"]
    assert type(ticket_id) is int
    if claim:
        response = await case.client.post(
            f"/api/v1/handoff/tickets/{ticket_id}/claim?expected_session_key={await _url_session_key(case, _staff_url(ticket_id))}",
            headers=case.headers[3],
        )
        assert response.status_code == 200
    return ticket_id


async def _bot(
    case: _Case,
    message: str,
    transport: str = "normal",
    *,
    session_id: int | None = None,
    actor: int = 1,
    **options: Any,
) -> Response:
    async with case.maker() as db:
        current = await db.get(ChatSession, session_id or case.private_id)
        session_key = current.session_key if current is not None else None
    return await case.client.post(
        "/api/v1/chat" + ("/stream" if transport == "stream" else ""),
        headers=case.headers.get(actor, {}),
        json={
            "session_id": session_id or case.private_id,
            "message": message,
            "expected_session_key": session_key,
            **options,
        },
    )


async def _send(
    case: _Case,
    url: str,
    actor: int,
    content: str,
    *,
    key: str | None = None,
    resolve: bool | None = None,
) -> Response:
    body: dict[str, Any] = {
        "client_message_id": key or str(uuid4()),
        "content": content,
        "expected_session_key": await _url_session_key(case, url),
    }
    if resolve is not None:
        body["resolve"] = resolve
    return await case.client.post(url, headers=case.headers.get(actor, {}), json=body)


def _ack(response: Response) -> dict[str, Any]:
    assert response.status_code == 200, response.text
    _no_store(response)
    body = response.json()
    assert isinstance(body, dict)
    message = body["message"]
    assert type(message["id"]) is int
    assert message["id"] > 0
    assert "message_id" not in message
    assert "status" not in message
    assert "user_rating" not in message
    assert message["author_kind"] in {"customer", "staff"}
    assert isinstance(message["timestamp"], str)
    return body


async def _feed(
    case: _Case, url: str, actor: int, *, after_id: int = 0, limit: int = 100
) -> dict[str, Any]:
    response = await _get(
        case, url, headers=case.headers[actor], params={"after_id": after_id, "limit": limit}
    )
    assert response.status_code == 200, response.text
    _no_store(response)
    body = response.json()
    assert isinstance(body, dict)
    return body


def _calls(case: _Case) -> tuple[int, int, int]:
    return case.calls.graph, case.calls.cache_epoch, len(case.calls.refunds)


@pytest.mark.parametrize("language", ["zh", "en"])
@pytest.mark.parametrize("transport", ["normal", "stream"])
async def test_claimed_conversation_delivers_both_authors_without_bot_execution(
    case: _Case, language: str, transport: str
) -> None:
    initial = await _bot(case, "human agent")
    assert initial.status_code == 200
    ticket_id = initial.json()["metadata"]["handoff_ticket_id"]
    assert (
        await case.client.post(
            f"/api/v1/handoff/tickets/{ticket_id}/claim?expected_session_key={await _url_session_key(case, _staff_url(ticket_id))}",
            headers=case.headers[3],
        )
    ).status_code == 200
    bot_before = [(row.id, row.content) for row in await _bot_rows(case)]
    counts = _calls(case)
    customer_text, staff_text = (
        ("请人工核实退款，不要自动办理。", "我已收到您的说明，正在核实。")
        if language == "zh"
        else ("Please review my refund manually.", "I have received your note and will review it.")
    )
    customer = _ack(await _send(case, _customer_url(case), 1, customer_text))
    staff_feed = await _feed(case, _staff_url(ticket_id), 3)
    assert staff_feed["messages"] == [customer["message"]]
    staff = _ack(await _send(case, _staff_url(ticket_id), 3, staff_text))
    owner_feed = await _feed(case, _customer_url(case), 1)
    assert owner_feed["messages"] == [customer["message"], staff["message"]]
    assert [row["author_kind"] for row in owner_feed["messages"]] == ["customer", "staff"]
    assert [row["author_user_id"] for row in owner_feed["messages"]] == [1, 3]
    assert owner_feed["ticket"]["status"] == "claimed"
    assert owner_feed["ticket"]["assigned_to"] == 3
    rows = await _human_rows(case)
    assert [(row.id, row.content, row.author_user_id) for row in rows] == [
        (customer["message"]["id"], customer_text, 1),
        (staff["message"]["id"], staff_text, 3),
    ]
    assert all(row.session_id == case.private_id and row.ticket_id == ticket_id for row in rows)
    rejected = await _bot(case, "confirm", transport, include_completion=True)
    assert rejected.status_code == 423
    assert "[DONE]" not in rejected.text
    assert _calls(case) == counts
    assert [(row.id, row.content) for row in await _bot_rows(case)] == bot_before
    history = await _get(
        case,
        "/api/v1/chat/history",
        params={"session_id": case.private_id},
        headers=case.headers[1],
    )
    assert history.status_code == 200
    assert [row["content"] for row in history.json()["messages"]] == [
        item[1] for item in bot_before
    ]


async def test_final_staff_reply_and_resolve_are_atomic_and_idempotent(case: _Case) -> None:
    ticket_id = await _ticket(case)
    key = str(uuid4())
    first = _ack(
        await _send(case, _staff_url(ticket_id), 3, "Review complete.", key=key, resolve=True)
    )
    assert first["reused"] is False
    assert first["ticket"]["status"] == "resolved"
    async with case.maker() as db:
        session = await db.get(ChatSession, case.private_id)
        assert session is not None
        assert session.automation_epoch == 1
    again = _ack(
        await _send(case, _staff_url(ticket_id), 3, "Review complete.", key=key, resolve=True)
    )
    assert again["reused"] is True
    assert again["message"] == first["message"]
    assert len(await _human_rows(case)) == 1
    async with case.maker() as db:
        session = await db.get(ChatSession, case.private_id)
        assert session is not None
        assert session.automation_epoch == 1
    conflict = await _send(case, _staff_url(ticket_id), 3, "Changed reply.", key=key, resolve=True)
    assert conflict.status_code == 409
    assert (
        await _send(case, _staff_url(ticket_id), 3, "Review complete.", key=key, resolve=False)
    ).status_code == 409
    assert (await _send(case, _staff_url(ticket_id), 3, "New post-close reply.")).status_code == 404
    assert (
        await _send(case, _staff_url(ticket_id), 4, "Review complete.", key=key, resolve=True)
    ).status_code == 404
    page = await _feed(case, _customer_url(case), 1)
    assert page["messages"] == [first["message"]]
    assert page["ticket"]["status"] == "resolved"


async def test_staff_insert_refusal_rolls_back_final_reply_close_and_epoch(case: _Case) -> None:
    ticket_id = await _ticket(case)
    key = str(uuid4())
    async with case.maker() as db:
        await db.execute(
            text(
                "CREATE TRIGGER reject_staff BEFORE INSERT ON handoff_messages "
                "WHEN NEW.author_kind = 'staff' BEGIN "
                "SELECT RAISE(ABORT, 'isolated staff write rejected'); END"
            )
        )
        await db.commit()
    response = await _send(case, _staff_url(ticket_id), 3, "Final reply.", key=key, resolve=True)
    assert response.status_code == 503
    assert "isolated staff write rejected" not in response.text
    assert await _human_rows(case) == []
    async with case.maker() as db:
        ticket = await db.get(HandoffTicket, ticket_id)
        session = await db.get(ChatSession, case.private_id)
        assert ticket is not None and ticket.status == "claimed"
        assert session is not None and session.automation_epoch == 0
        await db.execute(text("DROP TRIGGER reject_staff"))
        await db.commit()
    assert (
        _ack(await _send(case, _staff_url(ticket_id), 3, "Final reply.", key=key, resolve=True))[
            "reused"
        ]
        is False
    )


async def test_cursor_pages_are_stable_and_same_text_has_distinct_ids(case: _Case) -> None:
    ticket_id = await _ticket(case)
    sent = [
        _ack(await _send(case, url, actor, "Identical text."))["message"]
        for url, actor in [
            (_customer_url(case), 1),
            (_staff_url(ticket_id), 3),
            (_customer_url(case), 1),
        ]
    ]
    assert len({message["id"] for message in sent}) == 3
    cursor = 0
    received = []
    for index in range(3):
        page = await _feed(case, _customer_url(case), 1, after_id=cursor, limit=1)
        assert page["messages"] == [sent[index]]
        assert page["has_more"] is (index < 2)
        assert page["next_after_id"] > cursor
        cursor = page["next_after_id"]
        received.extend(page["messages"])
    assert received == sent
    empty = await _feed(case, _customer_url(case), 1, after_id=cursor, limit=1)
    assert empty["messages"] == []
    assert empty["next_after_id"] == cursor
    assert empty["has_more"] is False


@pytest.mark.parametrize("demo", [False, True])
@pytest.mark.parametrize("actor", [0, 2, 3])
async def test_customer_channel_requires_the_actual_owner_in_both_modes(
    case: _Case, monkeypatch: pytest.MonkeyPatch, demo: bool, actor: int
) -> None:
    await _ticket(case)
    monkeypatch.setattr(settings, "DEMO_MODE", demo)
    for response in [
        await _get(case, _customer_url(case), headers=case.headers.get(actor, {})),
        await _send(case, _customer_url(case), actor, "Unauthorized note."),
    ]:
        assert response.status_code == (401 if actor == 0 else 404)
        _no_store(response)
    assert await _human_rows(case) == []


@pytest.mark.parametrize("actor", [0, 1, 4])
async def test_staff_channel_requires_the_assigned_authenticated_admin(
    case: _Case, actor: int
) -> None:
    ticket_id = await _ticket(case)
    for response in [
        await _get(case, _staff_url(ticket_id), headers=case.headers.get(actor, {})),
        await _send(case, _staff_url(ticket_id), actor, "Unauthorized reply."),
    ]:
        assert response.status_code == {0: 401, 1: 403, 4: 404}[actor]
    assert await _human_rows(case) == []


async def test_open_conversation_accepts_customer_notes_but_only_claimed_staff_can_reply(
    case: _Case,
) -> None:
    ticket_id = await _ticket(case, claim=False)
    note = _ack(await _send(case, _customer_url(case), 1, "Please review this."))
    assert note["ticket"]["status"] == "open"
    assert (await _send(case, _staff_url(ticket_id), 3, "Unclaimed reply.")).status_code == 404
    assert (await _bot(case, "hello")).status_code == 200
    assert case.calls.graph == 1
    assert len(await _human_rows(case)) == 1


@pytest.mark.parametrize("actor", [0, 2, 3])
@pytest.mark.parametrize("endpoint", ["chat", "stream", "history", "clear"])
async def test_demo_mode_does_not_bypass_private_chat_history_or_clear(
    case: _Case, monkeypatch: pytest.MonkeyPatch, actor: int, endpoint: str
) -> None:
    assert (await _bot(case, "hello")).status_code == 200
    before = [(row.id, row.content) for row in await _bot_rows(case)]
    counts = _calls(case)
    monkeypatch.setattr(settings, "DEMO_MODE", True)
    if endpoint in {"chat", "stream"}:
        response = await _bot(
            case,
            "confirm",
            "stream" if endpoint == "stream" else "normal",
            actor=actor,
            user_id=1,
        )
    else:
        response = await case.client.request(
            "DELETE" if endpoint == "clear" else "GET",
            "/api/v1/chat/history",
            headers=case.headers.get(actor, {}),
            params={"session_id": case.private_id},
        )
    assert response.status_code == (401 if actor == 0 else 404)
    assert _calls(case) == counts
    assert [(row.id, row.content) for row in await _bot_rows(case)] == before


async def test_private_demo_tools_cannot_use_a_forged_body_identity(
    case: _Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "DEMO_MODE", True)
    response = await _bot(case, "query order ORD2001", user_id=2)
    assert response.status_code in {200, 403}
    if response.status_code == 200:
        assert case.calls.tool_users == [1]
        users = [row for row in await _bot_rows(case) if row.role == MessageRole.USER]
        assert users
        assert all((row.message_metadata or {}).get("user_id") == 1 for row in users)
    else:
        assert case.calls.tool_users == []


async def test_legacy_unmarked_demo_chat_keeps_its_existing_shared_behavior(
    case: _Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "DEMO_MODE", True)
    response = await _bot(case, "hello", session_id=10, actor=0, user_id=1)
    assert response.status_code == 200
    history = await _get(case, "/api/v1/chat/history", params={"session_id": 10})
    assert history.status_code == 200
    assert len(history.json()["messages"]) == 2


@pytest.mark.parametrize("transport", ["normal", "stream"])
async def test_inflight_bot_excludes_claim_until_the_turn_and_side_effect_settle(
    case: _Case, monkeypatch: pytest.MonkeyPatch, transport: str
) -> None:
    ticket_id = await _ticket(case, claim=False)
    for message in ["I want a refund", "ORD1001", "damaged item"]:
        assert (await _bot(case, message)).status_code == 200
    refund = case.registry.get_tool_for_intent("refund")
    assert refund is not None
    original = refund.handler
    entered, release = asyncio.Event(), asyncio.Event()

    async def paused(args: dict[str, Any]) -> Any:
        entered.set()
        await asyncio.wait_for(release.wait(), timeout=5)
        return original(args)

    monkeypatch.setattr(refund, "handler", paused)
    bot = asyncio.create_task(_bot(case, "confirm", transport, include_completion=True))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        claim = await case.client.post(
            f"/api/v1/handoff/tickets/{ticket_id}/claim?expected_session_key={await _url_session_key(case, _staff_url(ticket_id))}",
            headers=case.headers[3],
        )
        assert claim.status_code == 409
        clear = await _delete(
            case,
            "/api/v1/chat/history",
            headers=case.headers[1],
            params={"session_id": case.private_id},
        )
        assert clear.status_code == 409
        assert not bot.done()
        assert case.calls.refunds == []
        release.set()
        response = await asyncio.wait_for(bot, timeout=5)
        assert response.status_code == 200
        if transport == "stream":
            assert "data: [DONE]" in response.text
        assert len(case.calls.refunds) == 1
    finally:
        release.set()
        if not bot.done():
            bot.cancel()
        await asyncio.gather(bot, return_exceptions=True)
    assert (
        await case.client.post(
            f"/api/v1/handoff/tickets/{ticket_id}/claim?expected_session_key={await _url_session_key(case, _staff_url(ticket_id))}",
            headers=case.headers[3],
        )
    ).status_code == 200
    counts = _calls(case)
    assert (await _bot(case, "confirm", transport)).status_code == 423
    assert _calls(case) == counts


async def test_inflight_staff_send_excludes_resolve_and_postclose_new_writes(case: _Case) -> None:
    ticket_id = await _ticket(case)
    case.writer_gate.pause_next = True
    sender = asyncio.create_task(_send(case, _staff_url(ticket_id), 3, "Committed before close."))
    try:
        await asyncio.wait_for(case.writer_gate.entered.wait(), timeout=2)
        closing = await case.client.post(
            f"/api/v1/handoff/tickets/{ticket_id}/resolve?expected_session_key={await _url_session_key(case, _staff_url(ticket_id))}",
            headers=case.headers[3],
        )
        assert closing.status_code == 409
        assert not sender.done()
        case.writer_gate.release.set()
        reply = _ack(await asyncio.wait_for(sender, timeout=5))
    finally:
        case.writer_gate.release.set()
        if not sender.done():
            sender.cancel()
        await asyncio.gather(sender, return_exceptions=True)
    assert (
        await case.client.post(
            f"/api/v1/handoff/tickets/{ticket_id}/resolve?expected_session_key={await _url_session_key(case, _staff_url(ticket_id))}",
            headers=case.headers[3],
        )
    ).status_code == 200
    assert (await _send(case, _staff_url(ticket_id), 3, "Too late.")).status_code == 404
    assert (await _feed(case, _customer_url(case), 1))["messages"] == [reply["message"]]


async def test_claim_resolve_epoch_cannot_resume_a_stale_refund_confirmation(case: _Case) -> None:
    for message in ["I want a refund", "ORD1001", "damaged item"]:
        assert (await _bot(case, message)).status_code == 200
    assert case.calls.refunds == []
    ticket_id = await _ticket(case)
    assert (await _bot(case, "confirm")).status_code == 423
    _ack(await _send(case, _staff_url(ticket_id), 3, "Please start a fresh request.", resolve=True))
    assert (await _bot(case, "confirm")).status_code == 200
    assert case.calls.refunds == []
    for message in ["I want a refund", "ORD1001", "damaged item", "confirm"]:
        assert (await _bot(case, message)).status_code == 200
    assert len(case.calls.refunds) == 1
    old = await case.chat.graph.aget_state({"configurable": {"thread_id": str(case.private_id)}})
    fresh = await case.chat.graph.aget_state(
        {"configurable": {"thread_id": f"{case.private_id}:automation:1"}}
    )
    assert old.values.get("pending_confirmation")
    assert not fresh.values.get("pending_confirmation")


async def _stats(case: _Case) -> dict[str, Any]:
    response = await _get(case, "/api/v1/sessions/containment-stats", headers=case.headers[1])
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, dict)
    return body


async def test_human_only_channel_is_not_a_bot_cohort_or_feedback_target(case: _Case) -> None:
    ticket_id = await _ticket(case)
    _ack(await _send(case, _customer_url(case), 1, "Please help."))
    reply = _ack(await _send(case, _staff_url(ticket_id), 3, "I am reviewing it."))
    stats = await _stats(case)
    for key in [
        "sessions_served",
        "sessions_with_failed_replies",
        "sessions_with_completed_replies",
        "sessions_latest_reply_failed",
    ]:
        assert stats[key] == 0
    assert await _bot_rows(case) == []
    before = FEEDBACK_RATINGS.labels(rating="-1")._value.get()
    denied = await case.client.post(
        "/api/v1/feedback",
        headers=case.headers[1],
        json={"message_id": reply["message"]["id"], "rating": -1},
    )
    assert denied.status_code == 404  # No bot Message with this ID exists.
    assert FEEDBACK_RATINGS.labels(rating="-1")._value.get() == before
    assert case.calls.evictions == []


async def test_channel_id_collision_preserves_bot_identity_and_quality_counts(case: _Case) -> None:
    async with case.maker() as db:
        db.add_all(
            [
                Message(
                    session_id=10,
                    role=MessageRole.ASSISTANT,
                    status=MessageStatus.COMPLETED,
                    content="Earlier bot reply.",
                    created_at=datetime.now(UTC) - timedelta(minutes=2),
                ),
                Message(
                    session_id=10,
                    role=MessageRole.ASSISTANT,
                    status=MessageStatus.FAILED,
                    content="Failed bot reply.",
                    created_at=datetime.now(UTC) - timedelta(minutes=1),
                ),
            ]
        )
        await db.commit()
    ticket_id = await _ticket(case)
    human = _ack(await _send(case, _customer_url(case), 1, "Independent channel."))["message"]
    assert human["id"] == (await _bot_rows(case))[0].id
    assert "message_id" not in human
    before = await _stats(case)
    _ack(await _send(case, _staff_url(ticket_id), 3, "A newer human reply."))
    after = await _stats(case)
    assert {key: value for key, value in before.items() if key != "window_start"} == {
        key: value for key, value in after.items() if key != "window_start"
    }
    assert after["sessions_served"] == 1
    assert after["sessions_with_completed_replies"] == 1
    assert after["sessions_with_failed_replies"] == 1
    assert after["sessions_latest_reply_failed"] == 1
    # The same integer remains a legitimate BOT namespace identity;
    # human payloads must never advertise it as message_id for feedback.
    response = await case.client.post(
        "/api/v1/feedback",
        headers=case.headers[1],
        json={"message_id": human["id"], "rating": 1},
    )
    assert response.status_code == 200
    assert (await _bot_rows(case))[0].user_rating == 1
    assert (await _human_rows(case))[0].content == "Independent channel."


async def test_customer_send_replays_once_and_cannot_move_an_old_key_to_a_new_ticket(
    case: _Case,
) -> None:
    ticket_id = await _ticket(case)
    key = str(uuid4())
    first = _ack(await _send(case, _customer_url(case), 1, "One send.", key=key))
    again = _ack(await _send(case, _customer_url(case), 1, "One send.", key=key))
    assert first["reused"] is False
    assert again["reused"] is True
    assert first["message"] == again["message"]
    assert (
        await _send(case, _customer_url(case), 1, "Different send.", key=key)
    ).status_code == 409
    assert (await _send(case, _staff_url(ticket_id), 3, "One send.", key=key)).status_code == 409
    assert len(await _human_rows(case)) == 1
    assert (
        await case.client.post(
            f"/api/v1/handoff/tickets/{ticket_id}/resolve?expected_session_key={await _url_session_key(case, _staff_url(ticket_id))}",
            headers=case.headers[3],
        )
    ).status_code == 200
    replay = _ack(await _send(case, _customer_url(case), 1, "One send.", key=key))
    assert replay["message"] == first["message"]
    next_ticket = await _ticket(case)
    assert next_ticket != ticket_id
    assert (await _send(case, _customer_url(case), 1, "One send.", key=key)).status_code == 409
    assert len(await _human_rows(case)) == 1


async def test_empty_private_feed_does_not_invent_an_active_human_conversation(case: _Case) -> None:
    page = await _feed(case, _customer_url(case), 1)
    assert page["ticket"] is None
    assert page["messages"] == []
    assert page["next_after_id"] == 0
    assert page["has_more"] is False
    response = await _send(case, _customer_url(case), 1, "No active ticket.")
    assert response.status_code == 409
    _no_store(response)
    assert await _human_rows(case) == []
    assert await _bot_rows(case) == []


async def test_customer_cannot_choose_staff_author_session_or_resolution(case: _Case) -> None:
    ticket_id = await _ticket(case)
    response = await case.client.post(
        _customer_url(case),
        headers=case.headers[1],
        json={
            "client_message_id": str(uuid4()),
            "content": "Forged identity.",
            "author_kind": "staff",
            "author_user_id": 3,
            "session_id": 20,
            "ticket_id": ticket_id,
            "resolve": True,
        },
    )
    assert response.status_code == 422
    assert await _human_rows(case) == []
    async with case.maker() as db:
        ticket = await db.get(HandoffTicket, ticket_id)
        assert ticket is not None and ticket.status == "claimed"


async def test_claim_between_stream_preflight_and_body_emits_error_without_bot_work(
    case: _Case,
) -> None:
    ticket_id = await _ticket(case, claim=False)
    case.stream_gate.pause_next = True
    counts = _calls(case)
    request = asyncio.create_task(
        _bot(case, "confirm", "stream", include_completion=True, include_citations=True)
    )
    try:
        await asyncio.wait_for(case.stream_gate.entered.wait(), timeout=2)
        assert (
            await case.client.post(
                f"/api/v1/handoff/tickets/{ticket_id}/claim?expected_session_key={await _url_session_key(case, _staff_url(ticket_id))}",
                headers=case.headers[3],
            )
        ).status_code == 200
        case.stream_gate.release.set()
        response = await asyncio.wait_for(request, timeout=5)
    finally:
        case.stream_gate.release.set()
        if not request.done():
            request.cancel()
        await asyncio.gather(request, return_exceptions=True)
    assert response.status_code == 200  # The preflight already passed.
    data = [line[5:].strip() for line in response.text.splitlines() if line.startswith("data:")]
    assert "[STREAM_ERROR]" in data
    assert all(value in {"[STREAM_ERROR]", "[DONE]"} for value in data)
    assert "event: completion" not in response.text
    assert "event: citations" not in response.text
    assert _calls(case) == counts
    assert await _bot_rows(case) == []


@pytest.mark.parametrize("demo", [False, True])
async def test_shared_legacy_session_has_no_human_channel_or_human_context(
    case: _Case, monkeypatch: pytest.MonkeyPatch, demo: bool
) -> None:
    monkeypatch.setattr(settings, "DEMO_MODE", demo)
    result = await case.handoff.create_ticket_for_session(
        10, 1, "explicit", {"user_message": "legacy admin ticket"}
    )
    ticket_id = result["ticket_id"]
    assert type(ticket_id) is int
    assert (
        await case.client.post(
            f"/api/v1/handoff/tickets/{ticket_id}/claim?expected_session_key={await _url_session_key(case, _staff_url(ticket_id))}",
            headers=case.headers[3],
        )
    ).status_code == 200
    customer_url = "/api/v1/handoff/sessions/10/messages"
    assert (await _get(case, customer_url, headers=case.headers[1])).status_code == 404
    assert (await _send(case, customer_url, 1, "Public human message.")).status_code == 404
    assert (await _send(case, _staff_url(ticket_id), 3, "Public staff reply.")).status_code == 409
    assert await _human_rows(case) == []
    # An old/stray row must not turn a public anonymous bot conversation
    # into a private support-context reader, even though new writes deny it.
    async with case.maker() as db:
        db.add(
            HandoffMessage(
                session_id=10,
                ticket_id=ticket_id,
                author_kind="staff",
                author_user_id=3,
                client_message_id=str(uuid4()),
                content="PUBLIC_ROW_NOT_FOR_BOT_CONTEXT",
            )
        )
        await db.commit()
    assert await case.handoff.get_recent_context(10) == []


async def test_demo_private_quality_surfaces_reject_other_customers(
    case: _Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "DEMO_MODE", True)
    reply = await _bot(case, "你好", "chat")
    assert reply.status_code == 200
    message_id = reply.json()["message_id"]
    before = _calls(case)
    forbidden = await case.client.post(
        "/api/v1/feedback", headers=case.headers[2], json={"message_id": message_id, "rating": -1}
    )
    assert forbidden.status_code == 404
    assert _calls(case) == before
    async with case.maker() as db:
        message = await db.get(Message, message_id)
        assert message is not None and message.user_rating is None
    assert (
        await _get(case, "/api/v1/sessions/review-queue", headers=case.headers[2])
    ).status_code == 403
    assert (
        await case.client.post(
            f"/api/v1/sessions/{case.private_id}/review",
            headers=case.headers[2],
            json={"resolved": True},
        )
    ).status_code == 403


async def test_missing_and_foreign_session_delete_have_same_404(case: _Case) -> None:
    for session_id in (case.private_id, 999999):
        response = await _delete(case, f"/api/v1/sessions/{session_id}", headers=case.headers[2])
        assert response.status_code == 404


async def test_private_chat_and_session_responses_are_never_cached(case: _Case) -> None:
    responses = [
        await _bot(case, "你好"),
        await _get(
            case, f"/api/v1/chat/history?session_id={case.private_id}", headers=case.headers[1]
        ),
        await _get(case, "/api/v1/sessions", headers=case.headers[1]),
        await _get(case, f"/api/v1/sessions/{case.private_id}", headers=case.headers[1]),
        await _get(case, f"/api/v1/chat/history?session_id={case.private_id}"),
    ]
    assert [response.status_code for response in responses] == [200, 200, 200, 200, 401]
    for response in responses:
        _no_store(response)


async def test_deleted_private_session_cannot_reveal_old_checkpoint_to_demo(
    case: _Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    for message in ["I want a refund", "ORD1001", "damaged item"]:
        assert (await _bot(case, message)).status_code == 200
    assert (
        await _delete(case, f"/api/v1/sessions/{case.private_id}", headers=case.headers[1])
    ).status_code == 204
    monkeypatch.setattr(settings, "DEMO_MODE", True)
    assert (await _bot(case, "confirm", actor=0)).status_code == 200
    assert case.calls.refunds == []
    old = await case.chat.graph.aget_state({"configurable": {"thread_id": str(case.private_id)}})
    assert not old.values.get("pending_confirmation")


async def test_history_rechecks_private_owner_after_demo_preflight(
    case: _Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.services.session_service import SessionService

    monkeypatch.setattr(settings, "DEMO_MODE", True)

    # Represents a preflight that saw a missing/public row before private
    # creation. The actual guarded read must still reject the outsider.
    async def stale_preflight(self: SessionService, session_id: int) -> bool:
        return False

    monkeypatch.setattr(SessionService, "is_private_session", stale_preflight)
    response = await _get(
        case, f"/api/v1/chat/history?session_id={case.private_id}", headers=case.headers[2]
    )
    assert response.status_code == 404
