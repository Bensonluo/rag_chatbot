"""Transport cancellation must durably retain the text already emitted."""

import asyncio
import gc
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from langgraph.checkpoint.memory import MemorySaver
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from starlette.requests import ClientDisconnect
from starlette.types import Message as ASGIMessage

from app.api.deps import get_db
from app.api.v1 import chat as chat_api
from app.config.settings import settings
from app.core.security import create_access_token
from app.models.database.base import Base
from app.models.database.message import Message
from app.models.database.session import ChatSession
from app.models.database.user import User
from app.models.enums.message import MessageStatus
from app.services.chat import chat_service
from app.services.chat.chat_service import ChatService, CitationEvent
from app.services.chat.persistence import ChatMessagePersister
from app.services.dialogue.graph import build_dialogue_graph
from app.services.dialogue.tools import ToolRegistry
from app.services.guardrails.factory import GuardrailFactory
from app.services.intent.rule_based import RuleBasedIntentDetector
from app.services.llm.base import LLMMessage, LLMResponse, LLMServiceBase

_PARTIAL = "部分回复。"


class _InterruptedLLM(LLMServiceBase):
    def __init__(self) -> None:
        super().__init__(api_key="offline", model="offline-cancellation-fixture")
        self.closed = asyncio.Event()
        self.finish = asyncio.Event()

    async def generate(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        return LLMResponse(content=_PARTIAL, model=self.model)

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        try:
            yield _PARTIAL
            await self.finish.wait()
        finally:
            self.closed.set()

    def estimate_tokens(self, text: str) -> int:
        return len(text)

    async def count_tokens(self, messages: list[LLMMessage]) -> int:
        return sum(len(message.content) for message in messages)


@dataclass
class _StreamCase:
    chat: ChatService
    llm: _InterruptedLLM
    maker: async_sessionmaker[AsyncSession]
    producer_tasks: list[asyncio.Task[Any]]


@pytest.fixture
async def stream_case(tmp_path: Path, monkeypatch: Any) -> AsyncGenerator[_StreamCase, None]:
    monkeypatch.setattr(settings, "DEMO_MODE", False)
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", False)
    monkeypatch.setattr(settings, "CHAT_TRACE_STREAM_ENABLED", False)
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/cancellation.sqlite3")
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
        db.add(User(id=1, email="stream-owner@evaluation.invalid", hashed_password="!"))
        await db.flush()
        db.add(ChatSession(id=1, user_id=1))
        await db.commit()
    llm = _InterruptedLLM()
    graph = build_dialogue_graph(
        intent_detector=RuleBasedIntentDetector(),
        slot_filler=None,
        tool_registry=ToolRegistry(),
        retrieval_pipeline=None,
        llm_service=llm,
        guardrail_service=GuardrailFactory.create(),
        checkpointer=MemorySaver(),
    )
    invoke = graph.ainvoke
    producer_tasks: list[asyncio.Task[Any]] = []

    async def tracked_invoke(*args: Any, **kwargs: Any) -> Any:
        task = asyncio.current_task()
        assert task is not None
        producer_tasks.append(task)
        return await invoke(*args, **kwargs)

    monkeypatch.setattr(graph, "ainvoke", tracked_invoke)
    chat = ChatService(
        graph=graph,
        persister=ChatMessagePersister(maker),
    )
    try:
        yield _StreamCase(chat, llm, maker, producer_tasks)
    finally:
        await engine.dispose()


async def _assert_failed_partial(case: _StreamCase) -> None:
    async with case.maker() as db:
        rows = (await db.scalars(select(Message).order_by(Message.id))).all()
    assert [row.content for row in rows] == ["你好", _PARTIAL]
    assert rows[1].status == MessageStatus.FAILED
    assert not (rows[1].message_metadata or {}).get("citations")
    assert not (rows[1].message_metadata or {}).get("handoff_outcome")
    assert case.llm.closed.is_set()
    assert case.producer_tasks and all(task.done() for task in case.producer_tasks)


def _isolated_app(case: _StreamCase) -> FastAPI:
    app = FastAPI()
    app.include_router(chat_api.router, prefix="/api/v1")

    async def isolated_db() -> AsyncGenerator[AsyncSession, None]:
        async with case.maker() as db:
            yield db

    app.dependency_overrides[get_db] = isolated_db
    app.dependency_overrides[chat_api.get_chat_service] = lambda: case.chat
    return app


async def test_asyncio_cancellation_persists_partial_as_failed(stream_case: _StreamCase) -> None:
    emitted = asyncio.Event()
    events: list[Any] = []

    async def consume() -> None:
        async for chunk in stream_case.chat.process_message_stream(
            1, "你好", 1, include_citations=True
        ):
            events.append(chunk)
            emitted.set()

    task = asyncio.create_task(consume())
    await asyncio.wait_for(emitted.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 1)

    assert events == [_PARTIAL]
    assert not any(isinstance(event, CitationEvent) for event in events)
    await _assert_failed_partial(stream_case)


@pytest.mark.parametrize("asgi_spec", ["2.3", "2.4"])
async def test_real_asgi_disconnect_persists_partial_as_failed(
    stream_case: _StreamCase, caplog: pytest.LogCaptureFixture, asgi_spec: str
) -> None:
    app = _isolated_app(stream_case)
    token = create_access_token({"sub": "stream-owner@evaluation.invalid", "user_id": 1})
    body = json.dumps({"session_id": 1, "message": "你好", "include_citations": True}).encode()
    requested = False
    disconnected = asyncio.Event()
    packets: list[bytes] = []

    async def receive() -> ASGIMessage:
        nonlocal requested
        if not requested:
            requested = True
            return {"type": "http.request", "body": body, "more_body": False}
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message: ASGIMessage) -> None:
        if message["type"] == "http.response.body":
            packet = message.get("body", b"")
            packets.append(packet)
            if _PARTIAL.encode() in packet:
                disconnected.set()
                if asgi_spec == "2.4":
                    raise OSError("client connection closed during send")

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": asgi_spec},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/v1/chat/stream",
        "raw_path": b"/api/v1/chat/stream",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"content-type", b"application/json"),
            (b"authorization", f"Bearer {token}".encode()),
        ],
        "client": ("127.0.0.1", 1234),
        "server": ("isolated-evaluation.invalid", 80),
    }
    if asgi_spec == "2.4":
        with pytest.raises(ClientDisconnect):
            await asyncio.wait_for(app(scope, receive, send), 2)
    else:
        await asyncio.wait_for(app(scope, receive, send), 2)

    wire = b"".join(packets)
    assert disconnected.is_set()
    assert _PARTIAL.encode() in wire
    assert b"[DONE]" not in wire
    assert b"event: citations" not in wire
    await _assert_failed_partial(stream_case)
    # The API must explicitly close its nested stream in the producing
    # task. A GC finalizer can run later with a different tracing context.
    gc.collect()
    await asyncio.sleep(0)
    assert not any("Failed to detach context" in record.message for record in caplog.records)


async def test_cancelled_finalization_has_a_deadline_and_leaves_no_writer(
    stream_case: _StreamCase, monkeypatch: Any
) -> None:
    writer_closed = asyncio.Event()
    emitted = asyncio.Event()

    async def stalled_write(**kwargs: Any) -> None:
        try:
            await asyncio.Event().wait()
        finally:
            writer_closed.set()

    monkeypatch.setattr(chat_service, "_STREAM_FINALIZE_TIMEOUT_SECONDS", 0.02, raising=False)
    assert stream_case.chat.persister is not None
    monkeypatch.setattr(stream_case.chat.persister, "persist_turn", stalled_write)

    async def consume() -> None:
        async for _ in stream_case.chat.process_message_stream(1, "你好", 1):
            emitted.set()

    task = asyncio.create_task(consume())
    await asyncio.wait_for(emitted.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 0.5)
    assert writer_closed.is_set()
    assert stream_case.llm.closed.is_set()
    assert all(task.done() for task in stream_case.producer_tasks)


async def test_completed_finalization_does_not_hide_a_persistence_exception(
    stream_case: _StreamCase, monkeypatch: Any
) -> None:
    async def failed_write(**kwargs: Any) -> None:
        raise RuntimeError("unexpected persister failure")

    assert stream_case.chat.persister is not None
    monkeypatch.setattr(stream_case.chat.persister, "persist_turn", failed_write)
    stream_case.llm.finish.set()
    emitted: list[Any] = []
    with pytest.raises(RuntimeError, match="unexpected persister failure"):
        async for chunk in stream_case.chat.process_message_stream(
            1, "你好", 1, include_citations=True
        ):
            emitted.append(chunk)
    assert emitted == [_PARTIAL]
    assert stream_case.llm.closed.is_set()
    assert all(task.done() for task in stream_case.producer_tasks)


async def test_normal_http_finalization_timeout_emits_error_without_citations(
    stream_case: _StreamCase, monkeypatch: Any
) -> None:
    assert stream_case.chat.persister is not None
    persist = stream_case.chat.persister.persist_turn
    writer_closed = asyncio.Event()

    async def slow_write(**kwargs: Any) -> None:
        try:
            await asyncio.sleep(0.1)
            await persist(**kwargs)
        finally:
            writer_closed.set()

    monkeypatch.setattr(chat_service, "_STREAM_FINALIZE_TIMEOUT_SECONDS", 0.02)
    monkeypatch.setattr(stream_case.chat.persister, "persist_turn", slow_write)
    stream_case.llm.finish.set()
    token = create_access_token({"sub": "stream-owner@evaluation.invalid", "user_id": 1})
    async with AsyncClient(
        transport=ASGITransport(app=_isolated_app(stream_case)),
        base_url="http://isolated-evaluation.invalid",
    ) as client:
        response = await client.post(
            "/api/v1/chat/stream",
            headers={"Authorization": f"Bearer {token}"},
            json={"session_id": 1, "message": "你好", "include_citations": True},
        )
    assert response.status_code == 200
    assert _PARTIAL in response.text
    assert "data: [STREAM_ERROR]" in response.text
    # DONE ends the transport after an explicit failure, as on graph errors.
    assert "data: [DONE]" in response.text
    assert "event: citations" not in response.text
    assert writer_closed.is_set()
    assert stream_case.llm.closed.is_set()
    assert all(task.done() for task in stream_case.producer_tasks)
    async with stream_case.maker() as db:
        assert (await db.scalars(select(Message))).all() == []
