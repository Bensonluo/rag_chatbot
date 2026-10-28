"""Versioned source snapshots survive both transports and persisted history."""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.api.deps import get_current_user, get_db
from app.api.v1 import chat as chat_api
from app.config.settings import settings
from app.models.database.base import Base
from app.models.database.message import Message
from app.models.enums.message import MessageRole, MessageStatus
from app.services.chat.chat_service import STREAM_ERROR, ChatService, CitationEvent
from app.services.chat.persistence import ChatMessagePersister

pytestmark = pytest.mark.usefixtures("forbid_external_sockets")


class SnapshotGraph:
    def __init__(self) -> None:
        self.version = "policy-v1"
        self.failure: str | None = None
        self.cited = True

    async def ainvoke(self, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        queue = config["configurable"].get("stream_queue")
        if queue is not None:
            queue.put_nowait("Refunds within seven days.")
        if self.failure == "error":
            raise RuntimeError("generation failed")
        if self.failure == "timeout":
            await asyncio.Event().wait()
        return {
            "response": "Refunds within seven days.",
            "intent": "policy",
            "sources": ["policy"] if self.cited else [],
            "citations": [
                {
                    "source_type": "document",
                    "document_id": "policy",
                    "version_id": self.version,
                    "chunk_id": f"{self.version}-chunk-1",
                    "title": "Refund policy",
                    "excerpt": "Refunds within seven days.",
                    "page_start": 2,
                    "page_end": 2,
                    "start_pos": 0,
                    "end_pos": 25,
                    "section_path": ["Refunds"],
                }
            ]
            if self.cited
            else [],
        }


@pytest.fixture
async def citation_stack(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[tuple[Any, ...]]:
    monkeypatch.setattr(settings, "DEMO_MODE", True)
    monkeypatch.setattr(chat_api, "_enforce_chat_rate_limit", AsyncMock())
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    persister = ChatMessagePersister(session_maker=sessions)
    graph = SnapshotGraph()
    service = ChatService(graph=graph, persister=persister)
    app = FastAPI()
    app.include_router(chat_api.router, prefix="/api/v1")
    app.dependency_overrides[chat_api.get_chat_service] = lambda: service
    app.dependency_overrides[get_current_user] = lambda: None

    async def local_db():
        async with sessions() as db:
            yield db

    app.dependency_overrides[get_db] = local_db
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            yield client, service, graph, sessions
    finally:
        await engine.dispose()


def _citation_event(body: str) -> list[dict[str, Any]]:
    events = [frame for frame in body.split("\n\n") if frame.startswith("event: citations\n")]
    assert len(events) == 1
    payload = json.loads(events[0].split("data: ", 1)[1])
    citations: list[dict[str, Any]] = payload["citations"]
    return citations


@pytest.mark.parametrize("transport", ["rest", "stream"])
async def test_citation_snapshot_survives_transport_database_and_history(citation_stack, transport):
    client, _, graph, sessions = citation_stack
    request = {"session_id": 1, "message": "Refund policy?"}
    if transport == "rest":
        response = await client.post("/api/v1/chat", json=request)
        assert response.status_code == 200
        citations = response.json()["citations"]
        assert response.json()["sources"] == ["policy"]
    else:
        response = await client.post(
            "/api/v1/chat/stream", json={**request, "include_citations": True}
        )
        assert response.status_code == 200
        citations = _citation_event(response.text)
        assert response.text.index("event: citations") < response.text.index("data: [DONE]")
    assert len(citations) == 1
    assert citations[0]["version_id"] == "policy-v1"
    assert citations[0]["chunk_id"] == "policy-v1-chunk-1"
    assert citations[0]["excerpt"] == "Refunds within seven days."
    assert citations[0]["section_path"] == ["Refunds"]
    assert citations[0]["page_start"] == 2

    # A later answer uses a replacement revision; historical evidence stays v1.
    graph.version = "policy-v2"
    await client.post("/api/v1/chat", json=request)
    history = await client.get("/api/v1/chat/history", params={"session_id": 1})
    assert history.status_code == 200
    messages = history.json()["messages"]
    assert len(messages) == 4
    assert all(message["timestamp"] for message in messages)
    assert messages[0]["citations"] == []
    assert messages[0]["sources"] is None
    assert messages[1]["sources"] == ["policy"]
    assert messages[1]["citations"][0]["version_id"] == "policy-v1"
    assert messages[3]["citations"][0]["version_id"] == "policy-v2"
    for key, value in citations[0].items():
        assert messages[1]["citations"][0][key] == value
    async with sessions() as db:
        rows = (await db.execute(select(Message).order_by(Message.id))).scalars().all()
        assert rows[1].message_metadata["citations"][0]["version_id"] == "policy-v1"
        assert rows[3].message_metadata["citations"][0]["version_id"] == "policy-v2"


async def test_old_messages_and_uncited_turns_have_no_invented_citations(citation_stack):
    client, service, graph, sessions = citation_stack
    await service.persister.persist_turn(
        session_id=1,
        user_id=0,
        user_message="Old question",
        response="Old answer",
        sources=["legacy"],
    )
    graph.cited = False
    response = await client.post("/api/v1/chat", json={"session_id": 1, "message": "Hi"})
    assert response.json()["citations"] == []
    history = (await client.get("/api/v1/chat/history", params={"session_id": 1})).json()[
        "messages"
    ]
    assert all(message["citations"] == [] for message in history)
    assert history[1]["sources"] == ["legacy"]
    async with sessions() as db:
        rows = (await db.execute(select(Message))).scalars().all()
        assert all("citations" not in (row.message_metadata or {}) for row in rows)


async def test_stream_citations_require_opt_in(citation_stack):
    client, service, _, _ = citation_stack
    chunks = [chunk async for chunk in service.process_message_stream(1, "Policy?", 0)]
    assert chunks == ["Refunds within seven days."]
    response = await client.post(
        "/api/v1/chat/stream", json={"session_id": 2, "message": "Policy?"}
    )
    assert "event: citations" not in response.text
    assert "policy-v1" not in response.text
    assert "data: [DONE]" in response.text


@pytest.mark.parametrize("failure", ["error", "timeout", "disconnect"])
async def test_incomplete_stream_does_not_publish_or_persist_final_citations(
    citation_stack, failure
):
    _, service, graph, sessions = citation_stack
    graph.failure = failure if failure != "disconnect" else "timeout"
    stream = service.process_message_stream(
        1, "Policy?", 0, include_citations=True, stream_max_seconds=0.02
    )
    if failure == "disconnect":
        chunks = [await anext(stream)]
        await stream.aclose()
    else:
        chunks = [chunk async for chunk in stream]
        assert STREAM_ERROR in chunks
    assert not any(isinstance(chunk, CitationEvent) for chunk in chunks)
    async with sessions() as db:
        assistant = (
            await db.execute(select(Message).where(Message.role == MessageRole.ASSISTANT))
        ).scalar_one()
        assert assistant.status == MessageStatus.FAILED
        assert "citations" not in (assistant.message_metadata or {})
        assert "sources" not in (assistant.message_metadata or {})
