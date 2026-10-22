"""Real JWT and SQLite boundaries for the assigned-agent transcript route."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.api.deps import get_db
from app.api.v1.handoff import get_handoff_service, router
from app.config.settings import settings
from app.core.security import create_access_token
from app.models.database.base import Base
from app.models.database.message import Message
from app.models.database.session import ChatSession
from app.models.database.user import User
from app.models.enums.message import MessageRole
from app.services.chat.citations import normalize_citations
from app.services.handoff.service import HandoffService

CITATION = {
    "source_type": "document",
    "document_id": "returns",
    "version_id": "returns-v2",
    "excerpt": "Unused items may be returned within seven days.",
    "section_path": ["Returns"],
}


def headers(user_id: int) -> dict[str, str]:
    token = create_access_token({"sub": f"user-{user_id}@history.test", "user_id": user_id})
    return {"Authorization": f"Bearer {token}"}


@dataclass
class HistoryAPI:
    client: AsyncClient
    service: HandoffService
    maker: async_sessionmaker[AsyncSession]
    ticket_id: int


@pytest.fixture
async def history_api() -> AsyncIterator[HistoryAPI]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    maker = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with maker() as session:
            session.add_all(
                User(
                    id=user_id,
                    email=f"user-{user_id}@history.test",
                    hashed_password="!",
                    is_admin=user_id in {2, 3, 5},
                    is_active=user_id != 5,
                )
                for user_id in range(1, 6)
            )
            await session.flush()
            session.add_all([ChatSession(id=10, user_id=1), ChatSession(id=20, user_id=4)])
            await session.flush()
            timestamp = datetime.now(UTC)
            session.add_all(
                [
                    Message(
                        id=1,
                        session_id=10,
                        role=MessageRole.USER,
                        content="My return?",
                        created_at=timestamp,
                        message_metadata={
                            "sources": ["private-user-trace"],
                            "citations": [CITATION],
                        },
                    ),
                    Message(
                        id=2,
                        session_id=10,
                        role=MessageRole.ASSISTANT,
                        content="Seven days.",
                        created_at=timestamp,
                        message_metadata={
                            "sources": ["return-policy"],
                            "citations": [CITATION],
                            "private_debug": "INTERNAL_TRACE",
                            "session_id": 20,
                        },
                    ),
                    Message(
                        id=3,
                        session_id=10,
                        role=MessageRole.SYSTEM,
                        content="INTERNAL_SUMMARY",
                        created_at=timestamp,
                    ),
                    Message(
                        id=4,
                        session_id=10,
                        role=MessageRole.ASSISTANT,
                        content="Request recorded.",
                        created_at=timestamp,
                    ),
                    Message(
                        id=5,
                        session_id=20,
                        role=MessageRole.USER,
                        content="OTHER_CUSTOMER_SECRET",
                        created_at=timestamp,
                    ),
                ]
            )
            await session.commit()
        service = HandoffService(maker)
        created = await service.create_ticket_for_session(10, 1, "explicit", {})
        ticket_id = created["ticket_id"]
        assert isinstance(ticket_id, int)

        async def isolated_db() -> AsyncIterator[AsyncSession]:
            async with maker() as session:
                yield session

        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        app.dependency_overrides[get_db] = isolated_db
        app.dependency_overrides[get_handoff_service] = lambda: service
        async with AsyncClient(
            transport=ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://isolated-history.test",
        ) as client:
            yield HistoryAPI(client, service, maker, ticket_id)
    finally:
        await engine.dispose()


@pytest.mark.parametrize("demo_mode", [True, False])
async def test_assigned_admin_reads_only_the_ticket_session(
    history_api: HistoryAPI, monkeypatch: pytest.MonkeyPatch, demo_mode: bool
) -> None:
    monkeypatch.setattr(settings, "DEMO_MODE", demo_mode)
    api = history_api
    claimed = await api.client.post(
        f"/api/v1/handoff/tickets/{api.ticket_id}/claim", headers=headers(2)
    )
    assert claimed.status_code == 200
    response = await api.client.get(
        f"/api/v1/handoff/tickets/{api.ticket_id}/history?session_id=20&skip=0&limit=2",
        headers=headers(2),
    )
    assert response.status_code == 200
    assert response.headers["cache-control"] == "private, no-store"
    body = response.json()
    assert body["ticket_id"] == api.ticket_id
    assert body["session_id"] == 10
    assert body["total"] == 3
    assert [message["id"] for message in body["messages"]] == [1, 2]
    assert [message["role"] for message in body["messages"]] == ["user", "assistant"]
    assert body["messages"][0]["sources"] is None
    assert body["messages"][0]["citations"] == []
    assert body["messages"][1]["sources"] == ["return-policy"]
    assert normalize_citations(body["messages"][1]["citations"]) == [
        {**CITATION, "source_document_ids": [], "source_version_ids": [], "source_chunk_ids": []}
    ]
    assert "INTERNAL" not in response.text and "OTHER_CUSTOMER_SECRET" not in response.text
    assert "private_debug" not in response.text
    second = await api.client.get(
        f"/api/v1/handoff/tickets/{api.ticket_id}/history?skip=2&limit=2", headers=headers(2)
    )
    assert second.status_code == 200
    assert [message["id"] for message in second.json()["messages"]] == [4]
    empty = await api.client.get(
        f"/api/v1/handoff/tickets/{api.ticket_id}/history?skip=3", headers=headers(2)
    )
    assert empty.status_code == 200 and empty.json()["messages"] == []
    assert empty.json()["total"] == 3


async def test_ticket_states_and_foreign_assignments_share_one_404(history_api: HistoryAPI) -> None:
    api = history_api
    path = f"/api/v1/handoff/tickets/{api.ticket_id}/history"
    responses = [await api.client.get(path, headers=headers(2))]
    responses.append(
        await api.client.get("/api/v1/handoff/tickets/999/history", headers=headers(2))
    )
    await api.service.claim_ticket(api.ticket_id, 2)
    responses.append(await api.client.get(path, headers=headers(3)))
    await api.service.resolve_ticket(api.ticket_id, 2)
    responses.append(await api.client.get(path, headers=headers(2)))
    assert {response.status_code for response in responses} == {404}
    assert all(
        response.json() == {"detail": "Ticket history unavailable"} for response in responses
    )
    assert all(response.headers["cache-control"] == "private, no-store" for response in responses)


@pytest.mark.parametrize("actor,expected", [(None, 401), (1, 403), (4, 403), (5, 403)])
async def test_real_authentication_and_active_admin_role(
    history_api: HistoryAPI, actor: int | None, expected: int
) -> None:
    api = history_api
    await api.service.claim_ticket(api.ticket_id, 2)
    response = await api.client.get(
        f"/api/v1/handoff/tickets/{api.ticket_id}/history", headers=headers(actor) if actor else {}
    )
    assert response.status_code == expected
    assert "Seven days" not in response.text


@pytest.mark.parametrize("suffix", ["?limit=0", "?limit=101", "?skip=-1", "?skip=2147483648"])
async def test_pagination_bounds_are_rejected(history_api: HistoryAPI, suffix: str) -> None:
    api = history_api
    await api.service.claim_ticket(api.ticket_id, 2)
    response = await api.client.get(
        f"/api/v1/handoff/tickets/{api.ticket_id}/history{suffix}", headers=headers(2)
    )
    assert response.status_code == 422


async def test_database_failure_cannot_return_transcript_or_private_error(
    history_api: HistoryAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = history_api
    monkeypatch.setattr(
        api.service,
        "get_ticket_history",
        AsyncMock(side_effect=RuntimeError("private-db-password")),
    )
    response = await api.client.get(
        f"/api/v1/handoff/tickets/{api.ticket_id}/history", headers=headers(2)
    )
    assert response.status_code == 500
    assert "private-db-password" not in response.text
    assert "Seven days" not in response.text
