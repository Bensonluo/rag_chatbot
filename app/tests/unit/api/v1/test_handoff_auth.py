"""Authorization matrix for the handoff agent-workspace API.

The handoff endpoints are the internal human-agent surface: anonymous
requests must get 401, authenticated non-admins 403, admins full
access. This pins the matrix and the happy-path ticket lifecycle over
an in-memory database.
"""

from collections.abc import AsyncIterator, Iterator

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.api.deps import get_current_active_user
from app.api.v1.handoff import router as handoff_router
from app.api.v1.handoff import set_handoff_service
from app.models.database.base import Base
from app.models.database.session import ChatSession  # noqa: F401 (registers table)
from app.models.database.user import User
from app.services.handoff.service import HandoffService

ALL_PATHS = [
    ("/handoff/tickets", "get"),
    ("/handoff/stats", "get"),
    ("/handoff/tickets/1/claim", "post"),
    ("/handoff/tickets/1/resolve", "post"),
]


def _make_user(is_admin: bool) -> User:
    return User(
        id=1,
        email="admin@example.com" if is_admin else "user@example.com",
        hashed_password="x",
        is_active=True,
        is_admin=is_admin,
    )


@pytest.fixture
def app() -> FastAPI:
    app = FastAPI()
    app.include_router(handoff_router)
    return app


@pytest.fixture
def admin(app: FastAPI) -> Iterator[User]:
    app.dependency_overrides[get_current_active_user] = lambda: _make_user(is_admin=True)
    yield _make_user(is_admin=True)
    app.dependency_overrides.clear()


@pytest.fixture
def plain_user(app: FastAPI) -> Iterator[User]:
    app.dependency_overrides[get_current_active_user] = lambda: _make_user(is_admin=False)
    yield _make_user(is_admin=False)
    app.dependency_overrides.clear()


@pytest.fixture
async def db_service() -> AsyncIterator[HandoffService]:
    """Real HandoffService over an in-memory sqlite database."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    service = HandoffService(session_maker=maker)
    set_handoff_service(service)
    yield service
    set_handoff_service(None)  # never leak the test DB into other tests
    await engine.dispose()


async def _client(app: FastAPI) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


class TestAnonymousAccess:
    @pytest.mark.parametrize("path, method", ALL_PATHS)
    async def test_all_endpoints_require_auth(self, app, path, method):
        async with await _client(app) as client:
            resp = await getattr(client, method)(path)
        assert resp.status_code == 401


class TestNonAdminAccess:
    @pytest.mark.parametrize("path, method", ALL_PATHS)
    async def test_all_endpoints_require_admin(self, app, plain_user, path, method):
        async with await _client(app) as client:
            resp = await getattr(client, method)(path)
        assert resp.status_code == 403


class TestAdminLifecycle:
    async def test_list_claim_resolve_roundtrip(self, app, admin, db_service):
        created = await db_service.create_ticket_for_session(
            5, 7, "explicit", {"user_message": "转人工"}
        )

        async with await _client(app) as client:
            listed = await client.get("/handoff/tickets?status=open")
            assert listed.status_code == 200
            body = listed.json()
            assert body["count"] == 1
            assert body["tickets"][0]["id"] == created["ticket_id"]
            assert body["tickets"][0]["reason"] == "explicit"

            claimed = await client.post(f"/handoff/tickets/{created['ticket_id']}/claim")
            assert claimed.status_code == 200
            assert claimed.json()["status"] == "claimed"
            assert claimed.json()["assigned_to"] == 1  # admin user id

            stats = await client.get("/handoff/stats")
            assert stats.status_code == 200
            body = stats.json()
            assert body["open"] == 0
            assert body["claimed"] == 1
            assert body["resolved"] == 0
            # SLA dimensions ride along (additive keys; GB/T 47746 时效观测)
            assert body["open_sla_breaches"] == 0
            assert body["sla_wait_seconds"] == 30.0

            resolved = await client.post(f"/handoff/tickets/{created['ticket_id']}/resolve")
            assert resolved.status_code == 200
            assert resolved.json()["status"] == "resolved"

    async def test_double_claim_is_conflict(self, app, admin, db_service):
        created = await db_service.create_ticket_for_session(5, 7, "explicit", {})

        async with await _client(app) as client:
            first = await client.post(f"/handoff/tickets/{created['ticket_id']}/claim")
            second = await client.post(f"/handoff/tickets/{created['ticket_id']}/claim")
        assert first.status_code == 200
        assert second.status_code == 409

    async def test_missing_ticket_is_conflict(self, app, admin, db_service):
        async with await _client(app) as client:
            resp = await client.post("/handoff/tickets/999/resolve")
        assert resp.status_code == 409

    async def test_invalid_status_rejected(self, app, admin, db_service):
        async with await _client(app) as client:
            resp = await client.get("/handoff/tickets?status=dropped")
        assert resp.status_code == 422
