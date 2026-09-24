"""Authorization and contract tests for the knowledge-gap analytics API."""

from datetime import UTC
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.deps import get_current_active_user
from app.api.v1.analytics import _get_db
from app.api.v1.analytics import router as analytics_router
from app.models.database.user import User


def _make_user() -> User:
    return User(
        id=1,
        email="ops@example.com",
        hashed_password="x",
        is_active=True,
        is_admin=False,
    )


@pytest.fixture
def app() -> FastAPI:
    app = FastAPI()
    app.include_router(analytics_router)
    return app


async def _client(app: FastAPI) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


class TestAnonymousAccess:
    async def test_knowledge_gaps_requires_auth(self, app):
        async with await _client(app) as client:
            resp = await client.get("/analytics/knowledge-gaps")
        assert resp.status_code == 401


class TestAuthenticatedAccess:
    @pytest.fixture
    def authed(self, app):
        app.dependency_overrides[get_current_active_user] = _make_user
        app.dependency_overrides[_get_db] = lambda: None
        yield
        app.dependency_overrides.clear()

    async def test_returns_gap_report(self, app, authed):
        repo = AsyncMock()
        repo.top_gaps.return_value = [
            {
                "normalized_query": "发票怎么开",
                "hits": 12,
                "last_seen": "2026-09-24T01:00:00+00:00",
                "sample_query": "发票怎么开",
                "sample_intent": "faq",
            }
        ]
        with patch("app.api.v1.analytics.KnowledgeGapRepository", return_value=repo) as repo_cls:
            async with await _client(app) as client:
                resp = await client.get("/analytics/knowledge-gaps")
        assert resp.status_code == 200
        body = resp.json()
        assert body["days"] == 7
        assert body["total"] == 1
        assert body["gaps"][0]["hits"] == 12
        repo.top_gaps.assert_awaited_once_with(days=7, limit=20, include_resolved=False)
        repo_cls.assert_called_once()

    async def test_forwards_query_params(self, app, authed):
        repo = AsyncMock()
        repo.top_gaps.return_value = []
        with patch("app.api.v1.analytics.KnowledgeGapRepository", return_value=repo):
            async with await _client(app) as client:
                resp = await client.get("/analytics/knowledge-gaps?days=30&limit=5")
        assert resp.status_code == 200
        repo.top_gaps.assert_awaited_once_with(days=30, limit=5, include_resolved=False)

    @pytest.mark.parametrize("query", ["days=0", "days=91", "limit=0", "limit=101"])
    async def test_rejects_out_of_range_params(self, app, authed, query):
        async with await _client(app) as client:
            resp = await client.get(f"/analytics/knowledge-gaps?{query}")
        assert resp.status_code == 422


class TestResolveEndpoint:
    @pytest.fixture
    def authed(self, app):
        app.dependency_overrides[get_current_active_user] = _make_user
        app.dependency_overrides[_get_db] = lambda: None
        yield
        app.dependency_overrides.clear()

    async def test_requires_auth(self, app):
        async with await _client(app) as client:
            resp = await client.post(
                "/analytics/knowledge-gaps/resolve", json={"normalized_query": "q"}
            )
        assert resp.status_code == 401

    async def test_resolve_returns_payload(self, app, authed):
        from datetime import datetime
        from types import SimpleNamespace

        repo = AsyncMock()
        repo.resolve_query.return_value = SimpleNamespace(
            normalized_query="发票怎么开",
            resolved_by=1,
            resolved_at=datetime(2026, 9, 24, tzinfo=UTC),
        )
        with patch("app.api.v1.analytics.KnowledgeGapRepository", return_value=repo):
            async with await _client(app) as client:
                resp = await client.post(
                    "/analytics/knowledge-gaps/resolve",
                    json={"normalized_query": "发票怎么开"},
                )
        assert resp.status_code == 200
        body = resp.json()
        assert body["normalized_query"] == "发票怎么开"
        assert body["resolved_by"] == 1
        assert body["resolved_at"] == "2026-09-24T00:00:00+00:00"
        repo.resolve_query.assert_awaited_once_with("发票怎么开", resolved_by=1)

    async def test_blank_query_rejected(self, app, authed):
        async with await _client(app) as client:
            resp = await client.post(
                "/analytics/knowledge-gaps/resolve", json={"normalized_query": "   "}
            )
        assert resp.status_code == 422
