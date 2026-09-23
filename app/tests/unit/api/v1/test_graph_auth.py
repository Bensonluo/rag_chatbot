"""Authorization matrix for the GraphRAG admin API.

The graph router exposes destructive and expensive operations —
structured CSV import writes into Neo4j, community detection runs a
graph-wide Leiden computation. At 800K-1M daily requests, anonymous
access to either is a data-integrity and DoS vector. These tests pin
the auth matrix: anonymous → 401 everywhere except /health,
authenticated non-admin → 403 on admin endpoints, admin → full access,
and internal error details never leak into responses.
"""

from collections.abc import Iterator
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.deps import get_current_active_user
from app.api.v1.graph import router as graph_router
from app.models.database.user import User

STATS_PATHS = ["/graph/stats", "/graph/schema"]
ADMIN_PATHS = ["/graph/import/structured", "/graph/communities/detect"]


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
    app.include_router(graph_router)
    return app


@pytest.fixture
def user(app: FastAPI) -> Iterator[User]:
    """Override auth to a non-admin user for the duration of a test."""
    app.dependency_overrides[get_current_active_user] = lambda: _make_user(is_admin=False)
    yield _make_user(is_admin=False)
    app.dependency_overrides.clear()


@pytest.fixture
def admin(app: FastAPI) -> Iterator[User]:
    """Override auth to an admin for the duration of a test."""
    app.dependency_overrides[get_current_active_user] = lambda: _make_user(is_admin=True)
    yield _make_user(is_admin=True)
    app.dependency_overrides.clear()


async def _client(app: FastAPI) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


class TestAnonymousAccess:
    async def test_health_stays_anonymous(self, app):
        """Liveness probing must not require a token (client disabled → 'disabled')."""
        async with await _client(app) as client:
            resp = await client.get("/graph/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "disabled"

    @pytest.mark.parametrize("path", STATS_PATHS)
    async def test_introspection_endpoints_require_auth(self, app, path):
        async with await _client(app) as client:
            resp = await client.get(path)
        assert resp.status_code == 401

    @pytest.mark.parametrize(
        "path, method",
        [("/graph/query", "post")] + [(p, "post") for p in ADMIN_PATHS],
    )
    async def test_write_and_query_endpoints_require_auth(self, app, path, method):
        body = (
            {"query": "x"}
            if path == "/graph/query"
            else {
                "csv_content": "a,b\n1,2",
                "entity_mappings": [{"entity_type": "T", "name_column": "a"}],
            }
        )
        async with await _client(app) as client:
            resp = await client.request(method, path, json=body)
        assert resp.status_code == 401


class TestAuthenticatedNonAdmin:
    async def test_stats_and_schema_allowed(self, app, user):
        fake = AsyncMock()
        fake.get_stats.return_value = {"nodes": 5, "relationships": 7}
        fake.get_schema.return_value = {"entities": []}
        with patch("app.api.v1.graph.get_graph_client", return_value=fake):
            async with await _client(app) as client:
                stats = await client.get("/graph/stats")
                schema = await client.get("/graph/schema")
        assert stats.status_code == 200
        assert stats.json() == {"nodes": 5, "relationships": 7}
        assert schema.status_code == 200

    @pytest.mark.parametrize(
        "path, body",
        [
            (
                "/graph/import/structured",
                {
                    "csv_content": "a,b\n1,2",
                    "entity_mappings": [{"entity_type": "T", "name_column": "a"}],
                },
            ),
            ("/graph/communities/detect", None),
        ],
    )
    async def test_admin_endpoints_forbidden(self, app, user, path, body):
        with patch("app.api.v1.graph.get_graph_client", return_value=AsyncMock()):
            async with await _client(app) as client:
                resp = await client.post(path, json=body)
        assert resp.status_code == 403


class TestAdminAccess:
    async def test_structured_import_executes(self, app, admin):
        fake = AsyncMock()
        fake.add_entities.return_value = ["e1", "e2"]
        fake.add_relations.return_value = []
        with patch("app.api.v1.graph.get_graph_client", return_value=fake):
            async with await _client(app) as client:
                resp = await client.post(
                    "/graph/import/structured",
                    json={
                        "csv_content": "name,sku\nWidget,W1\nGadget,G2",
                        "entity_mappings": [{"entity_type": "Product", "name_column": "name"}],
                    },
                )
        assert resp.status_code == 200
        assert resp.json() == {
            "entities_imported": 2,
            "relations_imported": 0,
        }

    async def test_community_detection_executes(self, app, admin):
        fake = AsyncMock()
        fake.detect_communities = AsyncMock(return_value={0: [["a", "b"]]})
        service = AsyncMock()
        service.detect_communities.return_value = {0: [["a", "b"], ["c"]]}
        with (
            patch("app.api.v1.graph.get_graph_client", return_value=fake),
            patch(
                "app.services.graph.community.CommunityDetectionService",
                return_value=service,
            ),
        ):
            async with await _client(app) as client:
                resp = await client.post("/graph/communities/detect")
        assert resp.status_code == 200
        assert resp.json()["total_communities"] == 2


class TestErrorSanitization:
    async def test_internal_errors_do_not_leak_details(self, app, admin):
        """str(exc) must stay in logs, never in the HTTP response."""
        fake = AsyncMock()
        fake.get_stats.side_effect = RuntimeError("boom: neo4j auth failed with password=hunter2")
        with patch("app.api.v1.graph.get_graph_client", return_value=fake):
            async with await _client(app) as client:
                resp = await client.get("/graph/stats")
        assert resp.status_code == 500
        body = resp.json()["detail"]
        assert body == "Failed to load graph stats"
        assert "hunter2" not in body
        assert "neo4j" not in body

    async def test_health_error_reports_status_only(self, app, admin):
        fake = AsyncMock()
        fake.health_check.side_effect = RuntimeError("internal dsn secret")
        with patch("app.api.v1.graph.get_graph_client", return_value=fake):
            async with await _client(app) as client:
                resp = await client.get("/graph/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "error"}
