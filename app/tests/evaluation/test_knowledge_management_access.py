"""Open-registration customers cannot operate the shared knowledge registry.

Registration/login/JWT, document routes, SQLite foreign keys, retained sources,
durable jobs, ingestion and local Qdrant are real. Observers only record and
delegate actual calls. Embeddings and Redis use existing local boundaries.
"""

import socket
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.api.deps.publication import get_publication_guard
from app.api.v1 import auth, documents
from app.config.settings import settings
from app.core.security import create_access_token
from app.models.database.document import Document, DocumentVersion
from app.models.database.user import User
from app.services.retrieval.publication import PublicationGuard
from app.tests.evaluation.test_publication_visibility import _Case as PublicationCase
from app.tests.evaluation.test_publication_visibility import _factory
from app.tests.evaluation.test_publication_visibility import case as case


@dataclass
class _Observations:
    calls: list[str] = field(default_factory=list)


@dataclass
class _Management:
    case: PublicationCase
    client: AsyncClient
    admin: dict[str, str]
    customer: dict[str, str]
    active: str
    draft: str
    failed: str
    observations: _Observations


@pytest.fixture
async def management(
    case: PublicationCase, monkeypatch: pytest.MonkeyPatch
) -> AsyncGenerator[_Management, None]:
    dns: list[str] = []

    def no_dns(*args: Any, **kwargs: Any) -> Any:
        dns.append(repr(args))
        raise AssertionError("knowledge access evaluation forbids DNS")

    for name in ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr"):
        monkeypatch.setattr(socket, name, no_dns)
    async with case.maker() as db:
        db.add(
            User(id=50, email="operator@kb-access.example.com", hashed_password="!", is_admin=True)
        )
        await db.commit()

    async def submit(document_id: str, content: str) -> str:
        version = await case.jobs.submit(
            data=content.encode(),
            title=document_id,
            filename=document_id + ".txt",
            file_type="txt",
            metadata={"uploaded_by": 50},
            pipeline_config={},
            document_id=document_id,
        )
        result = version["version_id"]
        assert isinstance(result, str)
        return result

    active = await submit("active-policy", "Published policy for read-only search.")
    assert await case.jobs.process_next(lambda config: _factory(case, config))
    draft = await submit("operator-draft", "UNPUBLISHED_OPERATOR_DRAFT_BYTES")
    failed = await submit("failed-draft", "Retained failed draft bytes.")
    async with case.maker() as db:
        row = await db.get(DocumentVersion, failed)
        assert row is not None
        row.status = "failed"
        await db.commit()

    observations = _Observations()

    def observer(name: str, original: Callable[..., Any]) -> Callable[..., Any]:
        async def invoke(*args: Any, **kwargs: Any) -> Any:
            observations.calls.append(name)
            return await original(*args, **kwargs)

        return invoke

    for name in (
        "list_documents",
        "list_versions",
        "get_version",
        "source",
        "normalized",
        "retry",
        "submit",
        "delete",
    ):
        monkeypatch.setattr(case.jobs, name, observer("jobs." + name, getattr(case.jobs, name)))
    monkeypatch.setattr(case.vector, "search", observer("vector.search", case.vector.search))
    monkeypatch.setattr(
        case.vector,
        "delete_by_filter",
        observer("vector.delete_by_filter", case.vector.delete_by_filter),
    )

    app = FastAPI()
    app.include_router(auth.router, prefix="/api/v1")
    app.include_router(documents.router, prefix="/api/v1")

    async def local_db() -> AsyncGenerator[AsyncSession, None]:
        async with case.maker() as db:
            yield db

    def local_jobs() -> Any:
        observations.calls.append("jobs.dependency")
        return case.jobs

    async def local_storage() -> Any:
        observations.calls.append("storage.construct")
        return await _factory(case, {})

    def local_vector() -> Any:
        observations.calls.append("vector.dependency")
        return case.vector

    app.dependency_overrides[get_db] = local_db
    app.dependency_overrides[documents.get_document_jobs] = local_jobs
    app.dependency_overrides[documents.get_ingestion_service] = local_storage
    app.dependency_overrides[documents.get_qdrant_client] = local_vector
    app.dependency_overrides[get_publication_guard] = lambda: PublicationGuard(case.maker)
    admin = {
        "Authorization": "Bearer "
        + create_access_token(
            {
                "sub": "operator@kb-access.example.com",
                "user_id": 50,
            }
        )
    }
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://isolated-knowledge-access.invalid"
        ) as client:
            identity = {
                "email": "customer@kb-access.example.com",
                "password": "OnlyLocal22-StrongPass!",
            }
            registered = await client.post(
                "/api/v1/auth/register", json={**identity, "is_admin": True}
            )
            assert registered.status_code == 201
            assert registered.json()["is_admin"] is False
            logged_in = await client.post("/api/v1/auth/login", json=identity)
            assert logged_in.status_code == 200
            customer = {"Authorization": "Bearer " + logged_in.json()["access_token"]}
            me = await client.get("/api/v1/auth/me", headers=customer)
            assert me.status_code == 200 and me.json()["is_admin"] is False
            async with case.maker() as db:
                registered_user = await db.scalar(
                    select(User).where(User.email == identity["email"])
                )
                assert registered_user is not None and registered_user.is_admin is False
            yield _Management(case, client, admin, customer, active, draft, failed, observations)
    finally:
        assert dns == []


async def _registry(management: _Management) -> tuple[list[Any], list[Any]]:
    async with management.case.maker() as db:
        documents_ = (
            await db.execute(
                select(
                    Document.external_doc_id,
                    Document.active_version_id,
                    Document.is_active,
                ).order_by(Document.id)
            )
        ).all()
        versions = (
            await db.execute(
                select(
                    DocumentVersion.id,
                    DocumentVersion.status,
                    DocumentVersion.content_hash,
                ).order_by(DocumentVersion.document_id, DocumentVersion.revision)
            )
        ).all()
        return list(documents_), list(versions)


async def _command(
    management: _Management, command: str, headers: dict[str, str], *, published_text: bool = False
) -> Response:
    client = management.client
    prefix = "/api/v1/documents"
    if command == "list":
        return await client.get(prefix, headers=headers)
    if command == "versions":
        return await client.get(prefix + "/operator-draft/versions", headers=headers)
    if command in {"status", "source", "text"}:
        suffix = {"status": "", "source": "/source", "text": "/text"}[command]
        version = management.active if command == "text" and published_text else management.draft
        return await client.get(prefix + "/versions/" + version + suffix, headers=headers)
    if command == "retry":
        return await client.post(
            prefix + "/versions/" + management.failed + "/retry", headers=headers
        )
    if command in {"upload", "replace"}:
        return await client.post(
            prefix + "/upload",
            headers=headers,
            json={
                "document_id": "new-policy" if command == "upload" else "active-policy",
                "title": "Valid policy",
                "file_type": "txt",
                "content": "Valid queued replacement.",
            },
        )
    if command == "file":
        return await client.post(
            prefix + "/upload/file",
            headers=headers,
            files={"file": ("policy.txt", b"Valid local file.", "text/plain")},
        )
    assert command == "delete"
    return await client.delete(prefix + "/active-policy", headers=headers)


def _no_store(response: Response) -> None:
    header = response.headers.get("cache-control", "")
    assert "private" in header and "no-store" in header


@pytest.mark.parametrize("demo", [False, True])
@pytest.mark.parametrize(
    "command",
    [
        "list",
        "versions",
        "status",
        "source",
        "text",
        "retry",
        "upload",
        "replace",
        "file",
        "delete",
    ],
)
async def test_registered_customer_cannot_use_any_knowledge_management_route(
    management: _Management, monkeypatch: pytest.MonkeyPatch, demo: bool, command: str
) -> None:
    monkeypatch.setattr(settings, "DEMO_MODE", demo)
    before = await _registry(management)
    response = await _command(management, command, management.customer)
    assert response.status_code == 403
    _no_store(response)
    assert "UNPUBLISHED_OPERATOR_DRAFT_BYTES" not in response.text
    assert management.observations.calls == []
    assert await _registry(management) == before


@pytest.mark.parametrize("demo", [False, True])
async def test_admin_controls_retain_real_source_jobs_and_storage_operations(
    management: _Management, monkeypatch: pytest.MonkeyPatch, demo: bool
) -> None:
    monkeypatch.setattr(settings, "DEMO_MODE", demo)
    for command in [
        "list",
        "versions",
        "status",
        "source",
        "text",
        "retry",
        "upload",
        "file",
        "replace",
        "delete",
    ]:
        response = await _command(management, command, management.admin, published_text=True)
        assert response.status_code == (
            202 if command in {"retry", "upload", "file", "replace"} else 200
        ), (command, response.text)
        _no_store(response)
        if command == "source":
            assert response.content == b"UNPUBLISHED_OPERATOR_DRAFT_BYTES"
        if command == "text":
            assert response.text == "Published policy for read-only search."
        if command == "retry":
            assert response.json()["status"] == "queued"
        if command == "replace":
            assert response.json()["revision"] == 2
    calls = management.observations.calls
    assert "jobs.source" in calls and "jobs.normalized" in calls
    assert "jobs.retry" in calls and "jobs.submit" in calls
    assert "storage.construct" in calls and "jobs.delete" in calls
    assert "vector.delete_by_filter" in calls
    async with management.case.maker() as db:
        active = await db.scalar(
            select(Document).where(Document.external_doc_id == "active-policy")
        )
        versions = list(
            (
                await db.scalars(
                    select(DocumentVersion).where(DocumentVersion.document_id == "active-policy")
                )
            ).all()
        )
        assert active is not None and active.is_active is False
        assert len(versions) == 2 and all(row.status == "deleted" for row in versions)


@pytest.mark.parametrize("demo", [False, True])
async def test_customer_read_only_search_still_uses_published_local_knowledge(
    management: _Management, monkeypatch: pytest.MonkeyPatch, demo: bool
) -> None:
    monkeypatch.setattr(settings, "DEMO_MODE", demo)
    response = await management.client.post(
        "/api/v1/documents/search",
        headers=management.customer,
        json={"query": "Published policy", "top_k": 5},
    )
    assert response.status_code == 200
    _no_store(response)
    results = response.json()["results"]
    assert results and {row["document_id"] for row in results} == {"active-policy"}
    assert all(row["metadata"]["version_id"] == management.active for row in results)
    assert management.observations.calls == ["vector.dependency", "vector.search"]


@pytest.mark.parametrize("demo", [False, True])
async def test_anonymous_management_requests_still_require_authentication(
    management: _Management, monkeypatch: pytest.MonkeyPatch, demo: bool
) -> None:
    monkeypatch.setattr(settings, "DEMO_MODE", demo)
    for command in ["source", "upload", "delete"]:
        response = await _command(management, command, {})
        assert response.status_code == 401
        _no_store(response)
    assert management.observations.calls == []


@pytest.mark.parametrize("demo", [False, True])
@pytest.mark.parametrize("revocation", ["demoted", "inactive"])
async def test_existing_admin_token_observes_current_database_privileges(
    management: _Management,
    monkeypatch: pytest.MonkeyPatch,
    demo: bool,
    revocation: str,
) -> None:
    monkeypatch.setattr(settings, "DEMO_MODE", demo)
    allowed = await _command(management, "source", management.admin)
    assert allowed.status_code == 200
    assert allowed.content == b"UNPUBLISHED_OPERATOR_DRAFT_BYTES"
    async with management.case.maker() as db:
        admin = await db.get(User, 50)
        assert admin is not None
        if revocation == "demoted":
            admin.is_admin = False
        else:
            admin.is_active = False
        await db.commit()
    management.observations.calls.clear()
    before = await _registry(management)
    for command in ["list", "source", "upload", "delete"]:
        denied = await _command(management, command, management.admin)
        assert denied.status_code == 403
        _no_store(denied)
        assert "UNPUBLISHED_OPERATOR_DRAFT_BYTES" not in denied.text
    assert management.observations.calls == []
    assert await _registry(management) == before


@pytest.mark.parametrize("malformed_json", [False, True])
async def test_validation_failures_are_private_without_executing_storage(
    management: _Management, malformed_json: bool
) -> None:
    if malformed_json:
        response = await management.client.post(
            "/api/v1/documents/upload",
            headers={**management.admin, "Content-Type": "application/json"},
            content='{"broken"',
        )
    else:
        response = await management.client.post(
            "/api/v1/documents/upload",
            headers=management.admin,
            json={"title": "Valid title", "file_type": "txt", "content": ""},
        )
    assert response.status_code == 422
    _no_store(response)
    # Dependencies may run before model validation; malformed JSON can fail
    # before authentication. Neither result is authorization proof.
    assert all(name == "jobs.dependency" for name in management.observations.calls)
