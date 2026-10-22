"""Document lifecycle through ASGI with real SQLite jobs and no external services."""

from io import BytesIO
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI, HTTPException, UploadFile
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.api.deps import get_current_user
from app.api.v1 import documents
from app.config.settings import settings
from app.core.exceptions import ValidationError
from app.models.database.base import Base
from app.models.database.document import DocumentVersion
from app.models.database.user import User
from app.services.documents.jobs import DocumentJobService
from app.services.embeddings import EmbeddingFactory
from app.services.embeddings.base import EmbeddingServiceBase
from app.services.retrieval.factory import RetrievalFactory
from app.services.retrieval.vector_base import SearchResult

PREFIX = "/api/v1/documents"


@pytest.mark.parametrize("field", ["embedding_provider", "embedding_model"])
async def test_queued_job_with_previous_embedding_configuration_fails_before_clients(
    monkeypatch, field
):
    config = documents.pipeline_config()
    config[field] = "previous-deployment"
    constructor = AsyncMock()
    monkeypatch.setattr(EmbeddingFactory, "create", constructor)
    with pytest.raises(ValidationError, match="submit a new revision"):
        await documents.build_ingestion_service(config)
    constructor.assert_not_called()


@pytest.fixture
async def api(monkeypatch):
    monkeypatch.setattr(settings, "DEMO_MODE", False)
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", poolclass=StaticPool)
    async with engine.begin() as connection:
        await connection.run_sync(
            lambda sync: Base.metadata.create_all(
                sync,
                tables=[
                    Base.metadata.tables["documents"],
                    Base.metadata.tables["document_versions"],
                ],
            )
        )
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    jobs = DocumentJobService(sessions)
    app = FastAPI()
    app.include_router(documents.router, prefix="/api/v1")
    admin = User(id=7, email="admin@example.test", is_admin=True, is_active=True)
    app.dependency_overrides[get_current_user] = lambda: admin
    app.dependency_overrides[documents.get_document_jobs] = lambda: jobs
    service = SimpleNamespace(
        close=AsyncMock(),
        delete_document=AsyncMock(return_value={"document_id": "policy", "deleted_chunks": 2}),
    )
    app.dependency_overrides[documents.get_ingestion_service] = lambda: service
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield SimpleNamespace(
            app=app,
            client=client,
            jobs=jobs,
            sessions=sessions,
            admin=admin,
            service=service,
            engine=engine,
        )
    await engine.dispose()


async def upload(api: SimpleNamespace, **overrides: Any) -> httpx.Response:
    payload = {"title": "Return policy", "content": "Returns within seven days.", **overrides}
    response: httpx.Response = await api.client.post(PREFIX + "/upload", json=payload)
    return response


async def process(api: SimpleNamespace, *, error: Exception | None = None) -> None:
    service = SimpleNamespace(
        close=AsyncMock(),
        ingest_bytes=AsyncMock(
            return_value={
                "chunks_count": 1,
                "warnings": [],
                "_normalized_text": "退货政策\nNormalized policy.",
            },
            side_effect=error,
        ),
    )
    assert await api.jobs.process_next(AsyncMock(return_value=service))
    service.close.assert_awaited_once()


async def test_text_upload_is_queued_retained_and_scope_is_persisted(api):
    scope = {"lang": "zh", "topic": "returns", "product": "phone", "region": "CN"}
    response = await upload(api, metadata=scope, document_id="policy", file_type="md")
    assert response.status_code == 202
    job = response.json()
    assert (job["status"], job["attempts"], job["revision"]) == ("queued", 0, 1)
    assert job["filename"] == "Return policy.md"
    assert job["result"] is None
    async with api.sessions() as db:
        version = await db.get(DocumentVersion, job["version_id"])
        assert version.source_metadata == {**scope, "uploaded_by": 7}
        assert version.pipeline_config["chunking_strategy"] == "semantic"
    assert (await api.jobs.source(job["version_id"]))[0] == b"Returns within seven days."
    assert (await api.client.get(PREFIX + "/versions/" + job["version_id"])).json() == job
    listing = (await api.client.get(PREFIX)).json()
    assert listing[0]["document_id"] == "policy"
    assert listing[0]["is_active"] is False


async def test_duplicate_uploads_reuse_version_and_replacements_retain_originals(api):
    first = (await upload(api, document_id="policy")).json()
    duplicate = (await upload(api, document_id="policy")).json()
    second = (await upload(api, document_id="policy", content="Updated policy.")).json()
    assert duplicate["version_id"] == first["version_id"]
    assert second["revision"] == 2
    assert second["version_id"] != first["version_id"]
    versions = (await api.client.get(PREFIX + "/policy/versions")).json()
    assert [item["revision"] for item in versions] == [2, 1]
    original = await api.client.get(PREFIX + "/versions/" + first["version_id"] + "/source")
    assert original.content == b"Returns within seven days."


@pytest.mark.parametrize(
    "payload",
    [
        {"title": " "},
        {"content": "\n\t"},
        {"title": "x" * 501},
        {"document_id": "../escape"},
        {"file_type": "pdf"},
        {"metadata": {"region": 12}},
        {"metadata": {"region": "  "}},
        {"metadata": {"notes": "x" * 17_000}},
    ],
)
async def test_invalid_text_upload_returns_422_without_queuing(api, payload):
    assert (await upload(api, **payload)).status_code == 422
    assert await api.jobs.list_documents() == []


@pytest.mark.parametrize(
    "reserved",
    [
        "document_id",
        "version_id",
        "chunk_id",
        "uploaded_by",
        "source",
        "page",
        "source_type",
        "source_document_id",
        "source_document_ids",
        "source_version_id",
        "source_version_ids",
        "source_chunk_ids",
        "embedding_provider",
        "embedding_model",
    ],
)
async def test_protected_metadata_is_rejected_in_both_upload_formats(api, reserved):
    import json

    assert (await upload(api, metadata={reserved: "forged"})).status_code == 422
    response = await api.client.post(
        PREFIX + "/upload/file",
        params={"metadata": json.dumps({reserved: "forged"})},
        files={"file": ("policy.md", b"# Policy")},
    )
    assert response.status_code == 422
    assert await api.jobs.list_documents() == []


async def test_upload_byte_limits_count_utf8_and_stop_file_reading(api, monkeypatch):
    monkeypatch.setattr(settings, "DOCUMENT_MAX_BYTES", 16)
    assert (await upload(api, content="退货政策退货")).status_code == 413
    response = await api.client.post(
        PREFIX + "/upload/file", files={"file": ("policy.txt", b"x" * 17)}
    )
    assert response.status_code == 413
    assert await api.jobs.list_documents() == []

    file = UploadFile(filename="large.txt", file=BytesIO(b"x" * 1000))
    read = AsyncMock(wraps=file.read)
    monkeypatch.setattr(file, "read", read)
    with pytest.raises(HTTPException) as exc:
        await documents.upload_document_file(
            file=file,
            title=None,
            document_id=None,
            metadata=None,
            current_user=api.admin,
            jobs=api.jobs,
        )
    assert exc.value.status_code == 413
    assert [call.args[0] for call in read.await_args_list] == [17]
    assert file.file.closed


@pytest.mark.parametrize(
    "params",
    [
        {"title": " "},
        {"title": "x" * 501},
        {"metadata": "not-json"},
        {"metadata": "[]"},
        {"metadata": '{"product": null}'},
        {"document_id": "../escape"},
    ],
)
async def test_invalid_file_fields_return_422(api, params):
    response = await api.client.post(
        PREFIX + "/upload/file", params=params, files={"file": ("policy.txt", b"Policy")}
    )
    assert response.status_code == 422


async def test_file_upload_preserves_original_encoding_name_and_scope(api):
    data = "退货政策内容".encode("gbk")
    response = await api.client.post(
        PREFIX + "/upload/file",
        params={"metadata": '{"region":"CN","lang":"zh"}'},
        files={"file": ("../政策.txt", data, "text/plain")},
    )
    assert response.status_code == 202
    job = response.json()
    assert job["title"] == "政策"
    assert job["filename"] == "政策.txt"
    source = await api.client.get(PREFIX + "/versions/" + job["version_id"] + "/source")
    assert source.content == data
    assert source.headers["content-type"] == "application/octet-stream"
    assert source.headers["x-content-type-options"] == "nosniff"
    assert (
        source.headers["content-disposition"]
        == "attachment; filename*=UTF-8''%E6%94%BF%E7%AD%96.txt"
    )
    async with api.sessions() as db:
        version = await db.get(DocumentVersion, job["version_id"])
        assert version.source_metadata == {"region": "CN", "lang": "zh", "uploaded_by": 7}


async def test_unsupported_and_empty_files_do_not_create_jobs(api):
    response = await api.client.post(
        PREFIX + "/upload/file", files={"file": ("policy.docx", b"data")}
    )
    assert response.status_code == 400
    response = await api.client.post(PREFIX + "/upload/file", files={"file": ("policy.txt", b"")})
    assert response.status_code == 422
    assert await api.jobs.list_documents() == []


async def test_normalized_download_retry_and_delete_lifecycle(api):
    job = (await upload(api, document_id="policy")).json()
    version = PREFIX + "/versions/" + job["version_id"]
    assert (await api.client.get(version + "/text")).status_code == 409
    assert (await api.client.post(version + "/retry")).status_code == 409
    await process(api, error=RuntimeError("parser failed"))
    assert (await api.client.get(version)).json()["status"] == "failed"
    retry = await api.client.post(version + "/retry")
    assert retry.status_code == 202
    assert retry.json()["status"] == "queued"
    await process(api)
    text = await api.client.get(version + "/text")
    assert text.status_code == 200
    assert text.text == "退货政策\nNormalized policy."
    assert text.headers["content-type"] == "text/plain; charset=utf-8"
    assert ".normalized.txt" in text.headers["content-disposition"]
    deleted = await api.client.delete(PREFIX + "/policy")
    assert deleted.status_code == 200
    api.service.delete_document.assert_awaited_once_with("policy")
    api.service.close.assert_awaited_once()
    assert (await api.client.get(version + "/source")).status_code == 404
    assert (await api.client.get(version + "/text")).status_code == 404


LIFECYCLE_ROUTES = [
    ("POST", "/upload", {"json": {"title": "Policy", "content": "Body"}}),
    ("POST", "/upload/file", {"files": {"file": ("policy.txt", b"Body")}}),
    ("GET", "", {}),
    ("GET", "/versions/missing", {}),
    ("POST", "/versions/missing/retry", {}),
    ("GET", "/versions/missing/source", {}),
    ("GET", "/versions/missing/text", {}),
    ("GET", "/missing/versions", {}),
    ("DELETE", "/missing", {}),
]


@pytest.mark.parametrize("identity,expected", [(None, 401), ("inactive", 403), ("member", 403)])
@pytest.mark.parametrize("method,path,kwargs", LIFECYCLE_ROUTES)
async def test_lifecycle_requires_active_authorized_identity(
    api, monkeypatch, identity, expected, method, path, kwargs
):
    user = (
        None
        if identity is None
        else User(
            id=8, email="member@example.test", is_active=identity != "inactive", is_admin=False
        )
    )
    api.app.dependency_overrides[get_current_user] = lambda: user
    api.app.dependency_overrides.pop(documents.get_ingestion_service)
    builder = AsyncMock()
    monkeypatch.setattr(documents, "build_ingestion_service", builder)
    response = await api.client.request(method, PREFIX + path, **kwargs)
    assert response.status_code == expected
    builder.assert_not_awaited()


@pytest.mark.parametrize(
    "method,path",
    [
        ("GET", "/versions/missing"),
        ("POST", "/versions/missing/retry"),
        ("GET", "/versions/missing/source"),
        ("GET", "/versions/missing/text"),
        ("GET", "/missing/versions"),
    ],
)
async def test_missing_lifecycle_resources_return_404(api, method, path):
    assert (await api.client.request(method, PREFIX + path)).status_code == 404


async def test_delete_unregistered_legacy_document_is_idempotent(api):
    api.service.delete_document.return_value = {"document_id": "missing", "deleted_chunks": 0}
    response = await api.client.delete(PREFIX + "/missing")
    assert response.status_code == 200
    assert response.json()["deleted_chunks"] == 0
    api.service.delete_document.assert_awaited_once_with("missing")
    api.service.close.assert_awaited_once()


async def test_search_auth_rejects_anonymous_before_constructing_clients(api, monkeypatch):
    api.app.dependency_overrides[get_current_user] = lambda: None
    constructor = AsyncMock()
    monkeypatch.setattr(EmbeddingFactory, "create_from_settings", constructor)
    response = await api.client.post(PREFIX + "/search", json={"query": "policy"})
    assert response.status_code == 401
    constructor.assert_not_called()


async def test_search_cleanup_includes_embedding_cache_and_survives_vector_close_error(
    api, monkeypatch
):
    from app.services.embeddings.cached_embeddings import CachedEmbeddingService

    provider = Mock(spec=EmbeddingServiceBase, model="test", dimensions=2, close=AsyncMock())
    embeddings = CachedEmbeddingService(provider, redis_url="redis://unused")
    redis = Mock(spec=Redis, close=AsyncMock())
    embeddings._redis = redis
    vector = SimpleNamespace(
        client=SimpleNamespace(close=AsyncMock(side_effect=RuntimeError("close error"))),
        search=AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(EmbeddingFactory, "create_from_settings", lambda: embeddings)
    monkeypatch.setattr(RetrievalFactory, "create_vector_client", lambda **kwargs: vector)
    response = await api.client.post(PREFIX + "/search", json={"query": "policy"})
    assert response.status_code == 200
    vector.client.close.assert_awaited_once()
    redis.close.assert_awaited_once()
    provider.close.assert_awaited_once()


async def test_search_vector_construction_failure_still_closes_embedding_provider(monkeypatch):
    provider = SimpleNamespace(close=AsyncMock())
    monkeypatch.setattr(EmbeddingFactory, "create_from_settings", lambda: provider)

    def fail(**kwargs):
        raise RuntimeError("invalid vector configuration")

    monkeypatch.setattr(RetrievalFactory, "create_vector_client", fail)
    with pytest.raises(RuntimeError, match="invalid vector configuration"):
        await anext(documents.get_qdrant_client())
    provider.close.assert_awaited_once()


@pytest.mark.parametrize("fail_search", [False, True])
async def test_search_releases_owned_vector_and_embedding_clients(api, monkeypatch, fail_search):
    embeddings = SimpleNamespace(close=AsyncMock())
    vector = SimpleNamespace(
        client=SimpleNamespace(close=AsyncMock()),
        search=AsyncMock(
            return_value=[SearchResult("doc", "Policy", 0.9, {"chunk_id": "chunk"})],
            side_effect=RuntimeError("vector offline") if fail_search else None,
        ),
    )
    monkeypatch.setattr(EmbeddingFactory, "create_from_settings", lambda: embeddings)
    monkeypatch.setattr(RetrievalFactory, "create_vector_client", lambda **kwargs: vector)
    response = await api.client.post(PREFIX + "/search", json={"query": "Policy"})
    assert response.status_code == (500 if fail_search else 200)
    vector.client.close.assert_awaited_once()
    embeddings.close.assert_awaited_once()
    if not fail_search:
        assert response.json()["results"][0]["chunk_id"] == "chunk"


@pytest.mark.parametrize(
    "database,vector,worker,expected",
    [
        (True, True, True, 200),
        (False, True, True, 503),
        (True, False, True, 503),
        (True, True, False, 503),
    ],
)
async def test_health_reports_real_dependency_failures(
    api, monkeypatch, database, vector, worker, expected
):
    monkeypatch.setattr(documents, "async_session_maker", api.sessions)
    if not database:
        async with api.engine.begin() as connection:
            await connection.run_sync(
                lambda sync: Base.metadata.tables["document_versions"].drop(sync)
            )
    api.app.state.document_worker = SimpleNamespace(done=lambda: not worker)
    remote = AsyncMock()
    remote.get.return_value = httpx.Response(200 if vector else 503)
    remote.__aenter__.return_value = remote
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: remote)
    response = await api.client.get(PREFIX + "/health")
    assert response.status_code == expected
    body = response.json()
    assert (body["database"], body["vector_db"], body["worker"]) == (database, vector, worker)
    remote.get.assert_awaited_once()
