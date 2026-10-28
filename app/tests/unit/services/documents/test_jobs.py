"""Durable ingestion behavior against real SQLite persistence, without providers.

SQLite verifies sequential lifecycle and transaction failures here. Production
SKIP LOCKED coordination still requires PostgreSQL integration coverage.
"""

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy import String, event, inspect, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine import Dialect
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, undefer
from sqlalchemy.pool import StaticPool

from app.core.exceptions import ConflictError, ExternalServiceError, NotFoundError, ValidationError
from app.models.database.base import Base
from app.models.database.document import Document, DocumentVersion
from app.services.documents.ingestion import DocumentIngestionService
from app.services.documents.jobs import DocumentJobService
from app.services.embeddings.base import EmbeddingServiceBase
from app.services.graph.base import GraphClient, GraphClientError
from app.services.retrieval.qdrant_client import QdrantClient


@pytest.fixture
async def jobs(monkeypatch) -> AsyncIterator[DocumentJobService]:
    # These cases test SQL/job transitions; epoch IO has dedicated controls.
    monkeypatch.setattr("app.services.documents.jobs.bump_kb_epoch", AsyncMock())
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
    yield DocumentJobService(sessions)
    await engine.dispose()


async def submit(jobs: DocumentJobService, **changes: Any) -> dict[str, Any]:
    request: dict[str, Any] = {
        "data": b"Refunds are available within seven days.",
        "title": "Refund policy",
        "filename": "refund-policy.txt",
        "file_type": "txt",
        "metadata": {"uploaded_by": 1, "language": "en"},
        "pipeline_config": {"chunk_size": 512, "chunk_overlap": 50},
    }
    request.update(changes)
    return await jobs.submit(**request)


def provider(
    *,
    warnings: list[str] | None = None,
    graph_status: str = "disabled",
    error: BaseException | None = None,
) -> tuple[Mock, AsyncMock]:
    service = Mock(
        spec_set=DocumentIngestionService,
        ingest_bytes=AsyncMock(
            return_value={
                "chunks_count": 2,
                "warnings": warnings or [],
                "graph_status": graph_status,
                "_normalized_text": "Normalized refund policy.",
            },
            side_effect=error,
        ),
        close=AsyncMock(),
        delete_document=AsyncMock(return_value={"document_id": "doc-a", "deleted_chunks": 2}),
    )
    return service, AsyncMock(return_value=service)


@pytest.mark.asyncio
async def test_exact_reupload_deduplicates_but_different_pipeline_creates_revision(
    jobs: DocumentJobService,
):
    first = await submit(jobs)
    duplicate = await submit(jobs, metadata={"uploaded_by": 99, "language": "en"})
    assert duplicate["document_id"] == first["document_id"]
    assert duplicate["version_id"] == first["version_id"]
    assert duplicate["status"] == "queued"
    assert duplicate["attempts"] == 0
    assert len(await jobs.list_documents()) == 1

    revised = await submit(jobs, pipeline_config={"chunk_size": 256, "chunk_overlap": 30})
    assert revised["document_id"] == first["document_id"]
    assert revised["version_id"] != first["version_id"]
    assert revised["revision"] == 2


@pytest.mark.asyncio
async def test_changed_source_and_restoring_older_source_create_new_immutable_revisions(
    jobs: DocumentJobService,
):
    first = await submit(jobs, document_id="doc-a")
    second = await submit(jobs, document_id="doc-a", data=b"Updated refund policy.")
    third = await submit(jobs, document_id="doc-a")
    assert [version["revision"] for version in (first, second, third)] == [1, 2, 3]
    assert len({version["version_id"] for version in (first, second, third)}) == 3
    assert (await jobs.source(first["version_id"]))[0] == (await jobs.source(third["version_id"]))[
        0
    ]
    assert (await jobs.source(second["version_id"]))[0] == b"Updated refund policy."
    assert [v["revision"] for v in await jobs.list_versions("doc-a")] == [3, 2, 1]


@pytest.mark.asyncio
async def test_original_bytes_survive_new_service_and_status_does_not_load_large_columns(
    jobs: DocumentJobService,
):
    uploaded = await submit(jobs, data=b"\x00\x01original\xff", filename="..\\secret\\original.pdf")
    restarted = DocumentJobService(jobs.sessions)
    assert await restarted.source(uploaded["version_id"]) == (
        b"\x00\x01original\xff",
        "original.pdf",
        "txt",
    )
    async with jobs.sessions() as db:
        version = await db.get(DocumentVersion, uploaded["version_id"])
        assert version is not None
        state = inspect(version)
        assert state is not None
        assert {"source_content", "normalized_text"} <= state.unloaded
    status = await restarted.get_version(uploaded["version_id"])
    assert "source_content" not in status
    assert "normalized_text" not in status


@pytest.mark.asyncio
async def test_processing_is_visible_before_provider_finishes_and_success_becomes_active(
    jobs: DocumentJobService,
):
    uploaded = await submit(jobs, document_id="doc-a")
    service, factory = provider()

    async def ingest(**kwargs):
        visible = await jobs.get_version(uploaded["version_id"])
        assert visible["status"] == "processing"
        assert visible["attempts"] == 1
        assert kwargs["data"] == (await jobs.source(uploaded["version_id"]))[0]
        assert kwargs["metadata"]["version_id"] == uploaded["version_id"]
        assert kwargs["metadata"]["original_filename"] == "refund-policy.txt"
        assert kwargs["document_id"] == "doc-a"
        return {"chunks_count": 2, "warnings": [], "_normalized_text": "Normalized source."}

    service.ingest_bytes.side_effect = ingest
    assert await jobs.process_next(factory) is True
    status = await jobs.get_version(uploaded["version_id"])
    assert status["status"] == "indexed"
    assert "_normalized_text" not in status["result"]
    document = (await jobs.list_documents())[0]
    assert document["active_version_id"] == uploaded["version_id"]
    assert document["is_active"] is True
    assert document["chunks_count"] == 2
    async with jobs.sessions() as db:
        stored = await db.scalar(
            select(DocumentVersion)
            .where(DocumentVersion.id == uploaded["version_id"])
            .options(undefer(DocumentVersion.normalized_text))
        )
        assert stored is not None
        assert stored.normalized_text == "Normalized source."
    service.close.assert_awaited_once()
    assert await jobs.process_next(factory) is False


@pytest.mark.asyncio
async def test_partial_result_remains_active_and_retry_reuses_revision_and_source(
    jobs: DocumentJobService,
):
    uploaded = await submit(jobs, document_id="doc-a")
    service, factory = provider(warnings=["Graph extraction unavailable"], graph_status="failed")
    await jobs.process_next(factory)
    partial = await jobs.get_version(uploaded["version_id"])
    assert partial["status"] == "partial"
    assert (await jobs.list_documents())[0]["active_version_id"] == uploaded["version_id"]

    queued = await jobs.retry(uploaded["version_id"])
    assert queued["status"] == "queued"
    assert queued["revision"] == 1
    next_service, next_factory = provider()
    assert await jobs.process_next(next_factory)
    indexed = await jobs.get_version(uploaded["version_id"])
    assert indexed["status"] == "indexed"
    assert indexed["attempts"] == 2
    assert service.ingest_bytes.await_args.kwargs == next_service.ingest_bytes.await_args.kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [ValidationError("Unreadable PDF"), RuntimeError("private provider token")]
)
async def test_failed_replacement_preserves_active_version_and_can_retry(
    jobs: DocumentJobService, error
):
    active = await submit(jobs, document_id="doc-a")
    _, good_factory = provider()
    await jobs.process_next(good_factory)
    replacement = await submit(jobs, document_id="doc-a", data=b"Replacement source")
    failed_service, failed_factory = provider(error=error)
    await jobs.process_next(failed_factory)

    status = await jobs.get_version(replacement["version_id"])
    assert status["status"] == "failed"
    assert status["error"]
    assert "private provider token" not in status["error"]
    assert (await jobs.list_documents())[0]["active_version_id"] == active["version_id"]
    assert (await jobs.source(replacement["version_id"]))[0] == b"Replacement source"
    failed_service.close.assert_awaited_once()
    assert await jobs.process_next(failed_factory) is False

    await jobs.retry(replacement["version_id"])
    _, recovered_factory = provider()
    await jobs.process_next(recovered_factory)
    assert (await jobs.get_version(replacement["version_id"]))["attempts"] == 2
    assert (await jobs.list_documents())[0]["active_version_id"] == replacement["version_id"]


@pytest.mark.asyncio
async def test_restarted_worker_recovers_persisted_processing_job_with_same_ids(
    jobs: DocumentJobService,
):
    uploaded = await submit(jobs, document_id="doc-a")
    async with jobs.sessions() as db:
        version = await db.get(DocumentVersion, uploaded["version_id"])
        assert version is not None
        version.status, version.attempts = "processing", 1
        await db.commit()

    restarted = DocumentJobService(jobs.sessions)
    service, factory = provider()
    assert await restarted.process_next(factory)
    assert (
        service.ingest_bytes.await_args.kwargs["metadata"]["version_id"] == uploaded["version_id"]
    )
    assert service.ingest_bytes.await_args.kwargs["document_id"] == "doc-a"
    status = await restarted.get_version(uploaded["version_id"])
    assert status["status"] == "indexed"
    assert status["attempts"] == 2


@pytest.mark.asyncio
async def test_cancelled_worker_leaves_recoverable_processing_state(jobs: DocumentJobService):
    uploaded = await submit(jobs, document_id="doc-a")
    service, factory = provider(error=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await jobs.process_next(factory)
    assert (await jobs.get_version(uploaded["version_id"]))["status"] == "processing"
    service.close.assert_awaited_once()
    _, retry_factory = provider()
    assert await DocumentJobService(jobs.sessions).process_next(retry_factory)
    assert (await jobs.get_version(uploaded["version_id"]))["status"] == "indexed"


@pytest.mark.asyncio
async def test_retry_refuses_an_older_revision_even_when_newest_revision_failed(
    jobs: DocumentJobService,
):
    first = await submit(jobs, document_id="doc-a")
    _, failed_factory = provider(error=ValidationError("Unreadable source"))
    await jobs.process_next(failed_factory)
    second = await submit(jobs, document_id="doc-a", data=b"new revision")
    await jobs.process_next(failed_factory)
    assert (await jobs.get_version(second["version_id"]))["status"] == "failed"
    with pytest.raises(ConflictError, match="newer|latest"):
        await jobs.retry(first["version_id"])
    assert (await jobs.retry(second["version_id"]))["status"] == "queued"


@pytest.mark.asyncio
async def test_delete_purges_all_originals_and_prevents_queued_revision_reappearing(
    jobs: DocumentJobService,
):
    first = await submit(jobs, document_id="doc-a")
    _, factory = provider()
    await jobs.process_next(factory)
    second = await submit(jobs, document_id="doc-a", data=b"pending update")
    service, _ = provider()
    result = await jobs.delete("doc-a", service)
    assert result["deleted_chunks"] == 2
    service.delete_document.assert_awaited_once_with("doc-a")
    for version in (first, second):
        assert (await jobs.get_version(version["version_id"]))["status"] == "deleted"
        with pytest.raises(NotFoundError):
            await jobs.source(version["version_id"])
        with pytest.raises(ConflictError):
            await jobs.retry(version["version_id"])
    document = (await jobs.list_documents())[0]
    assert document["is_active"] is False
    assert document["active_version_id"] is None
    assert document["chunks_count"] == 0
    assert await jobs.process_next(factory) is False
    with pytest.raises(ConflictError, match="deleted"):
        await submit(jobs, document_id="doc-a")


@pytest.mark.asyncio
async def test_failed_external_delete_retains_sources_and_active_registry(jobs: DocumentJobService):
    first = await submit(jobs, document_id="doc-a")
    _, factory = provider()
    await jobs.process_next(factory)
    service, _ = provider()
    service.delete_document.side_effect = RuntimeError("Vector database unavailable")
    with pytest.raises(RuntimeError):
        await jobs.delete("doc-a", service)
    assert (await jobs.get_version(first["version_id"]))["status"] == "indexed"
    assert (await jobs.list_documents())[0]["active_version_id"] == first["version_id"]
    assert (await jobs.source(first["version_id"]))[0]


@pytest.mark.asyncio
async def test_final_registry_commit_failure_does_not_strand_an_indexed_job(
    jobs: DocumentJobService,
):
    uploaded = await submit(jobs, document_id="doc-a")
    _, factory = provider()

    def fail_final_commit(session):
        if any(
            isinstance(obj, Document) and obj.active_version_id
            for obj in session.identity_map.values()
        ):
            raise RuntimeError("Simulated loss before document commit")

    event.listen(Session, "before_commit", fail_final_commit)
    try:
        with pytest.raises(RuntimeError, match="Simulated loss"):
            await jobs.process_next(factory)
    finally:
        event.remove(Session, "before_commit", fail_final_commit)

    assert (await jobs.list_documents())[0]["active_version_id"] is None
    assert (await jobs.get_version(uploaded["version_id"]))["status"] == "processing"
    _, restarted_factory = provider()
    assert await DocumentJobService(jobs.sessions).process_next(restarted_factory)
    assert (await jobs.list_documents())[0]["active_version_id"] == uploaded["version_id"]


@pytest.mark.asyncio
async def test_missing_sources_and_invalid_submissions_are_explicit(jobs: DocumentJobService):
    with pytest.raises(ValidationError):
        await submit(jobs, data=b"")
    with pytest.raises(ValidationError):
        await submit(jobs, title="   ")
    with pytest.raises(NotFoundError):
        await jobs.get_version("absent")
    with pytest.raises(NotFoundError):
        await jobs.source("absent")
    with pytest.raises(NotFoundError):
        await jobs.list_versions("absent")


@pytest.mark.asyncio
async def test_normalized_source_availability_and_failed_retry_retention(jobs: DocumentJobService):
    uploaded = await submit(jobs, document_id="doc-a")
    with pytest.raises(NotFoundError):
        await jobs.normalized("absent")
    with pytest.raises(ConflictError, match="not available"):
        await jobs.normalized(uploaded["version_id"])
    _, partial_factory = provider(warnings=["Graph pending"], graph_status="failed")
    await jobs.process_next(partial_factory)
    assert await jobs.normalized(uploaded["version_id"]) == (
        "Normalized refund policy.",
        "refund-policy.txt",
    )

    await jobs.retry(uploaded["version_id"])
    _, failed_factory = provider(error=RuntimeError("Embeddings unavailable"))
    await jobs.process_next(failed_factory)
    assert (await jobs.get_version(uploaded["version_id"]))["status"] == "failed"
    assert (await jobs.normalized(uploaded["version_id"]))[0] == "Normalized refund policy."

    service, _ = provider()
    await jobs.delete("doc-a", service)
    with pytest.raises(NotFoundError):
        await jobs.normalized(uploaded["version_id"])


@pytest.mark.asyncio
async def test_retry_rechecks_status_after_concurrent_delete_finishes(
    jobs: DocumentJobService, monkeypatch
):
    uploaded = await submit(jobs, document_id="doc-a")
    _, failed_factory = provider(error=ValidationError("Unreadable source"))
    await jobs.process_next(failed_factory)
    original_scalar = AsyncSession.scalar
    deleted = False

    async def delete_before_lock(session, statement, *args, **kwargs):
        nonlocal deleted
        if not deleted and statement.column_descriptions[0].get("entity") is Document:
            deleted = True
            async with jobs.sessions() as other:
                version = await other.get(DocumentVersion, uploaded["version_id"])
                assert version is not None
                version.status, version.source_content = "deleted", None
                await other.commit()
        return await original_scalar(session, statement, *args, **kwargs)

    monkeypatch.setattr(AsyncSession, "scalar", delete_before_lock)
    with pytest.raises(ConflictError):
        await jobs.retry(uploaded["version_id"])
    assert deleted
    assert (await jobs.get_version(uploaded["version_id"]))["status"] == "deleted"


@pytest.mark.asyncio
async def test_invalid_provider_result_cannot_activate_a_failed_revision(jobs: DocumentJobService):
    first = await submit(jobs, document_id="doc-a")
    _, factory = provider()
    await jobs.process_next(factory)
    replacement = await submit(jobs, document_id="doc-a", data=b"new source")
    service, broken_factory = provider()
    service.ingest_bytes.return_value = {"warnings": []}  # No published chunk count.
    await jobs.process_next(broken_factory)
    assert (await jobs.get_version(replacement["version_id"]))["status"] == "failed"
    assert (await jobs.list_documents())[0]["active_version_id"] == first["version_id"]


@pytest.mark.asyncio
async def test_provider_cleanup_failure_does_not_lose_successful_job(jobs: DocumentJobService):
    uploaded = await submit(jobs, document_id="doc-a")
    service, factory = provider()
    service.close.side_effect = RuntimeError("Already closed")
    assert await jobs.process_next(factory)
    assert (await jobs.get_version(uploaded["version_id"]))["status"] == "indexed"
    assert (await jobs.list_documents())[0]["active_version_id"] == uploaded["version_id"]


@pytest.mark.asyncio
async def test_successful_ocr_warning_does_not_mark_a_fully_indexed_document_partial(
    jobs: DocumentJobService,
):
    uploaded = await submit(jobs, document_id="doc-a")
    _, factory = provider(warnings=["OCR was used for scanned pages"], graph_status="indexed")
    await jobs.process_next(factory)
    assert (await jobs.get_version(uploaded["version_id"]))["status"] == "indexed"


@pytest.mark.asyncio
async def test_delete_retry_purges_graph_when_vector_delete_already_succeeded(
    jobs: DocumentJobService, monkeypatch
):
    uploaded = await submit(jobs, document_id="doc-a")
    _, factory = provider()
    await jobs.process_next(factory)
    # Resume a partial/external deletion whose vector points are already gone.
    vector = Mock(spec_set=QdrantClient, delete_by_filter=AsyncMock(return_value=0))
    graph = Mock(
        spec_set=GraphClient,
        delete_document_graph=AsyncMock(side_effect=[GraphClientError("Graph offline"), {}]),
    )
    service = DocumentIngestionService(
        qdrant_client=vector,
        embedding_service=Mock(spec_set=EmbeddingServiceBase),
        graph_client=graph,
        graph_enabled=True,
    )
    monkeypatch.setattr("app.services.documents.ingestion.purge_keyword_document", AsyncMock())
    bump = AsyncMock()
    monkeypatch.setattr("app.services.documents.ingestion.bump_kb_epoch", bump)

    with pytest.raises(ExternalServiceError, match="Graph"):
        await jobs.delete("doc-a", service)
    assert (await jobs.source(uploaded["version_id"]))[0]
    assert (await jobs.get_version(uploaded["version_id"]))["status"] == "indexed"

    result = await DocumentJobService(jobs.sessions).delete("doc-a", service)
    assert result["deleted_chunks"] == 0
    assert graph.delete_document_graph.await_count == 2
    assert bump.await_count == 2
    assert (await jobs.get_version(uploaded["version_id"]))["status"] == "deleted"
    with pytest.raises(NotFoundError):
        await jobs.source(uploaded["version_id"])


@pytest.mark.asyncio
async def test_postgres_worker_claim_uses_non_key_exclusive_lock_and_skips_busy_documents(
    jobs: DocumentJobService, monkeypatch
):
    """Pin PostgreSQL SQL semantics that SQLite does not implement."""
    await submit(jobs, document_id="doc-a")
    statements = []
    original_scalar = AsyncSession.scalar

    async def record_claim(session, statement, *args, **kwargs):
        if statement.column_descriptions[0].get("entity") is Document:
            statements.append(
                str(
                    statement.compile(
                        dialect=cast(Callable[[], Dialect], postgresql.dialect)(),
                        compile_kwargs={"literal_binds": True},
                    )
                )
            )
        return await original_scalar(session, statement, *args, **kwargs)

    monkeypatch.setattr(AsyncSession, "scalar", record_claim)
    _, factory = provider()
    assert await jobs.process_next(factory)

    assert len(statements) == 1
    assert "FOR NO KEY UPDATE SKIP LOCKED" in statements[0]
    assert "FROM documents" in statements[0]
    assert "min(document_versions.created_at)" in statements[0]
    assert "LIMIT 1" in statements[0]


@pytest.mark.asyncio
async def test_maximum_accepted_document_id_fits_registry_source_column(jobs: DocumentJobService):
    document_id = "d" * 255
    uploaded = await submit(jobs, document_id=document_id)
    assert uploaded["document_id"] == document_id
    service, factory = provider()
    assert await jobs.process_next(factory)
    assert service.ingest_bytes.await_args.kwargs["document_id"] == document_id
    assert (await jobs.get_version(uploaded["version_id"]))["status"] == "indexed"
    async with jobs.sessions() as db:
        document = await db.scalar(select(Document).where(Document.external_doc_id == document_id))
        # SQLite does not enforce VARCHAR length, so assert the PostgreSQL
        # storage contract for the full API-accepted identifier explicitly.
        assert document is not None
        source_type = Document.__table__.c.source.type
        assert isinstance(source_type, String)
        assert source_type.length is not None
        assert len(document.source) <= source_type.length
    assert (await jobs.source(uploaded["version_id"]))[0]
