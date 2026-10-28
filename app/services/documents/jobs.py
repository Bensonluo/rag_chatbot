"""Postgres-backed ingestion queue, retaining immutable original sources.

A worker holds a document row lock while processing. Job status is committed
through a second session so it remains visible. A crashed worker releases its
lock; another worker resumes the persisted processing job with the same IDs.
External writes must therefore be idempotent. PostgreSQL SKIP LOCKED coordinates
replicas; SQLite is supported for sequential unit tests only.
"""

import asyncio
import hashlib
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from pathlib import PurePath
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import undefer

from app.core.exceptions import ConflictError, NotFoundError, ValidationError
from app.models.database.document import Document, DocumentVersion
from app.services.documents.ingestion import DocumentIngestionService
from app.services.documents.metadata import PIPELINE_VERSION
from app.services.retrieval.kb_epoch import bump_kb_epoch

logger = logging.getLogger(__name__)
ServiceFactory = Callable[[dict[str, Any]], Awaitable[DocumentIngestionService]]


def version_status(version: DocumentVersion) -> dict[str, Any]:
    return {
        "document_id": version.document_id,
        "version_id": version.id,
        "title": version.title,
        "filename": version.filename,
        "file_type": version.file_type,
        "revision": version.revision,
        "status": version.status,
        "attempts": version.attempts,
        "error": version.error,
        "result": version.result,
        "created_at": version.created_at,
        "updated_at": version.updated_at,
    }


class DocumentJobService:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self.sessions = sessions

    async def submit(
        self,
        *,
        data: bytes,
        title: str,
        filename: str,
        file_type: str,
        metadata: dict[str, Any],
        pipeline_config: dict[str, Any],
        document_id: str | None = None,
    ) -> dict[str, Any]:
        if not data or not title.strip():
            raise ValidationError("Document title and content cannot be empty")
        content_hash = hashlib.sha256(data).hexdigest()
        # Omitted IDs deduplicate exact sources with the same logical title/scope.
        identity = json.dumps(
            [
                title.strip(),
                file_type,
                content_hash,
                {k: v for k, v in metadata.items() if k != "uploaded_by"},
            ],
            sort_keys=True,
            ensure_ascii=False,
        )
        document_id = document_id or str(uuid.uuid5(uuid.NAMESPACE_URL, identity))
        config = {**pipeline_config, "pipeline_version": PIPELINE_VERSION}
        filename = PurePath(filename.replace("\\", "/")).name[:255]
        for attempt in range(2):
            async with self.sessions() as db:
                try:
                    doc = await db.scalar(
                        select(Document)
                        .where(Document.external_doc_id == document_id)
                        .with_for_update(skip_locked=True, key_share=True)
                    )
                    if doc is None:
                        exists = await db.scalar(
                            select(Document.id).where(Document.external_doc_id == document_id)
                        )
                        if exists:
                            raise ConflictError(
                                "Document is being processed; retry after it finishes"
                            )
                        doc = Document(
                            external_doc_id=document_id,
                            title=title.strip(),
                            # Source is a bounded locator; the authored ID can
                            # already fill external_doc_id's 255-character limit.
                            source=f"database://{uuid.uuid5(uuid.NAMESPACE_URL, document_id)}",
                            doc_type=file_type,
                            chunk_count=0,
                            is_active=False,
                        )
                        db.add(doc)
                        await db.flush()
                    latest = await db.scalar(
                        select(DocumentVersion)
                        .where(DocumentVersion.document_id == document_id)
                        .order_by(DocumentVersion.revision.desc())
                        .limit(1)
                    )
                    if latest is not None and latest.status == "deleted":
                        raise ConflictError(
                            "This source was deleted; use a new document_id to restore it"
                        )
                    if (
                        latest is not None
                        and latest.content_hash == content_hash
                        and latest.title == title.strip()
                        and latest.file_type == file_type
                        and latest.pipeline_config == config
                        and {k: v for k, v in latest.source_metadata.items() if k != "uploaded_by"}
                        == {k: v for k, v in metadata.items() if k != "uploaded_by"}
                    ):
                        return version_status(latest)
                    revision = latest.revision + 1 if latest is not None else 1
                    version_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{revision}"))
                    version = DocumentVersion(
                        id=version_id,
                        document_id=document_id,
                        title=title.strip(),
                        filename=filename,
                        file_type=file_type,
                        content_hash=content_hash,
                        source_content=data,
                        source_metadata=metadata,
                        pipeline_config=config,
                        status="queued",
                        attempts=0,
                        revision=revision,
                    )
                    db.add(version)
                    await db.commit()
                    await db.refresh(version)
                    return version_status(version)
                except IntegrityError:
                    await db.rollback()
                    if attempt:
                        raise
        raise RuntimeError("Unreachable submission state")

    async def get_version(self, version_id: str) -> dict[str, Any]:
        async with self.sessions() as db:
            version = await db.get(DocumentVersion, version_id)
            if version is None:
                raise NotFoundError("Document version", version_id)
            return version_status(version)

    async def list_documents(self, offset: int = 0, limit: int = 50) -> list[dict[str, Any]]:
        async with self.sessions() as db:
            docs = (
                await db.scalars(
                    select(Document).order_by(Document.id.desc()).offset(offset).limit(limit)
                )
            ).all()
            return [
                {
                    "document_id": d.external_doc_id,
                    "title": d.title,
                    "active_version_id": d.active_version_id,
                    "is_active": d.is_active,
                    "chunks_count": d.chunk_count,
                    "file_type": d.doc_type,
                }
                for d in docs
            ]

    async def list_versions(self, document_id: str) -> list[dict[str, Any]]:
        async with self.sessions() as db:
            versions = (
                await db.scalars(
                    select(DocumentVersion)
                    .where(DocumentVersion.document_id == document_id)
                    .order_by(DocumentVersion.revision.desc())
                )
            ).all()
            if not versions:
                raise NotFoundError("Document", document_id)
            return [version_status(v) for v in versions]

    async def source(self, version_id: str) -> tuple[bytes, str, str]:
        async with self.sessions() as db:
            version = await db.scalar(
                select(DocumentVersion)
                .where(DocumentVersion.id == version_id)
                .options(undefer(DocumentVersion.source_content))
            )
            if version is None or version.source_content is None:
                raise NotFoundError("Document source", version_id)
            return version.source_content, version.filename, version.file_type

    async def normalized(self, version_id: str) -> tuple[str, str]:
        async with self.sessions() as db:
            version = await db.scalar(
                select(DocumentVersion)
                .where(DocumentVersion.id == version_id)
                .options(undefer(DocumentVersion.normalized_text))
            )
            if version is None or version.status == "deleted":
                raise NotFoundError("Document version", version_id)
            if version.normalized_text is None:
                raise ConflictError("Normalized text is not available for this revision yet")
            return version.normalized_text, version.filename

    async def retry(self, version_id: str) -> dict[str, Any]:
        async with self.sessions() as db:
            version = await db.get(DocumentVersion, version_id)
            if version is None:
                raise NotFoundError("Document version", version_id)
            # Lock the document to serialize retry/delete/replacement across replicas.
            doc = await db.scalar(
                select(Document)
                .where(Document.external_doc_id == version.document_id)
                .with_for_update(skip_locked=True, key_share=True)
            )
            if doc is None:
                raise ConflictError("Document is currently being processed")
            # Status can change between the first read and obtaining the lock
            # (especially when a concurrent delete just finished).
            await db.refresh(version)
            if version.status not in {"failed", "partial"}:
                raise ConflictError("Only failed or partial ingestion jobs can be retried")
            newer = await db.scalar(
                select(DocumentVersion.id).where(
                    DocumentVersion.document_id == version.document_id,
                    DocumentVersion.revision > version.revision,
                )
            )
            if newer:
                raise ConflictError("A newer revision exists; retry the latest revision")
            version.status, version.error = "queued", None
            await db.commit()
            await db.refresh(version)
            return version_status(version)

    async def process_next(self, factory: ServiceFactory) -> bool:
        pending_since = (
            select(func.min(DocumentVersion.created_at))
            .where(
                DocumentVersion.document_id == Document.external_doc_id,
                DocumentVersion.status.in_(["queued", "processing"]),
            )
            .correlate(Document)
            .scalar_subquery()
        )
        async with self.sessions() as lock_db, lock_db.begin():
            # Claim an available document in PostgreSQL itself. A bounded scan
            # of versions can be filled by one busy document, starving all
            # unrelated jobs beyond that scan's limit.
            doc = await lock_db.scalar(
                select(Document)
                .where(pending_since.is_not(None))
                .order_by(pending_since, Document.id)
                .limit(1)
                .with_for_update(skip_locked=True, key_share=True)
            )
            if doc is None:
                return False
            document_id = doc.external_doc_id
            async with self.sessions() as db:
                version = await db.scalar(
                    select(DocumentVersion)
                    .where(
                        DocumentVersion.document_id == document_id,
                        DocumentVersion.status.in_(["queued", "processing"]),
                    )
                    .order_by(DocumentVersion.revision)
                    .limit(1)
                    .options(undefer(DocumentVersion.source_content))
                )
                if version is None:
                    return False
                version.status = "processing"
                version.attempts += 1
                version.error = None
                await db.commit()
                db.expunge(version)
            service = None
            try:
                service = await factory(version.pipeline_config)
                result = await service.ingest_bytes(
                    data=version.source_content or b"",
                    title=version.title,
                    file_type=version.file_type,
                    document_id=document_id,
                    metadata={
                        **version.source_metadata,
                        "version_id": version.id,
                        "original_filename": version.filename,
                        "content_hash": version.content_hash,
                        "pipeline_version": PIPELINE_VERSION,
                    },
                )
                chunks_count = result["chunks_count"]
                if (
                    isinstance(chunks_count, bool)
                    or not isinstance(chunks_count, int)
                    or chunks_count < 0
                ):
                    raise ValidationError("Ingestion returned an invalid chunk count")
                version.normalized_text = result.pop("_normalized_text", None)
                version.result = result
                version.status = "partial" if result.get("graph_status") == "failed" else "indexed"
                doc.active_version_id, doc.is_active = version.id, True
                doc.title, doc.doc_type = version.title, version.file_type
                doc.chunk_count = chunks_count
                doc.doc_metadata = version.source_metadata
            except ValidationError as exc:
                version.status, version.error = "failed", str(exc)
            except Exception:
                logger.exception("Document ingestion failed: version=%s", version.id)
                version.status, version.error = (
                    "failed",
                    "Processing failed; retry this revision or check server logs",
                )
            finally:
                if service is not None:
                    try:
                        await service.close()
                    except Exception:
                        logger.exception("Failed to close ingestion clients")
            # Completion and the active document pointer are one commit.
            # If it fails, the durable processing job is recovered again.
            completion: dict[str, Any] = {
                "status": version.status,
                "error": version.error,
                "result": version.result,
            }
            # A failed retry must retain the previously indexed normalized
            # source; an unloaded deferred attribute does not mean deletion.
            if "normalized_text" in version.__dict__:
                completion["normalized_text"] = version.normalized_text
            await lock_db.execute(
                update(DocumentVersion).where(DocumentVersion.id == version.id).values(**completion)
            )
        # The external-index write invalidates caches before this transaction
        # commits. Rotate again after the durable pointer/status is visible;
        # every reader still checks publication if Redis is unavailable.
        try:
            async with asyncio.timeout(1.0):
                await bump_kb_epoch()
        except Exception:
            logger.warning("Committed knowledge epoch publication unavailable", exc_info=True)
        return True

    async def delete(self, document_id: str, service: DocumentIngestionService) -> dict[str, Any]:
        async with self.sessions() as db, db.begin():
            doc = await db.scalar(
                select(Document)
                .where(Document.external_doc_id == document_id)
                .with_for_update(skip_locked=True, key_share=True)
            )
            if doc is None:
                exists = await db.scalar(
                    select(Document.id).where(Document.external_doc_id == document_id)
                )
                if exists:
                    raise ConflictError("Document is currently being processed")
                # Legacy seed documents may predate the registry.
                return await service.delete_document(document_id)
            result = await service.delete_document(document_id)
            versions = (
                await db.scalars(
                    select(DocumentVersion).where(DocumentVersion.document_id == document_id)
                )
            ).all()
            for version in versions:
                version.status = "deleted"
                version.source_content = None
                version.normalized_text = None
            doc.is_active, doc.active_version_id, doc.chunk_count = False, None, 0
            return result


async def run_document_worker(
    jobs: DocumentJobService, factory: ServiceFactory, interval: float
) -> None:
    while True:
        try:
            if await jobs.process_next(factory):
                continue
        except Exception:
            logger.exception("Document worker unavailable; durable jobs will be retried")
        await asyncio.sleep(interval)
