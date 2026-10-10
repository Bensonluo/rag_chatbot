"""Shared durable document import runner for standalone command-line tools."""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImportDocument:
    document_id: str
    title: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@asynccontextmanager
async def connected_graph() -> AsyncIterator[None]:
    """Supply the API builder's shared graph client for the entire CLI run."""
    from app.api.v1.graph import get_graph_client, set_graph_client
    from app.services.graph.factory import GraphFactory

    previous = get_graph_client()
    if previous is not None:
        # An embedded caller owns its existing connection.
        yield
        return
    client = GraphFactory.create_from_settings()
    if client is None:
        yield
        return
    try:
        await client.connect()
        set_graph_client(client)
        yield
    finally:
        set_graph_client(previous)
        try:
            # connect() may have allocated a driver before failing.
            await client.close()
        except Exception:
            logger.exception("Failed to close import graph connection")


async def ingest_documents(
    documents: list[ImportDocument], *, graph_enabled: bool | None = None
) -> list[dict[str, Any]]:
    """Retain, queue and process revisions, failing visibly on incomplete work.

    Imports are lazy so dry runs need only Python's standard library. Graph
    storage is connected even when extraction is disabled: replacements must
    still remove graph assertions belonging to the previous revision.
    """
    from app.api.database import async_session_maker, close_db
    from app.api.v1.documents import (
        DocumentUploadRequest,
        build_ingestion_service,
        pipeline_config,
    )
    from app.config.settings import get_settings
    from app.services.documents.jobs import DocumentJobService
    from app.services.documents.metadata import validate_metadata

    config = pipeline_config()
    if graph_enabled is not None:
        config["graph_enabled"] = graph_enabled
    jobs = DocumentJobService(async_session_maker)
    try:
        # Validate the complete batch before any source or index mutation.
        for doc in documents:
            DocumentUploadRequest(
                document_id=doc.document_id,
                title=doc.title,
                content=doc.content,
                metadata=doc.metadata,
            )
            validate_metadata(doc.metadata)
            if len(doc.content.encode("utf-8")) > get_settings().DOCUMENT_MAX_BYTES:
                raise ValueError(f"Document exceeds DOCUMENT_MAX_BYTES: {doc.document_id}")
        async with connected_graph():
            revisions = []
            for doc in documents:
                revision = await jobs.submit(
                    data=doc.content.encode("utf-8"),
                    title=doc.title,
                    filename=doc.document_id + ".txt",
                    file_type="txt",
                    document_id=doc.document_id,
                    pipeline_config=config,
                    metadata=doc.metadata,
                )
                revisions.append(revision)
            while True:
                revisions = [await jobs.get_version(r["version_id"]) for r in revisions]
                for revision in revisions:
                    if revision["status"] not in {"queued", "processing", "indexed"}:
                        raise RuntimeError(
                            f"Import incomplete for {revision['document_id']} "
                            f"(version {revision['version_id']}, {revision['status']}): "
                            f"{revision.get('error') or revision.get('result')}. "
                            "Inspect the revision and retry it through the document API."
                        )
                if all(r["status"] == "indexed" for r in revisions):
                    return revisions
                if not await jobs.process_next(build_ingestion_service):
                    # Another worker may own these document locks.
                    await asyncio.sleep(0.2)
    finally:
        await close_db()
