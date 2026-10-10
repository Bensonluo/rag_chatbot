"""Durable support-knowledge uploads, revision status, sources, and search."""

import asyncio
import logging
import shutil
from collections.abc import AsyncGenerator
from datetime import datetime
from pathlib import PurePath
from typing import Any
from urllib.parse import quote

import httpx
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import Response
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select

from app.api.database import async_session_maker
from app.api.deps import get_current_active_user
from app.api.deps.authorization import require_kb_writer
from app.config.settings import get_settings
from app.core.exceptions import ConflictError, NotFoundError, ValidationError
from app.models.database.document import Document, DocumentVersion
from app.models.database.user import User
from app.services.documents.ingestion import DocumentIngestionService
from app.services.documents.jobs import DocumentJobService
from app.services.documents.metadata import validate_metadata
from app.services.embeddings import EmbeddingFactory
from app.services.embeddings.cached_embeddings import CachedEmbeddingService
from app.services.retrieval.factory import RetrievalFactory
from app.services.retrieval.qdrant_client import QdrantClient
from app.services.retrieval.vector_base import VectorSearchRequest

router = APIRouter(prefix="/documents", tags=["documents"])
logger = logging.getLogger(__name__)


class DocumentUploadRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=1)
    metadata: dict[str, Any] | None = None
    document_id: str | None = Field(
        default=None, min_length=1, max_length=255, pattern=r"^[\w.-]+$"
    )
    file_type: str = Field(default="txt", pattern=r"^(txt|md)$")

    @field_validator("title", "content")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Must not be blank")
        return value


class DocumentUploadResponse(BaseModel):
    document_id: str
    version_id: str
    revision: int
    title: str
    filename: str
    file_type: str
    status: str
    attempts: int
    error: str | None = None
    result: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    filters: dict[str, Any] | None = None

    @field_validator("query")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Must not be blank")
        return value


class SearchResult(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: dict[str, Any] | None = None


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    total_count: int


class DeleteResponse(BaseModel):
    document_id: str
    deleted_chunks: int
    message: str = "Document and retained sources deleted"


def pipeline_config() -> dict[str, Any]:
    settings = get_settings()
    return {
        "embedding_provider": settings.EMBEDDING_PROVIDER,
        "embedding_model": settings.EMBEDDING_MODEL,
        "chunking_strategy": "semantic",
        "max_chunk_size": settings.DOCUMENT_CHUNK_SIZE,
        "chunk_overlap": settings.DOCUMENT_CHUNK_OVERLAP,
        "graph_enabled": settings.GRAPH_RAG_ENABLED and settings.GRAPH_RAG_EXTRACTION_ENABLED,
        "ocr_languages": settings.DOCUMENT_OCR_LANGUAGES,
    }


async def build_ingestion_service(config: dict[str, Any] | None = None) -> DocumentIngestionService:
    settings = get_settings()
    config = pipeline_config() if config is None else config
    if (
        config["embedding_provider"] != settings.EMBEDDING_PROVIDER
        or config["embedding_model"] != settings.EMBEDDING_MODEL
    ):
        raise ValidationError(
            "Embedding configuration changed; submit a new revision using current configuration"
        )
    embedding_service = EmbeddingFactory.create(
        provider=config["embedding_provider"],
        model=config["embedding_model"],
    )
    qdrant_client = RetrievalFactory.create_vector_client(
        client_type="qdrant",
        url=settings.VECTOR_DB_URL,
        collection_name=settings.VECTOR_COLLECTION_NAME,
        api_key=settings.VECTOR_API_KEY,
        embedding_service=embedding_service,
    )
    graph_client = None
    entity_extractor = None
    if settings.GRAPH_RAG_ENABLED:
        from app.api.v1.graph import get_graph_client

        graph_client = get_graph_client()
    if config["graph_enabled"] and graph_client is not None:
        try:
            from app.services.graph.extraction import LLMEntityExtractor
            from app.services.llm import LLMFactory

            entity_extractor = LLMEntityExtractor(
                llm_service=LLMFactory.create_from_settings(), owns_llm=True
            )
        except Exception:
            logger.exception("Graph extraction unavailable; ingestion will report partial status")
    return DocumentIngestionService(
        qdrant_client=qdrant_client,
        embedding_provider=config["embedding_provider"],
        embedding_service=embedding_service,
        chunking_strategy=config["chunking_strategy"],
        max_chunk_size=config["max_chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        graph_client=graph_client,
        entity_extractor=entity_extractor,
        graph_enabled=config["graph_enabled"],
        graph_storage_enabled=settings.GRAPH_RAG_ENABLED,
        ocr_languages=config["ocr_languages"],
    )


async def get_ingestion_service() -> DocumentIngestionService:
    return await build_ingestion_service()


def get_document_jobs() -> DocumentJobService:
    return DocumentJobService(async_session_maker)


async def get_document_manager(
    current_user: User = Depends(get_current_active_user),
) -> User:
    """Authorize before allocating dependencies for a document lifecycle operation."""
    require_kb_writer(current_user)
    return current_user


async def get_qdrant_client() -> AsyncGenerator[QdrantClient, None]:
    settings = get_settings()
    embeddings = EmbeddingFactory.create_from_settings()
    client = None
    try:
        client = RetrievalFactory.create_vector_client(
            client_type="qdrant",
            url=settings.VECTOR_DB_URL,
            collection_name=settings.VECTOR_COLLECTION_NAME,
            api_key=settings.VECTOR_API_KEY,
            embedding_service=embeddings,
        )
        yield client
    finally:
        if client is not None:
            try:
                await client.client.close()
            except Exception:
                logger.warning("Failed to close document search vector client", exc_info=True)
        if isinstance(embeddings, CachedEmbeddingService):
            try:
                await embeddings.close()
            except Exception:
                logger.warning("Failed to close document search embedding cache", exc_info=True)
            embeddings = embeddings.embedding_service
        close = getattr(embeddings, "close", None)
        if close is not None:
            try:
                await close()
            except Exception:
                logger.warning("Failed to close document search embedding client", exc_info=True)


def _api_error(exc: Exception) -> HTTPException:
    if isinstance(exc, ValidationError):
        return HTTPException(422, str(exc))
    if isinstance(exc, ConflictError):
        return HTTPException(409, str(exc))
    if isinstance(exc, NotFoundError):
        return HTTPException(404, str(exc))
    logger.error("Document operation failed", exc_info=exc)
    return HTTPException(500, "Document operation failed")


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document_text(
    request: DocumentUploadRequest,
    current_user: User = Depends(get_document_manager),
    jobs: DocumentJobService = Depends(get_document_jobs),
) -> dict[str, Any]:
    """Retain the source and queue ingestion. Poll /versions/{version_id}."""
    require_kb_writer(current_user)
    data = request.content.encode("utf-8")
    if len(data) > get_settings().DOCUMENT_MAX_BYTES:
        raise HTTPException(413, "Document exceeds the configured upload limit")
    try:
        metadata = validate_metadata(request.metadata)
        metadata["uploaded_by"] = current_user.id
        return await jobs.submit(
            data=data,
            title=request.title,
            filename=f"{request.title[:200]}.{request.file_type}",
            file_type=request.file_type,
            metadata=metadata,
            document_id=request.document_id,
            pipeline_config=pipeline_config(),
        )
    except Exception as exc:
        raise _api_error(exc) from exc


@router.post(
    "/upload/file", response_model=DocumentUploadResponse, status_code=status.HTTP_202_ACCEPTED
)
async def upload_document_file(
    file: UploadFile = File(...),
    title: str | None = Query(default=None, min_length=1, max_length=500),
    document_id: str | None = Query(
        default=None, min_length=1, max_length=255, pattern=r"^[\w.-]+$"
    ),
    metadata: str | None = Query(
        default=None, description="JSON policy scope, such as lang/topic/product/region"
    ),
    current_user: User = Depends(get_document_manager),
    jobs: DocumentJobService = Depends(get_document_jobs),
) -> dict[str, Any]:
    """Queue PDF/TXT/MD ingestion. Supply document_id to publish a replacement revision."""
    require_kb_writer(current_user)
    filename = PurePath((file.filename or "").replace("\\", "/")).name
    file_type = PurePath(filename).suffix.lower().lstrip(".")
    if not filename or file_type not in {"pdf", "txt", "md"}:
        raise HTTPException(400, "Supported file types: pdf, txt, md")
    data = bytearray()
    max_bytes = get_settings().DOCUMENT_MAX_BYTES
    try:
        while block := await file.read(min(1024 * 1024, max_bytes + 1 - len(data))):
            data.extend(block)
            if len(data) > max_bytes:
                raise HTTPException(413, "Document exceeds the configured upload limit")
    finally:
        await file.close()
    try:
        import json

        supplied = json.loads(metadata) if metadata else None
        if supplied is not None and not isinstance(supplied, dict):
            raise ValidationError("Metadata must be a JSON object")
        values = validate_metadata(supplied)
        values["uploaded_by"] = current_user.id
        return await jobs.submit(
            data=bytes(data),
            title=title or PurePath(filename).stem,
            filename=filename,
            file_type=file_type,
            metadata=values,
            document_id=document_id,
            pipeline_config=pipeline_config(),
        )
    except ValueError as exc:
        raise HTTPException(422, "Metadata must be valid JSON") from exc
    except Exception as exc:
        raise _api_error(exc) from exc


@router.get("", summary="List knowledge documents")
async def list_documents(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
    current_user: User = Depends(get_document_manager),
    jobs: DocumentJobService = Depends(get_document_jobs),
) -> list[dict[str, Any]]:
    require_kb_writer(current_user)
    return await jobs.list_documents(offset, limit)


@router.get("/versions/{version_id}", response_model=DocumentUploadResponse)
async def get_version(
    version_id: str,
    current_user: User = Depends(get_document_manager),
    jobs: DocumentJobService = Depends(get_document_jobs),
) -> dict[str, Any]:
    require_kb_writer(current_user)
    try:
        return await jobs.get_version(version_id)
    except Exception as exc:
        raise _api_error(exc) from exc


@router.post("/versions/{version_id}/retry", response_model=DocumentUploadResponse, status_code=202)
async def retry_version(
    version_id: str,
    current_user: User = Depends(get_document_manager),
    jobs: DocumentJobService = Depends(get_document_jobs),
) -> dict[str, Any]:
    require_kb_writer(current_user)
    try:
        return await jobs.retry(version_id)
    except Exception as exc:
        raise _api_error(exc) from exc


@router.get("/versions/{version_id}/source")
async def download_source(
    version_id: str,
    current_user: User = Depends(get_document_manager),
    jobs: DocumentJobService = Depends(get_document_jobs),
) -> Response:
    require_kb_writer(current_user)
    try:
        data, filename, file_type = await jobs.source(version_id)
        # Original bytes may use GBK/UTF-16; only normalized text is UTF-8.
        return Response(
            data,
            media_type="application/pdf" if file_type == "pdf" else "application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename, safe='')}",
                "X-Content-Type-Options": "nosniff",
            },
        )
    except Exception as exc:
        raise _api_error(exc) from exc


@router.get("/versions/{version_id}/text")
async def download_normalized_text(
    version_id: str,
    current_user: User = Depends(get_document_manager),
    jobs: DocumentJobService = Depends(get_document_jobs),
) -> Response:
    """Download the normalized text used for chunk offsets and retrieval."""
    require_kb_writer(current_user)
    try:
        text, filename = await jobs.normalized(version_id)
        filename = str(PurePath(filename).with_suffix(".normalized.txt"))
        return Response(
            text,
            media_type="text/plain; charset=utf-8",
            headers={
                "Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename, safe='')}",
                "X-Content-Type-Options": "nosniff",
            },
        )
    except Exception as exc:
        raise _api_error(exc) from exc


@router.get("/{document_id}/versions", response_model=list[DocumentUploadResponse])
async def list_versions(
    document_id: str,
    current_user: User = Depends(get_document_manager),
    jobs: DocumentJobService = Depends(get_document_jobs),
) -> list[dict[str, Any]]:
    require_kb_writer(current_user)
    try:
        return await jobs.list_versions(document_id)
    except Exception as exc:
        raise _api_error(exc) from exc


@router.post("/search", response_model=SearchResponse)
async def search_documents(
    request: SearchRequest,
    current_user: User = Depends(get_current_active_user),
    qdrant_client: QdrantClient = Depends(get_qdrant_client),
) -> SearchResponse:
    if current_user is None:
        raise HTTPException(401, "Authentication required")
    try:
        results = await qdrant_client.search(
            VectorSearchRequest(query=request.query, top_k=request.top_k, filters=request.filters)
        )
        return SearchResponse(
            query=request.query,
            total_count=len(results),
            results=[
                SearchResult(
                    chunk_id=(r.metadata or {}).get("chunk_id", ""),
                    document_id=r.document_id,
                    content=r.content,
                    score=r.score,
                    metadata=r.metadata,
                )
                for r in results
            ],
        )
    except Exception as exc:
        raise _api_error(exc) from exc


@router.delete("/{document_id}", response_model=DeleteResponse)
async def delete_document(
    document_id: str,
    current_user: User = Depends(get_document_manager),
    ingestion_service: DocumentIngestionService = Depends(get_ingestion_service),
    jobs: DocumentJobService = Depends(get_document_jobs),
) -> dict[str, Any]:
    require_kb_writer(current_user)
    try:
        return await jobs.delete(document_id, ingestion_service)
    except Exception as exc:
        raise _api_error(exc) from exc
    finally:
        await ingestion_service.close()


@router.get("/health")
async def health_check(request: Request) -> Response:
    from fastapi.responses import JSONResponse

    settings = get_settings()
    checks: dict[str, bool] = {}
    try:
        async with async_session_maker() as db:
            await asyncio.wait_for(
                db.execute(
                    select(DocumentVersion.id)
                    .join(Document, Document.external_doc_id == DocumentVersion.document_id)
                    .limit(1)
                ),
                timeout=3,
            )
        checks["database"] = True
    except Exception:
        checks["database"] = False
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            response = await client.get(
                settings.VECTOR_DB_URL.rstrip("/") + "/collections",
                headers={"api-key": settings.VECTOR_API_KEY} if settings.VECTOR_API_KEY else {},
            )
            checks["vector_db"] = response.is_success
    except Exception:
        checks["vector_db"] = False
    worker = getattr(request.app.state, "document_worker", None)
    checks["worker"] = worker is not None and not worker.done()
    healthy = all(checks.values())
    return JSONResponse(
        {
            "status": "healthy" if healthy else "degraded",
            **checks,
            "ocr_available": bool(shutil.which("pdftoppm") and shutil.which("tesseract")),
        },
        status_code=200 if healthy else 503,
    )
