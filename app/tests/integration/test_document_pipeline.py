"""Complete document revision lifecycle with SQLite and Qdrant's local engine.

No network: only embeddings and the Redis epoch transport are replaced. Parsing,
chunking, ingestion, queued jobs, vector operations, and hybrid search are real.
SQLite exercises sequential persistence; it does not validate PostgreSQL locks.
"""

import json

import httpx
import pytest
from fastapi import FastAPI
from qdrant_client import AsyncQdrantClient, models
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.api.deps import get_current_user
from app.api.v1 import documents
from app.config.settings import settings
from app.models.database.base import Base
from app.models.database.document import Document, DocumentVersion
from app.models.database.user import User
from app.services.documents.jobs import DocumentJobService
from app.services.embeddings.base import EmbeddingResult, EmbeddingServiceBase
from app.services.retrieval import kb_epoch, keyword_refresh
from app.services.retrieval.hybrid_search import HybridSearchService, KeywordSearch
from app.services.retrieval.qdrant_client import QdrantClient
from app.services.retrieval.vector_base import VectorSearchRequest


class DeterministicEmbeddings(EmbeddingServiceBase):
    def __init__(self):
        super().__init__(model="test-policy-embedding", dimensions=2)

    async def embed(self, texts):
        return EmbeddingResult(
            embeddings=[await self.embed_single(text) for text in texts],
            model=self.model,
            dimensions=self.dimensions,
            tokens_used=sum(len(text) for text in texts),
        )

    async def embed_single(self, text):
        return [1.0, 0.1 + text.lower().count("refund") / 10]


class BorrowedQdrant:
    """Delegate every SDK operation; fixture owns the local engine's lifetime."""

    def __init__(self, client):
        self.client = client

    def __getattr__(self, name):
        return getattr(self.client, name)

    async def close(self):
        pass


class LocalEpochStore:
    def __init__(self):
        self.values = {}

    async def get(self, key):
        return self.values.get(key)

    async def set(self, key, value, nx=False):
        if not nx or key not in self.values:
            self.values[key] = value


@pytest.mark.asyncio
async def test_queued_policy_replacement_and_delete_preserve_provenance(monkeypatch):
    monkeypatch.setattr(settings, "DEMO_MODE", False)
    monkeypatch.setattr(settings, "GRAPH_RAG_ENABLED", False)
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "local")
    monkeypatch.setattr(settings, "EMBEDDING_MODEL", "test-policy-embedding")
    monkeypatch.setattr(settings, "DOCUMENT_CHUNK_SIZE", 96)
    monkeypatch.setattr(settings, "DOCUMENT_CHUNK_OVERLAP", 12)
    monkeypatch.setattr(kb_epoch, "_client", LocalEpochStore())
    monkeypatch.setattr(keyword_refresh, "_keyword_hybrid", None)

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", poolclass=StaticPool)
    async with engine.begin() as connection:
        await connection.run_sync(
            lambda sync: Base.metadata.create_all(
                sync, tables=[Document.__table__, DocumentVersion.__table__]
            )
        )
    jobs = DocumentJobService(async_sessionmaker(engine, expire_on_commit=False))
    raw = AsyncQdrantClient(location=":memory:")
    await raw.create_collection(
        "policies",
        vectors_config=models.VectorParams(
            size=2,
            distance=models.Distance.COSINE,
        ),
    )
    embeddings = DeterministicEmbeddings()

    def vector_factory(**kwargs):
        return QdrantClient(
            url="unused",
            collection_name="policies",
            client=BorrowedQdrant(raw),
            embedding_service=embeddings,
        )

    monkeypatch.setattr(documents.EmbeddingFactory, "create", lambda **kwargs: embeddings)
    monkeypatch.setattr(documents.EmbeddingFactory, "create_from_settings", lambda: embeddings)
    monkeypatch.setattr(documents.RetrievalFactory, "create_vector_client", vector_factory)
    vector = vector_factory()
    hybrid = HybridSearchService(vector, KeywordSearch())
    keyword_refresh.register_keyword_index(hybrid)

    app = FastAPI()
    app.include_router(documents.router, prefix="/api/v1")
    app.dependency_overrides[documents.get_document_jobs] = lambda: jobs
    app.dependency_overrides[get_current_user] = lambda: User(
        id=9,
        email="author@example.test",
        is_active=True,
        is_admin=True,
    )
    prefix = "/api/v1/documents"
    original = (
        "# Refund policy\r\n\r\n## Eligibility\r\n\r\n"
        + "".join(
            f"Refund item {index}: unopened goods can be returned within seven days.\r\n\r\n"
            for index in range(8)
        )
    ).encode("utf-8")
    params = {
        "document_id": "refund-policy",
        "metadata": json.dumps({"region": "US", "lang": "en"}),
    }
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            queued = await client.post(
                prefix + "/upload/file",
                params=params,
                files={"file": ("refunds.md", original, "text/markdown")},
            )
            assert queued.status_code == 202
            first = queued.json()
            version_url = prefix + "/versions/" + first["version_id"]
            assert first["status"] == "queued"
            assert (await raw.count("policies", exact=True)).count == 0
            assert (await client.get(version_url + "/source")).content == original
            assert (await client.get(version_url + "/text")).status_code == 409

            assert await jobs.process_next(documents.build_ingestion_service)
            indexed = (await client.get(version_url)).json()
            assert indexed["status"] == "indexed"
            assert indexed["attempts"] == 1
            old_chunks = await vector.list_chunks()
            old_ids = {chunk["id"] for chunk in old_chunks}
            assert len(old_chunks) == indexed["result"]["chunks_count"] >= 3
            normalized = (await client.get(version_url + "/text")).text
            assert "\r" not in normalized
            assert normalized.count("Refund item") == 8
            covered = set()
            for chunk in old_chunks:
                metadata = chunk["metadata"]
                start, end = metadata["start_pos"], metadata["end_pos"]
                assert chunk["content"] == normalized[start:end]
                covered.update(range(start, end))
                assert metadata["document_id"] == "refund-policy"
                assert metadata["version_id"] == first["version_id"]
                assert metadata["file_type"] == "md"
                assert metadata["source"] == "refunds.md"
                assert metadata["region"] == "US"
                assert metadata["embedding_model"] == embeddings.model
                assert "uploaded_by" not in metadata
            assert covered == set(range(len(normalized)))
            assert any(
                chunk["metadata"]["section_path"] == ["Refund policy", "Eligibility"]
                for chunk in old_chunks
            )

            hits = await hybrid.search(
                VectorSearchRequest("Refund", top_k=100, filters={"region": "US"})
            )
            assert len(hits) == len(old_chunks)
            assert {hit.metadata["chunk_id"] for hit in hits} == old_ids
            assert {hit.document_id for hit in hits} == {"refund-policy"}
            assert (
                sum(
                    hit.metadata["vector_score"] > 0 and hit.metadata["keyword_score"] > 0
                    for hit in hits
                )
                >= 2
            )
            assert all(hit.metadata["source"] == "refunds.md" for hit in hits)
            searched = await client.post(prefix + "/search", json={"query": "Refund", "top_k": 20})
            assert searched.status_code == 200
            assert {hit["chunk_id"] for hit in searched.json()["results"]} == old_ids

            duplicate = await client.post(
                prefix + "/upload/file",
                params=params,
                files={"file": ("refunds.md", original, "text/markdown")},
            )
            assert duplicate.json()["version_id"] == first["version_id"]
            assert not await jobs.process_next(documents.build_ingestion_service)

            replacement = (
                b"# Refund policy\r\n\r\nRefund requests must arrive within three days.\r\n"
            )
            second = (
                await client.post(
                    prefix + "/upload/file",
                    params=params,
                    files={"file": ("refunds.md", replacement, "text/markdown")},
                )
            ).json()
            assert second["document_id"] == first["document_id"]
            assert second["revision"] == 2
            assert {chunk["id"] for chunk in await vector.list_chunks()} == old_ids
            assert await jobs.process_next(documents.build_ingestion_service)
            new_chunks = await vector.list_chunks()
            new_ids = {chunk["id"] for chunk in new_chunks}
            assert len(new_chunks) == 1
            assert not (old_ids & new_ids)
            assert set(hybrid.keyword_search.documents) == new_ids
            assert (await client.get(version_url + "/source")).content == original
            hits = await hybrid.search(VectorSearchRequest("Refund", top_k=20))
            assert len(hits) == 1
            assert "three days" in hits[0].content
            assert hits[0].metadata["version_id"] == second["version_id"]

            deleted = await client.delete(prefix + "/refund-policy")
            assert deleted.status_code == 200
            assert deleted.json()["deleted_chunks"] == 1
            assert (await raw.count("policies", exact=True)).count == 0
            assert hybrid.keyword_search.documents == {}
            assert await vector.search(VectorSearchRequest("Refund")) == []
            for version_id in (first["version_id"], second["version_id"]):
                url = prefix + "/versions/" + version_id
                assert (await client.get(url)).json()["status"] == "deleted"
                assert (await client.get(url + "/source")).status_code == 404
                assert (await client.get(url + "/text")).status_code == 404
            listing = (await client.get(prefix)).json()
            assert listing[0]["active_version_id"] is None
            assert listing[0]["chunks_count"] == 0
    finally:
        await raw.close()
        await engine.dispose()
