"""Production entry points cannot bypass committed-source verification."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy import select, text

from app.api.v1 import chat as chat_api
from app.api.v1 import graph as graph_api
from app.config.settings import settings
from app.models.database.document import Document, DocumentVersion
from app.services.chat.chat_service import ChatService
from app.services.chat.factory import ChatServiceFactory
from app.services.documents import jobs as jobs_module
from app.services.embeddings import EmbeddingFactory
from app.services.graph.base import GraphSearchResult
from app.services.graph.retrieval import TextToCypherService
from app.services.llm import LLMFactory
from app.services.retrieval.factory import RetrievalFactory
from app.services.retrieval.publication import PublicationGuard
from app.services.retrieval.vector_base import SearchResult
from app.tests.unit.api.v1.test_document_jobs_api import api as api
from app.tests.unit.api.v1.test_document_jobs_api import process, upload


async def _published(api: SimpleNamespace) -> str:
    revision = (await upload(api, document_id="policy")).json()
    await process(api)
    version_id = revision["version_id"]
    assert isinstance(version_id, str)
    return version_id


async def test_document_http_search_filters_before_delivery_and_preserves_business_filters(api):
    from app.api.v1 import documents

    version = await _published(api)
    hits = [
        SearchResult("policy", "candidate", 0.99, {"version_id": "not-published"}),
        SearchResult("policy", "published", 0.8, {"version_id": version, "chunk_id": "active"}),
        SearchResult("unregistered", "raw legacy", 0.7, {"version_id": "unknown"}),
    ]
    vector = SimpleNamespace(search=AsyncMock(return_value=hits))
    api.app.dependency_overrides[documents.get_qdrant_client] = lambda: vector
    response = await api.client.post(
        "/api/v1/documents/search",
        json={"query": "policy", "top_k": 1, "filters": {"region": "CN"}},
    )
    assert response.status_code == 200
    assert [hit["content"] for hit in response.json()["results"]] == ["published"]
    request = vector.search.await_args.args[0]
    assert request.filters == {"region": "CN"}
    assert request.top_k >= settings.RETRIEVAL_CANDIDATE_POOL


async def test_document_search_cannot_turn_registry_failure_into_unregistered_legacy_permission(
    api,
):
    from app.api.v1 import documents

    version = await _published(api)
    api.app.dependency_overrides[documents.get_qdrant_client] = lambda: SimpleNamespace(
        search=AsyncMock(
            return_value=[
                SearchResult("policy", "sensitive raw payload", 1, {"version_id": version})
            ]
        )
    )
    async with api.engine.begin() as connection:
        await connection.execute(text("DROP TABLE document_versions"))
    response = await api.client.post("/api/v1/documents/search", json={"query": "policy"})
    assert response.status_code == 503
    assert response.json() == {"detail": "Published knowledge is temporarily unavailable"}


@pytest.mark.parametrize("registry_available", [True, False])
async def test_graph_http_query_requires_complete_published_sources(
    api, monkeypatch, registry_available
):
    version = await _published(api)
    api.app.include_router(graph_api.router, prefix="/api/v1")
    active = GraphSearchResult(
        "published graph",
        [],
        [],
        1,
        "text_to_cypher",
        {"source_document_ids": ["policy"], "source_version_ids": [version]},
    )
    candidate = GraphSearchResult(
        "unpublished graph",
        [],
        [],
        2,
        "text_to_cypher",
        {"source_document_ids": ["policy"], "source_version_ids": ["candidate"]},
    )
    unattributed = GraphSearchResult("unattributed graph", [], [], 3, "text_to_cypher")
    monkeypatch.setattr(graph_api, "_graph_client", Mock())
    monkeypatch.setattr(
        "app.services.llm.LLMFactory.create_from_settings", Mock(return_value=Mock())
    )
    query = AsyncMock(return_value=[candidate, unattributed, active])
    monkeypatch.setattr(TextToCypherService, "query", query)
    if not registry_available:
        async with api.engine.begin() as connection:
            await connection.execute(text("DROP TABLE document_versions"))
    response = await api.client.post(
        "/api/v1/graph/query", json={"query": "policy", "max_results": 1}
    )
    if registry_available:
        assert response.status_code == 200
        assert [hit["content"] for hit in response.json()["results"]] == ["published graph"]
    else:
        assert response.status_code == 503
        assert response.json() == {"detail": "Published knowledge is temporarily unavailable"}


@pytest.mark.parametrize("retrieval_online", [True, False])
async def test_chat_startup_keeps_the_same_publication_policy_for_cache_and_retrieval(
    api, monkeypatch, retrieval_online
):
    guard = PublicationGuard(api.sessions)
    monkeypatch.setattr(chat_api, "get_publication_guard", lambda: guard)
    monkeypatch.setattr(chat_api, "_chat_service", None)
    for key in [
        "GRAPH_RAG_ENABLED",
        "SLOT_FILLING_ENABLED",
        "RERANKER_ENABLED",
        "KEYWORD_INDEX_WARMUP_ENABLED",
    ]:
        monkeypatch.setattr(settings, key, False)
    monkeypatch.setattr(settings, "CONTAINMENT_STATS_REFRESH_SECONDS", 0)
    monkeypatch.setattr(settings, "KEYWORD_INDEX_REFRESH_SECONDS", 0)
    monkeypatch.setattr(LLMFactory, "create_from_settings", Mock(return_value=Mock()))
    monkeypatch.setattr(LLMFactory, "create_light_from_settings", Mock(return_value=Mock()))
    embeddings = Mock(
        side_effect=None if retrieval_online else RuntimeError("local vector offline")
    )
    monkeypatch.setattr(EmbeddingFactory, "create_from_settings", embeddings)
    monkeypatch.setattr(RetrievalFactory, "create_vector_client", Mock(return_value=Mock()))
    hybrid = Mock()
    hybrid_factory = Mock(return_value=hybrid)
    monkeypatch.setattr(RetrievalFactory, "create_hybrid_search", hybrid_factory)
    monkeypatch.setattr(chat_api, "register_keyword_index", Mock())
    factory = Mock(return_value=ChatService(graph=Mock()))
    monkeypatch.setattr(ChatServiceFactory, "create_with_defaults", factory)
    await chat_api.initialize_chat_service(AsyncMock())
    pipeline = factory.call_args.kwargs["retrieval_pipeline"]
    assert pipeline["publication_guard"] is guard
    if retrieval_online:
        assert pipeline["hybrid_search"] is hybrid
        assert hybrid_factory.call_args.kwargs["publication_guard"] is guard
    else:
        assert "hybrid_search" not in pipeline
        hybrid_factory.assert_not_called()


async def test_worker_publishes_epoch_only_after_activation_commit_is_independently_visible(
    api, monkeypatch
):
    revision = (await upload(api, document_id="policy")).json()
    observed = []

    async def epoch_after_commit():
        async with api.sessions() as db:
            doc = await db.scalar(select(Document).where(Document.external_doc_id == "policy"))
            version = await db.get(DocumentVersion, revision["version_id"])
            assert doc is not None and version is not None
            observed.append((doc.active_version_id, version.status))

    monkeypatch.setattr(jobs_module, "bump_kb_epoch", epoch_after_commit)
    await process(api)
    assert observed == [(revision["version_id"], "indexed")]


async def test_failed_registry_commit_does_not_publish_a_success_epoch(api, monkeypatch):
    await upload(api, document_id="policy")
    async with api.sessions() as db:
        await db.execute(
            text(
                "CREATE TRIGGER reject_activation BEFORE UPDATE ON document_versions WHEN NEW.status = 'indexed' BEGIN SELECT RAISE(ABORT, 'synthetic registry rejection'); END"
            )
        )
        await db.commit()
    publisher = AsyncMock()
    monkeypatch.setattr(jobs_module, "bump_kb_epoch", publisher)
    with pytest.raises(Exception, match="synthetic registry rejection"):
        await process(api)
    publisher.assert_not_awaited()


async def test_post_commit_epoch_failure_cannot_revoke_durable_activation(api, monkeypatch):
    revision = (await upload(api, document_id="policy")).json()
    monkeypatch.setattr(
        jobs_module, "bump_kb_epoch", AsyncMock(side_effect=TimeoutError("local epoch failed"))
    )
    await process(api)
    async with api.sessions() as db:
        doc = await db.scalar(select(Document).where(Document.external_doc_id == "policy"))
        assert doc is not None and doc.active_version_id == revision["version_id"]


async def test_post_commit_epoch_wait_is_bounded(api, monkeypatch):
    await upload(api, document_id="policy")
    cancelled = asyncio.Event()

    async def hanging_epoch():
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(jobs_module, "bump_kb_epoch", hanging_epoch)
    await asyncio.wait_for(process(api), timeout=2)
    assert cancelled.is_set()
