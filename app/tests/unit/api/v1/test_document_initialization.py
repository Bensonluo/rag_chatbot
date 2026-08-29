"""Tests for document API dependency wiring."""

from unittest.mock import Mock, patch

import pytest


@pytest.mark.asyncio
async def test_ingestion_dependency_wires_connected_graphrag_services() -> None:
    from app.api.v1.documents import get_ingestion_service

    settings = Mock(
        VECTOR_DB_URL="http://qdrant:6333",
        VECTOR_COLLECTION_NAME="documents",
        VECTOR_API_KEY=None,
        EMBEDDING_PROVIDER="local",
        GRAPH_RAG_ENABLED=True,
        GRAPH_RAG_EXTRACTION_ENABLED=True,
    )
    embedding_service = Mock()
    qdrant_client = Mock()
    graph_client = Mock()
    llm_service = Mock()
    extractor = Mock()

    with (
        patch("app.api.v1.documents.get_settings", return_value=settings),
        patch(
            "app.api.v1.documents.EmbeddingFactory.create_from_settings",
            return_value=embedding_service,
        ),
        patch(
            "app.api.v1.documents.RetrievalFactory.create_vector_client",
            return_value=qdrant_client,
        ),
        patch("app.api.v1.graph.get_graph_client", return_value=graph_client),
        patch("app.services.llm.LLMFactory.create_from_settings", return_value=llm_service),
        patch(
            "app.services.graph.extraction.LLMEntityExtractor",
            return_value=extractor,
        ),
    ):
        service = await get_ingestion_service()

    assert service.embedding_service is embedding_service
    assert service.graph_client is graph_client
    assert service.entity_extractor is extractor
