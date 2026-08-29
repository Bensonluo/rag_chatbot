"""Tests for demo service wiring during application startup."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.config.settings import settings


@pytest.mark.asyncio
async def test_graph_client_is_exposed_to_graph_api_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The GraphRAG HTTP API should reuse the connected chat graph client."""
    from app.api.v1.chat import initialize_chat_service

    monkeypatch.setattr(settings, "GRAPH_RAG_ENABLED", True)
    monkeypatch.setattr(settings, "GRAPH_RAG_COMMUNITY_ENABLED", False)
    monkeypatch.setattr(settings, "GRAPH_RAG_TEXT_TO_CYPHER_ENABLED", False)
    monkeypatch.setattr(settings, "SLOT_FILLING_ENABLED", False)
    monkeypatch.setattr(settings, "RERANKER_ENABLED", False)

    graph_client = Mock()
    graph_client.connect = AsyncMock()
    embedding_service = Mock()

    with (
        patch("app.services.llm.LLMFactory.create_from_settings", return_value=Mock()),
        patch(
            "app.services.embeddings.EmbeddingFactory.create_from_settings",
            return_value=embedding_service,
        ),
        patch(
            "app.services.retrieval.RetrievalFactory.create_vector_client",
            return_value=Mock(),
        ),
        patch(
            "app.services.retrieval.RetrievalFactory.create_hybrid_search",
            return_value=Mock(),
        ),
        patch(
            "app.services.graph.GraphFactory.create_from_settings",
            return_value=graph_client,
        ),
        patch("app.api.v1.graph.set_graph_client") as set_graph_client,
        patch(
            "app.services.chat.factory.ChatServiceFactory.create_with_defaults",
            return_value=Mock(),
        ),
    ):
        await initialize_chat_service(AsyncMock())

    graph_client.connect.assert_awaited_once()
    set_graph_client.assert_called_once_with(graph_client)
