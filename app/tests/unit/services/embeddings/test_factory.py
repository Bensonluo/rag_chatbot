"""Tests for lightweight embedding dependency construction."""

from unittest.mock import Mock, patch


def test_local_embedding_factory_reuses_the_cached_model_service() -> None:
    from app.services.embeddings.factory import EmbeddingFactory

    local_service = Mock(model="bge-m3", dimensions=1024)

    with patch(
        "app.services.embeddings.factory.get_local_embedding_service",
        return_value=local_service,
    ) as get_local:
        result = EmbeddingFactory.create(
            provider="local",
            model="bge-m3",
            device="cpu",
            use_cache=False,
        )

    assert result is local_service
    get_local.assert_called_once_with(model="bge-m3", device="cpu")


def test_openai_embedding_factory_creates_service() -> None:
    from app.services.embeddings.factory import EmbeddingFactory
    from app.services.embeddings.openai_embeddings import OpenAIEmbeddingService

    result = EmbeddingFactory.create(
        provider="openai",
        api_key="test-key",
        use_cache=False,
    )

    assert isinstance(result, OpenAIEmbeddingService)
    assert result.model  # resolved from settings/model table
