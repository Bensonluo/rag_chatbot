"""GLM embedding endpoint selection.

Live finding (2026-11-12): the Coding Plan chat endpoint
(open.bigmodel.cn/api/coding/paas/v4/) does not serve embedding models
(HTTP 400, error 1211 "模型不存在"), while the standard PAAS endpoint
serves embedding-2/embedding-3 and accepts Coding Plan keys. The
embedding client must therefore default to the PAAS URL even when
GLM_BASE_URL points at the coding endpoint for chat.
"""

import pytest

from app.config.settings import settings
from app.services.embeddings.glm_embeddings import GLMEmbeddingService

PAAS_URL = "https://open.bigmodel.cn/api/paas/v4/"
CODING_URL = "https://open.bigmodel.cn/api/coding/paas/v4/"


class TestDefaultEndpoint:
    def test_embedding_endpoint_independent_of_chat_base_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Simulate the documented Coding Plan posture: chat on the coding
        # endpoint. Embeddings must still resolve to the PAAS endpoint.
        monkeypatch.setattr(settings, "GLM_BASE_URL", CODING_URL, raising=False)

        service = GLMEmbeddingService(api_key="test-key")

        assert str(service.client.base_url) == PAAS_URL
        assert str(service.client.base_url) != CODING_URL

    def test_settings_field_defaults_to_paas(self) -> None:
        assert settings.GLM_EMBEDDING_BASE_URL == PAAS_URL

    def test_explicit_base_url_override_wins(self) -> None:
        service = GLMEmbeddingService(api_key="test-key", base_url="https://glm.test/v4/")

        assert str(service.client.base_url) == "https://glm.test/v4/"


class TestFactoryModelResolution:
    """EMBEDDING_MODEL names local BGE checkpoints; an API provider must
    not receive it (GLM answers HTTP 1211 "模型不存在" for bge-* names).
    Live regression: 2026-11-12 fresh-stack chat failed retrieval because
    create_from_settings forwarded bge-small-zh-v1.5 to the GLM branch.
    """

    def test_local_model_name_not_forwarded_to_glm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from app.services.embeddings.cached_embeddings import CachedEmbeddingService
        from app.services.embeddings.factory import EmbeddingFactory

        monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "glm", raising=False)
        monkeypatch.setattr(settings, "EMBEDDING_MODEL", "bge-small-zh-v1.5", raising=False)
        monkeypatch.setattr(settings, "GLM_API_KEY", "test-key", raising=False)

        wrapped = EmbeddingFactory.create_from_settings()
        assert isinstance(wrapped, CachedEmbeddingService)
        assert isinstance(wrapped.embedding_service, GLMEmbeddingService)

        assert wrapped.embedding_service.model == "embedding-2"

    def test_local_provider_still_uses_embedding_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app.services.embeddings.factory import EmbeddingFactory

        monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "local", raising=False)
        monkeypatch.setattr(settings, "EMBEDDING_MODEL", "bge-small-zh-v1.5", raising=False)

        wrapped = EmbeddingFactory.create_from_settings(use_cache=False)

        assert wrapped.model == "bge-small-zh-v1.5"
