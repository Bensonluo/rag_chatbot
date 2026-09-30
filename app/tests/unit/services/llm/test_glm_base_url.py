"""GLM endpoint configurability (generation + embeddings).

The GLM clients hardcoded their endpoint with no way to override it.
Both clients now read GLM_BASE_URL from settings, so a deployment can
point at whichever Zhipu endpoint its key subscribes to. The default
is the Coding Plan endpoint — a deliberate choice: the Coding Plan is
the free package this deployment's key uses (transient 429s there are
the package's rate behavior, not a misconfiguration). Standard PAAS
keys set GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4/ instead.
"""

from typing import Any

from app.config.settings import settings
from app.services.embeddings.glm_embeddings import GLMEmbeddingService
from app.services.llm.glm_client import GLMClient

CODING_ENDPOINT = "https://open.bigmodel.cn/api/coding/paas/v4/"
STANDARD_ENDPOINT = "https://open.bigmodel.cn/api/paas/v4/"


class TestGlmBaseUrl:
    def test_settings_default_is_coding_plan_endpoint(self) -> None:
        # Deliberate default: the free Coding Plan package.
        assert settings.GLM_BASE_URL == CODING_ENDPOINT

    def test_generation_client_uses_configured_base_url(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(settings, "GLM_BASE_URL", STANDARD_ENDPOINT)
        client = GLMClient(api_key="id.secret")
        assert str(client.client.base_url) == STANDARD_ENDPOINT

    def test_embedding_client_uses_configured_base_url(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(settings, "GLM_BASE_URL", STANDARD_ENDPOINT)
        service = GLMEmbeddingService(api_key="id.secret")
        assert str(service.client.base_url) == STANDARD_ENDPOINT
