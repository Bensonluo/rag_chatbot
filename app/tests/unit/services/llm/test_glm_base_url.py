"""GLM endpoint configurability (generation + embeddings).

The GLM clients hardcoded the Coding Plan endpoint
(open.bigmodel.cn/api/coding/paas/v4/). A key with a standard PAAS
quota 429s there — the live demo lost every LLM-generated turn while
the same key returned 200 on the standard endpoint. The endpoint now
comes from settings (GLM_BASE_URL), defaulting to the standard PAAS
API; Coding Plan deployments override it via env.
"""

from typing import Any

from app.config.settings import settings
from app.services.embeddings.glm_embeddings import GLMEmbeddingService
from app.services.llm.glm_client import GLMClient

STANDARD_ENDPOINT = "https://open.bigmodel.cn/api/paas/v4/"
CODING_ENDPOINT = "https://open.bigmodel.cn/api/coding/paas/v4/"


class TestGlmBaseUrl:
    def test_settings_default_is_standard_endpoint(self) -> None:
        # The regression this pins: the default must NOT be the Coding
        # Plan endpoint — standard keys have no quota there.
        assert settings.GLM_BASE_URL == STANDARD_ENDPOINT

    def test_generation_client_uses_configured_base_url(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(settings, "GLM_BASE_URL", CODING_ENDPOINT)
        client = GLMClient(api_key="id.secret")
        assert str(client.client.base_url) == CODING_ENDPOINT

    def test_embedding_client_uses_configured_base_url(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(settings, "GLM_BASE_URL", CODING_ENDPOINT)
        service = GLMEmbeddingService(api_key="id.secret")
        assert str(service.client.base_url) == CODING_ENDPOINT
