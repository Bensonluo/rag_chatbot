"""Last-resort GLM fallback tier (user directive 2026-11-01).

A second GLM key (``GLM_FALLBACK_API_KEY``) is appended to the END of the
resilient chains so a quota-exhausted fleet — primary GLM, MiniMax, Kimi,
OpenAI, Anthropic — still has a domestic, reachable endpoint. Unset, the
chains keep their existing composition exactly.
"""

from app.services.llm.glm_client import GLMClient
from app.services.llm.resilience import ResilientLLMService


class TestGLMFallbackInMainChain:
    def test_fallback_is_last_entry(self):
        """glm-fallback closes the chain with the fallback key/model."""
        from unittest.mock import patch

        from app.config.settings import settings
        from app.services.llm.factory import LLMFactory

        with (
            patch.object(settings, "GLM_API_KEY", "primary"),
            patch.object(settings, "GLM_MODEL", "glm-5.3-flash"),
            patch.object(settings, "MINIMAX_API_KEY", "m"),
            patch.object(settings, "KIMI_API_KEY", "k"),
            patch.object(settings, "OPENAI_API_KEY", "o"),
            patch.object(settings, "ANTHROPIC_API_KEY", "a"),
            patch.object(settings, "GLM_FALLBACK_API_KEY", "fallback"),
            patch.object(settings, "LLM_RESILIENCE_ENABLED", True),
        ):
            svc = LLMFactory.create_from_settings()

        assert isinstance(svc, ResilientLLMService)
        assert svc.provider_names == [
            "glm",
            "minimax",
            "kimi",
            "openai",
            "anthropic",
            "glm-fallback",
        ]
        last = svc._providers[-1][1]
        assert isinstance(last, GLMClient)
        assert last.api_key == "fallback"
        assert last.model == "glm-5.3-flash"

    def test_chain_unchanged_without_fallback_key(self):
        from unittest.mock import patch

        from app.config.settings import settings
        from app.services.llm.factory import LLMFactory

        with (
            patch.object(settings, "GLM_API_KEY", "primary"),
            patch.object(settings, "MINIMAX_API_KEY", "m"),
            patch.object(settings, "OPENAI_API_KEY", None),
            patch.object(settings, "ANTHROPIC_API_KEY", None),
            patch.object(settings, "GLM_FALLBACK_API_KEY", None),
            patch.object(settings, "LLM_RESILIENCE_ENABLED", True),
        ):
            svc = LLMFactory.create_from_settings()

        assert isinstance(svc, ResilientLLMService)
        assert "glm-fallback" not in svc.provider_names

    def test_fallback_alone_still_builds_chain(self):
        from unittest.mock import patch

        from app.config.settings import settings
        from app.services.llm.factory import LLMFactory

        with (
            patch.object(settings, "GLM_API_KEY", None),
            patch.object(settings, "MINIMAX_API_KEY", None),
            patch.object(settings, "KIMI_API_KEY", None),
            patch.object(settings, "OPENAI_API_KEY", None),
            patch.object(settings, "ANTHROPIC_API_KEY", None),
            patch.object(settings, "GLM_FALLBACK_API_KEY", "fallback"),
            patch.object(settings, "LLM_RESILIENCE_ENABLED", True),
        ):
            svc = LLMFactory.create_from_settings()

        assert isinstance(svc, ResilientLLMService)
        assert svc.provider_names == ["glm-fallback"]


class TestGLMFallbackInLightChain:
    def test_light_chain_closes_with_fallback(self):
        from unittest.mock import patch

        from app.config.settings import settings
        from app.services.llm.factory import LLMFactory

        with (
            patch.object(settings, "MINIMAX_API_KEY", "m"),
            patch.object(settings, "GLM_API_KEY", "primary"),
            patch.object(settings, "GLM_LIGHT_MODEL", "glm-5.3"),
            patch.object(settings, "GLM_FALLBACK_API_KEY", "fallback"),
            patch.object(settings, "LLM_RESILIENCE_ENABLED", True),
        ):
            svc = LLMFactory._create_resilient_light_chain()

        assert isinstance(svc, ResilientLLMService)
        assert svc.provider_names == ["minimax", "glm", "glm-fallback"]
        last = svc._providers[-1][1]
        assert isinstance(last, GLMClient)
        assert last.api_key == "fallback"
        assert last.model == "glm-5.3"

    def test_light_chain_unchanged_without_fallback_key(self):
        from unittest.mock import patch

        from app.config.settings import settings
        from app.services.llm.factory import LLMFactory

        with (
            patch.object(settings, "MINIMAX_API_KEY", "m"),
            patch.object(settings, "GLM_API_KEY", "primary"),
            patch.object(settings, "GLM_FALLBACK_API_KEY", None),
            patch.object(settings, "LLM_RESILIENCE_ENABLED", True),
        ):
            svc = LLMFactory._create_resilient_light_chain()

        assert isinstance(svc, ResilientLLMService)
        assert svc.provider_names == ["minimax", "glm"]
