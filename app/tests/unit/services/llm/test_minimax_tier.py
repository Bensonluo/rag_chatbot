"""MiniMax provider wiring: backup failover tier + speed (light) tier.

The MiniMax coding-plan key serves two roles (user directive 2026-09-30):
1. Backup — enters the resilience chain right after GLM so a hard GLM
   failure still serves from a domestic, reachable endpoint.
2. Speed tier — when the key is configured, the light (classification)
   tier prefers MiniMax highspeed models: intent/slots/rerank calls are
   latency-sensitive and quality-tolerant, the exact profile the
   highspeed variants are built for.
"""

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import patch

import pytest

from app.core.exceptions import ValidationError
from app.services.llm.base import LLMMessage, LLMResponse, LLMServiceBase


class _Primary(LLMServiceBase):
    """Fake primary provider for wiring tests."""

    def __init__(self) -> None:
        super().__init__(api_key="fake", model="glm-5.3-flash")

    async def generate(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        return LLMResponse(content="ok", model=self.model)

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        yield "ok"

    async def generate_with_tools(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        return LLMResponse(content="ok", model=self.model)

    def estimate_tokens(self, text: str) -> int:
        return len(text)

    async def count_tokens(self, messages: list[LLMMessage]) -> int:
        return sum(len(m.content) for m in messages)


class TestMinimaxProvider:
    def test_create_builds_openai_compatible_client(self):
        from app.config.settings import settings
        from app.services.llm.factory import LLMFactory
        from app.services.llm.openai_client import OpenAIClient

        with (
            patch.object(settings, "MINIMAX_API_KEY", "k"),
            patch.object(settings, "MINIMAX_MODEL", "MiniMax-M3"),
            patch.object(settings, "MINIMAX_BASE_URL", "https://api.minimaxi.com/v1"),
        ):
            svc = LLMFactory.create(provider="minimax", api_key="k")

        assert isinstance(svc, OpenAIClient)
        assert svc.model == "MiniMax-M3"
        # Must point at the MiniMax endpoint, never the OpenAI default.
        # The openai SDK normalizes a trailing slash onto base_url.
        assert str(svc.client.base_url).rstrip("/") == "https://api.minimaxi.com/v1"

    def test_create_without_key_raises(self):
        from app.config.settings import settings
        from app.services.llm.factory import LLMFactory

        with (
            patch.object(settings, "MINIMAX_API_KEY", None),
            pytest.raises(ValidationError),
        ):
            LLMFactory.create(provider="minimax")


class TestMinimaxInResilienceChain:
    def test_minimax_is_second_after_glm(self):
        """Failover order: GLM primary, MiniMax backup, then the rest."""
        from app.config.settings import settings
        from app.services.llm.factory import LLMFactory
        from app.services.llm.resilience import ResilientLLMService

        with (
            patch.object(settings, "GLM_API_KEY", "g"),
            patch.object(settings, "MINIMAX_API_KEY", "m"),
            patch.object(settings, "OPENAI_API_KEY", None),
            patch.object(settings, "ANTHROPIC_API_KEY", None),
            patch.object(settings, "LLM_RESILIENCE_ENABLED", True),
        ):
            svc = LLMFactory.create_from_settings()

        assert isinstance(svc, ResilientLLMService)
        names = [name for name, _ in svc._providers]
        assert names == ["glm", "minimax"]

    def test_minimax_alone_still_configures(self):
        from app.config.settings import settings
        from app.services.llm.factory import LLMFactory
        from app.services.llm.openai_client import OpenAIClient

        with (
            patch.object(settings, "GLM_API_KEY", None),
            patch.object(settings, "MINIMAX_API_KEY", "m"),
            patch.object(settings, "OPENAI_API_KEY", None),
            patch.object(settings, "ANTHROPIC_API_KEY", None),
            patch.object(settings, "LLM_RESILIENCE_ENABLED", False),
        ):
            svc = LLMFactory.create_from_settings()

        assert isinstance(svc, OpenAIClient)
        assert svc.model == settings.MINIMAX_MODEL


class TestMinimaxSpeedTier:
    def test_light_tier_prefers_minimax_when_keyed(self):
        """With a MiniMax key, the light tier is MiniMax highspeed —
        speed-sensitive, quality-tolerant call sites get the fast
        provider even when CHAT_LLM_LIGHT_MODEL names a GLM model."""
        from app.config.settings import settings
        from app.services.llm.factory import LLMFactory
        from app.services.llm.openai_client import OpenAIClient

        primary = _Primary()
        with (
            patch.object(settings, "MINIMAX_API_KEY", "m"),
            patch.object(settings, "MINIMAX_LIGHT_MODEL", "MiniMax-M2.7-highspeed"),
            patch.object(settings, "MINIMAX_BASE_URL", "https://api.minimaxi.com/v1"),
            patch.object(settings, "GLM_API_KEY", "g"),
            patch.object(settings, "CHAT_LLM_LIGHT_MODEL", "glm-4-flash"),
        ):
            light = LLMFactory.create_light_from_settings(primary)

        assert light is not primary
        assert isinstance(light, OpenAIClient)
        assert light.model == "MiniMax-M2.7-highspeed"
        assert str(light.client.base_url).rstrip("/") == "https://api.minimaxi.com/v1"

    def test_light_tier_without_minimax_key_keeps_glm(self):
        from app.config.settings import settings
        from app.services.llm.factory import LLMFactory
        from app.services.llm.glm_client import GLMClient

        primary = _Primary()
        with (
            patch.object(settings, "MINIMAX_API_KEY", None),
            patch.object(settings, "GLM_API_KEY", "g"),
            patch.object(settings, "CHAT_LLM_LIGHT_MODEL", "glm-4-flash"),
        ):
            light = LLMFactory.create_light_from_settings(primary)

        assert isinstance(light, GLMClient)
        assert light.model == "glm-4-flash"

    def test_light_model_defaults_present(self):
        """Ship a working default: key alone is enough to turn the
        speed tier on (no extra env required)."""
        from app.config.settings import settings

        assert settings.MINIMAX_LIGHT_MODEL == "MiniMax-M2.7-highspeed"
        assert settings.MINIMAX_MODEL == "MiniMax-M3"
        assert settings.MINIMAX_BASE_URL == "https://api.minimaxi.com/v1"
