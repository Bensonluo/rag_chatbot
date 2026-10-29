"""Kimi Coding Plan wiring: explicit intent-only highspeed provider.

Kimi is not inserted into the general resilience chain from this configuration.
It exists so intent detection can use the membership's highspeed model without
changing generation, Agent tool rounds, slots, summaries, or MiniMax light-tier
consumers.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.exceptions import ValidationError
from app.services.llm.base import LLMMessage, LLMResponse, LLMServiceBase
from app.services.llm.kimi_client import KimiCodingClient
from app.services.llm.openai_client import OpenAIClient


class _RecordingLLM(LLMServiceBase):
    """Minimal tier marker for ChatServiceFactory wiring assertions."""

    def __init__(self, model: str) -> None:
        super().__init__(api_key="fake", model=model)

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
        return sum(len(message.content) for message in messages)


class TestKimiClient:
    async def test_client_normalizes_rejected_temperature_to_one(self):
        response = LLMResponse(content="pong", model="kimi-for-coding-highspeed")
        client = KimiCodingClient(
            api_key="test-key",
            model="kimi-for-coding-highspeed",
            client=object(),
        )

        with patch.object(OpenAIClient, "generate", new=AsyncMock(return_value=response)) as call:
            result = await client.generate(
                [LLMMessage(role="user", content="ping")],
                max_tokens=64,
                temperature=0.0,
            )

        assert result.content == "pong"
        call.assert_awaited_once()
        assert call.call_args.kwargs["temperature"] == 1.0


class TestKimiProviderFactory:
    def test_create_builds_coding_plan_client(self):
        from app.config.settings import settings
        from app.services.llm.factory import LLMFactory

        endpoint = "https://api.kimi.com/coding/v1"
        with (
            patch.object(settings, "KIMI_API_KEY", "kimi-key"),
            patch.object(settings, "KIMI_BASE_URL", endpoint),
            patch.object(settings, "KIMI_MODEL", "kimi-for-coding"),
        ):
            service = LLMFactory.create(provider="kimi")

        assert isinstance(service, KimiCodingClient)
        assert service.model == "kimi-for-coding"
        assert str(service.client.base_url).rstrip("/") == endpoint

    def test_create_without_key_raises(self):
        from app.config.settings import settings
        from app.services.llm.factory import LLMFactory

        with (
            patch.object(settings, "KIMI_API_KEY", None),
            pytest.raises(ValidationError, match="Kimi Coding API key not configured"),
        ):
            LLMFactory.create(provider="kimi")

    def test_kimi_is_not_implicitly_added_to_generation_failover(self):
        """Kimi serves intent only; a key must not alter the primary chain."""
        from app.config.settings import settings
        from app.services.llm.factory import LLMFactory
        from app.services.llm.resilience import ResilientLLMService

        with (
            patch.object(settings, "GLM_API_KEY", "g"),
            patch.object(settings, "MINIMAX_API_KEY", "m"),
            patch.object(settings, "OPENAI_API_KEY", None),
            patch.object(settings, "ANTHROPIC_API_KEY", None),
            patch.object(settings, "KIMI_API_KEY", "kimi"),
            patch.object(settings, "LLM_RESILIENCE_ENABLED", True),
        ):
            chain = LLMFactory.create_from_settings()

        assert isinstance(chain, ResilientLLMService)
        assert [name for name, _ in chain._providers] == ["glm", "minimax"]


class TestKimiIntentWiring:
    def _run_factory(self) -> None:
        from app.services.chat.factory import ChatServiceFactory

        ChatServiceFactory.create_with_defaults(
            llm_service=_RecordingLLM("glm-5.3-flash"),
            message_repo=MagicMock(),
            session_repo=MagicMock(),
            light_llm_service=_RecordingLLM("MiniMax-M2.7-highspeed"),
            memory_type="optimized",
            intent_type="hybrid",
        )

    def test_intent_detector_uses_kimi_while_other_light_calls_do_not(self):
        from app.config.settings import settings
        from app.services.llm.budget import BudgetedLLMService

        with (
            patch.object(settings, "INTENT_LLM_PROVIDER", "kimi"),
            patch.object(settings, "KIMI_API_KEY", "kimi-key"),
            patch.object(settings, "KIMI_INTENT_MODEL", "kimi-for-coding-highspeed"),
            patch.object(settings, "KIMI_BASE_URL", "https://api.kimi.com/coding/v1"),
            patch("app.services.chat.factory.IntentFactory.create") as intent_create,
            patch("app.services.handoff.create_handoff_service") as handoff_create,
        ):
            self._run_factory()

        wired = intent_create.call_args.kwargs["llm_service"]
        assert isinstance(wired, BudgetedLLMService)
        assert isinstance(wired._inner, KimiCodingClient)
        assert wired._inner.model == "kimi-for-coding-highspeed"
        assert str(wired._inner.client.base_url).rstrip("/") == "https://api.kimi.com/coding/v1"
        assert handoff_create.call_args.kwargs["llm_service"]._inner.model == (
            "MiniMax-M2.7-highspeed"
        )

    def test_kimi_intent_selection_without_key_fails_fast(self):
        from app.config.settings import settings

        with (
            patch.object(settings, "INTENT_LLM_PROVIDER", "kimi"),
            patch.object(settings, "KIMI_API_KEY", None),
            pytest.raises(ValidationError, match="KIMI_API_KEY"),
        ):
            self._run_factory()

    def test_defaults_keep_kimi_opt_in(self):
        from app.config.settings import settings

        assert settings.INTENT_LLM_PROVIDER == "light"
        assert settings.KIMI_INTENT_MODEL == "kimi-for-coding-highspeed"
        assert settings.KIMI_BASE_URL == "https://api.kimi.com/coding/v1"
