"""Tiered model routing: cheap classification tier, capable generation tier.

Industry practice (LiteLLM AutoRouter, OpenRouter routing, ICML 2025
cascade-routing) splits cost tiers per task, not per request: a cheap
model serves classification/extraction/summarization, the flagship
serves final generation. Our graph IS the router — intent detection and
conditional edges already classify every turn deterministically, so the
tier assignment happens at composition time with zero extra LLM calls
("never route with an LLM call if a regex will do" — the structural
analog). These tests pin the wiring: light tier for intent/slots/
rerank/handoff/memory, primary for generation/agent, one shared budget.
"""

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.services.llm.base import LLMMessage, LLMResponse, LLMServiceBase


class _RecordingLLM(LLMServiceBase):
    """Fake provider recording the model it serves."""

    def __init__(self, model: str) -> None:
        super().__init__(api_key="fake", model=model)
        self.generate_calls = 0

    async def generate(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.generate_calls += 1
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


class TestCreateLightFromSettings:
    def test_unset_light_model_returns_primary_unchanged(self):
        from app.config.settings import settings
        from app.services.llm.factory import LLMFactory

        primary = _RecordingLLM("glm-5.3-flash")
        with patch.object(settings, "CHAT_LLM_LIGHT_MODEL", None):
            assert LLMFactory.create_light_from_settings(primary) is primary

    def test_light_model_builds_distinct_glm_service(self):
        from app.config.settings import settings
        from app.services.llm.factory import LLMFactory
        from app.services.llm.glm_client import GLMClient

        primary = _RecordingLLM("glm-5.3-flash")
        with (
            patch.object(settings, "CHAT_LLM_LIGHT_MODEL", "glm-4-flash"),
            patch.object(settings, "GLM_API_KEY", "k"),
        ):
            light = LLMFactory.create_light_from_settings(primary)

        assert light is not primary
        assert isinstance(light, GLMClient)
        assert light.model == "glm-4-flash"

    def test_light_model_follows_primary_provider(self):
        """OpenAI-only deployments get an OpenAI light tier."""
        from app.config.settings import settings
        from app.services.llm.factory import LLMFactory
        from app.services.llm.openai_client import OpenAIClient

        primary = _RecordingLLM("gpt-4-turbo-preview")
        with (
            patch.object(settings, "CHAT_LLM_LIGHT_MODEL", "gpt-5-mini"),
            patch.object(settings, "GLM_API_KEY", None),
            patch.object(settings, "OPENAI_API_KEY", "k"),
        ):
            light = LLMFactory.create_light_from_settings(primary)

        assert isinstance(light, OpenAIClient)
        assert light.model == "gpt-5-mini"


class TestChatFactoryLightWiring:
    def _run_factory(self, **overrides: Any) -> None:
        from app.services.chat.factory import ChatServiceFactory

        ChatServiceFactory.create_with_defaults(
            llm_service=_RecordingLLM("glm-5.3-flash"),
            message_repo=MagicMock(),
            session_repo=MagicMock(),
            light_llm_service=_RecordingLLM("glm-4-flash"),
            memory_type="optimized",
            intent_type="hybrid",
            **overrides,
        )

    def test_intent_detector_receives_light_tier(self):
        from app.services.llm.budget import BudgetedLLMService

        with patch("app.services.chat.factory.IntentFactory.create") as intent_create:
            self._run_factory()
        kwargs = intent_create.call_args.kwargs
        wired = kwargs["llm_service"]
        assert isinstance(wired, BudgetedLLMService)
        assert wired._inner.model == "glm-4-flash"

    def test_handoff_summary_receives_light_tier(self):
        """The handoff 交接单 is a summarization task — light tier."""
        with patch("app.services.handoff.create_handoff_service") as handoff_create:
            self._run_factory()
        wired = handoff_create.call_args.kwargs["llm_service"]
        assert wired._inner.model == "glm-4-flash"

    def test_agent_receives_primary_tier(self):
        """Tool-calling rounds need capability — never the light tier."""
        from app.config.settings import settings

        with (
            patch.object(settings, "AGENT_TOOLS_ENABLED", True),
            patch("app.services.agent.AgentService") as agent_cls,
        ):
            self._run_factory()
        wired = agent_cls.call_args.kwargs["llm_service"]
        assert wired._inner.model == "glm-5.3-flash"


class TestSharedBudgetAcrossTiers:
    async def test_light_and_primary_count_one_budget(self):
        """Both tiers reserve against the same per-request scope."""
        from app.services.llm.budget import (
            BudgetedLLMService,
            LLMBudgetExceeded,
            enter_llm_budget,
        )

        light = BudgetedLLMService(_RecordingLLM("glm-4-flash"))
        primary = BudgetedLLMService(_RecordingLLM("glm-5.3-flash"))
        msg = [LLMMessage(role="user", content="q")]
        async with enter_llm_budget(max_calls=2):
            await light.generate(msg)
            await primary.generate(msg)
            with pytest.raises(LLMBudgetExceeded):
                await primary.generate(msg)


class TestLightSettings:
    def test_light_model_defaults_to_off(self):
        from app.config.settings import settings

        assert settings.CHAT_LLM_LIGHT_MODEL is None
