"""Kimi Coding Plan client using the service's OpenAI-compatible API.

The coding-plan highspeed models currently reject ``temperature`` values other
than ``1`` (verified live against ``kimi-for-coding-highspeed``). Callers such
as intent detection still request temperature ``0`` through the shared service
interface, so this adapter normalizes that provider constraint at the boundary
instead of leaking Kimi's restriction into every consumer.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from app.services.llm.base import LLMMessage, LLMResponse
from app.services.llm.openai_client import OpenAIClient

KIMI_REQUIRED_TEMPERATURE = 1.0


class KimiCodingClient(OpenAIClient):
    """OpenAI-compatible client for ``https://api.kimi.com/coding/v1``."""

    async def generate(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: object,
    ) -> LLMResponse:
        del temperature
        return await super().generate(
            messages,
            max_tokens=max_tokens,
            temperature=KIMI_REQUIRED_TEMPERATURE,
            **kwargs,
        )

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: object,
    ) -> AsyncGenerator[str, None]:
        del temperature
        async for chunk in super().generate_stream(
            messages,
            max_tokens=max_tokens,
            temperature=KIMI_REQUIRED_TEMPERATURE,
            **kwargs,
        ):
            yield chunk


__all__ = ["KIMI_REQUIRED_TEMPERATURE", "KimiCodingClient"]
