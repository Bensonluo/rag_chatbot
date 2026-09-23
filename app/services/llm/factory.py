"""
Factory for creating LLM service instances.

Provides a simple interface for creating LLM clients based on configuration.
"""

import logging

from app.config.settings import get_settings
from app.core.exceptions import ValidationError
from app.services.llm.anthropic_client import AnthropicClient
from app.services.llm.base import LLMServiceBase
from app.services.llm.glm_client import GLMClient
from app.services.llm.openai_client import OpenAIClient
from app.services.llm.resilience import ResilientLLMService

logger = logging.getLogger(__name__)


def _build_llm_tracer():
    """Construct the LLM tracer, degrading to None when OTel is broken.

    Tracing must never gate LLM availability: a broken tracing setup
    logs and disables spans rather than failing service creation.
    """
    try:
        from app.services.observability.llm_tracer import LLMTracer

        return LLMTracer()
    except Exception:  # noqa: BLE001 - observability is best-effort
        logger.warning("LLM tracer unavailable; spans disabled", exc_info=True)
        return None


class LLMFactory:
    """
    Factory for creating LLM service instances.

    Supports multiple LLM providers (OpenAI, Anthropic, GLM) with unified interface.
    """

    SUPPORTED_PROVIDERS = ["openai", "anthropic", "glm"]

    @staticmethod
    def create(
        provider: str = "openai",
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMServiceBase:
        """
        Create an LLM service instance for the specified provider.

        Args:
            provider: LLM provider name ("openai", "anthropic", "glm")
            api_key: API key (defaults to environment variable if not provided)
            model: Model name (defaults to provider default if not provided)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            LLMServiceBase: Configured LLM service instance

        Raises:
            ValidationError: If provider is not supported or configuration is invalid

        Examples:
            >>> # Create GLM client with environment variables
            >>> llm = LLMFactory.create(provider="glm")
            >>>
            >>> # Create GLM client with explicit API key
            >>> llm = LLMFactory.create(
            ...     provider="glm",
            ...     api_key="your-api-key",
            ...     model="glm-5.3-flash"
            ... )
            >>>
            >>> # Create OpenAI client
            >>> llm = LLMFactory.create(provider="openai")
        """
        settings = get_settings()

        # Validate provider
        provider = provider.lower()
        if provider not in LLMFactory.SUPPORTED_PROVIDERS:
            raise ValidationError(
                f"Unsupported LLM provider: {provider}. "
                f"Supported providers: {', '.join(LLMFactory.SUPPORTED_PROVIDERS)}"
            )

        # Create client based on provider
        if provider == "openai":
            api_key = api_key or settings.OPENAI_API_KEY
            if not api_key:
                raise ValidationError("OpenAI API key not configured")

            model = model or settings.OPENAI_MODEL
            return OpenAIClient(
                api_key=api_key,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                base_url=settings.OPENAI_BASE_URL,
            )

        elif provider == "anthropic":
            api_key = api_key or settings.ANTHROPIC_API_KEY
            if not api_key:
                raise ValidationError("Anthropic API key not configured")

            model = model or settings.ANTHROPIC_MODEL
            return AnthropicClient(
                api_key=api_key,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        elif provider == "glm":
            api_key = api_key or settings.GLM_API_KEY
            if not api_key:
                raise ValidationError("GLM API key not configured")

            model = model or settings.GLM_MODEL
            return GLMClient(
                api_key=api_key,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        else:
            raise ValidationError(f"Provider {provider} not implemented")

    @staticmethod
    def create_from_settings() -> LLMServiceBase:
        """
        Create LLM service from settings with automatic provider selection.

        Uses the first available API key in the order: GLM, OpenAI, Anthropic.

        Returns:
            LLMServiceBase: Configured LLM service instance

        Raises:
            ValidationError: If no LLM provider is configured

        Examples:
            >>> # Auto-select provider (checks GLM first, then OpenAI, then Anthropic)
            >>> llm = LLMFactory.create_from_settings()
        """
        settings = get_settings()

        # Build the full configured chain in priority order: a hard
        # failure on the primary serves from the next provider.
        providers: list[tuple[str, LLMServiceBase]] = []
        if settings.GLM_API_KEY:
            providers.append(
                (
                    "glm",
                    LLMFactory.create(
                        provider="glm",
                        api_key=settings.GLM_API_KEY,
                        model=settings.GLM_MODEL,
                    ),
                )
            )
        if settings.OPENAI_API_KEY:
            providers.append(
                (
                    "openai",
                    LLMFactory.create(
                        provider="openai",
                        api_key=settings.OPENAI_API_KEY,
                        model=settings.OPENAI_MODEL,
                    ),
                )
            )
        if settings.ANTHROPIC_API_KEY:
            providers.append(
                (
                    "anthropic",
                    LLMFactory.create(
                        provider="anthropic",
                        api_key=settings.ANTHROPIC_API_KEY,
                        model=settings.ANTHROPIC_MODEL,
                    ),
                )
            )

        if not providers:
            raise ValidationError(
                "No LLM provider configured. Please set at least one of: "
                "GLM_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY"
            )

        if not settings.LLM_RESILIENCE_ENABLED:
            return providers[0][1]

        return ResilientLLMService(
            providers,
            max_retries=settings.LLM_MAX_RETRIES,
            backoff_base=settings.LLM_RETRY_BACKOFF_BASE,
            circuit_failure_threshold=settings.LLM_CIRCUIT_FAILURE_THRESHOLD,
            circuit_recovery_seconds=settings.LLM_CIRCUIT_RECOVERY_SECONDS,
            tracer=_build_llm_tracer(),
        )
