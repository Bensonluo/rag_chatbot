"""Controlled purchase orchestration; external facts remain adapter owned."""

from functools import lru_cache

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config.settings import Settings, settings
from app.services.commerce.external_port import ExternalCommercePort
from app.services.commerce.mock_port import MockCommercePort
from app.services.commerce.port import CommercePort

__all__ = ["CommercePort", "ExternalCommercePort", "create_commerce_port"]


@lru_cache(maxsize=4)
def _shared_external_port(
    base_url: str, token: str, timeout_seconds: float
) -> ExternalCommercePort:
    """One pooled HTTP client per distinct provider configuration."""
    return ExternalCommercePort(base_url, token=token, timeout_seconds=timeout_seconds)


def create_commerce_port(
    sessions: async_sessionmaker[AsyncSession], *, config: Settings | None = None
) -> CommercePort:
    """Explicit provider selection from settings; mock and external never mix.

    External mode is a composition choice, not a runtime fallback: when the
    provider is missing or unreachable, the adapter fails every call loudly
    instead of consulting a local commerce implementation.
    """
    chosen = settings if config is None else config
    if chosen.COMMERCE_PROVIDER_MODE == "external":
        return _shared_external_port(
            chosen.COMMERCE_PROVIDER_URL,
            chosen.COMMERCE_PROVIDER_TOKEN,
            chosen.COMMERCE_PROVIDER_TIMEOUT_SECONDS,
        )
    return MockCommercePort(sessions)
