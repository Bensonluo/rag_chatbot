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
    """Explicit provider selection; an optional second supplier enables routing.

    External mode remains a composition choice, not a runtime fallback. When an
    external supplier URL is configured, taxonomy-driven routing composes the
    generated provider and external supplier behind one ``CommercePort``.
    """
    chosen = settings if config is None else config
    generated: CommercePort
    if chosen.COMMERCE_PROVIDER_MODE == "external":
        generated = _shared_external_port(
            chosen.COMMERCE_PROVIDER_URL,
            chosen.COMMERCE_PROVIDER_TOKEN,
            chosen.COMMERCE_PROVIDER_TIMEOUT_SECONDS,
        )
    else:
        generated = MockCommercePort(sessions)

    if not chosen.COMMERCE_EXTERNAL_PROVIDER_URL:
        return generated

    from app.services.commerce.router import RouterCommercePort
    from app.services.commerce.taxonomy import CommerceTaxonomyRegistry

    external = _shared_external_port(
        chosen.COMMERCE_EXTERNAL_PROVIDER_URL,
        chosen.COMMERCE_EXTERNAL_PROVIDER_TOKEN,
        chosen.COMMERCE_EXTERNAL_PROVIDER_TIMEOUT_SECONDS,
    )
    return RouterCommercePort(
        generated=generated,
        external=external,
        registry=CommerceTaxonomyRegistry(sessions),
        external_provider="external",
    )
