"""Runtime settings for the standalone commerce provider service.

Every value can be provided through ``COMMERCE_PROVIDER_*`` environment
variables (for example ``COMMERCE_PROVIDER_PORT``); the module entrypoint
additionally accepts CLI overrides. The settings intentionally stay separate
from the chat application's settings so the provider keeps running as an
independent internal service.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_AUTH_TOKEN = "commerce-provider-dev-token"


class CommerceProviderSettings(BaseSettings):
    """Configuration of the internal virtual commerce provider."""

    model_config = SettingsConfigDict(
        env_prefix="COMMERCE_PROVIDER_", extra="forbid", env_file_encoding="utf-8"
    )

    host: str = "127.0.0.1"
    port: int = Field(default=8071, ge=1, le=65535)
    database_path: str = "commerce_provider.sqlite3"
    taxonomy_path: str = Field(
        default="",
        description="Optional JSON taxonomy snapshot; empty uses the built-in generated taxonomy.",
    )
    auth_token: str = DEFAULT_AUTH_TOKEN
    quote_ttl_seconds: int = Field(default=300, ge=1)
    faults_enabled: bool = False


__all__ = ["DEFAULT_AUTH_TOKEN", "CommerceProviderSettings"]
