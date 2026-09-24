"""Settings field behavior on isolated instances.

These tests build fresh ``Settings(_env_file=None)`` objects rather
than touching the process-wide singleton: the singleton is constructed
once from the real ``.env`` at import time, so asserting against it
after ``monkeypatch.setenv`` tests nothing — the previous version of
this file rotted exactly that way while invisible to the gate, and
its "missing secret key" test even asserted an exception inside an
empty ``with`` block. The SECRET_KEY production guard has its own
curated contract in
``app/tests/unit/config/test_settings_secret_key.py`` — not
duplicated here.

Init kwargs outrank environment variables in pydantic-settings, so
the values asserted below cannot be polluted by the host environment.
"""

import pytest
from pydantic import ValidationError

from app.config.settings import Settings

_PG_URL = "postgresql+asyncpg://user:pass@localhost:5432/test"


class TestSettingsFields:
    def test_fields_load_from_init_kwargs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The production guard accepts SECRET_KEY only from the real
        # environment (init kwargs would defeat its purpose) — same
        # pattern as test_settings_secret_key.py.
        monkeypatch.setenv("SECRET_KEY", "x" * 48)
        settings = Settings(
            ENVIRONMENT="production",
            DATABASE_URL=_PG_URL,
            _env_file=None,
        )
        assert settings.ENVIRONMENT == "production"
        assert settings.DATABASE_URL == _PG_URL

    def test_invalid_database_url_scheme_is_rejected(self):
        with pytest.raises(ValidationError, match="postgresql"):
            Settings(DATABASE_URL="mysql://nope", _env_file=None)

    def test_database_pool_fields(self):
        settings = Settings(DATABASE_POOL_SIZE=50, DATABASE_MAX_OVERFLOW=20, _env_file=None)
        assert settings.DATABASE_POOL_SIZE == 50
        assert settings.DATABASE_MAX_OVERFLOW == 20

    def test_redis_url_field(self):
        settings = Settings(REDIS_URL="redis://localhost:6379/1", _env_file=None)
        assert settings.REDIS_URL == "redis://localhost:6379/1"

    def test_openai_configuration_fields(self):
        settings = Settings(OPENAI_API_KEY="sk-test-key", OPENAI_MODEL="gpt-4", _env_file=None)
        assert settings.OPENAI_API_KEY == "sk-test-key"
        assert settings.OPENAI_MODEL == "gpt-4"

    def test_jwt_expiration_fields(self):
        settings = Settings(
            JWT_ACCESS_TOKEN_EXPIRE_MINUTES=60,
            JWT_REFRESH_TOKEN_EXPIRE_DAYS=30,
            _env_file=None,
        )
        assert settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES == 60
        assert settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS == 30

    def test_rate_limiting_fields(self):
        settings = Settings(
            RATE_LIMIT_REQUESTS_PER_MINUTE=120,
            RATE_LIMIT_PER_HOUR=2000,
            _env_file=None,
        )
        assert settings.RATE_LIMIT_REQUESTS_PER_MINUTE == 120
        assert settings.RATE_LIMIT_PER_HOUR == 2000
