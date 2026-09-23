"""Production guard for the multi-replica SECRET_KEY contract.

The SECRET_KEY default is a per-process random value. In a multi-replica
deployment (the 800K-1M daily-requests target runs multiple API pods
behind a load balancer) that means replica A mints JWTs replica B
rejects — intermittent 401s that look random. Production boots must
refuse to start without an explicit, shared SECRET_KEY.
"""

import pytest

from app.config.settings import Settings


class TestSecretKeyProductionGuard:
    def test_production_without_env_secret_key_rejected(self, monkeypatch):
        monkeypatch.delenv("SECRET_KEY", raising=False)
        monkeypatch.delenv("ENVIRONMENT", raising=False)
        with pytest.raises(ValueError, match="SECRET_KEY must be set"):
            Settings(ENVIRONMENT="production", _env_file=None)

    def test_production_with_env_secret_key_accepted(self, monkeypatch):
        key = "x" * 48
        monkeypatch.setenv("SECRET_KEY", key)
        monkeypatch.delenv("ENVIRONMENT", raising=False)
        settings = Settings(ENVIRONMENT="production", _env_file=None)
        assert key == settings.SECRET_KEY

    def test_development_still_generates_per_process_key(self, monkeypatch):
        monkeypatch.delenv("SECRET_KEY", raising=False)
        monkeypatch.delenv("ENVIRONMENT", raising=False)
        first = Settings(ENVIRONMENT="development", _env_file=None)
        second = Settings(ENVIRONMENT="development", _env_file=None)
        assert first.SECRET_KEY != second.SECRET_KEY

    def test_short_secret_key_rejected_regardless_of_env(self, monkeypatch):
        monkeypatch.setenv("SECRET_KEY", "tooshort")
        with pytest.raises(ValueError, match="at least 32 characters"):
            Settings(ENVIRONMENT="development", _env_file=None)
