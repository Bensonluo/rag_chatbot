"""Public chat budget defaults.

The guest bootstrap endpoint must not consume the chat-specific budget, and the
remaining chat budget must accommodate several active visitors sharing one
public exit IP.
"""

from app.config.settings import Settings


def test_chat_budget_default_is_shared_exit_ip_friendly(monkeypatch):
    monkeypatch.delenv("RATE_LIMIT_REQUESTS_PER_MINUTE", raising=False)
    monkeypatch.delenv("CHAT_RATE_LIMIT_REQUESTS_PER_MINUTE", raising=False)
    settings = Settings(_env_file=None)
    assert settings.RATE_LIMIT_REQUESTS_PER_MINUTE == 120
    assert settings.CHAT_RATE_LIMIT_REQUESTS_PER_MINUTE == 30
