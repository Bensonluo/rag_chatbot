"""Chat-path rate limiting: Redis sliding window with in-memory fallback."""

from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient
from redis import exceptions as redis_exceptions

from app.api.v1 import chat as chat_module
from app.api.v1.chat import get_chat_service
from app.middleware.rate_limiter_redis import (
    EndpointRateLimiter,
    _RedisClientHolder,
)


class _FakeRedis:
    """Stateful fake emulating the sliding-window Lua script result."""

    def __init__(self, limit: int, fail: bool = False):
        self.limit = limit
        self.fail = fail
        self.calls = 0

    async def eval(self, script, numkeys, *args):  # noqa: ARG002
        if self.fail:
            raise redis_exceptions.ConnectionError("redis down")
        self.calls += 1
        if self.calls <= self.limit:
            return [1, self.limit - self.calls]
        return [0, 0]


class _KeyedFakeRedis:
    """Fake recording the exact Redis keys each check consumes."""

    def __init__(self):
        self.keys: list[str] = []

    async def eval(self, script, numkeys, key, *args):  # noqa: ARG002
        self.keys.append(key)
        return [1, 0]


class TestEndpointRateLimiter:
    async def test_allows_within_limit_and_rejects_beyond(self, monkeypatch):
        # One fake instance shared by every get() call (a fresh-per-call
        # lambda would reset the window counter each check).
        fake = _FakeRedis(limit=2)
        monkeypatch.setattr(_RedisClientHolder, "get", lambda: fake)
        limiter = EndpointRateLimiter(scope="chat", requests_per_minute=2)

        assert (await limiter.check("1.2.3.4"))[0] is True
        assert (await limiter.check("1.2.3.4"))[0] is True
        assert (await limiter.check("1.2.3.4"))[0] is False

    async def test_keys_are_scoped_per_limiter(self, monkeypatch):
        """The chat budget counter is separate from the global middleware one."""
        fake = _KeyedFakeRedis()
        monkeypatch.setattr(_RedisClientHolder, "get", lambda: fake)

        chat = EndpointRateLimiter(scope="chat", requests_per_minute=5)
        await chat.check("1.2.3.4")

        assert fake.keys == ["ratelimit:chat:1.2.3.4"]

    async def test_falls_back_to_memory_when_redis_down(self, monkeypatch):
        """A Redis outage degrades to a per-process bucket, not an error."""
        fake = _FakeRedis(limit=2, fail=True)
        monkeypatch.setattr(_RedisClientHolder, "get", lambda: fake)
        limiter = EndpointRateLimiter(scope="chat", requests_per_minute=2)

        assert (await limiter.check("1.2.3.4"))[0] is True
        assert (await limiter.check("1.2.3.4"))[0] is True
        assert (await limiter.check("1.2.3.4"))[0] is False


class _AlwaysDeny:
    async def check(self, client_ip):
        return False, 0


class _AlwaysAllow:
    async def check(self, client_ip):
        return True, 5


class TestChatEndpointRateLimit:
    @pytest.fixture
    def client(self):
        from app.main import create_app

        return TestClient(create_app())

    @pytest.fixture
    def mock_chat_service(self):
        service = Mock()
        service.process_message = AsyncMock(
            return_value=Mock(
                content="Hello!",
                session_id=1,
                intent="greeting",
                sources=None,
                metadata=None,
            )
        )
        return service

    def test_429_when_budget_exhausted(self, client, monkeypatch, mock_chat_service):
        # The dependency must resolve (so the handler runs and the
        # rate-limit check fires) even though the handler never proceeds.
        client.app.dependency_overrides[get_chat_service] = lambda: mock_chat_service
        monkeypatch.setattr(chat_module, "_chat_rate_limiter", _AlwaysDeny())

        resp = client.post("/api/v1/chat", json={"message": "hi", "session_id": 1})

        assert resp.status_code == 429
        assert resp.headers["Retry-After"] == "60"

    def test_allows_within_budget(self, client, monkeypatch, mock_chat_service):
        monkeypatch.setattr(chat_module, "_chat_rate_limiter", _AlwaysAllow())
        client.app.dependency_overrides[get_chat_service] = lambda: mock_chat_service

        resp = client.post("/api/v1/chat", json={"message": "hi", "session_id": 1})

        assert resp.status_code == 200
        mock_chat_service.process_message.assert_called_once()

    def test_disabled_rate_limit_skips_check(self, client, monkeypatch, mock_chat_service):
        """RATE_LIMIT_ENABLED=False turns the chat budget off entirely."""
        monkeypatch.setattr(chat_module, "_chat_rate_limiter", _AlwaysDeny())
        monkeypatch.setattr(chat_module.settings, "RATE_LIMIT_ENABLED", False)
        client.app.dependency_overrides[get_chat_service] = lambda: mock_chat_service

        resp = client.post("/api/v1/chat", json={"message": "hi", "session_id": 1})

        assert resp.status_code == 200

    def test_stream_endpoint_enforces_same_budget(self, client, monkeypatch, mock_chat_service):
        client.app.dependency_overrides[get_chat_service] = lambda: mock_chat_service
        monkeypatch.setattr(chat_module, "_chat_rate_limiter", _AlwaysDeny())

        resp = client.post("/api/v1/chat/stream", json={"message": "hi", "session_id": 1})

        assert resp.status_code == 429
