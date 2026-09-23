"""Tests for the Redis-backed distributed rate limiting middleware."""

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from redis import exceptions as redis_exceptions

from app.middleware.rate_limiter_redis import (
    DistributedRateLimiterMiddleware,
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


def _make_app(limit: int, window: int = 60, whitelist: list[str] | None = None) -> FastAPI:
    app = FastAPI()
    app.add_middleware(
        DistributedRateLimiterMiddleware,
        requests_per_minute=limit,
        window_seconds=window,
        whitelist_paths=whitelist,
    )

    @app.get("/ping")
    async def ping():
        return {"ok": True}

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    return app


def _patch_redis(monkeypatch: pytest.MonkeyPatch, fake: _FakeRedis) -> None:
    monkeypatch.setattr(_RedisClientHolder, "get", lambda: fake)


class TestDistributedRateLimiterRedis:
    """Redis-backed rate limiting behavior"""

    async def test_allows_within_limit_and_blocks_beyond(self, monkeypatch):
        """Requests up to the limit pass; the next one gets HTTP 429"""
        # Arrange
        _patch_redis(monkeypatch, _FakeRedis(limit=3))
        app = _make_app(limit=3)

        # Act / Assert
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            for _ in range(3):
                resp = await client.get("/ping")
                assert resp.status_code == 200
            resp = await client.get("/ping")
            assert resp.status_code == 429

    async def test_429_includes_standard_headers_and_body(self, monkeypatch):
        """The 429 response carries Retry-After and X-RateLimit headers"""
        # Arrange
        _patch_redis(monkeypatch, _FakeRedis(limit=1))
        app = _make_app(limit=1, window=30)

        # Act
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            await client.get("/ping")
            resp = await client.get("/ping")

        # Assert
        assert resp.status_code == 429
        assert resp.headers["Retry-After"] == "30"
        assert resp.headers["X-RateLimit-Limit"] == "1"
        assert resp.headers["X-RateLimit-Remaining"] == "0"
        assert resp.json()["message"] == "Rate limit exceeded"

    async def test_falls_back_to_memory_when_redis_down(self, monkeypatch):
        """A Redis outage degrades to per-process buckets instead of erroring"""
        # Arrange
        _patch_redis(monkeypatch, _FakeRedis(limit=2, fail=True))
        app = _make_app(limit=2)

        # Act / Assert
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            assert (await client.get("/ping")).status_code == 200
            assert (await client.get("/ping")).status_code == 200
            assert (await client.get("/ping")).status_code == 429

    async def test_options_preflight_bypasses_limit(self, monkeypatch):
        """CORS preflight requests are never rate limited"""
        # Arrange
        _patch_redis(monkeypatch, _FakeRedis(limit=1))
        app = _make_app(limit=1)

        # Act / Assert
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            assert (await client.get("/ping")).status_code == 200
            resp = await client.options("/ping")
            # 405 = route rejects OPTIONS; the point is it was not throttled
            assert resp.status_code != 429

    async def test_whitelisted_path_bypasses_limit(self, monkeypatch):
        """Health probes are exempt so load balancers are never throttled"""
        # Arrange
        _patch_redis(monkeypatch, _FakeRedis(limit=1))
        app = _make_app(limit=1, whitelist=["/health"])

        # Act / Assert
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            assert (await client.get("/ping")).status_code == 200
            for _ in range(5):
                resp = await client.get("/health")
                assert resp.status_code == 200

    def test_mounted_by_default_in_app_factory(self):
        """create_app wires the distributed limiter when RATE_LIMIT_ENABLED is on"""
        # Arrange / Act
        from app.main import create_app

        app = create_app()

        # Assert
        mounted = any(
            m.cls is DistributedRateLimiterMiddleware  # type: ignore[comparison-overlap]
            for m in app.user_middleware
        )
        assert mounted, "DistributedRateLimiterMiddleware should be mounted by default"
