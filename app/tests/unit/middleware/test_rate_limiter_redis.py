"""Tests for the Redis-backed distributed rate limiting middleware."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from redis import exceptions as redis_exceptions

from app.config.settings import settings
from app.middleware import rate_limiter_redis
from app.middleware.rate_limiter_redis import (
    DistributedRateLimiterMiddleware,
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


class _StalledRedis(_FakeRedis):
    """Redis call that never replies, including connection/pool waits."""

    def __init__(self) -> None:
        super().__init__(limit=1)
        self.started = asyncio.Event()
        self.cancelled = 0

    async def eval(self, script, numkeys, *args):
        self.calls += 1
        self.started.set()
        try:
            await asyncio.Future()
        finally:
            self.cancelled += 1


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

    async def test_stalled_redis_falls_back_within_deadline_and_still_limits(self, monkeypatch):
        fake = _StalledRedis()
        _patch_redis(monkeypatch, fake)
        monkeypatch.setattr(rate_limiter_redis, "_REDIS_TIMEOUT_SECONDS", 0.01)
        app = _make_app(limit=1)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            first = await asyncio.wait_for(client.get("/ping"), timeout=0.2)
            second = await asyncio.wait_for(client.get("/ping"), timeout=0.2)

        assert [first.status_code, second.status_code] == [200, 429]
        assert fake.calls == fake.cancelled == 2

    async def test_endpoint_stalled_redis_falls_back_and_keeps_per_ip_budget(self, monkeypatch):
        fake = _StalledRedis()
        _patch_redis(monkeypatch, fake)
        monkeypatch.setattr(rate_limiter_redis, "_REDIS_TIMEOUT_SECONDS", 0.01)
        limiter = EndpointRateLimiter(scope="chat", requests_per_minute=1)

        assert await asyncio.wait_for(limiter.check("client-a"), timeout=0.2) == (True, 0)
        assert await asyncio.wait_for(limiter.check("client-a"), timeout=0.2) == (False, 0)
        assert await asyncio.wait_for(limiter.check("client-b"), timeout=0.2) == (True, 0)
        assert fake.calls == fake.cancelled == 3

    @pytest.mark.parametrize("endpoint", [False, True])
    async def test_external_cancellation_propagates_without_consuming_fallback(
        self, monkeypatch, endpoint
    ):
        fake = _StalledRedis()
        _patch_redis(monkeypatch, fake)
        limiter: EndpointRateLimiter | DistributedRateLimiterMiddleware
        if endpoint:
            limiter = EndpointRateLimiter(scope="chat", requests_per_minute=1)
            check = limiter.check
        else:
            limiter = DistributedRateLimiterMiddleware(app=None, requests_per_minute=1)
            check = limiter._check

        task = asyncio.create_task(check("client-a"))
        await fake.started.wait()
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
        assert fake.cancelled == 1
        assert limiter._fallback_buckets == {}

    async def test_tcp_connected_redis_that_never_replies_has_bounded_fallback(self, monkeypatch):
        writers = []
        handlers = set()
        connections = 0

        async def stall(reader, writer):
            nonlocal connections
            connections += 1
            writers.append(writer)
            task = asyncio.current_task()
            assert task is not None
            handlers.add(task)
            try:
                await reader.read()
            finally:
                writer.close()
                await writer.wait_closed()
                handlers.discard(task)

        server = await asyncio.start_server(stall, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        monkeypatch.setattr(_RedisClientHolder, "_client", None)
        monkeypatch.setattr(settings, "RATE_LIMIT_REDIS_URL", f"redis://127.0.0.1:{port}/0")
        # Keep this real-socket regression test fast; the production bound is 250 ms.
        monkeypatch.setattr(rate_limiter_redis, "_REDIS_TIMEOUT_SECONDS", 0.02)

        try:
            app = _make_app(limit=1)
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
                first = await asyncio.wait_for(client.get("/ping"), timeout=0.3)
                second = await asyncio.wait_for(client.get("/ping"), timeout=0.3)

            assert [first.status_code, second.status_code] == [200, 429]
            assert connections == 2, "Each check must try Redis once without retry backoff"
        finally:
            await _RedisClientHolder.close()
            server.close()
            await server.wait_closed()
            for writer in writers:
                writer.close()
                await writer.wait_closed()
            if handlers:
                await asyncio.gather(*handlers)

    async def test_client_bounds_socket_waits_and_does_not_retry(self, monkeypatch):
        monkeypatch.setattr(_RedisClientHolder, "_client", None)
        monkeypatch.setattr(settings, "RATE_LIMIT_REDIS_URL", "redis://localhost:6379/0")
        client = _RedisClientHolder.get()
        kwargs = client.connection_pool.connection_kwargs
        assert kwargs["socket_connect_timeout"] == 0.25
        assert kwargs["socket_timeout"] == 0.25
        attempts = 0

        async def fail_command():
            nonlocal attempts
            attempts += 1
            raise redis_exceptions.TimeoutError("Redis read stalled")

        async def on_failure(error):
            pass

        try:
            with pytest.raises(redis_exceptions.TimeoutError):
                await kwargs["retry"].call_with_retry(fail_command, on_failure)
            assert attempts == 1
        finally:
            await _RedisClientHolder.close()

    @pytest.mark.parametrize("window, ttl", [(60, 60), (1.001, 2), (0.05, 1)])
    async def test_lua_expiry_uses_seconds_separate_from_millisecond_scores(
        self, monkeypatch, window, ttl
    ):
        client = Mock(eval=AsyncMock(return_value=[1, 0]))
        monkeypatch.setattr(_RedisClientHolder, "get", lambda: client)

        await rate_limiter_redis._redis_sliding_window(
            "client-a", key="ratelimit:client-a", limit=1, window_seconds=window
        )

        args = client.eval.call_args.args
        assert args[4] == int(window * 1000), "Sorted-set pruning uses milliseconds"
        assert args[7] == ttl, "EXPIRE must use rounded-up seconds, not milliseconds"
        assert "local ttl = tonumber(ARGV[5])" in args[0]
        assert "redis.call('EXPIRE', key, ttl)" in args[0]

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
