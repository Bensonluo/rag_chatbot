"""
Distributed rate limiting middleware backed by Redis.

Uses an atomic Lua script implementing a sliding-window counter so the
limit is enforced consistently across all API instances (horizontal
scaling safe). If Redis is unreachable, the middleware degrades to a
per-process in-memory token bucket and logs a warning, keeping the API
available while still providing best-effort protection.

Browser CORS preflight requests (OPTIONS) and whitelisted paths bypass
the limiter.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import status
from fastapi.responses import JSONResponse
from redis import exceptions as redis_exceptions
from redis.asyncio import Redis

from app.config.settings import settings
from app.middleware.error_handler import ErrorResponse
from app.middleware.rate_limiter import TokenBucket

logger = logging.getLogger(__name__)

# Atomic sliding-window check: prune expired entries, then admit or reject.
# Returns {allowed(0/1), remaining}.
_SLIDING_WINDOW_LUA = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local member = ARGV[4]
redis.call('ZREMRANGEBYSCORE', key, '-inf', now - window)
local count = redis.call('ZCARD', key)
if count < limit then
    redis.call('ZADD', key, now, member)
    redis.call('EXPIRE', key, window)
    return {1, limit - count - 1}
end
return {0, 0}
"""


class _RedisClientHolder:
    """Lazy module-level async Redis client shared by all middleware instances."""

    _client: Redis[Any] | None = None
    _last_warned: float = 0.0

    @classmethod
    def get(cls) -> Redis[Any]:
        if cls._client is None:
            cls._client = Redis.from_url(
                settings.RATE_LIMIT_REDIS_URL or settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
            )
        return cls._client

    @classmethod
    async def close(cls) -> None:
        if cls._client is not None:
            # types-redis stubs lag redis>=5 async API; aclose exists at runtime
            await cls._client.aclose()  # type: ignore[attr-defined]
            cls._client = None

    @classmethod
    def warn_degraded(cls) -> None:
        """Log Redis degradation at most once per 5 minutes to avoid log spam."""
        now = time.monotonic()
        if now - cls._last_warned > 300:
            cls._last_warned = now
            logger.warning(
                "Rate limit Redis unreachable; degrading to per-instance "
                "in-memory buckets (distributed limit not enforced)"
            )


async def close_rate_limit_redis() -> None:
    """Close the shared rate-limit Redis connection (call on app shutdown)."""
    await _RedisClientHolder.close()


class DistributedRateLimiterMiddleware:
    """
    Redis-backed sliding-window rate limiter (per client IP).

    Falls back to an in-memory token bucket per process when Redis is
    unavailable, so a Redis outage degrades protection but never takes
    the API down.
    """

    def __init__(
        self,
        app: Any,
        requests_per_minute: int = 60,
        window_seconds: int = 60,
        whitelist_paths: list[str] | None = None,
    ) -> None:
        self.app = app
        self.requests_per_minute = requests_per_minute
        self.window_seconds = window_seconds
        self.whitelist_paths = set(whitelist_paths or [])
        self._limit = requests_per_minute
        self._window = float(window_seconds)
        # Per-process fallback buckets used only while Redis is unreachable.
        self._fallback_buckets: dict[str, TokenBucket] = {}

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Let CORS preflights through; they carry no business payload.
        if scope.get("method") == "OPTIONS":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path in self.whitelist_paths:
            await self.app(scope, receive, send)
            return

        client_ip = self._client_ip(scope)
        allowed, remaining = await self._check(client_ip)

        if allowed:
            await self.app(scope, receive, send)
            return

        response = self._too_many_requests(client_ip, remaining)
        await response(scope, receive, send)

    async def _check(self, client_ip: str) -> tuple[bool, int]:
        """Check and consume one request slot. Returns (allowed, remaining)."""
        try:
            client = _RedisClientHolder.get()
            now = time.time()
            member = f"{now}-{client_ip}"
            result = await client.eval(  # type: ignore[no-untyped-call]
                _SLIDING_WINDOW_LUA,
                1,
                f"ratelimit:{client_ip}",
                int(now * 1000),
                int(self._window * 1000),
                self._limit,
                member,
            )
            allowed = int(result[0]) == 1
            remaining = int(result[1])
            return allowed, remaining
        except redis_exceptions.RedisError:
            _RedisClientHolder.warn_degraded()
            return self._fallback_check(client_ip)
        except (OSError, ValueError):
            _RedisClientHolder.warn_degraded()
            return self._fallback_check(client_ip)

    def _fallback_check(self, client_ip: str) -> tuple[bool, int]:
        """Per-process token bucket used when Redis is unavailable."""
        bucket = self._fallback_buckets.get(client_ip)
        if bucket is None:
            bucket = TokenBucket(
                capacity=self._limit,
                refill_rate=self._limit / self._window if self._window else 1.0,
            )
            self._fallback_buckets[client_ip] = bucket
        allowed = bucket.consume()
        return allowed, max(0, int(bucket.tokens))

    def _client_ip(self, scope: dict[str, Any]) -> str:
        """Resolve client IP, honouring X-Forwarded-For from trusted proxies."""
        headers: list[tuple[bytes, bytes]] = scope.get("headers") or []
        for name, value in headers:
            if name == b"x-forwarded-for":
                forwarded = value.decode("latin-1")
                return forwarded.split(",")[0].strip()
        client = scope.get("client")
        if client:
            return str(client[0])
        return "unknown"

    def _too_many_requests(self, client_ip: str, remaining: int) -> JSONResponse:
        error_response = ErrorResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            message="Rate limit exceeded",
            detail=f"Maximum {self.requests_per_minute} requests per "
            f"{self.window_seconds}s allowed",
        )
        logger.warning(
            "Rate limit exceeded for IP: %s",
            client_ip,
            extra={"ip": client_ip, "remaining": remaining},
        )
        response = JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content=error_response.model_dump(),
        )
        response.headers["Retry-After"] = str(self.window_seconds)
        response.headers["X-RateLimit-Limit"] = str(self._limit)
        response.headers["X-RateLimit-Remaining"] = "0"
        return response
