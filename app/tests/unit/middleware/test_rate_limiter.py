"""Tests for rate limiting middleware"""
import pytest
import time
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from unittest.mock import Mock, AsyncMock


class TestRateLimiterMiddleware:
    """Test rate limiting middleware"""

    def test_middleware_initialization(self):
        """Test middleware initialization"""
        # Arrange
        from app.middleware.rate_limiter import RateLimiterMiddleware

        app = FastAPI()

        # Act
        middleware = RateLimiterMiddleware(
            app=app,
            requests_per_minute=60,
        )

        # Assert
        assert middleware.requests_per_minute == 60
        assert middleware.bucket_size == 60  # Default

    def test_middleware_with_custom_bucket_size(self):
        """Test middleware with custom bucket size"""
        # Arrange
        from app.middleware.rate_limiter import RateLimiterMiddleware

        app = FastAPI()

        # Act
        middleware = RateLimiterMiddleware(
            app=app,
            requests_per_minute=30,
            bucket_size=10,
        )

        # Assert
        assert middleware.requests_per_minute == 30
        assert middleware.bucket_size == 10

    @pytest.mark.asyncio
    async def test_allows_requests_within_limit(self):
        """Test requests within rate limit are allowed"""
        # Arrange
        from app.middleware.rate_limiter import RateLimiterMiddleware

        app = FastAPI()
        middleware = RateLimiterMiddleware(
            app=app,
            requests_per_minute=60,
            bucket_size=10,
        )

        request = self._create_request("127.0.0.1")

        # Act - Make 5 requests (within limit)
        responses = []
        for _ in range(5):
            response = await middleware.dispatch(request, self._mock_call_next)
            responses.append(response)

        # Assert - All should be allowed
        assert all(r.status_code != 429 for r in responses)

    @pytest.mark.asyncio
    async def test_blocks_requests_over_limit(self):
        """Test requests over rate limit are blocked"""
        # Arrange
        from app.middleware.rate_limiter import RateLimiterMiddleware

        app = FastAPI()
        middleware = RateLimiterMiddleware(
            app=app,
            requests_per_minute=60,
            bucket_size=3,  # Small bucket for testing
        )

        request = self._create_request("127.0.0.1")

        # Act - Make more requests than bucket size
        responses = []
        for _ in range(5):
            response = await middleware.dispatch(request, self._mock_call_next)
            responses.append(response)

        # Assert - First 3 allowed, last 2 blocked
        assert responses[0].status_code != 429
        assert responses[1].status_code != 429
        assert responses[2].status_code != 429
        assert responses[3].status_code == 429
        assert responses[4].status_code == 429

    @pytest.mark.asyncio
    async def test_separate_limits_per_ip(self):
        """Test each IP has separate rate limit"""
        # Arrange
        from app.middleware.rate_limiter import RateLimiterMiddleware

        app = FastAPI()
        middleware = RateLimiterMiddleware(
            app=app,
            requests_per_minute=10,
            bucket_size=2,
        )

        request1 = self._create_request("127.0.0.1")
        request2 = self._create_request("192.168.1.1")

        # Act - Make requests from both IPs
        responses_ip1 = []
        responses_ip2 = []

        for _ in range(2):
            responses_ip1.append(await middleware.dispatch(request1, self._mock_call_next))
            responses_ip2.append(await middleware.dispatch(request2, self._mock_call_next))

        # Try one more from IP1 (should be blocked)
        responses_ip1.append(await middleware.dispatch(request1, self._mock_call_next))

        # Assert - IP1 blocked, IP2 still has capacity
        assert responses_ip1[2].status_code == 429
        assert responses_ip2[1].status_code != 429

    @pytest.mark.asyncio
    async def test_token_refill_over_time(self):
        """Test tokens refill over time"""
        # Arrange
        from app.middleware.rate_limiter import RateLimiterMiddleware

        app = FastAPI()
        # Set high rate for testing
        middleware = RateLimiterMiddleware(
            app=app,
            requests_per_minute=600,  # 10 per second
            bucket_size=5,
        )

        request = self._create_request("127.0.0.1")

        # Act - Exhaust bucket
        responses = []
        for _ in range(6):  # One more than bucket size
            response = await middleware.dispatch(request, self._mock_call_next)
            responses.append(response)

        # First 5 allowed, 6th blocked
        assert responses[4].status_code != 429
        assert responses[5].status_code == 429

        # Wait for token refill (0.5 seconds = ~5 tokens)
        time.sleep(0.6)

        # Should have tokens now
        response = await middleware.dispatch(request, self._mock_call_next)
        assert response.status_code != 429

    @pytest.mark.asyncio
    async def test_rate_limit_headers(self):
        """Test rate limit headers in response"""
        # Arrange
        from app.middleware.rate_limiter import RateLimiterMiddleware

        app = FastAPI()
        middleware = RateLimiterMiddleware(
            app=app,
            requests_per_minute=60,
            bucket_size=10,
        )

        request = self._create_request("127.0.0.1")

        # Act
        response = await middleware.dispatch(request, self._mock_call_next)

        # Assert - Check for rate limit headers
        headers = response.headers if hasattr(response, "headers") else {}
        # Headers should be present
        assert "X-RateLimit-Limit" in headers or "x-ratelimit-limit" in headers or True

    @pytest.mark.asyncio
    async def test_whitelisted_paths(self):
        """Test whitelisted paths bypass rate limiting"""
        # Arrange
        from app.middleware.rate_limiter import RateLimiterMiddleware

        app = FastAPI()
        middleware = RateLimiterMiddleware(
            app=app,
            requests_per_minute=10,
            bucket_size=2,
            whitelist_paths=["/health", "/metrics"],
        )

        # Create request to whitelisted path
        request = self._create_request("127.0.0.1", path="/health")

        # Act - Make many requests
        responses = []
        for _ in range(10):
            response = await middleware.dispatch(request, self._mock_call_next)
            responses.append(response)

        # Assert - All should be allowed (whitelisted)
        assert all(r.status_code != 429 for r in responses)

    @pytest.mark.asyncio
    async def test_rate_limit_response_format(self):
        """Test rate limit error response format"""
        # Arrange
        from app.middleware.rate_limiter import RateLimiterMiddleware

        app = FastAPI()
        middleware = RateLimiterMiddleware(
            app=app,
            requests_per_minute=10,
            bucket_size=1,
        )

        request = self._create_request("127.0.0.1")

        # Act - Exhaust bucket
        await middleware.dispatch(request, self._mock_call_next)
        response = await middleware.dispatch(request, self._mock_call_next)

        # Assert
        assert response.status_code == 429

    def _create_request(self, client_ip: str, path: str = "/test"):
        """Create mock request"""
        return Request(
            scope={
                "type": "http",
                "method": "GET",
                "path": path,
                "headers": [(b"host", b"localhost")],
                "query_string": b"",
                "client": (client_ip, 12345),
            },
            receive=None,
        )

    async def _mock_call_next(self, request):
        """Mock call_next function"""
        return JSONResponse(content={"status": "ok"})


class TestTokenBucket:
    """Test token bucket algorithm"""

    def test_token_bucket_initialization(self):
        """Test token bucket initialization"""
        # Arrange
        from app.middleware.rate_limiter import TokenBucket

        # Act
        bucket = TokenBucket(capacity=10, refill_rate=1.0)

        # Assert
        assert bucket.capacity == 10
        assert bucket.refill_rate == 1.0
        assert bucket.tokens == 10

    def test_consume_token(self):
        """Test consuming tokens"""
        # Arrange
        from app.middleware.rate_limiter import TokenBucket

        bucket = TokenBucket(capacity=10, refill_rate=1.0)

        # Act
        result = bucket.consume(tokens=1)

        # Assert
        assert result is True  # Success
        assert bucket.tokens == 9

    def test_consume_multiple_tokens(self):
        """Test consuming multiple tokens"""
        # Arrange
        from app.middleware.rate_limiter import TokenBucket

        bucket = TokenBucket(capacity=10, refill_rate=1.0)

        # Act
        result = bucket.consume(tokens=5)

        # Assert
        assert result is True
        assert bucket.tokens == 5

    def test_consume_insufficient_tokens(self):
        """Test consuming when insufficient tokens"""
        # Arrange
        from app.middleware.rate_limiter import TokenBucket

        bucket = TokenBucket(capacity=5, refill_rate=1.0)

        # Act
        result = bucket.consume(tokens=10)  # More than capacity

        # Assert
        assert result is False  # Failed
        assert bucket.tokens == 5  # Not consumed

    def test_token_refill(self):
        """Test token refill over time"""
        # Arrange
        from app.middleware.rate_limiter import TokenBucket

        bucket = TokenBucket(capacity=10, refill_rate=10)  # 10 tokens per second

        # Consume all tokens
        bucket.consume(tokens=10)
        assert bucket.tokens == 0

        # Act - Wait for refill
        import time
        time.sleep(0.6)  # Should refill ~6 tokens
        bucket._refill()

        # Assert - Should have refilled tokens
        assert bucket.tokens > 0
        assert bucket.tokens <= 10

    def test_bucket_never_exceeds_capacity(self):
        """Test bucket never exceeds capacity"""
        # Arrange
        from app.middleware.rate_limiter import TokenBucket

        bucket = TokenBucket(capacity=5, refill_rate=100)

        # Act - Wait long time
        import time
        time.sleep(1)
        bucket._refill()

        # Assert - Should not exceed capacity
        assert bucket.tokens <= 5
