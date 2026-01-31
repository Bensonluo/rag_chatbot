"""
Rate limiting middleware using token bucket algorithm.

Provides IP-based rate limiting for API endpoints.
"""
from fastapi import Request, status
from fastapi.responses import JSONResponse
from typing import Dict, Optional, list
import time
import logging

from app.middleware.error_handler import ErrorResponse


logger = logging.getLogger(__name__)


class TokenBucket:
    """
    Token bucket implementation for rate limiting.

    Tokens are added at a constant rate up to a maximum capacity.
    Each request consumes tokens.
    """

    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket.

        Args:
            capacity: Maximum number of tokens (bucket size)
            refill_rate: Tokens per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_refill = time.time()

    def consume(self, tokens: int = 1) -> bool:
        """
        Consume tokens from bucket.

        Args:
            tokens: Number of tokens to consume

        Returns:
            bool: True if tokens were consumed, False if insufficient
        """
        self._refill()

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill

        # Calculate tokens to add
        tokens_to_add = elapsed * self.refill_rate

        # Refill but don't exceed capacity
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now


class RateLimiterMiddleware:
    """
    Rate limiting middleware using token bucket algorithm.

    Limits requests per IP address.
    """

    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        bucket_size: Optional[int] = None,
        whitelist_paths: Optional[list[str]] = None,
    ):
        """
        Initialize rate limiter middleware.

        Args:
            app: FastAPI application
            requests_per_minute: Requests allowed per minute
            bucket_size: Maximum bucket size (default: requests_per_minute)
            whitelist_paths: Paths to exclude from rate limiting
        """
        self.app = app
        self.requests_per_minute = requests_per_minute
        self.bucket_size = bucket_size or requests_per_minute
        self.whitelist_paths = set(whitelist_paths or [])

        # Token bucket per IP
        self.buckets: Dict[str, TokenBucket] = {}

        # Calculate refill rate (tokens per second)
        self.refill_rate = requests_per_minute / 60.0

    async def dispatch(self, request: Request, call_next):
        """
        Process request and enforce rate limit.

        Args:
            request: Incoming request
            call_next: Next middleware/route handler

        Returns:
            Response: HTTP response or rate limit error
        """
        # Skip whitelisted paths
        if request.url.path in self.whitelist_paths:
            return await call_next(request)

        # Get client IP
        client_ip = self._get_client_ip(request)

        # Get or create bucket for this IP
        bucket = self._get_bucket(client_ip)

        # Try to consume token
        if bucket.consume():
            # Request allowed
            response = await call_next(request)

            # Add rate limit headers
            self._add_rate_limit_headers(response, bucket)

            return response
        else:
            # Rate limit exceeded
            return self._rate_limit_response(bucket, client_ip)

    def _get_client_ip(self, request: Request) -> str:
        """
        Get client IP from request.

        Args:
            request: Incoming request

        Returns:
            str: Client IP address
        """
        # Check for forwarded IP (behind proxy)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()

        # Use direct client IP
        if request.client:
            return request.client.host

        return "unknown"

    def _get_bucket(self, client_ip: str) -> TokenBucket:
        """
        Get or create token bucket for IP.

        Args:
            client_ip: Client IP address

        Returns:
            TokenBucket: Token bucket for this IP
        """
        if client_ip not in self.buckets:
            self.buckets[client_ip] = TokenBucket(
                capacity=self.bucket_size,
                refill_rate=self.refill_rate,
            )

        return self.buckets[client_ip]

    def _add_rate_limit_headers(
        self,
        response,
        bucket: TokenBucket,
    ) -> None:
        """
        Add rate limit headers to response.

        Args:
            response: HTTP response
            bucket: Token bucket
        """
        if hasattr(response, "headers"):
            response.headers["X-RateLimit-Limit"] = str(int(bucket.capacity))
            response.headers["X-RateLimit-Remaining"] = str(int(bucket.tokens))
            response.headers["X-RateLimit-Reset"] = str(int(bucket.last_refill + 1))

    def _rate_limit_response(
        self,
        bucket: TokenBucket,
        client_ip: str,
    ) -> JSONResponse:
        """
        Create rate limit error response.

        Args:
            bucket: Token bucket
            client_ip: Client IP

        Returns:
            JSONResponse: Rate limit error
        """
        error_response = ErrorResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            message="Rate limit exceeded",
            detail=f"Maximum {self.requests_per_minute} requests per minute allowed",
        )

        logger.warning(
            f"Rate limit exceeded for IP: {client_ip}",
            extra={"ip": client_ip, "tokens_remaining": bucket.tokens}
        )

        response = JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content=error_response.model_dump(),
        )

        # Add retry after header
        retry_after = int((1.0 / self.refill_rate) * (1 - bucket.tokens / bucket.capacity))
        response.headers["Retry-After"] = str(max(1, retry_after))

        return response
