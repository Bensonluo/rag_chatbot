"""
Simple rate limiter dependency using in-memory token bucket.

Provides rate limiting for API endpoints without external dependencies.
"""
from fastapi import Request, HTTPException, status
from typing import Dict
import time


class InMemoryRateLimiter:
    """
    In-memory rate limiter using token bucket algorithm.

    Simple implementation for demo purposes.
    """

    def __init__(self, requests_per_minute: int = 10):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute per IP
        """
        self.requests_per_minute = requests_per_minute
        self.bucket_size = requests_per_minute
        self.buckets: Dict[str, Dict] = {}

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP from request."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def check_rate_limit(self, request: Request) -> None:
        """
        Check if request is within rate limit.

        Args:
            request: FastAPI request

        Raises:
            HTTPException: If rate limit exceeded
        """
        client_ip = self._get_client_ip(request)
        now = time.time()

        # Get or create bucket
        if client_ip not in self.buckets:
            self.buckets[client_ip] = {
                "tokens": self.bucket_size - 1,
                "last_update": now,
            }
            return

        bucket = self.buckets[client_ip]

        # Refill tokens based on elapsed time
        elapsed = now - bucket["last_update"]
        tokens_to_add = elapsed * (self.requests_per_minute / 60.0)
        bucket["tokens"] = min(self.bucket_size, bucket["tokens"] + tokens_to_add)
        bucket["last_update"] = now

        # Check if we have tokens
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
        else:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Maximum {self.requests_per_minute} requests per minute.",
            )


# Global rate limiter instance
_rate_limiter = InMemoryRateLimiter(requests_per_minute=10)


def check_rate_limit(request: Request):
    """
    FastAPI dependency for rate limiting.

    Args:
        request: FastAPI request

    Raises:
        HTTPException: If rate limit exceeded
    """
    _rate_limiter.check_rate_limit(request)
