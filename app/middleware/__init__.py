"""
Middleware package.

Exports all middleware components including error handling,
rate limiting, request ID tracking, and metrics collection.
"""
from app.middleware.error_handler import ErrorHandlerMiddleware, ErrorResponse
from app.middleware.rate_limiter import RateLimiterMiddleware, TokenBucket
from app.middleware.request_id import RequestIDMiddleware, get_request_id
from app.middleware.metrics import (
    PrometheusMiddleware,
    metrics_endpoint,
    health_check_endpoint,
    ready_check_endpoint,
)

__all__ = [
    # Error handling
    "ErrorHandlerMiddleware",
    "ErrorResponse",
    # Rate limiting
    "RateLimiterMiddleware",
    "TokenBucket",
    # Request ID
    "RequestIDMiddleware",
    "get_request_id",
    # Metrics
    "PrometheusMiddleware",
    "metrics_endpoint",
    "health_check_endpoint",
    "ready_check_endpoint",
]
