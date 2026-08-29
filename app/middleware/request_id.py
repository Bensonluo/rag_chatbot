"""
Request ID middleware.

Generates unique request IDs for tracing and debugging.
"""
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Optional
import uuid
import logging


logger = logging.getLogger(__name__)


def get_request_id(request: Request) -> Optional[str]:
    """
    Get request ID from request state.

    Args:
        request: FastAPI request

    Returns:
        str | None: Request ID if set
    """
    return getattr(request.state, "request_id", None)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add unique request IDs.
    """

    def __init__(self, app, header_name: str = "X-Request-ID"):
        super().__init__(app)
        self.header_name = header_name

    async def dispatch(self, request: Request, call_next):
        """
        Process request and add request ID.

        Args:
            request: Incoming request
            call_next: Next middleware/route handler

        Returns:
            Response: HTTP response with request ID
        """
        # Get or generate request ID
        request_id = self._get_or_generate_request_id(request)

        # Add to request state for logging
        request.state.request_id = request_id

        # Process request
        response: Response = await call_next(request)

        # Add to response headers
        response.headers[self.header_name] = request_id

        return response

    def _get_or_generate_request_id(self, request: Request) -> str:
        """
        Get existing request ID from headers or generate new one.

        Args:
            request: Incoming request

        Returns:
            str: Request ID
        """
        # Check for existing request ID in headers
        existing_id = request.headers.get(self.header_name)
        if existing_id:
            return existing_id

        # Generate new request ID
        return self._generate_request_id()

    def _generate_request_id(self) -> str:
        """
        Generate unique request ID.

        Returns:
            str: UUID v4
        """
        return str(uuid.uuid4())
