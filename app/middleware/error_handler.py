"""
Error handler middleware.

Catches and formats exceptions into proper HTTP responses.
"""
from fastapi import Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, list
import traceback
import logging

from app.core.exceptions import BaseServiceError, ValidationError


logger = logging.getLogger(__name__)


class ErrorResponse(BaseModel):
    """Standard error response format."""
    status_code: int = Field(..., description="HTTP status code")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    errors: Optional[list[dict]] = Field(None, description="Validation errors")
    path: Optional[str] = Field(None, description="Request path")


class ErrorHandlerMiddleware:
    """
    Middleware to handle errors globally.

    Catches exceptions and converts them to appropriate HTTP responses
    with consistent error format.
    """

    def __init__(self, app, debug: bool = False):
        """
        Initialize error handler middleware.

        Args:
            app: FastAPI application
            debug: Whether to include debug information
        """
        self.app = app
        self.debug = debug

    async def dispatch(self, request: Request, call_next):
        """
        Process request and handle errors.

        Args:
            request: Incoming request
            call_next: Next middleware/route handler

        Returns:
            Response: HTTP response
        """
        try:
            response = await call_next(request)
            return response

        except ValidationError as e:
            # Pydantic validation error
            return self._handle_validation_error(e, request)

        except BaseServiceError as e:
            # Custom service error
            return self._handle_service_error(e, request)

        except Exception as e:
            # Unexpected error
            return self._handle_unexpected_error(e, request)

    def _handle_validation_error(
        self,
        error: ValidationError,
        request: Request,
    ) -> JSONResponse:
        """
        Handle validation errors.

        Args:
            error: Validation error
            request: Request that caused error

        Returns:
            JSONResponse: Formatted error response
        """
        errors = []
        if hasattr(error, "errors"):
            for err in error.errors():
                errors.append({
                    "field": ".".join(str(loc) for loc in err["loc"]),
                    "message": err["msg"],
                    "type": err["type"],
                })

        response = ErrorResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            message="Validation error",
            errors=errors,
            path=request.url.path,
        )

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=response.model_dump(),
        )

    def _handle_service_error(
        self,
        error: BaseServiceError,
        request: Request,
    ) -> JSONResponse:
        """
        Handle service errors.

        Args:
            error: Service error
            request: Request that caused error

        Returns:
            JSONResponse: Formatted error response
        """
        response = ErrorResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=str(error),
            detail=error.details if hasattr(error, "details") else None,
            path=request.url.path,
        )

        logger.error(
            f"Service error: {error}",
            extra={"path": request.url.path, "details": error.details if hasattr(error, "details") else None}
        )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=response.model_dump(),
        )

    def _handle_unexpected_error(
        self,
        error: Exception,
        request: Request,
    ) -> JSONResponse:
        """
        Handle unexpected errors.

        Args:
            error: Unexpected error
            request: Request that caused error

        Returns:
            JSONResponse: Formatted error response
        """
        # Log error with traceback
        logger.error(
            f"Unexpected error: {error}",
            exc_info=True,
            extra={"path": request.url.path}
        )

        # Build error response
        error_detail = str(error)

        if self.debug:
            # Include traceback in debug mode
            error_detail += "\n\n" + "".join(traceback.format_tb(error.__traceback__))

        response = ErrorResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message="Internal server error",
            detail=error_detail if self.debug else None,
            path=request.url.path,
        )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=response.model_dump(),
        )
