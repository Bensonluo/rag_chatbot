"""Tests for error handler middleware"""
import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from unittest.mock import Mock, patch


class TestErrorHandlerMiddleware:
    """Test error handler middleware"""

    def test_middleware_initialization(self):
        """Test middleware initialization"""
        # Arrange
        from app.middleware.error_handler import ErrorHandlerMiddleware

        app = FastAPI()

        # Act
        middleware = ErrorHandlerMiddleware(app)

        # Assert
        assert middleware.app == app
        assert middleware.debug is False

    def test_middleware_with_debug_enabled(self):
        """Test middleware with debug mode"""
        # Arrange
        from app.middleware.error_handler import ErrorHandlerMiddleware

        app = FastAPI()

        # Act
        middleware = ErrorHandlerMiddleware(app, debug=True)

        # Assert
        assert middleware.debug is True

    @pytest.mark.asyncio
    async def test_handles_http_exception(self):
        """Test handling HTTP exceptions"""
        # Arrange
        from app.middleware.error_handler import ErrorHandlerMiddleware
        from fastapi import HTTPException

        app = FastAPI()
        middleware = ErrorHandlerMiddleware(app)

        @app.get("/test")
        async def test_route():
            raise HTTPException(status_code=404, detail="Not found")

        # Create request
        request = Request(
            scope={
                "type": "http",
                "method": "GET",
                "path": "/test",
                "headers": [],
                "query_string": b"",
            },
            receive=None,
        )

        # Act
        response = await middleware.dispatch(request, self.call_next)

        # Assert
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_handles_validation_error(self):
        """Test handling validation errors"""
        # Arrange
        from app.middleware.error_handler import ErrorHandlerMiddleware
        from pydantic import ValidationError

        app = FastAPI()
        middleware = ErrorHandlerMiddleware(app)

        @app.get("/test")
        async def test_route():
            raise ValidationError(
                model=Mock,
                errors=[{"loc": ("field",), "msg": "error"}]
            )

        request = Request(
            scope={
                "type": "http",
                "method": "GET",
                "path": "/test",
                "headers": [],
                "query_string": b"",
            },
            receive=None,
        )

        # Act
        response = await middleware.dispatch(request, self.call_next)

        # Assert
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_handles_base_service_error(self):
        """Test handling BaseServiceError"""
        # Arrange
        from app.middleware.error_handler import ErrorHandlerMiddleware
        from app.core.exceptions import BaseServiceError

        app = FastAPI()
        middleware = ErrorHandlerMiddleware(app)

        @app.get("/test")
        async def test_route():
            raise BaseServiceError("Service error occurred")

        request = Request(
            scope={
                "type": "http",
                "method": "GET",
                "path": "/test",
                "headers": [],
                "query_string": b"",
            },
            receive=None,
        )

        # Act
        response = await middleware.dispatch(request, self.call_next)

        # Assert
        assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_handles_generic_exception(self):
        """Test handling generic exceptions"""
        # Arrange
        from app.middleware.error_handler import ErrorHandlerMiddleware

        app = FastAPI()
        middleware = ErrorHandlerMiddleware(app)

        @app.get("/test")
        async def test_route():
            raise ValueError("Unexpected error")

        request = Request(
            scope={
                "type": "http",
                "method": "GET",
                "path": "/test",
                "headers": [],
                "query_string": b"",
            },
            receive=None,
        )

        # Act
        response = await middleware.dispatch(request, self.call_next)

        # Assert
        assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_debug_mode_includes_traceback(self):
        """Test debug mode includes traceback"""
        # Arrange
        from app.middleware.error_handler import ErrorHandlerMiddleware

        app = FastAPI()
        middleware = ErrorHandlerMiddleware(app, debug=True)

        @app.get("/test")
        async def test_route():
            raise ValueError("Test error")

        request = Request(
            scope={
                "type": "http",
                "method": "GET",
                "path": "/test",
                "headers": [],
                "query_string": b"",
            },
            receive=None,
        )

        # Act
        response = await middleware.dispatch(request, self.call_next)
        body = await self._get_response_body(response)

        # Assert - Debug mode should include error details
        assert "Test error" in body or "traceback" in body.lower()

    @pytest.mark.asyncio
    async def test_successful_request_passthrough(self):
        """Test successful requests pass through"""
        # Arrange
        from app.middleware.error_handler import ErrorHandlerMiddleware

        app = FastAPI()
        middleware = ErrorHandlerMiddleware(app)

        @app.get("/test")
        async def test_route():
            return {"status": "ok"}

        request = Request(
            scope={
                "type": "http",
                "method": "GET",
                "path": "/test",
                "headers": [],
                "query_string": b"",
            },
            receive=None,
        )

        # Act
        response = await middleware.dispatch(request, self.call_next)

        # Assert - Should not be modified by middleware
        assert response is not None

    async def call_next(self, request):
        """Mock call_next for testing"""
        # Find route handler
        for route in request.app.routes:
            if hasattr(route, "path") and route.path == request.url.path:
                return await route.endpoint()
        return JSONResponse(content={"error": "Not found"}, status_code=404)

    async def _get_response_body(self, response):
        """Extract response body for testing"""
        if hasattr(response, "body"):
            body = await response.body()
            return body.decode()
        return ""


class TestErrorResponse:
    """Test error response models"""

    def test_error_response_creation(self):
        """Test creating error response"""
        # Arrange
        from app.middleware.error_handler import ErrorResponse

        # Act
        response = ErrorResponse(
            status_code=500,
            message="Internal server error",
            detail="Database connection failed",
            path="/api/v1/chat"
        )

        # Assert
        assert response.status_code == 500
        assert response.message == "Internal server error"
        assert response.detail == "Database connection failed"
        assert response.path == "/api/v1/chat"

    def test_validation_error_response(self):
        """Test validation error response structure"""
        # Arrange
        from app.middleware.error_handler import ErrorResponse

        # Act
        response = ErrorResponse(
            status_code=422,
            message="Validation error",
            errors=[
                {"field": "email", "message": "Invalid email format"},
                {"field": "password", "message": "Password too short"}
            ],
            path="/api/v1/register"
        )

        # Assert
        assert response.status_code == 422
        assert len(response.errors) == 2
        assert response.errors[0]["field"] == "email"

    def test_error_response_serialization(self):
        """Test error response JSON serialization"""
        # Arrange
        from app.middleware.error_handler import ErrorResponse
        import json

        response = ErrorResponse(
            status_code=404,
            message="Not found",
            detail="Resource not found",
            path="/api/v1/users/123"
        )

        # Act
        data = response.model_dump()

        # Assert
        assert data["status_code"] == 404
        assert data["message"] == "Not found"
        assert "detail" in data
