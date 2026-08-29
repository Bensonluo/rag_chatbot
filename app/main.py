"""
FastAPI application entry point.

This is the main application file that sets up the FastAPI app,
configures middleware, and includes all routers.
"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config.settings import settings
from app.config.logging import logger


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.

    Handles startup and shutdown events.
    """
    # Startup
    logger.info(
        "Starting RAG Chatbot",
        environment=settings.ENVIRONMENT,
        debug=settings.DEBUG,
    )

    app.state.chat_ready = False

    # Initialize chat service
    try:
        from app.api.database import async_session_maker
        from app.api.v1.chat import initialize_chat_service
        async with async_session_maker() as db:
            await initialize_chat_service(db)
        app.state.chat_ready = True
        logger.info("Chat service initialized successfully")
    except Exception as e:
        logger.warning("Failed to initialize chat service: %s", e)

    yield
    # Shutdown
    logger.info("Shutting down RAG Chatbot")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        FastAPI: Configured FastAPI application instance
    """
    app = FastAPI(
        title="RAG Chatbot",
        description="GraphRAG chatbot technical demo with dialogue state and hybrid retrieval",
        version="0.1.0",
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        lifespan=lifespan,
    )
    app.state.chat_ready = False

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request ID middleware
    from app.middleware.request_id import RequestIDMiddleware
    app.add_middleware(RequestIDMiddleware)

    # Prometheus metrics middleware
    if settings.ENABLE_METRICS:
        from app.middleware.metrics import PrometheusMiddleware
        app.add_middleware(PrometheusMiddleware)

    # OpenTelemetry tracing
    if settings.ENABLE_TRACING:
        from app.middleware.tracing import setup_tracing
        setup_tracing(app=app, endpoint=settings.OTEL_ENDPOINT)

    # Include API v1 router
    from app.api.v1.router import api_router
    app.include_router(api_router, prefix=settings.API_PREFIX)

    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Basic health check endpoint"""
        return {
            "status": "healthy",
            "service": "rag_chatbot",
            "environment": settings.ENVIRONMENT,
        }

    @app.get("/ready")
    async def readiness_check():
        """Report whether the demo chat graph finished initialization."""
        if not app.state.chat_ready:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready",
                    "service": "rag_chatbot",
                    "chat": "not_initialized",
                },
            )
        return {
            "status": "ready",
            "service": "rag_chatbot",
            "chat": "initialized",
        }

    # Prometheus metrics endpoint
    if settings.ENABLE_METRICS:
        from app.middleware.metrics import metrics_endpoint
        app.add_route("/metrics", metrics_endpoint)

    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with API information"""
        return {
            "name": "RAG Chatbot API",
            "version": "0.1.0",
            "status": "running",
            "docs": "/docs" if settings.DEBUG else "disabled in production",
        }

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        """Global exception handler for unhandled exceptions"""
        logger.error(
            "Unhandled exception",
            path=request.url.path,
            method=request.method,
            error=str(exc),
            exc_info=exc,
        )
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "error": "SERVER_ERROR" if not settings.DEBUG else str(exc),
            },
        )

    return app


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
