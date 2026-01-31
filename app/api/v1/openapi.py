"""
OpenAPI/Swagger documentation configuration.

Enhances FastAPI auto-generated documentation with detailed descriptions.
"""
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi


def custom_openapi(app: FastAPI):
    """
    Custom OpenAPI schema with enhanced documentation.

    Args:
        app: FastAPI application

    Returns:
        dict: OpenAPI schema
    """
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="RAG Chatbot API",
        version="1.0.0",
        description=openapi_description,
        routes=app.routes,
    )

    # Add tags
    openapi_schema["tags"] = [
        {
            "name": "auth",
            "description": "Authentication and user management",
        },
        {
            "name": "sessions",
            "description": "Chat session management",
        },
        {
            "name": "chat",
            "description": "Chat interactions and streaming",
        },
        {
            "name": "health",
            "description": "Health checks and monitoring",
        },
    ]

    # Add contact info
    openapi_schema["info"]["contact"] = {
        "name": "API Support",
        "email": "support@example.com",
    }

    # Add license
    openapi_schema["info"]["license"] = {
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    }

    # Add servers
    openapi_schema["servers"] = [
        {
            "url": "http://localhost:8000",
            "description": "Development server",
        },
        {
            "url": "https://api.example.com",
            "description": "Production server",
        },
    ]

    # Add components/schemas
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        },
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


# OpenAPI description
openapi_description = """
# RAG Chatbot API

Production-grade Retrieval-Augmented Generation (RAG) chatbot API with:

- **Intelligent Intent Detection**: Hybrid rule-based and LLM-powered intent classification
- **Memory Management**: Sliding window, summarization, and hybrid memory strategies
- **Vector Search**: Hybrid semantic + keyword search with reranking
- **Streaming Responses**: Real-time server-sent events
- **Authentication**: JWT-based user authentication
- **Rate Limiting**: Token bucket algorithm
- **Monitoring**: Prometheus metrics and health checks

## Architecture

The API integrates multiple services:

```
Client Request
    ↓
[API Gateway]
    ↓
[Intent Detection] → Route to appropriate handler
    ↓
[Memory Service] → Retrieve conversation context
    ↓
[Retrieval Service] → Fetch relevant documents (if needed)
    ↓
[LLM Service] → Generate response
    ↓
[Memory Service] → Store conversation
    ↓
Client Response
```

## Authentication

Most endpoints require JWT authentication:

```bash
# Register
curl -X POST http://localhost:8000/api/v1/register \\
  -H "Content-Type: application/json" \\
  -d '{"email": "user@example.com", "password": "securepass"}'

# Login
curl -X POST http://localhost:8000/api/v1/login \\
  -H "Content-Type: application/json" \\
  -d '{"email": "user@example.com", "password": "securepass"}'

# Use token
curl http://localhost:8000/api/v1/chat \\
  -H "Authorization: Bearer <token>" \\
  -H "Content-Type: application/json" \\
  -d '{"message": "Hello!", "session_id": 1}'
```

## Rate Limiting

- **Default**: 60 requests per minute
- **Headers**: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `Retry-After`
- **Whitelisted**: `/health`, `/metrics`, `/docs`

## Error Responses

All errors follow this format:

```json
{
  "status_code": 400,
  "message": "Error message",
  "detail": "Detailed error information",
  "path": "/api/v1/endpoint"
}
```

## Chat Features

### 1. Simple Chat

```json
POST /api/v1/chat
{
  "message": "What is Python?",
  "session_id": 1
}
```

### 2. Chat with Retrieval

Automatically retrieves relevant documents for questions:

```json
POST /api/v1/chat
{
  "message": "How does FastAPI work?",
  "session_id": 1
}

Response:
{
  "content": "FastAPI is a modern web framework...",
  "sources": ["doc1", "doc2"],
  "intent": "question"
}
```

### 3. Streaming Chat

Real-time response streaming:

```bash
curl -X POST http://localhost:8000/api/v1/chat/stream \\
  -H "Authorization: Bearer <token>" \\
  -H "Content-Type: application/json" \\
  -d '{"message": "Tell me a story", "session_id": 1}'
```

## Memory Strategies

Choose the best memory strategy for your use case:

- **Sliding Window**: Fast, keeps last N messages (default for short chats)
- **Summarization**: LLM-powered summaries (best for long conversations)
- **Hybrid**: Adaptive strategy (recommended for production)

## Monitoring

- **Health**: `GET /health`
- **Readiness**: `GET /ready`
- **Metrics**: `GET /metrics` (Prometheus format)

## Support

For issues and questions:
- GitHub: https://github.com/example/rag-chatbot
- Email: support@example.com
- Docs: https://docs.example.com
"""


def setup_openapi(app: FastAPI):
    """
    Setup custom OpenAPI documentation for FastAPI app.

    Args:
        app: FastAPI application
    """
    app.openapi = lambda: custom_openapi(app)
