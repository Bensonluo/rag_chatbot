"""Custom OpenAPI schema for the DEBUG-gated /docs storefront.

The single source of truth for title, description, and version stays the
FastAPI constructor in ``app.main`` — this module reads them from the app
instance instead of duplicating (and eventually contradicting) them. It only
adds what FastAPI cannot infer from the code:

- tag metadata for every router tag that exists in the tree,
- servers (local development + the public storefront),
- license, contact, and the JWT bearer security scheme.

Placeholder values (``example.com`` addresses, hardcoded versions) are what
the previous revision of this file shipped; ``app/tests/unit/api/v1/test_openapi_schema.py``
pins them out.
"""

from typing import Any

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

REPOSITORY_URL = "https://github.com/Bensonluo/rag_chatbot"
STOREFRONT_URL = "https://benluo.art"

TAG_METADATA: list[dict[str, str]] = [
    {
        "name": "public visitor",
        "description": "Cookie-authenticated shopping and support conversations "
        "for anonymous visitors — no registration, no shared IDs.",
    },
    {
        "name": "chat",
        "description": "Authenticated chat turns and SSE streaming through the full RAG pipeline.",
    },
    {
        "name": "sessions",
        "description": "Chat session lifecycle, history, and the QA review queue.",
    },
    {
        "name": "authentication",
        "description": "User registration, login, and JWT issuance.",
    },
    {
        "name": "documents",
        "description": "Knowledge-base document upload, ingestion, and deletion.",
    },
    {
        "name": "feedback",
        "description": "Thumbs up/down ratings on assistant messages.",
    },
    {
        "name": "handoff",
        "description": "Human-support tickets created when conversations escalate.",
    },
    {
        "name": "analytics",
        "description": "Containment and satisfaction aggregates for operations.",
    },
    {
        "name": "commerce-admin",
        "description": "Admin views over purchases and commerce coverage.",
    },
    {
        "name": "graph",
        "description": "GraphRAG queries and schema (included only when GRAPH_RAG_ENABLED=true).",
    },
]


def custom_openapi(app: FastAPI) -> dict[str, Any]:
    """Build the enriched schema once and cache it on the app instance."""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    openapi_schema["tags"] = TAG_METADATA

    openapi_schema["info"]["contact"] = {
        "name": "Agent Store maintainers",
        "url": f"{REPOSITORY_URL}/issues",
    }
    openapi_schema["info"]["license"] = {
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    }

    openapi_schema["servers"] = [
        {"url": "http://localhost:8000", "description": "Local development"},
        {"url": STOREFRONT_URL, "description": "Public storefront (nginx routes /api here)"},
    ]

    # get_openapi only emits "components" when routes need it; a route-less
    # app (tests) legitimately has none.
    components = openapi_schema.setdefault("components", {})
    components["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        },
    }

    app.openapi_schema = openapi_schema
    return openapi_schema


def setup_openapi(app: FastAPI) -> None:
    """Install the custom schema generator (documented FastAPI idiom)."""
    app.openapi = lambda: custom_openapi(app)  # type: ignore[method-assign]
