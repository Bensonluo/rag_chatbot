"""
API router aggregation for v1 endpoints.

Combines all v1 routers into a single router for inclusion in the main app.
"""
from fastapi import APIRouter

from app.api.v1 import auth, sessions, documents, chat, openapi

# Create API v1 router
api_router = APIRouter()

# Include sub-routers
api_router.include_router(auth.router)
api_router.include_router(sessions.router)
api_router.include_router(documents.router)
api_router.include_router(chat.router)
api_router.include_router(openapi.router)
