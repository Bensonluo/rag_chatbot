"""
FastAPI dependencies for API endpoints.

Provides authentication and other common dependencies.
"""
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.models.database.user import User

# HTTP Bearer token scheme
security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[User]:
    """
    Get current authenticated user (optional for demo).

    For demo mode, this allows anonymous access without authentication.
    In production, you would validate the JWT token and return the user.

    Args:
        credentials: Optional HTTP Bearer credentials

    Returns:
        Optional[User]: User if authenticated, None otherwise
    """
    # Demo mode: allow anonymous access
    # In production, validate JWT token here
    if credentials is None:
        return None

    # TODO: Implement JWT token validation for production
    # For now, return None to allow anonymous access
    return None
