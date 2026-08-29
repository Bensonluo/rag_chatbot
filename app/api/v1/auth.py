"""
Authentication API endpoints.

Provides endpoints for user registration, login, token refresh,
and getting current user information.
"""
from typing import Annotated, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from app.models.schemas.user import (
    UserCreate,
    UserResponse,
    UserLogin,
    TokenResponse,
    RefreshTokenRequest,
)
from app.api.deps import (
    get_auth_service,
    get_current_active_user,
)
from app.core.exceptions import ConflictError, ValidationError
from app.services.auth_service import AuthenticationService
from app.models.database.user import User

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
)
async def register(
    user_data: UserCreate,
    auth_service: Annotated[AuthenticationService, Depends(get_auth_service)],
) -> User:
    """
    Register a new user account.

    - **email**: User email address (must be unique)
    - **password**: Password (min 8 characters, must contain uppercase, lowercase, and digit)
    - **full_name**: User's full name (optional)

    Returns the created user information.
    """
    try:
        user = await auth_service.register_user(
            email=user_data.email,
            password=user_data.password,
            full_name=user_data.full_name,
        )
        return user
    except HTTPException:
        raise
    except ConflictError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        ) from e
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}",
        ) from e


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Login with email and password",
)
async def login(
    user_credentials: UserLogin,
    auth_service: Annotated[AuthenticationService, Depends(get_auth_service)],
) -> Dict[str, str | int]:
    """
    Authenticate a user and return access and refresh tokens.

    - **email**: User email address
    - **password**: User password

    Returns JWT access token and refresh token.
    """
    try:
        # Authenticate user
        user = await auth_service.authenticate_user(
            email=user_credentials.email,
            password=user_credentials.password,
        )

        # Create tokens
        token_data = await auth_service.create_access_token(user)
        refresh_token = await auth_service.create_refresh_token(user)

        return {
            **token_data,
            "refresh_token": refresh_token,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        ) from e


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Refresh access token",
)
async def refresh_token(
    token_data: RefreshTokenRequest,
    auth_service: Annotated[AuthenticationService, Depends(get_auth_service)],
) -> Dict[str, str | int]:
    """
    Refresh an access token using a refresh token.

    - **refresh_token**: Valid refresh token from login

    Returns a new access token.
    """
    try:
        new_token_data = await auth_service.refresh_access_token(
            refresh_token=token_data.refresh_token
        )
        return new_token_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        ) from e


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user",
)
async def get_current_user_info(
    current_user: Annotated[User, Depends(get_current_active_user)],
) -> User:
    """
    Get information about the currently authenticated user.

    Requires a valid access token in the Authorization header.
    """
    return current_user
