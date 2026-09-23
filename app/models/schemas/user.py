"""User-related Pydantic schemas"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserBase(BaseModel):
    """Base user schema with common fields"""

    email: EmailStr
    full_name: str | None = None


class UserCreate(UserBase):
    """Schema for user registration"""

    password: str = Field(..., min_length=8, max_length=100)


class UserLogin(BaseModel):
    """Schema for user login"""

    email: EmailStr
    password: str = Field(..., min_length=1)


class UserResponse(UserBase):
    """Schema for user response"""

    id: int
    is_active: bool
    is_admin: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class TokenResponse(BaseModel):
    """Schema for authentication token response"""

    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: str | None = None


class RefreshTokenRequest(BaseModel):
    """Schema for token refresh request"""

    refresh_token: str


class UserUpdate(BaseModel):
    """Schema for updating user information"""

    full_name: str | None = None
    email: EmailStr | None = None


class ChangePasswordRequest(BaseModel):
    """Schema for changing password"""

    old_password: str
    new_password: str = Field(..., min_length=8, max_length=100)
