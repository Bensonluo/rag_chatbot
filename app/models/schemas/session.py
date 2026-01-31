"""Session-related Pydantic schemas"""
from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Optional
from datetime import datetime


class SessionBase(BaseModel):
    """Base session schema"""
    title: Optional[str] = Field(None, max_length=255)
    memory_type: str = Field("sliding_window", pattern="^(sliding_window|summarization|hybrid)$")
    context_window: int = Field(10, ge=1, le=100)


class SessionCreate(SessionBase):
    """Schema for creating a new session"""
    pass


class SessionUpdate(BaseModel):
    """Schema for updating a session"""
    title: Optional[str] = Field(None, max_length=255)
    memory_type: Optional[str] = Field(None, pattern="^(sliding_window|summarization|hybrid)$")
    context_window: Optional[int] = Field(None, ge=1, le=100)


class SessionResponse(SessionBase):
    """Schema for session response"""
    id: int
    user_id: int
    created_at: datetime
    updated_at: datetime
    message_count: int = 0

    model_config = ConfigDict(from_attributes=True)


class SessionListResponse(BaseModel):
    """Schema for paginated session list"""
    items: list[SessionResponse]
    total: int
    page: int
    page_size: int
