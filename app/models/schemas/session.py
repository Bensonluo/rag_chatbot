"""Session-related Pydantic schemas"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class SessionBase(BaseModel):
    """Base session schema"""

    title: str | None = Field(None, max_length=255)
    memory_type: str = Field("sliding_window", pattern="^(sliding_window|summarization|hybrid)$")
    context_window: int = Field(10, ge=1, le=100)


class SessionCreate(SessionBase):
    """Schema for creating a new session"""

    pass


class SessionUpdate(BaseModel):
    """Schema for updating a session"""

    title: str | None = Field(None, max_length=255)
    memory_type: str | None = Field(None, pattern="^(sliding_window|summarization|hybrid)$")
    context_window: int | None = Field(None, ge=1, le=100)


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
