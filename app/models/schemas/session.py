"""Session-related Pydantic schemas"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.schemas.session_key import SessionKey


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

    expected_session_key: SessionKey | None = None

    title: str | None = Field(None, max_length=255)
    memory_type: str | None = Field(None, pattern="^(sliding_window|summarization|hybrid)$")
    context_window: int | None = Field(None, ge=1, le=100)


class SessionResponse(SessionBase):
    """Schema for session response"""

    id: int
    user_id: int
    session_key: SessionKey
    created_at: datetime
    updated_at: datetime
    message_count: int = 0
    is_private: bool = False

    model_config = ConfigDict(from_attributes=True)


class SessionListResponse(BaseModel):
    """Schema for paginated session list"""

    items: list[SessionResponse]
    total: int
    page: int
    page_size: int


class SessionReviewVerdict(BaseModel):
    """QA reviewer's verdict on a sampled session (review 2026-09-26, #11).

    ``resolved`` closes the resolution loop the user-confirmation path
    opened: confirmed resolutions stamp resolved_at and join
    sessions_resolved in the containment cross-tab.
    """

    expected_session_key: SessionKey | None = None
    expected_review_version: str | None = Field(
        None, strict=True, min_length=64, max_length=64, pattern="^[0-9a-f]{64}$"
    )
    resolved: bool
    note: str | None = Field(None, max_length=500)
