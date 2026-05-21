"""Pydantic schemas for feedback API."""
from pydantic import BaseModel, Field


class FeedbackCreate(BaseModel):
    """Request body for submitting feedback."""
    message_id: int = Field(..., description="ID of the assistant message to rate")
    rating: int = Field(..., ge=-1, le=1, description="Rating: 1=thumbs up, -1=thumbs down")
    text: str | None = Field(None, max_length=1000, description="Optional feedback text")


class FeedbackResponse(BaseModel):
    """Response after submitting feedback."""
    success: bool
    message_id: int
    rating: int
