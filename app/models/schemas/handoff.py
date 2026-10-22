"""Public transcript fields available to the agent handling a handoff ticket."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models.schemas.citation import SourceCitation


class HandoffHistoryMessage(BaseModel):
    """A persisted customer-facing turn, excluding internal message metadata."""

    model_config = ConfigDict(extra="forbid", strict=True)

    id: int
    role: Literal["user", "assistant"]
    content: str
    timestamp: str
    sources: list[str] | None = None
    citations: list[SourceCitation] = Field(default_factory=list)


class HandoffHistoryResponse(BaseModel):
    """One page of the conversation associated with an authorized ticket."""

    model_config = ConfigDict(extra="forbid", strict=True)

    ticket_id: int
    session_id: int
    skip: int = Field(ge=0, le=2_147_483_647)
    limit: int = Field(ge=1, le=100)
    total: int = Field(ge=0)
    messages: list[HandoffHistoryMessage]
