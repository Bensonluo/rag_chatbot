"""Public transcript fields available to the agent handling a handoff ticket."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.schemas.citation import SourceCitation
from app.models.schemas.session_key import SessionKey


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
    session_key: SessionKey | None = None
    skip: int = Field(ge=0, le=2_147_483_647)
    limit: int = Field(ge=1, le=100)
    total: int = Field(ge=0)
    messages: list[HandoffHistoryMessage]


class HandoffMessageCreate(BaseModel):
    """A client identifies one immutable send attempt; identity comes from auth."""

    model_config = ConfigDict(extra="forbid", strict=True)

    client_message_id: str = Field(min_length=36, max_length=36)
    content: str = Field(min_length=1, max_length=8000)

    @field_validator("client_message_id")
    @classmethod
    def canonical_uuid(cls, value: str) -> str:
        return str(UUID(value))

    @field_validator("content")
    @classmethod
    def nonblank_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Message content cannot be blank")
        return value


class HandoffCustomerMessageCreate(HandoffMessageCreate):
    # New clients bind a composed message to its original human ticket.
    # Omitted for legacy clients; those calls retain session-only routing.
    expected_ticket_id: int | None = Field(default=None, gt=0, le=2_147_483_647)


class HandoffStaffMessageCreate(HandoffMessageCreate):
    resolve: bool = False


class HandoffCustomerMessageRequest(HandoffCustomerMessageCreate):
    expected_session_key: SessionKey


class HandoffStaffMessageRequest(HandoffStaffMessageCreate):
    expected_session_key: SessionKey


class HandoffConversationTicket(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: int
    status: Literal["open", "claimed", "resolved"]
    assigned_to: int | None


class HandoffConversationMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: int
    session_id: int
    ticket_id: int
    author_kind: Literal["customer", "staff"]
    author_user_id: int
    client_message_id: str
    content: str
    timestamp: str


class HandoffConversationPage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    session_key: SessionKey
    session_id: int
    ticket: HandoffConversationTicket | None
    messages: list[HandoffConversationMessage]
    next_after_id: int
    has_more: bool


class HandoffMessageAcknowledgement(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    session_key: SessionKey
    message: HandoffConversationMessage
    ticket: HandoffConversationTicket
    reused: bool
