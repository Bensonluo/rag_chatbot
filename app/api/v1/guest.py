"""Cookie-authenticated public conversations without registration or shared IDs.

Only this boundary converts an opaque visitor credential into a trusted user
principal. Existing chat/history services keep their owner, key, and coordinator
checks. A numeric conversation ID or a model-provided identity grants no access.
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.api.deps.private_response import PrivateAPIRoute
from app.api.v1 import chat as chat_api
from app.api.v1 import feedback as feedback_api
from app.config.settings import settings
from app.core.exceptions import ConflictError, NotFoundError, ValidationError
from app.models.schemas.feedback import FeedbackCreate, FeedbackResponse
from app.models.schemas.session_key import SessionKey
from app.services.chat.chat_service import ChatService
from app.services.chat.coordination import (
    ConversationAccessError,
    ConversationBusyError,
    CoordinationUnavailableError,
    HumanHandlingError,
    SessionCoordinator,
    get_session_coordinator,
)
from app.services.chat.persistence import ChatMessagePersister
from app.services.chat.session_identity import ConversationChangedError, require_session_key
from app.services.commerce.conversation import CommerceConversation
from app.services.commerce.port import ProviderRejected
from app.services.commerce.service import PurchaseService
from app.services.commerce.tools import current_purchase
from app.services.guest_identity import (
    GuestAuthenticationError,
    GuestIdentityService,
    GuestIdentityUnavailableError,
    GuestPrincipal,
)
from app.services.session_service import SessionService

COOKIE_NAME = "rag_guest_session"
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/guest", tags=["public visitor"], route_class=PrivateAPIRoute)


class GuestBootstrapRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GuestChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=8000)
    session_id: int = Field(gt=0, le=2_147_483_647, strict=True)
    expected_session_key: SessionKey
    max_tokens: int | None = Field(None, gt=0, le=4096)
    include_citations: bool = False
    include_completion: bool = False

    def chat_request(self) -> chat_api.ChatRequest:
        return chat_api.ChatRequest(**self.model_dump())


def get_guest_chat_service(
    coordinator: Annotated[SessionCoordinator, Depends(get_session_coordinator)],
) -> ChatService:
    try:
        return chat_api.get_chat_service()
    except HTTPException as exc:
        if exc.status_code != 503:
            raise
    # Mock shopping and durable history do not require a configured remote LLM.
    # Ordinary RAG requests still return503 from the existing chat handler.
    return ChatService(
        graph=None,
        coordinator=coordinator,
        persister=ChatMessagePersister(coordinator.sessions),
    )


def get_guest_purchase_service() -> PurchaseService:
    from app.api.database import async_session_maker

    return PurchaseService(async_session_maker)


def get_guest_commerce_conversation(
    purchases: Annotated[PurchaseService, Depends(get_guest_purchase_service)],
    coordinator: Annotated[SessionCoordinator, Depends(get_session_coordinator)],
    chat_service: Annotated[ChatService, Depends(get_guest_chat_service)],
) -> CommerceConversation:
    return CommerceConversation(
        purchases,
        coordinator,
        chat_service.persister or ChatMessagePersister(coordinator.sessions),
        provider=chat_service.llm_service,
    )


@contextmanager
def commerce_errors() -> Iterator[None]:
    try:
        yield
    except (NotFoundError, ConversationAccessError) as exc:
        raise HTTPException(404, "Purchase or conversation not found") from exc
    except (
        ConflictError,
        ConversationBusyError,
        ConversationChangedError,
        ProviderRejected,
    ) as exc:
        raise HTTPException(
            409, "Purchase conditions changed; explicitly refresh before continuing"
        ) from exc
    except HumanHandlingError as exc:
        raise HTTPException(423, "Human support is handling this conversation") from exc
    except CoordinationUnavailableError as exc:
        raise HTTPException(503, "Conversation coordination is unavailable") from exc
    except ValidationError as exc:
        raise HTTPException(422, "Invalid mock purchase input") from exc
    except SQLAlchemyError as exc:
        raise HTTPException(503, "Mock purchases are temporarily unavailable") from exc


def validate_guest_browser_request(request: Request) -> None:
    """Cookie writes require a same-origin/allowlisted fetch, not an HTML form."""
    if request.headers.get("x-rag-guest") != "1":
        raise HTTPException(403, "A public visitor request header is required")
    origin = request.headers.get("origin")
    own_origin = str(request.base_url).rstrip("/")
    allowed = {str(value).rstrip("/") for value in settings.CORS_ORIGINS}
    if origin is not None and origin.rstrip("/") not in allowed | {own_origin}:
        raise HTTPException(403, "Visitor request origin is not allowed")
    if request.method not in {"GET", "HEAD"} and origin is None:
        raise HTTPException(403, "Visitor writes require an Origin header")


def get_guest_identity_service() -> GuestIdentityService:
    from app.api.database import async_session_maker

    return GuestIdentityService(
        async_session_maker, ttl=timedelta(seconds=settings.GUEST_SESSION_TTL_SECONDS)
    )


async def get_guest_principal(
    request: Request,
    service: Annotated[GuestIdentityService, Depends(get_guest_identity_service)],
) -> GuestPrincipal:
    validate_guest_browser_request(request)
    try:
        return await service.authenticate(request.cookies.get(COOKIE_NAME))
    except GuestAuthenticationError as exc:
        raise HTTPException(
            401, "Anonymous session expired; explicitly start a new session"
        ) from exc
    except (GuestIdentityUnavailableError, SQLAlchemyError) as exc:
        raise HTTPException(503, "Anonymous sessions are temporarily unavailable") from exc


async def require_guest_conversation(
    principal: GuestPrincipal,
    session_service: SessionService,
    session_id: int,
    expected_session_key: str,
) -> None:
    """The guest boundary never inherits the legacy shared-Demo exception."""
    try:
        row = await session_service.get_session_by_id(session_id, principal.user.id)
        if not row.is_private:
            raise NotFoundError("Session")
        require_session_key(row.session_key, expected_session_key, required=True)
    except NotFoundError as exc:
        raise HTTPException(404, "Session not found") from exc
    except ConversationChangedError as exc:
        raise HTTPException(409, "Conversation changed; explicitly reopen it") from exc


@router.post("/bootstrap")
async def bootstrap(
    _body: GuestBootstrapRequest,
    request: Request,
    response: Response,
    service: Annotated[GuestIdentityService, Depends(get_guest_identity_service)],
) -> dict[str, object]:
    validate_guest_browser_request(request)
    await chat_api._enforce_chat_rate_limit(request)
    try:
        issued = await service.bootstrap(request.cookies.get(COOKIE_NAME))
    except (GuestIdentityUnavailableError, SQLAlchemyError) as exc:
        raise HTTPException(503, "Anonymous sessions are temporarily unavailable") from exc
    if issued.token_to_set is not None:
        # Deliberately a session cookie. Server expiry remains authoritative,
        # including when a browser restores cookies after a session restore.
        response.set_cookie(
            COOKIE_NAME,
            issued.token_to_set,
            httponly=True,
            secure=settings.GUEST_COOKIE_SECURE,
            samesite="lax",
            path="/",
        )
    return {
        "visitor_id": issued.principal.visitor_id,
        "expires_at": issued.principal.expires_at.isoformat(),
        "session": {"id": issued.session.id, "session_key": issued.session.session_key},
    }


@router.post("/chat", response_model=chat_api.ChatResponse)
async def chat(
    body: GuestChatRequest,
    request: Request,
    principal: Annotated[GuestPrincipal, Depends(get_guest_principal)],
    chat_service: Annotated[ChatService, Depends(get_guest_chat_service)],
    session_service: Annotated[SessionService, Depends(chat_api.get_session_service)],
    commerce: Annotated[CommerceConversation, Depends(get_guest_commerce_conversation)],
) -> chat_api.ChatResponse:
    await require_guest_conversation(
        principal, session_service, body.session_id, body.expected_session_key
    )
    with commerce_errors():
        if await commerce.should_handle(principal.user.id, body.message):
            await chat_api._enforce_chat_rate_limit(request)
            reply = await commerce.respond(
                principal.user.id, body.session_id, body.expected_session_key, body.message
            )
            return chat_api.ChatResponse(
                content=reply.content,
                session_id=body.session_id,
                session_key=body.expected_session_key,
                intent="shopping",
                metadata={"commerce_mode": "mock"},
                ui=reply.ui.model_dump(mode="json") if reply.ui else None,
                message_id=reply.receipt.message_id if reply.receipt else None,
                status=reply.receipt.status if reply.receipt else None,
            )
    return await chat_api.chat(
        body.chat_request(),
        request,
        current_user=principal.user,
        chat_service=chat_service,
        session_service=session_service,
    )


@router.post("/chat/stream")
async def chat_stream(
    body: GuestChatRequest,
    request: Request,
    principal: Annotated[GuestPrincipal, Depends(get_guest_principal)],
    chat_service: Annotated[ChatService, Depends(get_guest_chat_service)],
    session_service: Annotated[SessionService, Depends(chat_api.get_session_service)],
    commerce: Annotated[CommerceConversation, Depends(get_guest_commerce_conversation)],
) -> StreamingResponse:
    await require_guest_conversation(
        principal, session_service, body.session_id, body.expected_session_key
    )
    with commerce_errors():
        shopping = await commerce.should_handle(principal.user.id, body.message)
        if shopping:
            await chat_api._enforce_chat_rate_limit(request)
            async with commerce.coordinator.hold(body.session_id):
                await commerce.validate(
                    principal.user.id, body.session_id, body.expected_session_key, write=True
                )
    if shopping:

        async def generate() -> AsyncIterator[str]:
            try:
                with commerce_errors():
                    async with asyncio.timeout(settings.CHAT_STREAM_MAX_SECONDS):
                        reply = await commerce.respond(
                            principal.user.id,
                            body.session_id,
                            body.expected_session_key,
                            body.message,
                        )
                yield chat_api._sse_frame(reply.content)
                if body.include_completion and reply.receipt is not None:
                    payload = json.dumps(
                        {
                            "message_id": reply.receipt.message_id,
                            "session_id": body.session_id,
                            "session_key": body.expected_session_key,
                            "status": reply.receipt.status,
                        }
                    )
                    yield f"event: completion\ndata: {payload}\n\n"
            except HTTPException:
                yield "data: [STREAM_ERROR]\n\n"
            except Exception as exc:
                # Error text can contain provider diagnostics. Expose only the
                # protocol failure and never turn a failed planner into a reply.
                # CancelledError remains a BaseException and is not swallowed.
                logger.warning("Public shopping stream failed (%s)", type(exc).__name__)
                yield "data: [STREAM_ERROR]\n\n"
            yield "data: [DONE]\n\n"

        return chat_api._ClosingStreamingResponse(generate(), media_type="text/event-stream")
    return await chat_api.chat_stream(
        body.chat_request(),
        request,
        current_user=principal.user,
        chat_service=chat_service,
        session_service=session_service,
    )


@router.get("/history", response_model=chat_api.ChatHistoryResponse)
async def history(
    principal: Annotated[GuestPrincipal, Depends(get_guest_principal)],
    chat_service: Annotated[ChatService, Depends(get_guest_chat_service)],
    session_service: Annotated[SessionService, Depends(chat_api.get_session_service)],
    session_id: int = Query(gt=0, le=2_147_483_647),
    expected_session_key: SessionKey = Query(),
    limit: int = Query(50, gt=0, le=100),
) -> chat_api.ChatHistoryResponse:
    await require_guest_conversation(principal, session_service, session_id, expected_session_key)
    return await chat_api.get_chat_history(
        session_id=session_id,
        expected_session_key=expected_session_key,
        limit=limit,
        current_user=principal.user,
        session_service=session_service,
        chat_service=chat_service,
    )


@router.delete("/history", status_code=status.HTTP_204_NO_CONTENT)
async def clear_history(
    principal: Annotated[GuestPrincipal, Depends(get_guest_principal)],
    chat_service: Annotated[ChatService, Depends(get_guest_chat_service)],
    session_service: Annotated[SessionService, Depends(chat_api.get_session_service)],
    session_id: int = Query(gt=0, le=2_147_483_647),
    expected_session_key: SessionKey = Query(),
) -> None:
    await require_guest_conversation(principal, session_service, session_id, expected_session_key)
    await chat_api.clear_chat_history(
        session_id=session_id,
        expected_session_key=expected_session_key,
        current_user=principal.user,
        session_service=session_service,
        chat_service=chat_service,
    )


@router.post("/feedback", response_model=FeedbackResponse)
async def feedback(
    body: FeedbackCreate,
    principal: Annotated[GuestPrincipal, Depends(get_guest_principal)],
    db: Annotated[AsyncSession, Depends(get_db)],
    coordinator: Annotated[SessionCoordinator, Depends(get_session_coordinator)],
) -> FeedbackResponse:
    return await feedback_api.submit_scoped_feedback(
        body,
        db,
        principal.user,
        coordinator,
        allow_shared=False,
    )


@router.get("/commerce/purchases/active")
async def active_purchase(
    principal: Annotated[GuestPrincipal, Depends(get_guest_principal)],
    commerce: Annotated[CommerceConversation, Depends(get_guest_commerce_conversation)],
    session_id: int = Query(gt=0, le=2_147_483_647),
    expected_session_key: SessionKey = Query(),
) -> dict[str, object]:
    with commerce_errors():
        async with commerce.coordinator.hold(session_id):
            await commerce.validate(
                principal.user.id, session_id, expected_session_key, write=False
            )
            view = await current_purchase(commerce.purchases, principal.user.id)
    return {
        "session_id": session_id,
        "session_key": expected_session_key,
        "ui": view.model_dump(mode="json") if view else None,
    }


@router.get("/commerce/purchases/{purchase_id}")
async def purchase_status(
    purchase_id: SessionKey,
    principal: Annotated[GuestPrincipal, Depends(get_guest_principal)],
    commerce: Annotated[CommerceConversation, Depends(get_guest_commerce_conversation)],
    session_id: int = Query(gt=0, le=2_147_483_647),
    expected_session_key: SessionKey = Query(),
) -> dict[str, object]:
    with commerce_errors():
        async with commerce.coordinator.hold(session_id):
            await commerce.validate(
                principal.user.id, session_id, expected_session_key, write=False
            )
            view = await commerce.purchases.status(principal.user.id, purchase_id)
    return {
        "session_id": session_id,
        "session_key": expected_session_key,
        "ui": view.model_dump(mode="json"),
    }
