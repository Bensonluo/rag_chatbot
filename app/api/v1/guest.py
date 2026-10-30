"""Cookie-authenticated public conversations without registration or shared IDs.

Only this boundary converts an opaque visitor credential into a trusted user
principal. Existing chat/history services keep their owner, key, and coordinator
checks. A numeric conversation ID or a model-provided identity grants no access.
"""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from contextlib import aclosing, contextmanager
from datetime import timedelta
from time import perf_counter
from typing import Annotated, cast

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
from app.services.commerce import create_commerce_port
from app.services.commerce.conversation import CommerceConversation, CommerceReply
from app.services.commerce.port import ProviderRejected, ProviderUnknown
from app.services.commerce.service import PurchaseService
from app.services.commerce.tools import current_purchase
from app.services.guest_identity import (
    GuestAuthenticationError,
    GuestIdentityService,
    GuestIdentityUnavailableError,
    GuestPrincipal,
)
from app.services.observability.trace_events import TraceEvent
from app.services.session_service import SessionService

COOKIE_NAME = "rag_guest_session"
logger = logging.getLogger(__name__)


def _pipeline_trace(started: float, status: str, detail: dict[str, object]) -> str:
    return chat_api._sse_trace_frame(
        TraceEvent(
            stage="cs.pipeline",
            status=status,
            ms=(perf_counter() - started) * 1000,
            detail=detail,
        )
    )


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
    # Ordinary guest history remains readable while the chat graph is unavailable.
    # Shopping deliberately does not fall back: without a tool-calling LLM it
    # must report unavailable rather than route or parse user semantics mechanically.
    return ChatService(
        graph=None,
        coordinator=coordinator,
        persister=ChatMessagePersister(coordinator.sessions),
    )


def get_guest_purchase_service() -> PurchaseService:
    from app.api.database import async_session_maker

    # Explicit composition from settings: mock stays local, external is the
    # HTTP provider. An unreachable external provider surfaces as an explicit
    # error (ProviderUnknown) — never as a silent local commerce fallback.
    return PurchaseService(async_session_maker, create_commerce_port(async_session_maker))


def get_guest_commerce_conversation(
    purchases: Annotated[PurchaseService, Depends(get_guest_purchase_service)],
    coordinator: Annotated[SessionCoordinator, Depends(get_session_coordinator)],
    chat_service: Annotated[ChatService, Depends(get_guest_chat_service)],
) -> CommerceConversation:
    from app.api.database import async_session_maker
    from app.services.commerce.coverage import SqlCommerceCoverageRecorder
    from app.services.commerce.taxonomy import CommerceTaxonomyRegistry

    taxonomy_registry = CommerceTaxonomyRegistry(async_session_maker)
    return CommerceConversation(
        purchases,
        coordinator,
        chat_service.persister or ChatMessagePersister(coordinator.sessions),
        provider=chat_service.llm_service,
        commerce_graph_service=getattr(chat_service, "commerce_graph_service", None),
        coverage_recorder=SqlCommerceCoverageRecorder(taxonomy_registry),
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
    except ProviderUnknown as exc:
        # The provider may have committed the write; the result stays unconfirmed
        # and no local substitute answers for it.
        raise HTTPException(
            503, "Shopping provider did not confirm the result; retry the same request"
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(503, "Shopping agent is temporarily unavailable") from exc
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
        routing_decision = await commerce.route(principal.user.id, body.message)
        if routing_decision.route == "shopping":
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

    async def generate() -> AsyncIterator[str]:
        pipeline_started = perf_counter()
        yield chat_api._sse_trace_frame(
            TraceEvent(stage="cs.pipeline", status="start", detail={"path": "guest"})
        )

        router_started = perf_counter()
        yield chat_api._sse_trace_frame(
            TraceEvent(stage="cs.intent", status="start", detail={"router": "provider"})
        )
        try:
            with commerce_errors():
                decision = await commerce.route(principal.user.id, body.message)
        except HTTPException as exc:
            yield chat_api._sse_trace_frame(
                TraceEvent(
                    stage="cs.intent",
                    status="error",
                    ms=(perf_counter() - router_started) * 1000,
                    detail={"error_type": type(exc).__name__},
                )
            )
            yield chat_api._sse_trace_frame(
                TraceEvent(
                    stage="cs.pipeline",
                    status="error",
                    ms=(perf_counter() - pipeline_started) * 1000,
                    detail={"error_type": type(exc).__name__},
                )
            )
            yield "data: [STREAM_ERROR]\n\n"
            return
        except Exception as exc:
            logger.warning("Public stream router failed (%s)", type(exc).__name__, exc_info=True)
            yield chat_api._sse_trace_frame(
                TraceEvent(
                    stage="cs.intent",
                    status="error",
                    ms=(perf_counter() - router_started) * 1000,
                    detail={"error_type": type(exc).__name__},
                )
            )
            yield chat_api._sse_trace_frame(
                TraceEvent(
                    stage="cs.pipeline",
                    status="error",
                    ms=(perf_counter() - pipeline_started) * 1000,
                    detail={"error_type": type(exc).__name__},
                )
            )
            yield "data: [STREAM_ERROR]\n\n"
            return

        route = decision.route
        yield chat_api._sse_trace_frame(
            TraceEvent(
                stage="cs.intent",
                status="detail",
                detail={
                    "intent": route,
                    "confidence": decision.confidence,
                    "router": "provider",
                    "reason": decision.reason,
                },
            )
        )
        yield chat_api._sse_trace_frame(
            TraceEvent(
                stage="cs.intent",
                status="end",
                ms=(perf_counter() - router_started) * 1000,
                detail={"intent": route, "router": "provider"},
            )
        )

        shopping = route == "shopping"

        if not shopping:
            try:
                support_response = await chat_api.chat_stream(
                    body.chat_request(),
                    request,
                    current_user=principal.user,
                    chat_service=chat_service,
                    session_service=session_service,
                )
            except Exception as exc:
                if not isinstance(exc, HTTPException):
                    logger.warning("Public support stream failed (%s)", type(exc).__name__)
                yield _pipeline_trace(pipeline_started, "error", {"error_type": type(exc).__name__})
                yield "data: [STREAM_ERROR]\n\ndata: [DONE]\n\n"
                return

            saw_done = False
            support_failed = False
            try:
                async with aclosing(
                    cast(AsyncGenerator[str, None], support_response.body_iterator)
                ) as support_stream:
                    async for raw_chunk in support_stream:
                        chunk = raw_chunk.decode() if isinstance(raw_chunk, bytes) else raw_chunk
                        if chunk == "data: [STREAM_ERROR]\n\n":
                            support_failed = True
                            yield _pipeline_trace(
                                pipeline_started, "error", {"error_type": "StreamError"}
                            )
                            yield chunk
                            continue
                        if chunk == "data: [DONE]\n\n":
                            saw_done = True
                            if not support_failed:
                                yield _pipeline_trace(pipeline_started, "end", {"route": route})
                            yield chunk
                            return
                        yield chunk
            except Exception as exc:
                logger.warning("Public support stream failed (%s)", type(exc).__name__)
                yield _pipeline_trace(pipeline_started, "error", {"error_type": type(exc).__name__})
                yield "data: [STREAM_ERROR]\n\ndata: [DONE]\n\n"
                return
            if not saw_done:
                if not support_failed:
                    yield _pipeline_trace(pipeline_started, "end", {"route": route})
                yield "data: [DONE]\n\n"
            return

        events: asyncio.Queue[TraceEvent | None] = asyncio.Queue()

        async def execute() -> CommerceReply:
            try:
                return await commerce.respond(
                    principal.user.id,
                    body.session_id,
                    body.expected_session_key,
                    body.message,
                    on_trace=events.put_nowait,
                )
            finally:
                events.put_nowait(None)

        task: asyncio.Task[CommerceReply] | None = None
        try:
            # The limiter precedes the session fence and billable Agent run.
            await chat_api._enforce_chat_rate_limit(request)
            task = asyncio.create_task(execute())

            async with asyncio.timeout(
                settings.CHAT_STREAM_MAX_SECONDS if settings.CHAT_STREAM_MAX_SECONDS > 0 else None
            ):
                while True:
                    event = await events.get()
                    if event is None:
                        break
                    yield chat_api._sse_trace_frame(event)
                reply = await task
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
            yield chat_api._sse_trace_frame(
                TraceEvent(
                    stage="cs.pipeline",
                    status="end",
                    ms=(perf_counter() - pipeline_started) * 1000,
                    detail={"route": route},
                )
            )
        except Exception as exc:
            # Error text can contain provider diagnostics. Expose only the
            # protocol failure and never turn a failed planner into a reply.
            if not isinstance(exc, HTTPException):
                logger.warning("Public shopping stream failed (%s)", type(exc).__name__)
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                while not events.empty():
                    event = events.get_nowait()
                    if event is not None:
                        yield chat_api._sse_trace_frame(event)
            yield chat_api._sse_trace_frame(
                TraceEvent(
                    stage="cs.pipeline",
                    status="error",
                    ms=(perf_counter() - pipeline_started) * 1000,
                    detail={"error_type": type(exc).__name__},
                )
            )
            yield "data: [STREAM_ERROR]\n\n"
        finally:
            if task is not None and not task.done():
                task.cancel()
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)
        yield "data: [DONE]\n\n"

    return chat_api._ClosingStreamingResponse(generate(), media_type="text/event-stream")


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
