"""Identity and object-level authorization boundary (review 2026-09-26, #1).

One switch, ``DEMO_MODE``, selects the posture:

- demo (default): unmarked public sessions keep shared behavior —
  anonymous visitors chat under a body-supplied ``user_id`` and read
  or clear history by bare session id (the demo DB's sessions are
  owned by seeded users). Private sessions always require their owner.
- strict (``DEMO_MODE=false``): identity comes only from the
  authenticated context. Chat requires login, a body ``user_id``
  contradicting the token is refused, and history and message-processing
  endpoints only touch sessions owned by the caller — foreign and missing sessions
  both answer 404 so existence is not leaked (OWASP BOLA guidance).

Knowledge-base management requires an authenticated active admin in both
modes. The public-demo chat posture does not grant document-management access.

Endpoint wiring calls these helpers *before* their try blocks: an
HTTPException raised inside would be swallowed by the generic
``except Exception`` handler and resurface as a 500.
"""

from typing import Protocol

from fastapi import HTTPException, status

from app.config.settings import settings
from app.core.exceptions import NotFoundError
from app.models.database.user import User
from app.services.chat.session_identity import ConversationChangedError, require_session_key


class AnySessionService(Protocol):
    """Structural type: anything answering the scoped session lookup.

    Avoids importing SessionService here (which would drag its
    repositories into API-layer tests for no gain).
    """

    async def get_session_by_id(self, session_id: int, user_id: int) -> object:
        """Return the session when the user owns it, else raise NotFoundError."""
        ...

    async def is_private_session(self, session_id: int) -> bool:
        """Whether this session carries the server-owned private marker."""
        ...


def resolve_chat_identity(
    current_user: User | None,
    requested_user_id: int | None,
) -> int:
    """The user identity a chat turn runs under.

    Strict posture: the token wins absolutely — anonymous is 401 and a
    body ``user_id`` pointing at someone else is 403. Demo posture:
    token identity when present, otherwise the body id (demo traffic
    is anonymous), otherwise 0 as before.
    """
    if settings.DEMO_MODE:
        if current_user is not None:
            return current_user.id
        return requested_user_id or 0
    if current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )
    if requested_user_id is not None and requested_user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="user_id does not match the authenticated user",
        )
    return current_user.id


async def ensure_session_access(
    session_id: int,
    current_user: User | None,
    session_service: AnySessionService,
    *,
    expected_session_key: str | None = None,
    require_key: bool = False,
) -> str | None:
    """Object-level check before chat processing or history access (strict posture).

    Demo posture checks the server-owned privacy marker first. Private and strict:
    the caller must be authenticated and own the session; a foreign or
    missing session is 404 (indistinguishable on purpose).
    """
    if settings.DEMO_MODE:
        privacy_reader = getattr(session_service, "is_private_session", None)
        # Internal legacy service adapters have no private-session feature;
        # the production SessionService always supplies the scoped reader.
        if privacy_reader is None or not await privacy_reader(session_id):
            if expected_session_key is None:
                return None
            reader = getattr(session_service, "get_session_key", None)
            key = await reader(session_id) if reader is not None else None
            try:
                require_session_key(key, expected_session_key)
            except ConversationChangedError as exc:
                raise HTTPException(
                    409, "Conversation changed; reopen it before continuing"
                ) from exc
            return key if isinstance(key, str) else None
    if current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )
    try:
        row = await session_service.get_session_by_id(session_id, current_user.id)
    except NotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        ) from exc

    key = getattr(row, "session_key", None)
    key = key if isinstance(key, str) else None
    try:
        require_session_key(
            key,
            expected_session_key,
            required=require_key and getattr(row, "is_private", False) is True,
        )
    except ConversationChangedError as exc:
        raise HTTPException(409, "Conversation changed; reopen it before continuing") from exc
    return key


def require_kb_writer(current_user: User | None) -> None:
    """Require an admin for knowledge-base management in both modes.

    Endpoint dependencies already require an active authenticated user.
    Open-registration customers must not be able to inspect unpublished
    source documents, poison answers, or delete the shared knowledge base.
    """
    if current_user is None or current_user.is_admin is not True:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator role required",
        )


def require_qa_reviewer(current_user: User | None) -> None:
    """Gate for QA review endpoints (review queue sampling + verdicts).

    The review queue and verdicts read other users' transcripts and
    shape the resolution KPI — an ops surface, not a customer one.
    This surface can include private customer content. Admin is required
    in both public Demo and strict modes.
    """
    if current_user is None or not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator role required",
        )
