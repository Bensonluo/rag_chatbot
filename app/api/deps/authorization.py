"""Identity and object-level authorization boundary (review 2026-09-26, #1).

One switch, ``DEMO_MODE``, selects the posture:

- demo (default): the public demo keeps its documented behavior —
  anonymous visitors chat under a body-supplied ``user_id`` and read
  or clear history by bare session id (the demo DB's sessions are
  owned by seeded users, so strict ownership would break it).
- strict (``DEMO_MODE=false``): identity comes only from the
  authenticated context. Chat requires login, a body ``user_id``
  contradicting the token is refused, and the history endpoints only
  touch sessions owned by the caller — foreign and missing sessions
  both answer 404 so existence is not leaked (OWASP BOLA guidance).
  Knowledge-base writes additionally require admin.

Endpoint wiring calls these helpers *before* their try blocks: an
HTTPException raised inside would be swallowed by the generic
``except Exception`` handler and resurface as a 500.
"""

from typing import Protocol

from fastapi import HTTPException, status

from app.config.settings import settings
from app.core.exceptions import NotFoundError
from app.models.database.user import User


class AnySessionService(Protocol):
    """Structural type: anything answering the scoped session lookup.

    Avoids importing SessionService here (which would drag its
    repositories into API-layer tests for no gain).
    """

    async def get_session_by_id(self, session_id: int, user_id: int) -> object:
        """Return the session when the user owns it, else raise NotFoundError."""
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
) -> None:
    """Object-level check for chat history endpoints (strict posture).

    Demo posture returns without touching the session service. Strict:
    the caller must be authenticated and own the session; a foreign or
    missing session is 404 (indistinguishable on purpose).
    """
    if settings.DEMO_MODE:
        return
    if current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )
    try:
        await session_service.get_session_by_id(session_id, current_user.id)
    except NotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        ) from exc


def require_kb_writer(current_user: User | None) -> None:
    """Gate for knowledge-base write endpoints.

    Demo posture: any authenticated user may manage the demo KB (the
    endpoint's auth dependency already rejected anonymous). Strict:
    admin only — an open-registration customer must not be able to
    poison or delete the shared service knowledge base.
    """
    if settings.DEMO_MODE:
        return
    if current_user is None or not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator role required",
        )


def require_qa_reviewer(current_user: User | None) -> None:
    """Gate for QA review endpoints (review queue sampling + verdicts).

    The review queue and verdicts read other users' transcripts and
    shape the resolution KPI — an ops surface, not a customer one.
    Demo posture: any authenticated user (the endpoint's auth
    dependency already rejected anonymous; the demo DB has no secrets
    worth gatekeeping from its own users). Strict: admin only.
    """
    if settings.DEMO_MODE:
        return
    if current_user is None or not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator role required",
        )
