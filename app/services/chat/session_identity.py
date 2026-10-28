"""Compare object incarnation after authorization; a key never grants access."""

from app.core.exceptions import ConflictError


class ConversationChangedError(ConflictError):
    """The caller's conversation was deleted or explicitly reset."""


def require_session_key(
    actual: str | None, expected: str | None, *, required: bool = False
) -> None:
    if (required and expected is None) or (expected is not None and actual != expected):
        raise ConversationChangedError("Conversation identity changed; reopen it before continuing")
