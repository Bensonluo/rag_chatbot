"""Strict server-issued conversation incarnation keys used by private clients."""

from typing import Annotated
from uuid import UUID

from pydantic import AfterValidator, StringConstraints


def canonical_session_key(value: str) -> str:
    if str(UUID(value)) != value:
        raise ValueError("Session key must be a canonical UUID")
    return value


SessionKey = Annotated[
    str,
    StringConstraints(strict=True, min_length=36, max_length=36),
    AfterValidator(canonical_session_key),
]
