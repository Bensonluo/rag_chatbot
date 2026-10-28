"""Provider-owned error taxonomy mapped to HTTP statuses by the API layer.

These classes are the provider's only outcome vocabulary: business refusals
are definitive and commit nothing, conflicts guard idempotency keys, and
"unknown" means the write may have committed and must be reconciled via the
operation lookup.
"""

from __future__ import annotations


class ProviderError(Exception):
    """Base class for every provider outcome that is not a normal result."""

    code = "rejected"
    status = 422

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ProviderRejectedError(ProviderError):
    """Definitive business refusal; no external operation was committed."""

    code = "rejected"
    status = 422


class ProviderNotFoundError(ProviderError):
    """The requested provider object does not exist for this owner."""

    code = "not_found"
    status = 404


class ProviderConflictError(ProviderError):
    """Idempotency conflict: the request key is bound to different arguments."""

    code = "conflict"
    status = 409


class ProviderUnknownError(ProviderError):
    """Outcome not confirmed; the request may have committed. Reconcile it."""

    code = "unknown"
    status = 504


class ProviderAuthError(ProviderError):
    """Missing or invalid internal service token."""

    code = "unauthorized"
    status = 401
