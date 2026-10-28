"""Shared committed-knowledge policy for customer-facing retrieval entry points."""

from app.services.retrieval.publication import PublicationGuard


def get_publication_guard() -> PublicationGuard:
    from app.api.database import async_session_maker

    return PublicationGuard(async_session_maker)
