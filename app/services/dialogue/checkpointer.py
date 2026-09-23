"""
Shared dialogue checkpointer lifecycle.

LangGraph checkpointers hold conversation state (intent, slots, staged
confirmations). ``MemorySaver`` pins that state to one process, so under
horizontal scaling a user's consecutive turns hit different replicas and
the dialogue — including a staged irreversible-action confirmation — is
silently lost. The manager below builds a Postgres-backed shared
checkpointer for production and degrades to ``MemorySaver`` for local
development, tests, or on connection failure so the app always starts.
"""

from __future__ import annotations

import logging
from contextlib import AbstractAsyncContextManager
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)


def normalize_db_url(url: str) -> str:
    """Translate a SQLAlchemy async URL into a psycopg-compatible one.

    ``postgresql+asyncpg://`` is SQLAlchemy syntax; the Postgres
    checkpointer speaks plain ``postgresql://``.
    """
    return url.replace("postgresql+asyncpg://", "postgresql://", 1)


class DialogueCheckpointerManager:
    """Owns creation and shutdown of the app-wide dialogue checkpointer."""

    def __init__(self, checkpointer_type: str, db_url: str | None) -> None:
        self._type = checkpointer_type
        self._db_url = db_url
        self._entered_cm: AbstractAsyncContextManager[Any] | None = None

    async def start(self) -> BaseCheckpointSaver[Any]:
        """Create and initialize the shared checkpointer.

        Returns a Postgres-backed saver in ``postgres`` mode (tables
        created via ``setup()``, connection pool kept open), otherwise —
        or when the database is unreachable — a process-local
        ``MemorySaver`` with a logged warning so behavior degrades
        visibly rather than fatally.
        """
        if self._type != "postgres":
            return MemorySaver()

        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            if not self._db_url:
                raise ValueError("postgres checkpointer requires a database URL")

            cm = AsyncPostgresSaver.from_conn_string(normalize_db_url(self._db_url))
            saver = await cm.__aenter__()
            await saver.setup()
            self._entered_cm = cm
            logger.info("Shared Postgres dialogue checkpointer initialized")
            return saver
        except Exception as exc:
            logger.warning(
                "Postgres dialogue checkpointer unavailable (%s); "
                "falling back to MemorySaver — dialogue state will NOT "
                "survive replica changes or restarts",
                exc,
            )
            self._entered_cm = None
            return MemorySaver()

    async def stop(self) -> None:
        """Close the checkpointer's connection pool (idempotent)."""
        if self._entered_cm is None:
            return
        try:
            await self._entered_cm.__aexit__(None, None, None)
        except Exception as exc:
            logger.warning("Failed to close dialogue checkpointer: %s", exc)
        finally:
            self._entered_cm = None


def create_checkpointer_manager_from_settings() -> DialogueCheckpointerManager:
    """Build the manager from application settings."""
    from app.config.settings import get_settings

    settings = get_settings()
    return DialogueCheckpointerManager(
        checkpointer_type=settings.DIALOGUE_CHECKPOINTER_TYPE,
        db_url=settings.DIALOGUE_CHECKPOINTER_DB_URL or settings.DATABASE_URL,
    )
