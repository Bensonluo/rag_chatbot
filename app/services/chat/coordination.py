"""Serialize conversation actions without locking the session's parent row.

Production PostgreSQL uses transaction-scoped advisory locks on a dedicated
pool. Ordinary history/persistence/handoff queries retain their business pool,
so a turn does not hold a parent-row lock or consume that pool while invoking
nested services. SQLite only provides a process-local gate for local use.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from app.models.database.session import ChatSession
from app.models.database.ticket import HandoffTicket

_LOCK_NAMESPACE = 1_381_190_739
_SQLITE_LOCKS: dict[tuple[object, int], asyncio.Lock] = {}
_default: SessionCoordinator | None = None
_lock_engine: AsyncEngine | None = None


class ConversationBusyError(RuntimeError):
    """Another conversation action has not yet settled; do not replay it."""


class CoordinationUnavailableError(RuntimeError):
    """The conversation gate cannot be established safely."""


class HumanHandlingError(RuntimeError):
    """An assigned human is handling this session; automation is paused."""


class ConversationAccessError(LookupError):
    """A private session is unavailable to the caller's authenticated identity."""


@dataclass(frozen=True)
class ConversationState:
    epoch: int
    owner_id: int
    is_private: bool
    claimed_ticket_id: int | None
    session_key: str | None = None

    def thread_id(self, session_id: int) -> str:
        return str(session_id) if self.epoch == 0 else f"{session_id}:automation:{self.epoch}"


class SessionCoordinator:
    def __init__(
        self,
        session_maker: Callable[[], Any],
        lock_session_maker: Callable[[], Any] | None = None,
        *,
        dialect: str | None = None,
        lock_timeout_seconds: float = 1.0,
    ) -> None:
        if (
            isinstance(lock_timeout_seconds, bool)
            or not math.isfinite(lock_timeout_seconds)
            or lock_timeout_seconds <= 0
        ):
            raise ValueError("Conversation lock timeout must be finite and positive")
        self.sessions = session_maker
        self._lock_sessions = lock_session_maker or session_maker
        bind = getattr(session_maker, "kw", {}).get("bind")
        self._dialect = dialect or getattr(getattr(bind, "dialect", None), "name", None)
        if self._dialect not in {"sqlite", "postgresql"}:
            raise ValueError("Conversation coordination requires PostgreSQL or local SQLite")
        self._key = bind if bind is not None else session_maker
        self._timeout = lock_timeout_seconds

    @asynccontextmanager
    async def hold(self, session_id: int) -> AsyncIterator[None]:
        if type(session_id) is not int or not 1 <= session_id <= 2_147_483_647:
            raise ValueError("Conversation ID must be a positive 31-bit integer")
        if self._dialect == "sqlite":
            lock = _SQLITE_LOCKS.setdefault((self._key, session_id), asyncio.Lock())
            if lock.locked():
                raise ConversationBusyError("Conversation is busy")
            await lock.acquire()
            try:
                yield
            finally:
                lock.release()
                _SQLITE_LOCKS.pop((self._key, session_id), None)
            return
        body_failed = False
        try:
            async with self._lock_sessions() as db, db.begin():
                async with asyncio.timeout(self._timeout):
                    acquired = await db.scalar(
                        text("SELECT pg_try_advisory_xact_lock(:namespace, :session_id)"),
                        {"namespace": _LOCK_NAMESPACE, "session_id": session_id},
                    )
                if acquired is not True:
                    raise ConversationBusyError("Conversation is busy")
                try:
                    yield
                except BaseException:
                    body_failed = True
                    raise
        except (ConversationBusyError, asyncio.CancelledError):
            raise
        except Exception as exc:
            if body_failed:
                raise
            raise CoordinationUnavailableError("Conversation coordination unavailable") from exc

    async def state(self, session_id: int) -> ConversationState:
        claimed = (
            select(HandoffTicket.id)
            .where(HandoffTicket.session_id == session_id, HandoffTicket.status == "claimed")
            .limit(1)
            .scalar_subquery()
        )
        try:
            async with asyncio.timeout(3.0):
                async with self.sessions() as db:
                    row = (
                        await db.execute(
                            select(ChatSession, claimed).where(ChatSession.id == session_id)
                        )
                    ).one_or_none()
                    if row is None:
                        raise ConversationAccessError("Session unavailable")
                    session, ticket_id = row
                    return ConversationState(
                        session.automation_epoch,
                        session.user_id,
                        session.is_private,
                        ticket_id,
                        session.session_key,
                    )
        except ConversationAccessError:
            raise
        except Exception as exc:
            raise CoordinationUnavailableError("Conversation state unavailable") from exc


def get_session_coordinator() -> SessionCoordinator:
    global _default, _lock_engine
    if _default is None:
        from app.api.database import async_session_maker
        from app.config.settings import settings

        bind = async_session_maker.kw.get("bind")
        if getattr(getattr(bind, "dialect", None), "name", None) == "postgresql":
            _lock_engine = create_async_engine(
                settings.DATABASE_URL,
                pool_size=settings.DATABASE_POOL_SIZE,
                max_overflow=0,
                pool_timeout=1.0,
                pool_pre_ping=True,
            )
            locks = async_sessionmaker(_lock_engine, expire_on_commit=False)
            _default = SessionCoordinator(async_session_maker, locks)
        else:
            _default = SessionCoordinator(async_session_maker)
    return _default


async def close_session_coordinator() -> None:
    global _default, _lock_engine
    engine = _lock_engine
    _default, _lock_engine = None, None
    if engine is not None:
        await engine.dispose()
