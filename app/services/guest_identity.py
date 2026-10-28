"""Browser possession authorizes a guest; numeric user/session IDs never do."""

import hashlib
import re
import secrets
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import AuthenticationError
from app.models.database.guest import GuestVisitor
from app.models.database.session import ChatSession
from app.models.database.user import User


class GuestAuthenticationError(AuthenticationError):
    def __init__(self) -> None:
        super().__init__("Guest session is unavailable")


class GuestIdentityUnavailableError(RuntimeError):
    """The identity cannot be safely reused; do not issue a replacement on DB failure."""


class GuestPrincipal(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    visitor_id: str
    user: User = Field(exclude=True, repr=False)
    expires_at: datetime


class IssuedGuest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    principal: GuestPrincipal
    token_to_set: str | None = Field(default=None, exclude=True, repr=False)
    session: ChatSession = Field(exclude=True, repr=False)


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _token_hash(raw_token: str | None) -> str:
    if type(raw_token) is not str or re.fullmatch(r"[A-Za-z0-9_-]{43}", raw_token) is None:
        raise GuestAuthenticationError()
    return hashlib.sha256(raw_token.encode("ascii")).hexdigest()


class GuestIdentityService:
    def __init__(
        self,
        session_maker: Callable[[], AsyncSession],
        *,
        ttl: timedelta = timedelta(days=7),
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(ttl, timedelta) or ttl <= timedelta(0):
            raise ValueError("Guest credential TTL must be positive")
        self._sessions = session_maker
        self._ttl = ttl
        self._clock = clock or (lambda: datetime.now(UTC))

    async def _load(
        self, db: AsyncSession, token_hash: str, *, lock: bool = False
    ) -> tuple[GuestVisitor, User] | None:
        stmt = (
            select(GuestVisitor, User)
            .join(User, User.id == GuestVisitor.user_id)
            .where(
                GuestVisitor.token_hash == token_hash,
                GuestVisitor.expires_at > _utc(self._clock()),
                User.is_guest.is_(True),
                User.is_active.is_(True),
                User.is_admin.is_(False),
                User.email.is_(None),
                User.hashed_password.is_(None),
            )
            .execution_options(populate_existing=True)
        )
        if lock:
            stmt = stmt.with_for_update(of=GuestVisitor)
        row = (await db.execute(stmt)).one_or_none()
        return (row[0], row[1]) if row is not None else None

    @staticmethod
    def _principal(visitor: GuestVisitor, user: User) -> GuestPrincipal:
        return GuestPrincipal(visitor_id=visitor.id, user=user, expires_at=_utc(visitor.expires_at))

    async def authenticate(self, raw_token: str | None) -> GuestPrincipal:
        digest = _token_hash(raw_token)
        async with self._sessions() as db:
            found = await self._load(db, digest)
            if found is None:
                raise GuestAuthenticationError()
            return self._principal(*found)

    async def bootstrap(self, raw_token: str | None) -> IssuedGuest:
        try:
            return await self._reuse(_token_hash(raw_token))
        except GuestAuthenticationError:
            return await self._create()

    async def _create(self) -> IssuedGuest:
        raw = secrets.token_urlsafe(32)
        expires_at = _utc(self._clock()) + self._ttl
        async with self._sessions() as db, db.begin():
            user = User(is_guest=True, email=None, hashed_password=None, is_admin=False)
            db.add(user)
            await db.flush()
            chat = ChatSession(
                user_id=user.id,
                title="Guest consultation",
                session_metadata={"access_mode": "private"},
            )
            db.add(chat)
            await db.flush()
            visitor = GuestVisitor(
                user_id=user.id,
                token_hash=_token_hash(raw),
                expires_at=expires_at,
                current_session_id=chat.id,
            )
            db.add(visitor)
            await db.flush()
            user_id, session_id, visitor_id = user.id, chat.id, visitor.id
        # Refresh scalars even when a supplied maker uses expire_on_commit=True.
        async with self._sessions() as db:
            fresh_user = await db.get(User, user_id)
            fresh_chat = await db.get(ChatSession, session_id)
            if fresh_user is None or fresh_chat is None:
                raise GuestIdentityUnavailableError("Guest creation could not be confirmed")
            return IssuedGuest(
                principal=GuestPrincipal(
                    visitor_id=visitor_id, user=fresh_user, expires_at=expires_at
                ),
                token_to_set=raw,
                session=fresh_chat,
            )

    async def _reuse(self, digest: str) -> IssuedGuest:
        for _attempt in range(3):
            async with self._sessions() as db:
                found = await self._load(db, digest, lock=True)
                if found is None:
                    raise GuestAuthenticationError()
                visitor, user = found
                chat = (
                    await db.get(ChatSession, visitor.current_session_id)
                    if visitor.current_session_id is not None
                    else None
                )
                if chat is not None and chat.user_id == user.id and chat.is_private:
                    return IssuedGuest(principal=self._principal(visitor, user), session=chat)
                original_session_id = visitor.current_session_id
                chat = ChatSession(
                    user_id=user.id,
                    title="Guest consultation",
                    session_metadata={"access_mode": "private"},
                )
                db.add(chat)
                await db.flush()
                changed = await db.execute(
                    update(GuestVisitor)
                    .where(
                        GuestVisitor.id == visitor.id,
                        GuestVisitor.token_hash == digest,
                        GuestVisitor.expires_at > _utc(self._clock()),
                        GuestVisitor.current_session_id == original_session_id
                        if original_session_id is not None
                        else GuestVisitor.current_session_id.is_(None),
                        select(User.id)
                        .where(
                            User.id == GuestVisitor.user_id,
                            User.is_guest.is_(True),
                            User.is_active.is_(True),
                            User.is_admin.is_(False),
                            User.email.is_(None),
                            User.hashed_password.is_(None),
                        )
                        .exists(),
                    )
                    .values(current_session_id=chat.id)
                    .returning(GuestVisitor.id)
                    .execution_options(synchronize_session=False)
                )
                if changed.scalar_one_or_none() != visitor.id:
                    await db.rollback()
                    continue
                principal = self._principal(visitor, user)
                await db.commit()
                await db.refresh(user)
                await db.refresh(chat)
                return IssuedGuest(principal=principal, session=chat)
        raise GuestIdentityUnavailableError("Guest conversation changed; retry bootstrap")
