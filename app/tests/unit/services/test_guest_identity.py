"""Guest credentials have isolated ownership, expiry and no account password."""

import hashlib
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from fastapi.encoders import jsonable_encoder
from sqlalchemy import event, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.exceptions import AuthenticationError
from app.core.security import create_refresh_token
from app.models.database.base import Base
from app.models.database.guest import GuestVisitor
from app.models.database.message import Message
from app.models.database.session import ChatSession
from app.models.database.user import User
from app.models.enums.message import MessageRole, MessageStatus
from app.repositories.user_repository import UserRepository
from app.services.auth_service import AuthenticationService
from app.services.guest_identity import GuestAuthenticationError, GuestIdentityService


@pytest.fixture
async def guest_db(tmp_path) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/guest.sqlite3")

    @event.listens_for(engine.sync_engine, "connect")
    def foreign_keys(connection, _record):
        connection.execute("PRAGMA foreign_keys=ON")

    async with engine.begin() as db:
        await db.run_sync(Base.metadata.create_all)
    try:
        # Expiration is intentional: returned principals must have loaded scalars.
        yield async_sessionmaker(engine, expire_on_commit=True)
    finally:
        await engine.dispose()


async def test_new_visitor_is_atomic_private_owned_and_has_no_fake_credentials(guest_db):
    issued = await GuestIdentityService(guest_db).bootstrap(None)
    raw = issued.token_to_set
    assert raw is not None and len(raw) == 43
    assert str(UUID(issued.principal.visitor_id)) == issued.principal.visitor_id
    assert issued.principal.user.is_guest is True
    assert issued.principal.user.email is None and issued.principal.user.hashed_password is None
    assert issued.principal.user.is_admin is False
    assert issued.session.is_private and issued.session.user_id == issued.principal.user.id
    assert str(UUID(issued.session.session_key)) == issued.session.session_key
    assert raw not in repr(issued)
    assert raw not in json.dumps(jsonable_encoder(issued))
    async with guest_db() as db:
        visitor = await db.get(GuestVisitor, issued.principal.visitor_id)
        assert visitor is not None
        assert visitor.token_hash == hashlib.sha256(raw.encode()).hexdigest()
        assert visitor.current_session_id == issued.session.id
        assert await db.scalar(select(func.count()).select_from(User)) == 1
        assert await db.scalar(select(func.count()).select_from(ChatSession)) == 1
        assert await db.scalar(select(func.count()).select_from(GuestVisitor)) == 1


async def test_existing_cookie_reuses_identity_and_session_without_secret_rotation(guest_db):
    service = GuestIdentityService(guest_db)
    first = await service.bootstrap(None)
    second = await GuestIdentityService(guest_db).bootstrap(first.token_to_set)
    assert second.token_to_set is None
    assert second.principal.visitor_id == first.principal.visitor_id
    assert second.principal.user.id == first.principal.user.id
    assert second.session.id == first.session.id
    assert second.session.session_key == first.session.session_key
    assert second.principal.expires_at == first.principal.expires_at


async def test_two_cookie_principals_are_isolated_and_secret_is_authority(guest_db):
    service = GuestIdentityService(guest_db)
    first, second = await service.bootstrap(None), await service.bootstrap(None)
    assert first.principal.visitor_id != second.principal.visitor_id
    assert first.principal.user.id != second.principal.user.id
    assert first.session.id != second.session.id
    assert first.token_to_set != second.token_to_set
    assert (await service.authenticate(first.token_to_set)).user.id == first.principal.user.id
    for public_identifier in (
        first.principal.visitor_id,
        str(first.session.id),
        first.session.session_key,
    ):
        with pytest.raises(GuestAuthenticationError):
            await service.authenticate(public_identifier)


@pytest.mark.parametrize("raw", [None, "", "wrong", "a" * 43, "a" * 4096, "bad\nvalue"])
async def test_authentication_invalid_credentials_fail_closed_and_create_nothing(guest_db, raw):
    with pytest.raises(GuestAuthenticationError, match="unavailable"):
        await GuestIdentityService(guest_db).authenticate(raw)
    async with guest_db() as db:
        assert await db.scalar(select(func.count()).select_from(User)) == 0
        assert await db.scalar(select(func.count()).select_from(GuestVisitor)) == 0


async def test_expired_authentication_rejects_but_bootstrap_issues_a_fresh_identity(guest_db):
    now = datetime(2026, 10, 28, tzinfo=UTC)
    first = await GuestIdentityService(
        guest_db, ttl=timedelta(seconds=10), clock=lambda: now
    ).bootstrap(None)
    service = GuestIdentityService(guest_db, clock=lambda: now + timedelta(seconds=10))
    with pytest.raises(GuestAuthenticationError):
        await service.authenticate(first.token_to_set)
    second = await service.bootstrap(first.token_to_set)
    assert second.token_to_set is not None
    assert second.principal.visitor_id != first.principal.visitor_id
    async with guest_db() as db:
        assert await db.scalar(select(func.count()).select_from(GuestVisitor)) == 2


async def test_inactive_guest_is_not_authenticated(guest_db):
    service = GuestIdentityService(guest_db)
    issued = await service.bootstrap(None)
    async with guest_db() as db:
        await db.execute(
            update(User).where(User.id == issued.principal.user.id).values(is_active=False)
        )
        await db.commit()
    with pytest.raises(GuestAuthenticationError):
        await service.authenticate(issued.token_to_set)


async def test_deleted_conversation_reopens_for_same_cookie_without_reusing_key(guest_db):
    service = GuestIdentityService(guest_db)
    first = await service.bootstrap(None)
    async with guest_db() as db:
        chat = await db.get(ChatSession, first.session.id)
        assert chat is not None
        await db.delete(chat)
        await db.commit()
    reopened = await service.bootstrap(first.token_to_set)
    assert reopened.token_to_set is None
    assert reopened.principal.visitor_id == first.principal.visitor_id
    assert reopened.principal.user.id == first.principal.user.id
    assert reopened.session.session_key != first.session.session_key
    async with guest_db() as db:
        assert await db.scalar(select(func.count()).select_from(ChatSession)) == 1


async def test_foreign_pointer_is_not_adopted_as_guest_conversation(guest_db):
    service = GuestIdentityService(guest_db)
    first, second = await service.bootstrap(None), await service.bootstrap(None)
    async with guest_db() as db:
        await db.execute(
            update(GuestVisitor)
            .where(GuestVisitor.id == first.principal.visitor_id)
            .values(current_session_id=second.session.id)
        )
        await db.commit()
    repaired = await service.bootstrap(first.token_to_set)
    assert repaired.session.id not in {first.session.id, second.session.id}
    assert repaired.session.user_id == first.principal.user.id


@pytest.mark.parametrize(
    "values",
    [
        {"is_guest": True, "email": "fake@guest.invalid", "hashed_password": None},
        {"is_guest": True, "email": None, "hashed_password": "fake"},
        {"is_guest": True, "email": None, "hashed_password": None, "is_admin": True},
        {"is_guest": False, "email": None, "hashed_password": "registered"},
        {"is_guest": False, "email": "registered@test.invalid", "hashed_password": None},
    ],
)
async def test_database_enforces_guest_and_registered_identity_shapes(guest_db, values):
    async with guest_db() as db:
        db.add(User(**values))
        with pytest.raises(IntegrityError):
            await db.commit()


async def test_registered_repositories_and_all_token_minting_exclude_guest(guest_db):
    issued = await GuestIdentityService(guest_db).bootstrap(None)
    async with guest_db() as db:
        repo = UserRepository(db)
        assert await repo.get_by_id(issued.principal.user.id) is None
        assert await repo.get_with_sessions(issued.principal.user.id) is None
        assert await repo.list_users() == []
        auth = AuthenticationService(repo)
        with pytest.raises(AuthenticationError):
            await auth.create_access_token(issued.principal.user)
        with pytest.raises(AuthenticationError):
            await auth.create_refresh_token(issued.principal.user)
        forged_account_refresh = create_refresh_token(
            {"user_id": issued.principal.user.id, "sub": "not-an-account"}
        )
        with pytest.raises(AuthenticationError):
            await auth.refresh_access_token(forged_account_refresh)


async def test_guest_creation_rolls_back_all_three_records_on_insert_failure(guest_db):
    async with guest_db() as db:
        bind = db.get_bind()

    def refuse_visitor(_connection, _cursor, statement, _parameters, _context, _many):
        if statement.lstrip().startswith("INSERT INTO guest_visitors"):
            raise RuntimeError("synthetic guest insert refusal")

    event.listen(bind, "before_cursor_execute", refuse_visitor)
    try:
        with pytest.raises(RuntimeError, match="synthetic"):
            await GuestIdentityService(guest_db).bootstrap(None)
    finally:
        event.remove(bind, "before_cursor_execute", refuse_visitor)
    async with guest_db() as db:
        assert await db.scalar(select(func.count()).select_from(User)) == 0
        assert await db.scalar(select(func.count()).select_from(ChatSession)) == 0
        assert await db.scalar(select(func.count()).select_from(GuestVisitor)) == 0


async def test_existing_cookie_database_failure_does_not_issue_a_new_identity(guest_db):
    service = GuestIdentityService(guest_db)
    first = await service.bootstrap(None)
    async with guest_db() as db:
        bind = db.get_bind()

    def refuse_read(_connection, _cursor, statement, _parameters, _context, _many):
        if statement.lstrip().startswith("SELECT") and "FROM guest_visitors" in statement:
            raise RuntimeError("synthetic credential read refusal")

    event.listen(bind, "before_cursor_execute", refuse_read)
    try:
        with pytest.raises(RuntimeError, match="synthetic"):
            await service.bootstrap(first.token_to_set)
    finally:
        event.remove(bind, "before_cursor_execute", refuse_read)
    async with guest_db() as db:
        assert await db.scalar(select(func.count()).select_from(User)) == 1
        assert await db.scalar(select(func.count()).select_from(GuestVisitor)) == 1


async def test_token_collision_rolls_back_new_identity_without_adopting_the_existing_one(
    guest_db, monkeypatch
):
    service = GuestIdentityService(guest_db)
    first = await service.bootstrap(None)
    monkeypatch.setattr(
        "app.services.guest_identity.secrets.token_urlsafe", lambda _size: first.token_to_set
    )
    with pytest.raises(IntegrityError):
        await service.bootstrap(None)
    assert (await service.authenticate(first.token_to_set)).visitor_id == first.principal.visitor_id
    async with guest_db() as db:
        assert await db.scalar(select(func.count()).select_from(User)) == 1
        assert await db.scalar(select(func.count()).select_from(ChatSession)) == 1
        assert await db.scalar(select(func.count()).select_from(GuestVisitor)) == 1


async def test_visitor_record_cannot_turn_a_registered_user_into_a_guest_principal(guest_db):
    raw = "z" * 43
    async with guest_db() as db:
        user = User(email="registered@unit.invalid", hashed_password="!", is_guest=False)
        db.add(user)
        await db.flush()
        db.add(
            GuestVisitor(
                user_id=user.id,
                token_hash=hashlib.sha256(raw.encode()).hexdigest(),
                expires_at=datetime(2099, 1, 1, tzinfo=UTC),
            )
        )
        await db.commit()
    with pytest.raises(GuestAuthenticationError):
        await GuestIdentityService(guest_db).authenticate(raw)


async def test_guest_conversation_can_hold_real_durable_messages(guest_db):
    issued = await GuestIdentityService(guest_db).bootstrap(None)
    async with guest_db() as db:
        db.add(
            Message(
                session_id=issued.session.id,
                role=MessageRole.USER,
                status=MessageStatus.COMPLETED,
                content="Guest owns this message",
            )
        )
        await db.commit()
    principal = await GuestIdentityService(guest_db).authenticate(issued.token_to_set)
    assert principal.user.id == issued.session.user_id


@pytest.mark.parametrize("ttl", [timedelta(0), timedelta(seconds=-1)])
def test_guest_ttl_must_be_positive(guest_db, ttl):
    with pytest.raises(ValueError, match="positive"):
        GuestIdentityService(guest_db, ttl=ttl)
