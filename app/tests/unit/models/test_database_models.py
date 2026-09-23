"""Database model contracts: defaults, uniqueness, relationships.

Rewritten against the async SQLAlchemy 2.0 stack: the original file
predated the async migration (sync ``session.query`` calls, a removed
``Intent.QUESTION`` member) and had silently rotted outside the
curated testpaths. These tests pin the column-level contracts the
repositories and API layers build on: nullable/unique constraints,
server-side defaults, and FK relationships.
"""

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.database.document import Document
from app.models.database.message import Message
from app.models.database.session import ChatSession
from app.models.database.user import User
from app.models.enums.message import MessageRole, MessageStatus


class TestBaseModel:
    """Test base database model"""

    def test_base_model_has_timestamps(self):
        """TimestampMixin exposes created_at / updated_at columns"""
        from app.models.database.base import TimestampMixin

        assert hasattr(TimestampMixin, "created_at")
        assert hasattr(TimestampMixin, "updated_at")


class TestUserModel:
    """Test User model"""

    async def test_user_creation(self, db_session: AsyncSession) -> None:
        """Creating a user fills PK and timestamps"""
        user = User(
            email="test@example.com",
            hashed_password="hashed_password_here",
            full_name="Test User",
            is_active=True,
            is_admin=False,
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        assert user.id is not None
        assert user.email == "test@example.com"
        assert user.hashed_password == "hashed_password_here"
        assert user.full_name == "Test User"
        assert user.is_active is True
        assert user.is_admin is False
        assert user.created_at is not None
        assert user.updated_at is not None

    async def test_user_email_unique(self, db_session: AsyncSession) -> None:
        """Duplicate emails are rejected at flush time"""
        db_session.add(User(email="test@example.com", hashed_password="p1"))
        db_session.add(User(email="test@example.com", hashed_password="p2"))

        with pytest.raises(IntegrityError):
            await db_session.commit()
        await db_session.rollback()

    async def test_user_email_required(self, db_session: AsyncSession) -> None:
        """Missing email violates NOT NULL"""
        db_session.add(User(hashed_password="hashed_password_here"))

        with pytest.raises(IntegrityError):
            await db_session.commit()
        await db_session.rollback()

    async def test_user_defaults(self, db_session: AsyncSession) -> None:
        """is_active defaults True, is_admin/full_name default off/None"""
        user = User(email="test@example.com", hashed_password="hashed_password_here")
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        assert user.is_active is True
        assert user.is_admin is False
        assert user.full_name is None


class TestChatSessionModel:
    """Test ChatSession model"""

    async def _make_user(self, db_session: AsyncSession) -> User:
        user = User(email="test@example.com", hashed_password="hashed_password_here")
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)
        return user

    async def test_session_creation(self, db_session: AsyncSession) -> None:
        """Creating a session fills PK, timestamps, and FK"""
        user = await self._make_user(db_session)

        session = ChatSession(
            user_id=user.id,
            title="Test Session",
            memory_type="sliding_window",
            context_window=10,
        )
        db_session.add(session)
        await db_session.commit()
        await db_session.refresh(session)

        assert session.id is not None
        assert session.user_id == user.id
        assert session.title == "Test Session"
        assert session.memory_type == "sliding_window"
        assert session.context_window == 10
        assert session.created_at is not None
        assert session.updated_at is not None

    async def test_session_relationship_with_user(self, db_session: AsyncSession) -> None:
        """The session.user relationship resolves via eager loading"""
        from sqlalchemy import func

        user = await self._make_user(db_session)

        session = ChatSession(user_id=user.id, title="Test Session")
        db_session.add(session)
        await db_session.commit()

        loaded = await db_session.scalar(
            select(ChatSession)
            .options(selectinload(ChatSession.user))
            .where(ChatSession.user_id == user.id)
        )
        assert loaded is not None
        assert loaded.user.id == user.id

        count = await db_session.scalar(
            select(func.count()).select_from(ChatSession).where(ChatSession.user_id == user.id)
        )
        assert count == 1

    async def test_session_defaults(self, db_session: AsyncSession) -> None:
        """title/memory_type/context_window defaults match the memory layer"""
        user = await self._make_user(db_session)

        session = ChatSession(user_id=user.id)
        db_session.add(session)
        await db_session.commit()
        await db_session.refresh(session)

        assert session.title == "New Chat"
        assert session.memory_type == "sliding_window"
        assert session.context_window == 10


class TestMessageModel:
    """Test Message model"""

    async def _make_session(self, db_session: AsyncSession) -> ChatSession:
        user = User(email="test@example.com", hashed_password="hashed_password_here")
        db_session.add(user)
        await db_session.commit()
        session = ChatSession(user_id=user.id)
        db_session.add(session)
        await db_session.commit()
        await db_session.refresh(session)
        return session

    async def test_message_creation(self, db_session: AsyncSession) -> None:
        """Creating a message persists role/content/intent (string column)"""
        session = await self._make_session(db_session)

        message = Message(
            session_id=session.id,
            role=MessageRole.USER,
            content="Hello, world!",
            intent="question",
            status=MessageStatus.COMPLETED,
        )
        db_session.add(message)
        await db_session.commit()
        await db_session.refresh(message)

        assert message.id is not None
        assert message.session_id == session.id
        assert message.role == MessageRole.USER
        assert message.content == "Hello, world!"
        assert message.intent == "question"
        assert message.status == MessageStatus.COMPLETED
        assert message.created_at is not None

    async def test_message_defaults(self, db_session: AsyncSession) -> None:
        """status defaults COMPLETED; intent/token_count/metadata stay None"""
        session = await self._make_session(db_session)

        message = Message(
            session_id=session.id,
            role=MessageRole.USER,
            content="Test message",
        )
        db_session.add(message)
        await db_session.commit()
        await db_session.refresh(message)

        assert message.status == MessageStatus.COMPLETED
        assert message.intent is None
        assert message.token_count is None
        assert message.message_metadata is None


class TestDocumentModel:
    """Test Document model"""

    async def test_document_creation(self, db_session: AsyncSession) -> None:
        """Creating a document fills PK and metadata columns"""
        document = Document(
            external_doc_id="doc_123",
            title="Test Document",
            source="/path/to/doc.pdf",
            doc_type="pdf",
            chunk_count=10,
        )
        db_session.add(document)
        await db_session.commit()
        await db_session.refresh(document)

        assert document.id is not None
        assert document.external_doc_id == "doc_123"
        assert document.title == "Test Document"
        assert document.source == "/path/to/doc.pdf"
        assert document.doc_type == "pdf"
        assert document.chunk_count == 10
        assert document.is_active is True

    async def test_document_external_id_unique(self, db_session: AsyncSession) -> None:
        """Duplicate external_doc_id is rejected at flush time"""
        db_session.add(
            Document(external_doc_id="doc_123", title="Document 1", source="s1", doc_type="pdf")
        )
        db_session.add(
            Document(external_doc_id="doc_123", title="Document 2", source="s2", doc_type="txt")
        )

        with pytest.raises(IntegrityError):
            await db_session.commit()
        await db_session.rollback()

    async def test_document_defaults(self, db_session: AsyncSession) -> None:
        """chunk_count defaults 0, is_active True, doc_metadata None"""
        document = Document(
            external_doc_id="doc_123",
            title="Test Document",
            source="source",
            doc_type="pdf",
        )
        db_session.add(document)
        await db_session.commit()
        await db_session.refresh(document)

        assert document.chunk_count == 0
        assert document.is_active is True
        assert document.doc_metadata is None
