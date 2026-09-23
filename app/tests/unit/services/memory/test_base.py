"""Tests for memory management base interface"""

from typing import Any
from unittest.mock import Mock

import pytest

from app.services.memory.base import MemoryContent, MemoryStrategy, MessageContent


class _StubStrategy(MemoryStrategy):
    """Minimal concrete strategy so the ABC's own method bodies can be exercised."""

    async def get_context(
        self,
        session_id: int,
        max_tokens: int | None = None,
        current_query: str | None = None,
    ) -> list[dict[str, Any]]:
        # safe-super: calling the ABC's own body (raises
        # NotImplementedError) is exactly the behavior under test.
        return await super().get_context(  # type: ignore[safe-super]
            session_id, max_tokens=max_tokens, current_query=current_query
        )

    async def add_message(self, session_id: int, message: MessageContent) -> None:
        return await super().add_message(session_id, message)  # type: ignore[safe-super]

    async def clear_session(self, session_id: int) -> None:
        return await super().clear_session(session_id)  # type: ignore[safe-super]


class TestMemoryStrategy:
    """Test Memory strategy base class"""

    async def test_base_class_not_implemented_get_context(self):
        """Test that the ABC body for get_context raises NotImplementedError"""
        strategy = _StubStrategy(message_repo=Mock())

        with pytest.raises(NotImplementedError):
            await strategy.get_context(session_id=1)

    async def test_base_class_not_implemented_add_message(self):
        """Test that the ABC body for add_message raises NotImplementedError"""
        strategy = _StubStrategy(message_repo=Mock())
        message = MessageContent(role="user", content="Hello")

        with pytest.raises(NotImplementedError):
            await strategy.add_message(session_id=1, message=message)

    async def test_base_class_not_implemented_clear_session(self):
        """Test that the ABC body for clear_session raises NotImplementedError"""
        strategy = _StubStrategy(message_repo=Mock())

        with pytest.raises(NotImplementedError):
            await strategy.clear_session(session_id=1)

    async def test_abc_cannot_be_instantiated_directly(self):
        """Test that the abstract class itself cannot be instantiated"""
        with pytest.raises(TypeError):
            MemoryStrategy(message_repo=Mock())  # type: ignore[abstract]

    def test_base_class_initialization(self):
        """Test base class initialization with message repo"""
        # Arrange
        from app.repositories.message_repository import MessageRepository

        mock_repo = Mock(spec=MessageRepository)

        # Act
        strategy = _StubStrategy(message_repo=mock_repo)

        # Assert
        assert strategy.message_repo == mock_repo

    async def test_estimate_tokens_counts_dict_messages(self):
        """estimate_tokens is concrete and keys into MessageContent dicts"""
        # Arrange
        strategy = _StubStrategy(message_repo=Mock())
        messages = [
            MessageContent(role="user", content="Hello!"),  # 6 chars -> 1 token
            MessageContent(role="assistant", content="Hi"),  # 2 chars -> 0 tokens
        ]

        # Act
        count = await strategy.estimate_tokens(messages)

        # Assert
        assert count == 1

    async def test_estimate_tokens_empty_list(self):
        """estimate_tokens returns 0 for an empty list"""
        strategy = _StubStrategy(message_repo=Mock())

        assert await strategy.estimate_tokens([]) == 0

    async def test_truncate_by_tokens_keeps_newest(self):
        """truncate_by_tokens keeps the most recent messages that fit"""
        # Arrange
        strategy = _StubStrategy(message_repo=Mock())
        messages = [
            MessageContent(role="user", content="a" * 40),  # 10 tokens
            MessageContent(role="assistant", content="b" * 40),  # 10 tokens
            MessageContent(role="user", content="c" * 40),  # 10 tokens
        ]

        # Act - only room for the two newest
        truncated = await strategy.truncate_by_tokens(messages, max_tokens=20)

        # Assert - oldest dropped, newest two kept, order preserved
        assert [m["content"] for m in truncated] == ["b" * 40, "c" * 40]

    async def test_truncate_by_tokens_empty(self):
        """truncate_by_tokens returns [] for empty input"""
        strategy = _StubStrategy(message_repo=Mock())

        assert await strategy.truncate_by_tokens([], max_tokens=10) == []


class TestMemoryContent:
    """Test memory content data structures"""

    def test_create_memory_content(self):
        """Test creating memory content"""
        # Act
        content = MemoryContent(
            messages=[], summary="Conversation summary", metadata={"token_count": 100}
        )

        # Assert
        assert content.messages == []
        assert content.summary == "Conversation summary"
        metadata = content.metadata
        assert metadata is not None
        assert metadata["token_count"] == 100

    def test_create_memory_content_defaults(self):
        """Test creating memory content with defaults"""
        # Act
        content = MemoryContent(
            messages=[],
        )

        # Assert
        assert content.messages == []
        assert content.summary is None
        assert content.metadata is None
