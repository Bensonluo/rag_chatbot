"""Tests for sliding window memory strategy"""

from datetime import datetime
from unittest.mock import AsyncMock, Mock

import pytest

from app.services.memory.base import MessageContent


def _repo_with_messages(messages):
    """MessageRepository mock whose get_recent_messages returns the newest N."""
    repo = Mock()
    repo.get_recent_messages = AsyncMock(side_effect=lambda session_id, limit: messages[-limit:])
    return repo


class TestSlidingWindowMemory:
    """Test sliding window memory strategy"""

    @pytest.mark.asyncio
    async def test_get_context_returns_recent_messages(self):
        """Test that get_context returns last N messages"""
        # Arrange
        from app.models.database.message import Message
        from app.services.memory.sliding_window import SlidingWindowMemory

        messages = [
            Message(id=i, role="user", content=f"Msg {i}", created_at=datetime.now())
            for i in range(1, 6)
        ]
        mock_repo = _repo_with_messages(messages)
        memory = SlidingWindowMemory(message_repo=mock_repo, window_size=3)

        # Act
        context = await memory.get_context(session_id=1)

        # Assert - window size respected, newest messages kept, dict format
        assert len(context) == 3
        assert [m["content"] for m in context] == ["Msg 3", "Msg 4", "Msg 5"]
        mock_repo.get_recent_messages.assert_called_once_with(session_id=1, limit=3)

    @pytest.mark.asyncio
    async def test_add_message_stores_in_database(self):
        """Test that add_message stores the dict message as an ORM row"""
        # Arrange
        from app.services.memory.sliding_window import SlidingWindowMemory

        mock_repo = Mock()
        mock_repo.create = AsyncMock()
        memory = SlidingWindowMemory(message_repo=mock_repo, window_size=3)

        message = MessageContent(role="user", content="New message")

        # Act
        await memory.add_message(session_id=1, message=message)

        # Assert
        mock_repo.create.assert_called_once()
        stored = mock_repo.create.call_args.args[0]
        assert stored.session_id == 1
        assert stored.role.value == "user"
        assert stored.content == "New message"

    @pytest.mark.asyncio
    async def test_clear_session_deletes_messages(self):
        """Test that clear_session deletes all messages"""
        # Arrange
        from app.services.memory.sliding_window import SlidingWindowMemory

        mock_repo = Mock()
        mock_repo.delete_by_session = AsyncMock()
        memory = SlidingWindowMemory(message_repo=mock_repo, window_size=3)

        # Act
        await memory.clear_session(session_id=1)

        # Assert
        mock_repo.delete_by_session.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_truncate_by_max_tokens(self):
        """Test truncating messages to fit max tokens"""
        # Arrange
        from app.services.memory.sliding_window import SlidingWindowMemory

        mock_repo = Mock()
        memory = SlidingWindowMemory(message_repo=mock_repo, window_size=10)

        # Each message body is 40 chars -> 10 estimated tokens
        messages = [MessageContent(role="user", content=f"{i:02d}" + "x" * 38) for i in range(5)]

        # Act - only room for the two newest
        truncated = await memory.truncate_by_tokens(messages, max_tokens=20)

        # Assert - keeps the most recent messages that fit
        assert len(truncated) == 2
        assert truncated[0]["content"].startswith("03")
        assert truncated[1]["content"].startswith("04")

    @pytest.mark.asyncio
    async def test_get_context_with_max_tokens(self):
        """Test get_context respects max_tokens parameter"""
        # Arrange
        from app.models.database.message import Message
        from app.services.memory.sliding_window import SlidingWindowMemory

        # "Message number {i}" is 16-17 chars -> 4 estimated tokens each
        messages = [
            Message(id=i, role="user", content=f"Message number {i}", created_at=datetime.now())
            for i in range(1, 6)
        ]
        mock_repo = _repo_with_messages(messages)
        memory = SlidingWindowMemory(message_repo=mock_repo, window_size=10)

        # Act - 9 tokens fits exactly two 4-token messages (8), not three (12)
        context = await memory.get_context(session_id=1, max_tokens=9)

        # Assert
        assert len(context) == 2
        assert context[-1]["content"] == "Message number 5"

    @pytest.mark.asyncio
    async def test_estimate_tokens(self):
        """Test token estimation over dict messages"""
        # Arrange
        from app.services.memory.sliding_window import SlidingWindowMemory

        mock_repo = Mock()
        memory = SlidingWindowMemory(message_repo=mock_repo, window_size=10)

        messages = [MessageContent(role="user", content="Hello world!")]

        # Act
        count = await memory.estimate_tokens(messages)

        # Assert
        # "Hello world!" is 12 chars, 12 // 4 = 3 tokens
        assert count == 3

    def test_initialization(self):
        """Test strategy initialization"""
        # Arrange
        from app.services.memory.sliding_window import SlidingWindowMemory

        mock_repo = Mock()

        # Act
        memory = SlidingWindowMemory(message_repo=mock_repo, window_size=15)

        # Assert
        assert memory.window_size == 15
        assert memory.message_repo == mock_repo

    def test_initialization_default_window_size(self):
        """Test default window size"""
        # Arrange
        from app.services.memory.sliding_window import SlidingWindowMemory

        mock_repo = Mock()

        # Act
        memory = SlidingWindowMemory(message_repo=mock_repo)

        # Assert
        assert memory.window_size == 10  # Default value

    @pytest.mark.asyncio
    async def test_empty_session_context(self):
        """Test getting context for empty session"""
        # Arrange
        from app.services.memory.sliding_window import SlidingWindowMemory

        mock_repo = _repo_with_messages([])
        memory = SlidingWindowMemory(message_repo=mock_repo, window_size=10)

        # Act
        context = await memory.get_context(session_id=1)

        # Assert
        assert context == []
