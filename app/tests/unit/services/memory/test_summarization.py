"""Tests for summarization memory strategy"""

from datetime import datetime
from unittest.mock import AsyncMock, Mock

import pytest

from app.services.memory.base import MessageContent


def _llm_responding(text):
    """LLM mock whose generate returns a response object with .content."""
    llm = Mock()
    llm.generate = AsyncMock(return_value=Mock(content=text))
    return llm


class TestSummarizationMemory:
    """Test summarization memory strategy"""

    @pytest.mark.asyncio
    async def test_get_context_with_summary(self):
        """Test that get_context includes summary and recent messages"""
        # Arrange
        from app.models.database.message import Message
        from app.services.memory.summarization import SummarizationMemory

        mock_repo = Mock()
        memory = SummarizationMemory(
            message_repo=mock_repo, llm_service=Mock(), summary_threshold=5, summary_interval=3
        )

        summary_msg = Message(
            id=1,
            role="system",
            content="Conversation summary: User asked about AI",
            created_at=datetime.now(),
        )
        recent_messages = [
            Message(
                id=2, role="user", content="What is machine learning?", created_at=datetime.now()
            ),
            Message(
                id=3, role="assistant", content="ML is a subset of AI", created_at=datetime.now()
            ),
        ]

        mock_repo.get_latest_summary = AsyncMock(return_value=summary_msg)
        mock_repo.get_recent_messages = AsyncMock(return_value=recent_messages)

        # Act
        context = await memory.get_context(session_id=1)

        # Assert - summary first as a system dict, then recent messages
        assert len(context) == 3
        assert context[0]["role"] == "system"
        assert "summary" in context[0]["content"].lower()
        assert [m["content"] for m in context[1:]] == [
            "What is machine learning?",
            "ML is a subset of AI",
        ]

    @pytest.mark.asyncio
    async def test_get_context_without_summary(self):
        """Test getting context when no summary exists yet"""
        # Arrange
        from app.models.database.message import Message
        from app.services.memory.summarization import SummarizationMemory

        mock_repo = Mock()
        memory = SummarizationMemory(
            message_repo=mock_repo,
            llm_service=Mock(),
            summary_threshold=10,
            summary_interval=5,
        )

        mock_repo.get_latest_summary = AsyncMock(return_value=None)
        recent_messages = [
            Message(id=1, role="user", content="Recent msg", created_at=datetime.now()),
        ]
        mock_repo.get_recent_messages = AsyncMock(return_value=recent_messages)

        # Act
        context = await memory.get_context(session_id=1)

        # Assert
        assert len(context) == 1
        assert context[0]["content"] == "Recent msg"

    @pytest.mark.asyncio
    async def test_add_message_triggers_summary_when_threshold_reached(self):
        """Test that summary is created when message count reaches threshold"""
        # Arrange
        from app.models.database.message import Message
        from app.services.memory.summarization import SummarizationMemory

        mock_repo = Mock()
        llm_service = _llm_responding("Summary of conversation")
        memory = SummarizationMemory(
            message_repo=mock_repo, llm_service=llm_service, summary_threshold=5, summary_interval=3
        )

        mock_repo.create = AsyncMock()
        mock_repo.count_messages = AsyncMock(return_value=5)
        mock_repo.get_messages_before_summary = AsyncMock(
            return_value=[
                Message(id=1, role="user", content="Hello", created_at=datetime.now()),
                Message(id=2, role="assistant", content="Hi!", created_at=datetime.now()),
            ]
        )
        mock_repo.create_summary = AsyncMock()
        mock_repo.archive_messages = AsyncMock()

        message = MessageContent(role="user", content="Test")

        # Act
        await memory.add_message(session_id=1, message=message)

        # Assert
        mock_repo.create.assert_called_once()
        llm_service.generate.assert_called_once()
        mock_repo.create_summary.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_message_below_threshold(self):
        """Test that summary is NOT created when below threshold"""
        # Arrange
        from app.services.memory.summarization import SummarizationMemory

        mock_repo = Mock()
        llm_service = _llm_responding("unused")
        memory = SummarizationMemory(
            message_repo=mock_repo,
            llm_service=llm_service,
            summary_threshold=10,
            summary_interval=5,
        )

        mock_repo.create = AsyncMock()
        mock_repo.count_messages = AsyncMock(return_value=3)

        message = MessageContent(role="user", content="Test")

        # Act
        await memory.add_message(session_id=1, message=message)

        # Assert
        llm_service.generate.assert_not_called()
        mock_repo.create_summary.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_message_stores_dict_message(self):
        """add_message accepts a MessageContent dict, not an ORM object"""
        # Arrange
        from app.services.memory.summarization import SummarizationMemory

        mock_repo = Mock()
        memory = SummarizationMemory(
            message_repo=mock_repo, llm_service=_llm_responding("unused"), summary_threshold=10
        )

        mock_repo.create = AsyncMock()
        mock_repo.count_messages = AsyncMock(return_value=1)

        # Act - a plain dict must not raise AttributeError
        await memory.add_message(session_id=7, message=MessageContent(role="user", content="Hi"))

        # Assert
        stored = mock_repo.create.call_args.args[0]
        assert stored.session_id == 7
        assert stored.role.value == "user"
        assert stored.content == "Hi"

    @pytest.mark.asyncio
    async def test_clear_session(self):
        """Test that clear_session works correctly"""
        # Arrange
        from app.services.memory.summarization import SummarizationMemory

        mock_repo = Mock()
        memory = SummarizationMemory(message_repo=mock_repo, llm_service=Mock())

        mock_repo.delete_by_session = AsyncMock()

        # Act
        await memory.clear_session(session_id=1)

        # Assert
        mock_repo.delete_by_session.assert_called_once_with(1)

    def test_initialization(self):
        """Test strategy initialization"""
        # Arrange
        from app.services.memory.summarization import SummarizationMemory

        mock_repo = Mock()

        # Act
        memory = SummarizationMemory(
            message_repo=mock_repo,
            llm_service=Mock(),
            summary_threshold=20,
            summary_interval=10,
        )

        # Assert
        assert memory.summary_threshold == 20
        assert memory.summary_interval == 10

    @pytest.mark.asyncio
    async def test_create_summary_generates_summary(self):
        """Test that _create_summary builds an LLM prompt from the old messages"""
        # Arrange
        from app.models.database.message import Message
        from app.services.memory.summarization import SummarizationMemory

        mock_repo = Mock()
        llm_service = _llm_responding("Summary: User greeted assistant")
        memory = SummarizationMemory(message_repo=mock_repo, llm_service=llm_service)

        messages = [
            Message(id=1, role="user", content="Hello", created_at=datetime.now()),
            Message(id=2, role="assistant", content="Hi there!", created_at=datetime.now()),
        ]

        mock_repo.get_messages_before_summary = AsyncMock(return_value=messages)
        mock_repo.create_summary = AsyncMock()
        mock_repo.archive_messages = AsyncMock()

        # Act
        await memory._create_summary(session_id=1)

        # Assert
        llm_service.generate.assert_called_once()
        prompt_messages = llm_service.generate.call_args.kwargs["messages"]
        assert len(prompt_messages) == 1
        assert "Hello" in prompt_messages[0].content
        assert "Hi there!" in prompt_messages[0].content

    @pytest.mark.asyncio
    async def test_create_summary_stores_as_system_message(self):
        """Test that the summary is stored as a system message"""
        # Arrange
        from app.models.database.message import Message
        from app.services.memory.summarization import SummarizationMemory

        mock_repo = Mock()
        llm_service = _llm_responding("Test summary")
        memory = SummarizationMemory(message_repo=mock_repo, llm_service=llm_service)

        mock_repo.get_messages_before_summary = AsyncMock(
            return_value=[Message(id=1, role="user", content="Old", created_at=datetime.now())]
        )
        mock_repo.create_summary = AsyncMock()
        mock_repo.archive_messages = AsyncMock()

        # Act
        await memory._create_summary(session_id=1)

        # Assert
        mock_repo.create_summary.assert_called_once()
        stored = mock_repo.create_summary.call_args.args[0]
        assert stored.role.value == "system"
        assert stored.content == "Test summary"

    @pytest.mark.asyncio
    async def test_create_summary_archives_old_messages(self):
        """Test that old messages are archived after summary"""
        # Arrange
        from app.models.database.message import Message
        from app.services.memory.summarization import SummarizationMemory

        mock_repo = Mock()
        llm_service = _llm_responding("Summary")
        memory = SummarizationMemory(message_repo=mock_repo, llm_service=llm_service)

        old_messages = [
            Message(id=1, role="user", content="Old msg 1", created_at=datetime.now()),
            Message(id=2, role="assistant", content="Old msg 2", created_at=datetime.now()),
        ]

        mock_repo.get_messages_before_summary = AsyncMock(return_value=old_messages)
        mock_repo.create_summary = AsyncMock()
        mock_repo.archive_messages = AsyncMock()

        # Act
        await memory._create_summary(session_id=1)

        # Assert - archives exactly the summarized count
        mock_repo.archive_messages.assert_called_once_with(session_id=1, count=2)

    @pytest.mark.asyncio
    async def test_create_summary_no_messages_noop(self):
        """Test that _create_summary does nothing when there are no messages"""
        # Arrange
        from app.services.memory.summarization import SummarizationMemory

        mock_repo = Mock()
        llm_service = _llm_responding("unused")
        memory = SummarizationMemory(message_repo=mock_repo, llm_service=llm_service)

        mock_repo.get_messages_before_summary = AsyncMock(return_value=[])

        # Act
        await memory._create_summary(session_id=1)

        # Assert
        llm_service.generate.assert_not_called()
        mock_repo.create_summary.assert_not_called()

    @pytest.mark.asyncio
    async def test_estimate_tokens(self):
        """Test token estimation over dict messages"""
        # Arrange
        from app.services.memory.summarization import SummarizationMemory

        memory = SummarizationMemory(message_repo=Mock(), llm_service=Mock())

        messages = [MessageContent(role="user", content="Hello world!")]

        # Act
        count = await memory.estimate_tokens(messages)

        # Assert - 12 chars // 4 = 3 tokens
        assert count == 3
