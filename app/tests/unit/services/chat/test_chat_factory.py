"""Tests for chat service factory"""

from unittest.mock import Mock, patch

import pytest

import app.services.dialogue.graph  # noqa: F401 (patch target must be importable)
import app.services.dialogue.tools  # noqa: F401
from app.core.exceptions import ValidationError
from app.services.chat.chat_service import ChatService
from app.services.chat.factory import ChatServiceFactory


class TestChatServiceFactory:
    """Test chat service factory"""

    def test_create_chat_service(self):
        mock_llm = Mock()
        mock_memory = Mock()
        mock_intent = Mock()
        service = ChatServiceFactory.create(
            llm_service=mock_llm,
            memory_strategy=mock_memory,
            intent_detector=mock_intent,
        )
        assert isinstance(service, ChatService)
        assert service.llm_service is mock_llm
        assert service.memory_strategy is mock_memory

    def test_create_chat_service_without_retrieval(self):
        mock_llm = Mock()
        mock_memory = Mock()
        mock_intent = Mock()
        service = ChatServiceFactory.create(
            llm_service=mock_llm,
            memory_strategy=mock_memory,
            intent_detector=mock_intent,
        )
        assert service.graph is None

    @patch("app.services.chat.factory.MemoryFactory")
    @patch("app.services.chat.factory.IntentFactory")
    @patch("app.services.embeddings.EmbeddingFactory")
    def test_create_chat_service_with_defaults(self, mock_emb, mock_intent_fac, mock_mem_fac):
        mock_llm = Mock()
        mock_repo = Mock()
        mock_session_repo = Mock()
        mock_memory = Mock()
        mock_intent = Mock()
        mock_mem_fac.create.return_value = mock_memory
        mock_intent_fac.create.return_value = mock_intent
        mock_emb.create_from_settings.return_value = Mock()

        with (
            patch("app.services.dialogue.tools.create_default_tool_registry"),
            patch("app.services.dialogue.graph.build_dialogue_graph", return_value=Mock()),
        ):
            service = ChatServiceFactory.create_with_defaults(
                llm_service=mock_llm,
                message_repo=mock_repo,
                session_repo=mock_session_repo,
                memory_type="optimized",
                intent_type="hybrid",
            )
        assert isinstance(service, ChatService)
        # LLM calls flow through the budget wrapper; the provider is
        # its delegate.
        from app.services.llm.budget import BudgetedLLMService

        assert isinstance(service.llm_service, BudgetedLLMService)
        assert service.llm_service._inner is mock_llm

    @patch("app.services.chat.factory.MemoryFactory")
    @patch("app.services.chat.factory.IntentFactory")
    def test_create_with_memory_factory(self, mock_intent_fac, mock_mem_fac):
        mock_llm = Mock()
        mock_repo = Mock()
        mock_session_repo = Mock()
        mock_memory = Mock()
        mock_intent = Mock()
        mock_mem_fac.create.return_value = mock_memory
        mock_intent_fac.create.return_value = mock_intent

        with (
            patch("app.services.dialogue.tools.create_default_tool_registry"),
            patch("app.services.dialogue.graph.build_dialogue_graph", return_value=Mock()),
        ):
            service = ChatServiceFactory.create_with_defaults(
                llm_service=mock_llm,
                message_repo=mock_repo,
                session_repo=mock_session_repo,
                memory_type="sliding_window",
            )
        assert isinstance(service, ChatService)
        assert service.memory_strategy is mock_memory

    @patch("app.services.chat.factory.MemoryFactory")
    @patch("app.services.chat.factory.IntentFactory")
    def test_create_with_intent_factory(self, mock_intent_fac, mock_mem_fac):
        mock_llm = Mock()
        mock_repo = Mock()
        mock_session_repo = Mock()
        mock_memory = Mock()
        mock_intent = Mock()
        mock_mem_fac.create.return_value = mock_memory
        mock_intent_fac.create.return_value = mock_intent

        with (
            patch("app.services.dialogue.tools.create_default_tool_registry"),
            patch("app.services.dialogue.graph.build_dialogue_graph", return_value=Mock()),
        ):
            service = ChatServiceFactory.create_with_defaults(
                llm_service=mock_llm,
                message_repo=mock_repo,
                session_repo=mock_session_repo,
                intent_type="rule_based",
            )
        assert isinstance(service, ChatService)

    def test_factory_validation(self):
        with pytest.raises(ValidationError):
            ChatServiceFactory.create(
                llm_service=None,  # type: ignore[arg-type]  # deliberate invalid input
                memory_strategy=Mock(),
                intent_detector=Mock(),
            )

    @patch("app.services.chat.factory.MemoryFactory")
    @patch("app.services.chat.factory.IntentFactory")
    def test_create_graph_build_failure_graceful(self, mock_intent_fac, mock_mem_fac):
        mock_llm = Mock()
        mock_repo = Mock()
        mock_session_repo = Mock()
        mock_memory = Mock()
        mock_mem_fac.create.return_value = mock_memory
        mock_intent_fac.create.return_value = Mock()

        with patch(
            "app.services.dialogue.tools.create_default_tool_registry",
            side_effect=ImportError("no langgraph"),
        ):
            service = ChatServiceFactory.create_with_defaults(
                llm_service=mock_llm,
                message_repo=mock_repo,
                session_repo=mock_session_repo,
            )
        assert isinstance(service, ChatService)
        assert service.graph is None

    @patch("app.services.embeddings.EmbeddingFactory")
    @patch("app.services.chat.factory.MemoryFactory")
    @patch("app.services.chat.factory.IntentFactory")
    @patch("app.services.handoff.create_handoff_service", return_value=Mock())
    def test_create_with_defaults_builds_real_graph(
        self, mock_handoff, mock_intent_fac, mock_mem_fac, mock_emb
    ):
        """The graph-build try block must never die on a NameError.

        Regression guard: a shipped commit referenced ``settings`` inside
        the factory without importing it; the exception was swallowed by
        the graceful-fallback except and every deployment silently ran
        graphless (legacy pipeline only). Run the real graph builder and
        pin that the compiled LangGraph is wired in.
        """
        mock_mem_fac.create.return_value = Mock()
        mock_intent_fac.create.return_value = Mock()
        mock_emb.create_from_settings.return_value = Mock()

        service = ChatServiceFactory.create_with_defaults(
            llm_service=Mock(),
            message_repo=Mock(),
            session_repo=Mock(),
            memory_type="optimized",
            intent_type="rule_based",
        )

        assert isinstance(service, ChatService)
        assert service.graph is not None
