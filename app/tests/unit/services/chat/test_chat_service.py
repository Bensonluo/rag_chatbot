"""Tests for chat service"""

from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from app.services.chat.chat_service import STREAM_ERROR, ChatResponse, ChatService


def _make_graph(return_value: dict[str, Any]) -> Mock:
    """Create a mock compiled LangGraph that returns the given value on ainvoke."""
    graph = Mock()
    graph.ainvoke = AsyncMock(return_value=return_value)
    return graph


class TestChatService:
    """Test chat orchestration service"""

    def test_chat_service_initialization(self):
        mock_graph = Mock()
        mock_llm = Mock()
        mock_memory = Mock()
        service = ChatService(
            graph=mock_graph,
            llm_service=mock_llm,
            memory_strategy=mock_memory,
        )
        assert service.graph is mock_graph
        assert service.llm_service is mock_llm
        assert service.memory_strategy is mock_memory

    @pytest.mark.asyncio
    async def test_process_message_simple(self):
        mock_graph = _make_graph(
            {
                "response": "您好！有什么可以帮您的？",
                "intent": "greeting",
                "confidence": 0.95,
            }
        )
        service = ChatService(graph=mock_graph)
        response = await service.process_message(
            session_id=1,
            message="你好",
            user_id=1,
        )
        assert response.content == "您好！有什么可以帮您的？"
        assert response.session_id == 1
        assert response.intent == "greeting"
        mock_graph.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_message_with_retrieval(self):
        mock_graph = _make_graph(
            {
                "response": "退货政策是7天无理由退货。",
                "intent": "policy",
                "sources": ["doc1"],
                "confidence": 0.9,
            }
        )
        service = ChatService(graph=mock_graph)
        response = await service.process_message(
            session_id=1,
            message="退货政策是什么",
            user_id=1,
        )
        assert "退货" in response.content
        assert response.sources is not None
        assert len(response.sources) > 0

    @pytest.mark.asyncio
    async def test_process_message_with_memory(self):
        mock_graph = _make_graph({"response": "好的", "intent": "chitchat"})
        mock_memory = Mock()
        # Memory strategies return MessageContent dicts, not attribute objects
        msg1 = {"role": "user", "content": "My name is Alice"}
        msg2 = {"role": "assistant", "content": "Hello Alice!"}
        mock_memory.get_context = AsyncMock(return_value=[msg1, msg2])
        service = ChatService(graph=mock_graph, memory_strategy=mock_memory)
        history = await service.get_chat_history(session_id=1)
        assert len(history) == 2
        assert history[0].role == "user"

    @pytest.mark.asyncio
    async def test_process_message_with_streaming(self):
        async def mock_ainvoke(state, config):
            queue = config["configurable"]["stream_queue"]
            queue.put_nowait("您好")
            queue.put_nowait("有什么可以帮您")
            return {"response": "您好有什么可以帮您"}

        mock_graph = Mock()
        mock_graph.ainvoke = mock_ainvoke
        service = ChatService(graph=mock_graph)
        chunks = []
        async for chunk in service.process_message_stream(
            session_id=1,
            message="你好",
            user_id=1,
        ):
            chunks.append(chunk)
        assert chunks == ["您好", "有什么可以帮您"]

    @pytest.mark.asyncio
    async def test_get_chat_history(self):
        mock_graph = Mock()
        mock_memory = Mock()
        mock_memory.get_context = AsyncMock(
            return_value=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ]
        )
        service = ChatService(graph=mock_graph, memory_strategy=mock_memory)
        history = await service.get_chat_history(session_id=1)
        assert len(history) == 3
        assert history[0].content == "Hello"

    @pytest.mark.asyncio
    async def test_clear_chat_history(self):
        mock_graph = Mock()
        mock_memory = Mock()
        mock_memory.clear_session = AsyncMock()
        service = ChatService(graph=mock_graph, memory_strategy=mock_memory)
        await service.clear_chat_history(session_id=1)
        mock_memory.clear_session.assert_called_once_with(session_id=1)

    @pytest.mark.asyncio
    async def test_get_chat_history_no_memory(self):
        service = ChatService(graph=Mock())
        history = await service.get_chat_history(session_id=1)
        assert history == []

    @pytest.mark.asyncio
    async def test_process_message_returns_metadata(self):
        mock_graph = _make_graph(
            {
                "response": "请提供您的订单号",
                "intent": "refund",
                "confidence": 0.9,
                "pending_slots": ["order_id", "reason"],
                "filled_slots": {},
            }
        )
        service = ChatService(graph=mock_graph)
        response = await service.process_message(
            session_id=1,
            message="我要退款",
            user_id=1,
        )
        metadata = response.metadata
        assert metadata is not None
        assert metadata["pending_slots"] == ["order_id", "reason"]


class TestKnowledgeGapWiring:
    """ChatService forwards every turn to the gap recorder; the recorder
    (not the service) decides whether a turn is a knowledge gap."""

    @pytest.mark.asyncio
    async def test_recorder_called_with_graph_result(self):
        mock_graph = _make_graph({"response": "抱歉", "intent": "faq", "retrieved_docs": []})
        recorder = AsyncMock()
        service = ChatService(graph=mock_graph, gap_recorder=recorder)
        await service.process_message(session_id=3, message="发票怎么开", user_id=9)
        recorder.record_if_gap.assert_awaited_once_with(
            query="发票怎么开",
            intent="faq",
            retrieved_docs=[],
            session_id=3,
            user_id=9,
        )

    @pytest.mark.asyncio
    async def test_recorder_forwarded_for_non_knowledge_intent_too(self):
        # The recorder owns gap gating; the service forwards verbatim.
        mock_graph = _make_graph({"response": "您好！", "intent": "greeting"})
        recorder = AsyncMock()
        service = ChatService(graph=mock_graph, gap_recorder=recorder)
        await service.process_message(session_id=1, message="你好", user_id=1)
        recorder.record_if_gap.assert_awaited_once_with(
            query="你好",
            intent="greeting",
            retrieved_docs=None,
            session_id=1,
            user_id=1,
        )

    @pytest.mark.asyncio
    async def test_recorder_called_in_stream_path(self):
        async def mock_ainvoke(state, config):
            queue = config["configurable"]["stream_queue"]
            queue.put_nowait("抱歉")
            return {"response": "抱歉", "intent": "policy", "retrieved_docs": []}

        mock_graph = Mock()
        mock_graph.ainvoke = mock_ainvoke
        recorder = AsyncMock()
        service = ChatService(graph=mock_graph, gap_recorder=recorder)
        chunks = [
            chunk
            async for chunk in service.process_message_stream(
                session_id=5, message="退货政策", user_id=2
            )
        ]
        assert chunks == ["抱歉"]
        recorder.record_if_gap.assert_awaited_once_with(
            query="退货政策",
            intent="policy",
            retrieved_docs=[],
            session_id=5,
            user_id=2,
        )

    @pytest.mark.asyncio
    async def test_recorder_not_called_when_graph_fails_in_stream(self):
        async def mock_ainvoke(state, config):
            queue = config["configurable"]["stream_queue"]
            queue.put_nowait("部分")
            raise RuntimeError("graph exploded")


class TestChatResponse:
    """Test ChatResponse dataclass"""

    def test_chat_response_creation(self):
        response = ChatResponse(
            content="Hello!",
            session_id=1,
            intent="greeting",
            sources=["doc1", "doc2"],
            metadata={"tokens": 50},
        )
        assert response.content == "Hello!"
        assert response.session_id == 1
        assert response.intent == "greeting"
        sources = response.sources
        assert sources is not None
        assert len(sources) == 2

    def test_chat_response_without_sources(self):
        response = ChatResponse(
            content="Hi!",
            session_id=1,
            intent="greeting",
        )
        assert response.sources is None


class TestTurnAuditMetadata:
    """Agent tool executions are persisted with the turn — irreversible
    support actions must be traceable after the fact."""

    @pytest.mark.asyncio
    async def test_persists_executed_tools_in_metadata(self):
        trace = [{"tool": "process_refund", "ok": True, "args": {"order_id": "O1"}, "summary": "s"}]
        mock_graph = _make_graph(
            {
                "response": "退款已发起。",
                "intent": "refund",
                "executed_tools": trace,
            }
        )
        persister = AsyncMock()
        service = ChatService(graph=mock_graph, persister=persister)
        response = await service.process_message(session_id=1, message="退款", user_id=1)

        persisted_meta = persister.persist_turn.await_args.kwargs["metadata"]
        assert persisted_meta["executed_tools"] == trace
        metadata = response.metadata
        assert metadata is not None
        assert metadata["executed_tools"] == trace

    @pytest.mark.asyncio
    async def test_omits_executed_tools_when_none_ran(self):
        mock_graph = _make_graph({"response": "您好！", "intent": "greeting"})
        persister = AsyncMock()
        service = ChatService(graph=mock_graph, persister=persister)
        await service.process_message(session_id=1, message="你好", user_id=1)

        persisted_meta = persister.persist_turn.await_args.kwargs["metadata"]
        assert "executed_tools" not in persisted_meta


class TestStreamGuardrails:
    """Streaming hard limits: total-duration cap and failure error frame.

    A heartbeat keeps idle connections alive, so a wedged graph task
    (hung LLM upstream) would otherwise hold the connection forever.
    The stream must end with an explicit error sentinel instead — both
    on timeout and on graph failure — so the SSE layer can tell the
    client why the stream stopped.
    """

    @pytest.mark.asyncio
    async def test_stream_timeout_yields_error_and_ends(self):
        import asyncio

        async def mock_ainvoke(state, config):
            await asyncio.sleep(999)  # wedged upstream, never yields

        mock_graph = Mock()
        mock_graph.ainvoke = mock_ainvoke
        service = ChatService(graph=mock_graph)

        chunks = [
            chunk
            async for chunk in service.process_message_stream(
                session_id=1, message="你好", user_id=1, stream_max_seconds=0.05
            )
        ]

        assert chunks == [STREAM_ERROR]

    @pytest.mark.asyncio
    async def test_stream_timeout_persists_partial_turn(self):
        import asyncio

        async def mock_ainvoke(state, config):
            queue = config["configurable"]["stream_queue"]
            queue.put_nowait("部分")
            await asyncio.sleep(999)

        mock_graph = Mock()
        mock_graph.ainvoke = mock_ainvoke
        persister = AsyncMock()
        service = ChatService(graph=mock_graph, persister=persister)

        chunks = [
            chunk
            async for chunk in service.process_message_stream(
                session_id=1, message="你好", user_id=1, stream_max_seconds=0.05
            )
        ]

        assert chunks == ["部分", STREAM_ERROR]
        persister.persist_turn.assert_awaited_once()
        assert persister.persist_turn.await_args.kwargs["response"] == "部分"

    @pytest.mark.asyncio
    async def test_graph_failure_yields_error_sentinel_not_raise(self):
        async def mock_ainvoke(state, config):
            queue = config["configurable"]["stream_queue"]
            queue.put_nowait("部分")
            raise RuntimeError("graph exploded")

        mock_graph = Mock()
        mock_graph.ainvoke = mock_ainvoke
        recorder = AsyncMock()
        service = ChatService(graph=mock_graph, gap_recorder=recorder)

        chunks = [
            chunk
            async for chunk in service.process_message_stream(
                session_id=1, message="你好", user_id=1
            )
        ]

        assert chunks == ["部分", STREAM_ERROR]
        recorder.record_if_gap.assert_not_awaited()
