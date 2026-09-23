"""Integration contract: ChatService adapter over the dialogue graph.

process_message maps the graph result into ChatResponse, threads the
persister and knowledge-gap recorder side effects, and surfaces the
executed-tool audit trail for irreversible support actions. These
tests run the adapter against a scripted in-process graph; the
node-level pipeline itself is pinned in
app/tests/unit/services/dialogue/.
"""

from typing import Any
from unittest.mock import AsyncMock

from app.services.chat.chat_service import ChatService


class _ScriptedGraph:
    """Fake compiled graph returning a canned result dict."""

    def __init__(self, result: dict[str, Any]) -> None:
        self._result = result
        self.calls: list[tuple[dict[str, Any], dict[str, Any]]] = []

    async def ainvoke(self, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((state, config))
        return self._result


GRAPH_RESULT: dict[str, Any] = {
    "response": "退款已发起，预计 1-3 个工作日到账。",
    "intent": "refund",
    "sources": ["doc1", "doc2"],
    "confidence": 0.91,
    "pending_slots": [],
    "filled_slots": {"order_id": "A1"},
    "retrieved_docs": [{"id": "doc1"}],
    # Durable audit trail: refunds must stay traceable.
    "executed_tools": [{"name": "create_refund", "args": {"order_id": "A1"}}],
}


class TestProcessMessageContract:
    async def test_graph_result_mapped_to_chat_response(self):
        service = ChatService(graph=_ScriptedGraph(GRAPH_RESULT))

        response = await service.process_message(session_id=7, message="我要退款", user_id=42)

        assert response.content == GRAPH_RESULT["response"]
        assert response.session_id == 7
        assert response.intent == "refund"
        assert response.sources == ["doc1", "doc2"]
        assert response.metadata is not None
        assert response.metadata["executed_tools"] == GRAPH_RESULT["executed_tools"]
        assert response.metadata["confidence"] == 0.91
        assert response.metadata["filled_slots"] == {"order_id": "A1"}

    async def test_state_and_thread_forwarded_to_graph(self):
        graph = _ScriptedGraph(GRAPH_RESULT)
        service = ChatService(graph=graph)

        await service.process_message(session_id=7, message="我要退款", user_id=42)

        state, config = graph.calls[0]
        assert state == {"message": "我要退款", "session_id": 7, "user_id": 42}
        assert config["configurable"]["thread_id"] == "7"

    async def test_persister_receives_full_turn(self):
        graph = _ScriptedGraph(GRAPH_RESULT)
        persister = AsyncMock()
        service = ChatService(graph=graph, persister=persister)

        await service.process_message(session_id=7, message="我要退款", user_id=42)

        persister.persist_turn.assert_awaited_once()
        kwargs = persister.persist_turn.await_args.kwargs
        assert kwargs["user_message"] == "我要退款"
        assert kwargs["response"] == GRAPH_RESULT["response"]
        assert kwargs["intent"] == "refund"
        assert kwargs["sources"] == ["doc1", "doc2"]
        assert kwargs["metadata"]["executed_tools"]

    async def test_gap_recorder_receives_retrieval_inputs(self):
        graph = _ScriptedGraph(GRAPH_RESULT)
        recorder = AsyncMock()
        service = ChatService(graph=graph, gap_recorder=recorder)

        await service.process_message(session_id=7, message="我要退款", user_id=42)

        recorder.record_if_gap.assert_awaited_once()
        kwargs = recorder.record_if_gap.await_args.kwargs
        assert kwargs["query"] == "我要退款"
        assert kwargs["retrieved_docs"] == GRAPH_RESULT["retrieved_docs"]
        assert kwargs["session_id"] == 7
        assert kwargs["user_id"] == 42

    async def test_no_optional_services_is_fine(self):
        service = ChatService(graph=_ScriptedGraph(GRAPH_RESULT))

        response = await service.process_message(session_id=7, message="我要退款", user_id=42)

        assert response.content == GRAPH_RESULT["response"]

    async def test_sparse_graph_result_defaults_safely(self):
        service = ChatService(graph=_ScriptedGraph({}))

        response = await service.process_message(session_id=7, message="你好", user_id=42)

        assert response.content == ""
        assert response.intent == "unknown"
        assert response.sources is None
        assert response.metadata is not None
        assert "executed_tools" not in response.metadata
