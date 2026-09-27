"""Turn finalization parity and history-clear semantics (review #8).

Two gaps. (a) The stream path persisted only text and LLM counters —
no intent, sources, executed tools — and abnormal exits (client
disconnect, graph error, budget exhaustion) stored partial content
under the default COMPLETED status, so the audit trail could not tell
a finished turn from a cut-off one. (b) clear_chat_history only wiped
the legacy memory strategy: a staged irreversible action lived on in
the dialogue checkpoint, and a later bare "确认" would execute a
refund the user believed they had discarded by resetting the chat.

Contracts under test: a completed stream persists exactly what the
non-stream path persists; an interrupted stream persists its partial
content as FAILED; clearing history resets the whole dialogue — the
checkpoint thread dies with the messages.
"""

from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

from langgraph.checkpoint.memory import MemorySaver

from app.models.enums.intent import Intent
from app.models.enums.message import MessageStatus
from app.services.chat.chat_service import STREAM_ERROR, ChatService
from app.services.dialogue.graph import build_dialogue_graph
from app.services.llm.base import LLMMessage, LLMResponse, LLMServiceBase


class _ChunkedLLM(LLMServiceBase):
    """Streams a fixed chunk sequence; non-stream joins them."""

    def __init__(self, chunks: list[str]) -> None:
        super().__init__(api_key="test", model="test")
        self._chunks = chunks

    async def generate(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        return LLMResponse(content="".join(self._chunks), model=self.model)

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        for chunk in self._chunks:
            yield chunk

    def estimate_tokens(self, text: str) -> int:
        return len(text)

    async def count_tokens(self, messages: list[LLMMessage]) -> int:
        return sum(len(m.content) for m in messages)


def _detector_returning(intent: Intent) -> Mock:
    probe = Mock()
    probe.intent = intent
    probe.confidence = 0.9
    detector = Mock()
    detector.detect_with_confidence = AsyncMock(return_value=probe)
    return detector


def _fake_hybrid_search() -> Mock:
    hit = SimpleNamespace(document_id="doc-1", content="退货政策全文", score=0.9, metadata=None)
    search = Mock()
    search.search = AsyncMock(return_value=[hit])
    return search


def _recorder() -> Mock:
    persister = Mock()
    persister.persist_turn = AsyncMock()
    return persister


def _rag_chat(persister: Mock) -> ChatService:
    return ChatService(
        graph=build_dialogue_graph(
            intent_detector=_detector_returning(Intent.POLICY),
            slot_filler=None,
            tool_registry=Mock(),
            retrieval_pipeline={"hybrid_search": _fake_hybrid_search()},
            llm_service=_ChunkedLLM(["退货政策回答完毕。"]),
        ),
        persister=persister,
    )


# ── Part A: streaming persistence parity ────────────────────────────────────


class TestStreamPersistenceParity:
    async def test_completed_stream_persists_full_turn_metadata(self) -> None:
        """A normally completed stream must persist the same turn
        record as the non-stream path: intent, sources, executed-tool
        audit trail, and COMPLETED status."""
        persister = _recorder()
        chat = _rag_chat(persister)

        chunks = [c async for c in chat.process_message_stream(1, "退货政策是什么", 0)]
        assert STREAM_ERROR not in chunks

        kwargs = persister.persist_turn.await_args.kwargs
        assert kwargs["intent"] == "policy"
        assert kwargs["sources"] == ["doc-1"]
        assert kwargs["metadata"].get("turn_id"), "turn id must survive into the audit trail"
        assert kwargs["status"] is MessageStatus.COMPLETED

    async def test_disconnected_stream_persists_partial_as_failed(self) -> None:
        """Client disconnect mid-stream: the partial content the user
        saw is still persisted, but marked FAILED — the audit trail
        must distinguish a cut-off turn from a completed one."""
        persister = _recorder()
        chat = ChatService(
            graph=build_dialogue_graph(
                intent_detector=_detector_returning(Intent.POLICY),
                slot_filler=None,
                tool_registry=Mock(),
                retrieval_pipeline={"hybrid_search": _fake_hybrid_search()},
                llm_service=_ChunkedLLM(["已发货。", "第二段未完成"]),
            ),
            persister=persister,
        )

        stream = chat.process_message_stream(1, "退货政策是什么", 0)
        first: str | None = None
        async for chunk in stream:
            # trace events ride the same queue; only content counts
            if isinstance(chunk, str):
                first = chunk
                break
        assert first, "a real content chunk was emitted"
        await stream.aclose()

        kwargs = persister.persist_turn.await_args.kwargs
        assert kwargs["response"] == first
        assert kwargs["status"] is MessageStatus.FAILED
        assert kwargs["intent"] is None  # graph never settled a verdict

    async def test_non_stream_turn_persists_full_metadata(self) -> None:
        """Pin the non-stream reference behavior the stream path must
        match."""
        persister = _recorder()
        chat = _rag_chat(persister)

        await chat.process_message(1, "退货政策是什么", 0)

        kwargs = persister.persist_turn.await_args.kwargs
        assert kwargs["intent"] == "policy"
        assert kwargs["sources"] == ["doc-1"]


# ── Part B: clearing history resets the dialogue ────────────────────────────


class TestClearHistoryResetsDialogue:
    async def test_clear_history_removes_staged_confirmation(self) -> None:
        """The checkpoint thread dies with the messages: a staged
        irreversible action must not survive a history clear."""
        graph = build_dialogue_graph(
            intent_detector=_detector_returning(Intent.CHITCHAT),
            slot_filler=None,
            tool_registry=Mock(),
            retrieval_pipeline={},
            llm_service=_ChunkedLLM(["好的。"]),
            checkpointer=MemorySaver(),
        )
        chat = ChatService(graph=graph)
        cfg = {"configurable": {"thread_id": "9"}}
        await graph.aupdate_state(
            cfg, {"pending_confirmation": {"intent": "refund", "args": {"order_id": "ORD1001"}}}
        )
        snapshot = await graph.aget_state(cfg)
        assert snapshot.values.get("pending_confirmation"), "seed sanity"

        await chat.clear_chat_history(9)

        snapshot = await graph.aget_state(cfg)
        assert not snapshot.values.get("pending_confirmation")
        assert not snapshot.next

    async def test_confirm_after_clear_does_not_execute(self) -> None:
        """Product-level guarantee: after clearing history, a bare
        "确认" must not execute a refund staged before the clear."""
        registry = Mock()
        registry.execute = AsyncMock(
            return_value=SimpleNamespace(success=True, data={"refund_id": "RF1"})
        )
        graph = build_dialogue_graph(
            intent_detector=_detector_returning(Intent.CONFIRM),
            slot_filler=None,
            tool_registry=registry,
            retrieval_pipeline={},
            llm_service=_ChunkedLLM(["好的。"]),
            checkpointer=MemorySaver(),
        )
        chat = ChatService(graph=graph)
        cfg = {"configurable": {"thread_id": "9"}}
        await graph.aupdate_state(
            cfg, {"pending_confirmation": {"intent": "refund", "args": {"order_id": "ORD1001"}}}
        )

        await chat.clear_chat_history(9)
        response = await chat.process_message(9, "确认", 0)

        registry.execute.assert_not_awaited()
        content = cast(str, response.content)
        assert "RF1" not in content

    async def test_clear_history_survives_graphless_service(self) -> None:
        """Legacy wiring (no graph) keeps working; the memory strategy
        is still cleared."""
        memory = Mock()
        memory.clear_session = AsyncMock()
        await ChatService(graph=None, memory_strategy=memory).clear_chat_history(5)
        memory.clear_session.assert_awaited_once()
