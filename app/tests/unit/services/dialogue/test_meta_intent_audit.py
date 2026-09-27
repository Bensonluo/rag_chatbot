"""Deterministic confirm executions leave an audit trail (review #9).

The agent path records every tool call in ``executed_tools`` (shape:
tool / ok / args / summary). The deterministic confirm branch in
``_handle_meta_intent`` executed the very tools that most need
auditing — irreversible refunds — without writing that field, so the
persisted turn (and the human-handoff context) showed no trace of
what the bot executed. These tests pin: the confirm path records the
same trail shape, failures are audited with ``ok=False``, and a
double "确认" never re-executes (idempotent within the demo's
staged-action lifecycle).
"""

from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

from langgraph.checkpoint.memory import MemorySaver

from app.models.enums.intent import Intent
from app.services.chat.chat_service import ChatService
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


def _registry(
    success: bool = True,
    data: dict[str, Any] | None = None,
    message: str = "",
) -> Mock:
    registry = Mock()
    registry.execute = AsyncMock(
        return_value=SimpleNamespace(
            success=success,
            data=data if data is not None else {"refund_id": "RF1"},
            message=message,
        )
    )
    registry.get_tool_for_intent = Mock(return_value=SimpleNamespace(name="process_refund"))
    return registry


def _confirm_chat(registry: Mock, persister: Mock) -> ChatService:
    return ChatService(
        graph=build_dialogue_graph(
            intent_detector=_detector_returning(Intent.CONFIRM),
            slot_filler=None,
            tool_registry=registry,
            retrieval_pipeline={},
            llm_service=_ChunkedLLM(["退款已受理。"]),
            checkpointer=MemorySaver(),
        ),
        persister=persister,
    )


def _recorder() -> Mock:
    persister = Mock()
    persister.persist_turn = AsyncMock()
    return persister


async def _stage_refund(graph: Any, thread: str = "9") -> None:
    await graph.aupdate_state(
        {"configurable": {"thread_id": thread}},
        {"pending_confirmation": {"intent": "refund", "args": {"order_id": "ORD1001"}}},
    )


class TestConfirmExecutionAudit:
    async def test_confirm_execution_records_executed_tools(self) -> None:
        """A confirmed refund executed on the deterministic path must
        leave the same tool/ok/args/summary trail the agent path
        records — in the persisted turn and the response metadata."""
        registry = _registry()
        persister = _recorder()
        chat = _confirm_chat(registry, persister)
        await _stage_refund(chat.graph)

        response = await chat.process_message(9, "确认", 1)

        kwargs = persister.persist_turn.await_args.kwargs
        trace = kwargs["metadata"]["executed_tools"]
        assert len(trace) == 1
        entry = trace[0]
        assert entry["tool"] == "process_refund"
        assert entry["ok"] is True
        assert entry["args"] == {"order_id": "ORD1001"}
        assert "RF1" in entry["summary"]
        # End to end: the response carries the same trail.
        assert response.metadata["executed_tools"] == trace

    async def test_failed_execution_is_audited_as_not_ok(self) -> None:
        """A failed irreversible action is still an execution attempt —
        the audit trail must record it with ok=False and the error in
        the summary, not silently omit it."""
        registry = _registry(success=False, data={}, message="订单状态不可退款")
        persister = _recorder()
        chat = _confirm_chat(registry, persister)
        await _stage_refund(chat.graph)

        await chat.process_message(9, "确认", 1)

        entry = persister.persist_turn.await_args.kwargs["metadata"]["executed_tools"][0]
        assert entry["ok"] is False
        assert "不可退款" in entry["summary"]


class TestConfirmIdempotency:
    async def test_double_confirm_executes_once(self) -> None:
        """Double-click / repeated 确认 must not re-execute the staged
        action: the first confirm consumes the gate, the second is an
        acknowledgment-only turn (review #9 double-click scenario)."""
        registry = _registry()
        persister = _recorder()
        chat = _confirm_chat(registry, persister)
        await _stage_refund(chat.graph)

        first = await chat.process_message(9, "确认", 1)
        second = await chat.process_message(9, "确认", 1)

        assert registry.execute.await_count == 1
        assert first.metadata.get("executed_tools")
        assert not (second.metadata.get("executed_tools") or [])
