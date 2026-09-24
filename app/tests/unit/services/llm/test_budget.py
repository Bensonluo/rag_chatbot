"""Per-request LLM call budget: hard cap + graceful agent degradation.

At 800K-1M daily requests an uncapped pathological turn (agent loop +
intent + rerank + generation) multiplies LLM spend; LangGraph's
recursion limit only stops runaway graphs after dozens of expensive
calls. The budget enforces the cap at the LLM boundary — every call
path (generate / generate_stream / generate_with_tools) reserves
before the provider is hit, and the agent loop degrades gracefully
instead of failing the turn.
"""

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("opentelemetry.sdk.trace")  # noqa: E402

from opentelemetry.sdk.trace.export import (  # noqa: E402
    SpanExporter,
    SpanExportResult,
)

from app.services.llm.base import LLMMessage, LLMResponse, LLMServiceBase  # noqa: E402


class _ListExporter(SpanExporter):
    """SDK-conformant exporter collecting ReadableSpans in memory."""

    def __init__(self) -> None:
        super().__init__()
        self.exported: list[Any] = []

    def export(self, spans: Any) -> SpanExportResult:
        self.exported.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:  # noqa: ARG002
        return True


class _CountingLLM(LLMServiceBase):
    """Fake provider recording every call it serves."""

    def __init__(self) -> None:
        super().__init__(api_key="fake", model="fake")
        self.generate_calls = 0
        self.stream_calls = 0
        self.tool_calls = 0

    async def generate(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.generate_calls += 1
        return LLMResponse(content=f"r{self.generate_calls}", model="fake")

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        self.stream_calls += 1
        yield f"s{self.stream_calls}"

    async def generate_with_tools(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.tool_calls += 1
        return LLMResponse(content=f"t{self.tool_calls}", model="fake")

    def estimate_tokens(self, text: str) -> int:
        return len(text)

    async def count_tokens(self, messages: list[LLMMessage]) -> int:
        return sum(len(m.content) for m in messages)


class TestBudgetScope:
    async def test_calls_within_budget_pass(self):
        from app.services.llm.budget import BudgetedLLMService, enter_llm_budget

        inner = _CountingLLM()
        llm = BudgetedLLMService(inner)
        async with enter_llm_budget(max_calls=2):
            assert (await llm.generate([LLMMessage(role="user", content="q")])).content == "r1"
            assert (await llm.generate([LLMMessage(role="user", content="q")])).content == "r2"

    async def test_call_beyond_budget_raises(self):
        from app.services.llm.budget import (
            BudgetedLLMService,
            LLMBudgetExceeded,
            enter_llm_budget,
        )

        inner = _CountingLLM()
        llm = BudgetedLLMService(inner)
        async with enter_llm_budget(max_calls=1):
            await llm.generate([LLMMessage(role="user", content="q")])
            try:
                await llm.generate([LLMMessage(role="user", content="q")])
                raise AssertionError("budget not enforced")
            except LLMBudgetExceeded:
                pass

    async def test_exhausted_call_never_reaches_provider(self):
        from app.services.llm.budget import (
            BudgetedLLMService,
            LLMBudgetExceeded,
            enter_llm_budget,
        )

        inner = _CountingLLM()
        llm = BudgetedLLMService(inner)
        async with enter_llm_budget(max_calls=1):
            await llm.generate([LLMMessage(role="user", content="q")])
            with pytest.raises(LLMBudgetExceeded):
                await llm.generate([LLMMessage(role="user", content="q")])
        assert inner.generate_calls == 1  # refused call was pre-provider

    async def test_no_scope_means_unlimited(self):
        from app.services.llm.budget import BudgetedLLMService

        inner = _CountingLLM()
        llm = BudgetedLLMService(inner)
        for _ in range(5):
            await llm.generate([LLMMessage(role="user", content="q")])
        assert inner.generate_calls == 5

    async def test_scope_resets_between_requests(self):
        from app.services.llm.budget import BudgetedLLMService, enter_llm_budget

        inner = _CountingLLM()
        llm = BudgetedLLMService(inner)
        for _ in range(2):
            async with enter_llm_budget(max_calls=1):
                await llm.generate([LLMMessage(role="user", content="q")])
        assert inner.generate_calls == 2  # fresh budget each request

    async def test_concurrent_requests_have_isolated_budgets(self):
        """Two in-flight turns must not share one budget."""
        from app.services.llm.budget import BudgetedLLMService, enter_llm_budget

        inner = _CountingLLM()
        llm = BudgetedLLMService(inner)

        async def one_request() -> None:
            async with enter_llm_budget(max_calls=2):
                await llm.generate([LLMMessage(role="user", content="a")])
                await llm.generate([LLMMessage(role="user", content="b")])

        await asyncio.gather(one_request(), one_request())
        assert inner.generate_calls == 4

    async def test_stream_iteration_consumes_one_call(self):
        from app.services.llm.budget import BudgetedLLMService, enter_llm_budget

        inner = _CountingLLM()
        llm = BudgetedLLMService(inner)
        async with enter_llm_budget(max_calls=1):
            chunks = [c async for c in llm.generate_stream([LLMMessage(role="user", content="q")])]
        assert chunks == ["s1"]
        assert inner.stream_calls == 1

    async def test_stream_refused_when_budget_exhausted(self):
        from app.services.llm.budget import (
            BudgetedLLMService,
            LLMBudgetExceeded,
            enter_llm_budget,
        )

        inner = _CountingLLM()
        llm = BudgetedLLMService(inner)
        async with enter_llm_budget(max_calls=1):
            await llm.generate([LLMMessage(role="user", content="q")])
            try:
                async for _ in llm.generate_stream([LLMMessage(role="user", content="q")]):
                    pass
                raise AssertionError("stream budget not enforced")
            except LLMBudgetExceeded:
                pass
        assert inner.stream_calls == 0

    async def test_tool_calls_are_budgeted_too(self):
        """generate_with_tools must not bypass the cap."""
        from app.services.llm.budget import (
            BudgetedLLMService,
            LLMBudgetExceeded,
            enter_llm_budget,
        )

        inner = _CountingLLM()
        llm = BudgetedLLMService(inner)
        async with enter_llm_budget(max_calls=1):
            await llm.generate_with_tools([LLMMessage(role="user", content="q")], tools=[])
            try:
                await llm.generate_with_tools([LLMMessage(role="user", content="q")], tools=[])
                raise AssertionError("tool budget not enforced")
            except LLMBudgetExceeded:
                pass
        assert inner.tool_calls == 1


class TestAgentBudgetGraceful:
    async def test_budget_exhaustion_returns_truncated_result(self):
        """Agent loop must degrade, not fail the turn."""
        from app.services.agent.service import AgentService
        from app.services.llm.budget import LLMBudgetExceeded

        class _BlowingAgentLLM(_CountingLLM):
            async def generate_with_tools(
                self,
                messages: list[LLMMessage],
                tools: list[dict[str, Any]],
                max_tokens: int | None = None,
                temperature: float | None = None,
                **kwargs: Any,
            ) -> LLMResponse:
                self.tool_calls += 1
                if self.tool_calls > 1:
                    raise LLMBudgetExceeded("llm budget exhausted")
                return LLMResponse(
                    content="先查订单",
                    model="fake",
                    tool_calls=[
                        {
                            "id": "c1",
                            "name": "query_order_status",
                            "arguments": json.dumps({"order_id": "A1"}),
                        }
                    ],
                )

        from app.services.dialogue.tools import ToolRegistry

        agent = AgentService(
            llm_service=_BlowingAgentLLM(),
            tool_registry=ToolRegistry(),
            max_steps=3,
        )
        result = await agent.run(user_message="查订单然后退款")
        assert result.truncated is True
        assert result.response  # user still gets an answer, not an error


class TestBudgetReport:
    async def test_scope_yields_state_with_used_and_refused(self):
        from app.services.llm.budget import (
            BudgetedLLMService,
            LLMBudgetExceeded,
            enter_llm_budget,
        )

        inner = _CountingLLM()
        llm = BudgetedLLMService(inner)
        async with enter_llm_budget(max_calls=1) as state:
            assert state is not None
            await llm.generate([LLMMessage(role="user", content="q")])
            with pytest.raises(LLMBudgetExceeded):
                await llm.generate([LLMMessage(role="user", content="q")])
            assert state.used == 1
            assert state.refused == 1

    async def test_disabled_scope_yields_none(self):
        from app.services.llm.budget import enter_llm_budget

        async with enter_llm_budget(0) as state:
            assert state is None

    async def test_budget_attrs_reach_active_span(self):
        """Scope exit stamps llm.* attrs on the current (root) span."""
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        from app.services.llm.budget import BudgetedLLMService, enter_llm_budget

        provider = TracerProvider()
        exporter = _ListExporter()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("t")

        from app.services.llm.budget import record_budget_on_span

        inner = _CountingLLM()
        llm = BudgetedLLMService(inner)
        try:
            with tracer.start_as_current_span("cs.pipeline"):
                async with enter_llm_budget(max_calls=2) as state:
                    assert state is not None
                    await llm.generate([LLMMessage(role="user", content="q")])
                    # ChatService stamps at the moment the turn's LLM work
                    # is done — for streams that is NOT scope exit (the
                    # scope only wraps task creation there).
                    record_budget_on_span(state)
        finally:
            provider.shutdown()

        root = next(s for s in exporter.exported if s.name == "cs.pipeline")
        assert root.attributes["llm.calls"] == 1
        assert root.attributes["llm.refused"] == 0
        assert root.attributes["llm.budget"] == 2


class TestBudgetMetadata:
    async def test_process_message_reports_budget_in_metadata(self):
        from app.services.chat.chat_service import ChatService
        from app.tests.unit.services.chat.test_persistence import _FakeGraph

        service = ChatService(graph=_FakeGraph(result={"response": "好的", "intent": "chitchat"}))
        resp = await service.process_message(session_id=1, message="你好", user_id=1)
        assert resp.metadata is not None
        assert resp.metadata["llm_calls"] == 0
        assert resp.metadata["llm_refused"] == 0

    async def test_stream_persist_carries_budget_metadata(self):
        from app.services.chat.chat_service import ChatService

        class _Graph:
            async def ainvoke(
                self, state: dict[str, Any], config: dict[str, Any]
            ) -> dict[str, Any]:
                config["configurable"]["stream_queue"].put_nowait("好")
                return {}

        persister = AsyncMock()
        service = ChatService(graph=_Graph(), persister=persister)
        chunks = [c async for c in service.process_message_stream(1, "你好", 1)]
        assert chunks == ["好"]
        persister.persist_turn.assert_awaited_once()
        metadata = persister.persist_turn.await_args.kwargs.get("metadata")
        assert metadata is not None
        assert metadata["llm_calls"] == 0
        assert metadata["llm_refused"] == 0


class TestBudgetSettings:
    def test_default_budget_is_finite_and_sane(self):
        """Default must cap the pathological tail but fit the normal path."""
        from app.config.settings import settings

        assert 4 <= settings.CHAT_LLM_CALL_BUDGET <= 12
