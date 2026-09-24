"""Trace-event multiplexing on the chat stream queue.

The demo cockpit needs stage events and content on ONE channel in
arrival order. These tests pin the consumer contract: TraceEvent items
flow through ``process_message_stream`` in FIFO order with content,
never pollute the persisted turn, and do not count as TTFT "tokens".
The emission flag gates the whole channel, so a pure-content consumer
can turn the panel data off with one env var.
"""

import asyncio
from typing import Any
from unittest.mock import Mock

from app.services.chat.chat_service import ChatService
from app.services.observability.pipeline_tracer import traced_stage
from app.services.observability.trace_events import TraceEvent


def _make_persister() -> tuple[Mock, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []

    async def persist_turn(**kwargs: Any) -> None:
        calls.append(kwargs)

    return Mock(spec=["persist_turn"], persist_turn=persist_turn), calls


class TestStreamTraceMultiplex:
    async def test_trace_events_flow_in_order_with_content(self):
        async def mock_ainvoke(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
            queue = config["configurable"]["stream_queue"]
            queue.put_nowait(TraceEvent(stage="cs.intent", status="start"))
            queue.put_nowait(TraceEvent(stage="cs.intent", status="end", ms=1.5))
            queue.put_nowait("您好，")
            queue.put_nowait(TraceEvent(stage="cs.claim_gate", status="detail"))
            queue.put_nowait("请问有什么可以帮您")
            return {"response": "您好，请问有什么可以帮您"}

        graph = Mock()
        graph.ainvoke = mock_ainvoke
        service = ChatService(graph=graph)

        items = [item async for item in service.process_message_stream(1, "你好", 1)]

        kinds = ["event" if isinstance(i, TraceEvent) else "text" for i in items]
        assert kinds == ["event", "event", "text", "event", "text"]
        stages = [i.stage for i in items if isinstance(i, TraceEvent)]
        assert stages == ["cs.intent", "cs.intent", "cs.claim_gate"]

    async def test_trace_events_excluded_from_persisted_turn(self):
        """The persisted response is the user-visible text — trace
        payloads must never leak into stored conversation content."""
        persister, calls = _make_persister()

        async def mock_ainvoke(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
            queue = config["configurable"]["stream_queue"]
            queue.put_nowait(TraceEvent(stage="cs.intent", status="start"))
            queue.put_nowait("答案内容")
            return {"response": "答案内容"}

        graph = Mock()
        graph.ainvoke = mock_ainvoke
        service = ChatService(graph=graph, persister=persister)

        _ = [item async for item in service.process_message_stream(1, "你好", 1)]

        assert len(calls) == 1
        assert calls[0]["response"] == "答案内容"

    async def test_trace_events_do_not_count_as_first_token(self):
        """TTFT measures time to content the user can read; pipeline
        stage events arriving earlier must not reset the clock."""

        async def mock_ainvoke(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
            queue = config["configurable"]["stream_queue"]
            queue.put_nowait(TraceEvent(stage="cs.intent", status="start"))
            await asyncio.sleep(0.05)
            queue.put_nowait("真正的内容")
            return {"response": "真正的内容"}

        graph = Mock()
        graph.ainvoke = mock_ainvoke
        service = ChatService(graph=graph)

        from prometheus_client import REGISTRY

        before = REGISTRY.get_sample_value("chat_first_token_seconds_count") or 0.0
        items = [item async for item in service.process_message_stream(1, "你好", 1)]
        after = REGISTRY.get_sample_value("chat_first_token_seconds_count") or 0.0

        assert after == before + 1  # observed once, at the content chunk
        assert sum(1 for i in items if isinstance(i, TraceEvent)) == 1

    async def test_real_traced_stage_events_reach_the_stream(self):
        """End-to-end wiring: a graph whose node is decorated with
        @traced_stage pushes its own stage events onto the stream when
        the trace channel is enabled."""

        @traced_stage("cs.demo_node")
        async def demo_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
            queue = config["configurable"]["stream_queue"]
            queue.put_nowait("好的")
            return {"response": "好的"}

        graph = Mock()
        graph.ainvoke = demo_node
        service = ChatService(graph=graph)

        items = [item async for item in service.process_message_stream(1, "你好", 1)]

        events = [i for i in items if isinstance(i, TraceEvent)]
        assert [e.stage for e in events] == ["cs.demo_node", "cs.demo_node"]
        assert [e.status for e in events] == ["start", "end"]
        assert [i for i in items if not isinstance(i, TraceEvent)] == ["好的"]

    async def test_disabled_flag_suppresses_stage_events(self):
        """CHAT_TRACE_STREAM_ENABLED=False keeps the stream pure content —
        the panel data channel is opt-out for consumers that want it."""

        @traced_stage("cs.demo_node")
        async def demo_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
            queue = config["configurable"]["stream_queue"]
            queue.put_nowait("好的")
            return {"response": "好的"}

        graph = Mock()
        graph.ainvoke = demo_node

        from app.config.settings import settings

        service = ChatService(graph=graph)
        original = settings.CHAT_TRACE_STREAM_ENABLED
        settings.CHAT_TRACE_STREAM_ENABLED = False
        try:
            items = [item async for item in service.process_message_stream(1, "你好", 1)]
        finally:
            settings.CHAT_TRACE_STREAM_ENABLED = original

        assert items == ["好的"]
