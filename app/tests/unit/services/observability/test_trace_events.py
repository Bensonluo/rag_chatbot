"""Live trace events for the execution-chain demo panel.

The backend already records every pipeline stage as an OTel span, but
spans land in a collector — invisible to a live demo viewer. These
tests pin the trace-event channel that makes the pipeline observable
IN-BAND: typed ``TraceEvent`` objects multiplexed onto the per-request
stream queue, published through a ContextVar sink so nodes never change
their signatures. Observability made visible is the demo's core story.
"""

import asyncio
import contextlib
import json

from app.services.observability.pipeline_tracer import traced_stage
from app.services.observability.trace_events import (
    CallbackTraceSink,
    TraceEvent,
    TraceSink,
    emit_trace,
    trace_sink_active,
)


class TestTraceEventModel:
    def test_payload_is_json_serializable_with_chinese_detail(self):
        event = TraceEvent(
            stage="cs.claim_gate",
            status="detail",
            ms=12.5,
            detail={"clause": "30天无理由", "reason": "numeric_mismatch"},
        )
        decoded = json.loads(json.dumps(event.payload(), ensure_ascii=False))
        assert decoded["stage"] == "cs.claim_gate"
        assert decoded["status"] == "detail"
        assert decoded["ms"] == 12.5
        assert decoded["detail"]["clause"] == "30天无理由"

    def test_defaults_are_safe(self):
        event = TraceEvent(stage="cs.intent", status="start")
        assert event.ms is None
        assert event.payload()["detail"] == {}


class TestTraceSink:
    async def test_emit_pushes_event_onto_queue(self):
        queue: asyncio.Queue[TraceEvent] = asyncio.Queue()
        sink = TraceSink(queue)

        emit_trace("cs.intent", status="start")

        assert queue.empty()  # no sink active in this context

        with trace_sink_active(sink):
            emit_trace("cs.intent", status="start")
            emit_trace("cs.intent", intent="question", confidence=0.92)

        first = queue.get_nowait()
        second = queue.get_nowait()
        assert isinstance(first, TraceEvent)
        assert first.stage == "cs.intent"
        assert first.status == "start"
        assert second.detail == {"intent": "question", "confidence": 0.92}

    async def test_emit_never_raises_on_broken_queue(self):
        """Observability must not be able to break chat — a sink whose
        queue is somehow unusable swallows instead of propagating."""

        class PoisonQueue:
            def put_nowait(self, item: object) -> None:
                raise RuntimeError("queue is closed")

        with trace_sink_active(TraceSink(PoisonQueue())):  # type: ignore[arg-type]
            emit_trace("cs.anything")  # must not raise

    async def test_task_created_inside_scope_inherits_sink(self):
        """The propagation contract: asyncio copies context at task
        creation, so a graph task created inside the active scope (same
        mechanism as the LLM budget ContextVar) sees the sink."""
        queue: asyncio.Queue[TraceEvent] = asyncio.Queue()
        sink = TraceSink(queue)

        async def node_like_coroutine() -> None:
            emit_trace("cs.node", status="start")

        with trace_sink_active(sink):
            task = asyncio.create_task(node_like_coroutine())
        await task

        event = queue.get_nowait()
        assert isinstance(event, TraceEvent)
        assert event.stage == "cs.node"

    async def test_sink_resets_after_scope(self):
        queue: asyncio.Queue[TraceEvent] = asyncio.Queue()
        with trace_sink_active(TraceSink(queue)):
            pass
        emit_trace("cs.gone")
        assert queue.empty()


class TestCallbackTraceSink:
    async def test_emit_forwards_event_to_callback(self):
        """Pipelines that forward events via an on_trace observer (the
        guest shopping stream) must be able to bridge ContextVar
        emissions onto the same channel."""
        seen: list[TraceEvent] = []
        sink = CallbackTraceSink(seen.append)

        with trace_sink_active(sink):
            emit_trace("cs.tool", status="start", tool="commerce_graph_upsert")
            emit_trace("cs.tool", status="end", ms=640, tool="commerce_graph_upsert", ok=True)

        assert [e.status for e in seen] == ["start", "end"]
        assert seen[-1].detail["tool"] == "commerce_graph_upsert"
        assert seen[-1].detail["ok"] is True

    async def test_emit_never_raises_on_broken_callback(self):
        def broken(event: TraceEvent) -> None:
            raise RuntimeError("observer detached")

        with trace_sink_active(CallbackTraceSink(broken)):
            emit_trace("cs.anything")  # must not raise

    async def test_background_task_created_inside_scope_inherits_callback(self):
        seen: list[TraceEvent] = []

        async def fire_and_forget() -> None:
            emit_trace("cs.tool", status="end", ms=1, tool="background")

        with trace_sink_active(CallbackTraceSink(seen.append)):
            task = asyncio.create_task(fire_and_forget())
        await task

        assert len(seen) == 1
        assert seen[0].detail["tool"] == "background"


class TestTracedStageEmission:
    async def test_stage_emits_start_and_end_with_timing(self):
        queue: asyncio.Queue[TraceEvent] = asyncio.Queue()
        sink = TraceSink(queue)

        @traced_stage("cs.demo")
        async def node() -> dict[str, object]:
            await asyncio.sleep(0.01)
            return {"intent": "x"}

        with trace_sink_active(sink):
            await node()

        events: list[TraceEvent] = [queue.get_nowait() for _ in range(2)]
        assert events[0].stage == "cs.demo"
        assert events[0].status == "start"
        assert events[1].status == "end"
        assert events[1].ms is not None and events[1].ms >= 0

    async def test_stage_emits_error_status_on_exception(self):
        queue: asyncio.Queue[TraceEvent] = asyncio.Queue()
        sink = TraceSink(queue)

        @traced_stage("cs.bad")
        async def bad_node() -> dict[str, object]:
            raise RuntimeError("boom")

        with trace_sink_active(sink), contextlib.suppress(RuntimeError):
            await bad_node()

        events: list[TraceEvent] = [queue.get_nowait() for _ in range(2)]
        assert [e.status for e in events] == ["start", "error"]

    async def test_stage_runs_unchanged_without_sink(self):
        @traced_stage("cs.bare")
        async def bare_node() -> dict[str, object]:
            return {"ok": True}

        assert await bare_node() == {"ok": True}
