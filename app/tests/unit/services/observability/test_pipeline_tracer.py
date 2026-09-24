"""Tests for full-pipeline OTel stage tracing (traced_stage).

Industry baseline: one trace per chat request with a span per pipeline
stage (guardrail → intent → slots → retrieval/FAQ/agent → generation →
handoff) so latency regressions and error hotspots are attributable to
a stage, not the pipeline blob. The legacy setup traced LLM calls only.
"""

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import patch

import pytest

pytest.importorskip("opentelemetry.sdk.trace")  # noqa: E402

from opentelemetry.sdk.trace.export import (  # noqa: E402
    SpanExporter,
    SpanExportResult,
)

from app.services.observability.pipeline_tracer import traced_stage  # noqa: E402


class _RecordingSpan:
    def __init__(self, name: str) -> None:
        self.name = name
        self.attributes: dict[str, Any] = {}
        self.events: list[tuple[str, BaseException]] = []
        self.ended = False
        self.exit_exc_type: type[BaseException] | None = None

    def set_attribute(self, key: str, value: Any) -> None:
        if self.ended:
            raise AssertionError(f"set_attribute after end on {self.name}")
        self.attributes[key] = value

    def record_exception(self, exception: BaseException) -> None:
        self.events.append(("exception", exception))

    def end(self) -> None:
        self.ended = True


class _SpanCM:
    """Models the real OTel contract: the tracer returns a context
    manager, and the span only becomes usable through __enter__."""

    def __init__(self, span: _RecordingSpan) -> None:
        self._span = span

    def __enter__(self) -> _RecordingSpan:
        return self._span

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is not None:
            self._span.exit_exc_type = exc_type
        self._span.ended = True


class _RecordingTracer:
    def __init__(self) -> None:
        self.spans: list[_RecordingSpan] = []

    def start_as_current_span(self, name: str, **kwargs: Any) -> _SpanCM:
        span = _RecordingSpan(name)
        self.spans.append(span)
        return _SpanCM(span)


class TestTracedStageDecorator:
    async def test_wraps_method_in_named_span(self):
        tracer = _RecordingTracer()

        class _Svc:
            @traced_stage("cs.intent")
            async def node(self, state: dict[str, Any]) -> dict[str, Any]:
                return {"intent": "order_query"}

        with patch(
            "app.services.observability.pipeline_tracer.get_tracer",
            return_value=tracer,
        ):
            result = await _Svc().node({"message": "查订单"})

        assert result == {"intent": "order_query"}
        assert [s.name for s in tracer.spans] == ["cs.intent"]
        assert tracer.spans[0].ended is True

    async def test_kwargs_and_result_become_attributes(self):
        tracer = _RecordingTracer()

        class _Svc:
            @traced_stage("cs.pipeline")
            async def run(self, message: str, session_id: int = 0) -> dict[str, Any]:
                return {"intent": "faq", "response": "好的"}

        with patch(
            "app.services.observability.pipeline_tracer.get_tracer",
            return_value=tracer,
        ):
            await _Svc().run(message="hi", session_id=42)

        attrs = tracer.spans[0].attributes
        assert attrs["session_id"] == 42
        assert attrs["intent"] == "faq"
        # Noisy fields must not leak into telemetry
        assert "response" not in attrs
        assert "message" not in attrs

    async def test_async_generator_methods_stay_iterable(self):
        """Streaming entrypoints must remain async generators."""
        tracer = _RecordingTracer()

        class _Svc:
            @traced_stage("cs.pipeline.stream")
            async def stream(self, session_id: int = 0) -> AsyncIterator[str]:
                yield "a"
                yield "b"

        with patch(
            "app.services.observability.pipeline_tracer.get_tracer",
            return_value=tracer,
        ):
            chunks = [c async for c in _Svc().stream(session_id=5)]

        assert chunks == ["a", "b"]
        span = tracer.spans[0]
        assert span.attributes["session_id"] == 5
        assert span.ended is True

    async def test_async_generator_exception_reraised_and_ended(self):
        tracer = _RecordingTracer()

        class _Svc:
            @traced_stage("cs.pipeline.stream")
            async def stream(self) -> AsyncIterator[str]:
                yield "a"
                raise RuntimeError("stream blew up")

        with (
            patch(
                "app.services.observability.pipeline_tracer.get_tracer",
                return_value=tracer,
            ),
            pytest.raises(RuntimeError, match="stream blew up"),
        ):
            async for _ in _Svc().stream():
                pass

        assert tracer.spans[0].ended is True

    async def test_exception_recorded_and_reraised(self):
        tracer = _RecordingTracer()

        class _Svc:
            @traced_stage("cs.agent")
            async def node(self, state: dict[str, Any]) -> dict[str, Any]:
                raise RuntimeError("tool loop blew up")

        with (
            patch(
                "app.services.observability.pipeline_tracer.get_tracer",
                return_value=tracer,
            ),
            pytest.raises(RuntimeError, match="tool loop blew up"),
        ):
            await _Svc().node({})

        span = tracer.spans[0]
        assert span.ended is True
        assert span.events[0][1].args[0] == "tool loop blew up"

    async def test_span_exit_receives_exception_type(self):
        """The CM __exit__ must carry the exception for OTel status."""
        tracer = _RecordingTracer()

        class _Svc:
            @traced_stage("cs.tool")
            async def node(self) -> dict[str, Any]:
                raise RuntimeError("tool blew up")

        with (
            patch(
                "app.services.observability.pipeline_tracer.get_tracer",
                return_value=tracer,
            ),
            pytest.raises(RuntimeError, match="tool blew up"),
        ):
            await _Svc().node()

        span = tracer.spans[0]
        assert span.exit_exc_type is RuntimeError
        assert span.ended is True


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


class TestRealSDKCompatibility:
    async def test_decorated_calls_produce_real_exported_spans(self):
        """Regression: real OTel tracers return a context manager, not
        a span — the decorator must honor the CM protocol or spans are
        silently never ended/exported."""
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        provider = TracerProvider()
        exporter = _ListExporter()
        provider.add_span_processor(SimpleSpanProcessor(exporter))

        class _Svc:
            @traced_stage("cs.intent")
            async def node(self, session_id: int = 0) -> dict[str, Any]:
                return {"intent": "order_query"}

        try:
            with patch(
                "app.services.observability.pipeline_tracer.get_tracer",
                return_value=provider.get_tracer("rag-chatbot.dialogue"),
            ):
                result = await _Svc().node(session_id=7)
        finally:
            provider.shutdown()

        assert result == {"intent": "order_query"}
        exported = [s for s in exporter.exported if s.name == "cs.intent"]
        assert len(exported) == 1
        span = exported[0]
        assert span.attributes["session_id"] == 7
        assert span.end_time is not None  # actually ended, so exported


class TestPipelineSpansWired:
    """Root span + representative node spans fire on a real turn."""

    async def test_process_message_creates_root_and_intent_spans(self):
        from app.services.chat.chat_service import ChatService
        from app.tests.unit.services.chat.test_persistence import _FakeGraph

        tracer = _RecordingTracer()
        service = ChatService(
            graph=_FakeGraph(result={"response": "已发货", "intent": "order_query"}),
        )

        with patch(
            "app.services.observability.pipeline_tracer.get_tracer",
            return_value=tracer,
        ):
            await service.process_message(session_id=3, message="查订单", user_id=9)

        names = [s.name for s in tracer.spans]
        assert "cs.pipeline" in names
        assert tracer.spans[0].name == "cs.pipeline"  # root comes first

    async def test_agent_node_creates_stage_span(self):
        from app.services.agent.service import AgentResult
        from app.tests.unit.services.dialogue.test_agent_node import (
            _agent_returning,
            _make_factory,
        )

        tracer = _RecordingTracer()
        agent = _agent_returning(AgentResult(response="已处理"))
        factory = _make_factory(agent_service=agent)

        with patch(
            "app.services.observability.pipeline_tracer.get_tracer",
            return_value=tracer,
        ):
            await factory.handle_agent_node({"message": "查订单", "session_id": 1, "user_id": 7})

        assert "cs.agent" in [s.name for s in tracer.spans]
