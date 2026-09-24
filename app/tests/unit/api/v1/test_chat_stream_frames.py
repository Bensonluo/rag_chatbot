"""SSE encoding of trace events (chat stream demo panel channel).

Trace events ride the same SSE connection as content, as NAMED events
(``event: trace``). Per the SSE spec, EventSource consumers listening
only to ``message`` never see them — the existing content contract is
unchanged — while the demo frontend subscribes to ``trace`` for the
execution-chain panel. These tests pin the frame encoding.
"""

import json

from app.api.v1.chat import _sse_trace_frame
from app.services.observability.trace_events import TraceEvent


class TestSseTraceFrame:
    def test_frame_is_named_event_with_parseable_json_payload(self):
        event = TraceEvent(
            stage="cs.claim_gate",
            status="detail",
            ms=8.3,
            detail={"clause": "30天无理由退货", "reason": "numeric_mismatch"},
        )

        frame = _sse_trace_frame(event)

        lines = frame.splitlines()
        assert lines[0] == "event: trace"
        assert lines[1].startswith("data: ")
        payload = json.loads(lines[1][len("data: ") :])
        assert payload["stage"] == "cs.claim_gate"
        assert payload["status"] == "detail"
        assert payload["detail"]["clause"] == "30天无理由退货"
        # One SSE frame = terminated by a blank line
        assert frame.endswith("\n\n")

    def test_payload_keeps_chinese_readable(self):
        """ensure_ascii=False keeps the demo panel payload compact and
        human-readable in the network tab."""
        event = TraceEvent(stage="cs.handoff", status="detail", detail={"reason": "情绪激动"})
        frame = _sse_trace_frame(event)
        assert "情绪激动" in frame
