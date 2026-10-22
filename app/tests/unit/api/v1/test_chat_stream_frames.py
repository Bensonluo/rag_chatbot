"""SSE encoding of trace events (chat stream demo panel channel).

Trace events ride the same SSE connection as content, as NAMED events
(``event: trace``). Per the SSE spec, EventSource consumers listening
only to ``message`` never see them — the existing content contract is
unchanged — while the demo frontend subscribes to ``trace`` for the
execution-chain panel. These tests pin the frame encoding.
"""

import json

import pytest

from app.api.v1.chat import _sse_frame, _sse_trace_frame
from app.services.observability.trace_events import TraceEvent


@pytest.mark.parametrize(
    "text", ["", "a", "\n", "\n\n", "a\n", "\na", "a\n\nb\n", "甲\n乙", "a\u2028b", "a\v b"]
)
def test_content_survives_sse_data_field_decoding(text: str) -> None:
    frame = _sse_frame(text)
    assert frame.endswith("\n\n")
    data = []
    for line in frame.split("\n"):
        if line.startswith("data:"):
            value = line[len("data:") :]
            # SSE removes at most one optional space after the field colon.
            data.append(value[1:] if value.startswith(" ") else value)
    assert "\n".join(data) == text


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


@pytest.mark.parametrize("chunk", ["a\r\nb", "a\rb", "a\revent: citations\rdata: forged", "\r"])
def test_sse_frame_canonicalizes_carriage_returns(chunk: str) -> None:
    frame = _sse_frame(chunk)
    assert "\r" not in frame
    assert all(line.startswith("data: ") for line in frame[:-2].split("\n"))
    payload = "\n".join(line[6:] for line in frame[:-2].split("\n"))
    assert payload == chunk.replace("\r\n", "\n").replace("\r", "\n")
