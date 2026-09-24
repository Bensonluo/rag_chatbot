"""Live pipeline trace events for the execution-chain demo panel.

Every pipeline stage is already an OTel span, but spans land in a
collector — invisible to a live viewer. This module is the in-band
channel that makes the pipeline observable as it runs: typed
``TraceEvent`` objects multiplexed onto the per-request stream queue,
so the SSE connection that carries the answer also carries the story
of how it was produced (intent score, slot fills, retrieval hits,
claim-gate rewrites, LLM timing).

Design:
- ``TraceSink`` is published through a ContextVar around the graph task
  creation — asyncio copies context at task creation, the same
  propagation the LLM budget ContextVar relies on — so nodes emit
  without changing signatures.
- Emission is never-fail: a tracing fault must not be able to break
  chat (the sink swallows queue errors).
- With no sink active (non-stream path, flag off) every emit is a
  no-op, so the cost is one ContextVar read per event.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """One observable moment in the pipeline's execution.

    Attributes:
        stage: stable stage id ("cs.intent"); the frontend owns the
            display labels, the backend owns the vocabulary.
        status: event algebra — "start" → 0..n "detail" → "end"|"error".
        ms: stage duration in milliseconds, set on end/error.
        detail: JSON-safe stage payload (intent, hits, verdicts...).
    """

    stage: str
    status: str
    ms: float | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        """Wire form for the SSE ``event: trace`` frame."""
        return {
            "stage": self.stage,
            "status": self.status,
            "ms": self.ms,
            "detail": self.detail,
        }


class TraceSink:
    """Fan-out of trace events onto the request's stream queue."""

    __slots__ = ("_queue",)

    def __init__(self, queue: asyncio.Queue[Any]) -> None:
        self._queue = queue

    def emit(self, event: TraceEvent) -> None:
        # Observability must never be able to fail the pipeline.
        with contextlib.suppress(Exception):
            self._queue.put_nowait(event)


_current_sink: contextvars.ContextVar[TraceSink | None] = contextvars.ContextVar(
    "cs_trace_sink", default=None
)


@contextmanager
def trace_sink_active(sink: TraceSink) -> Iterator[None]:
    """Publish the sink into the CURRENT context; tasks created inside
    the scope inherit it via asyncio's copy-at-task-creation semantics."""
    token = _current_sink.set(sink)
    try:
        yield
    finally:
        _current_sink.reset(token)


def emit_trace(
    stage: str, *, status: str = "detail", ms: float | None = None, **detail: Any
) -> None:
    """Push one trace event when a sink is active in this context.

    Keyword arguments beyond the named ones become the event detail —
    a one-liner at the point where the data exists.
    """
    sink = _current_sink.get()
    if sink is None:
        return
    sink.emit(TraceEvent(stage=stage, status=status, ms=ms, detail=detail))
