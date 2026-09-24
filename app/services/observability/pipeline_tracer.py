"""Full-pipeline stage tracing for the dialogue graph.

Industry baseline: one trace per chat request with a span per pipeline
stage (guardrail → intent → slots → retrieval/agent → generation →
handoff) so latency regressions and error hotspots are attributable to
a stage, not the pipeline blob.

Real OpenTelemetry tracers return a CONTEXT MANAGER from
``start_as_current_span`` — the span only becomes usable via
``__enter__`` and must be closed via ``__exit__(exc_type, exc, tb)``
with attributes set BEFORE exit, or the span leaks and is never
exported. The no-op tracer in ``app.middleware.tracing`` returns a
span that doubles as its own context manager; ``_StageSpan`` handles
both contracts.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any, TypeVar, cast

from app.middleware.tracing import get_tracer

logger = logging.getLogger(__name__)

# PII-safe allowlists: only these kwargs/results become span attributes.
_KWARG_ATTRS = frozenset({"session_id", "user_id"})
_RESULT_ATTRS = frozenset({"intent", "confidence"})

_F = TypeVar("_F", bound=Callable[..., Any])


class _StageSpan:
    """Bridges OTel's context-manager span protocol for stage spans."""

    __slots__ = ("_cm", "_span")

    def __init__(self, name: str, kwargs: dict[str, Any]) -> None:
        self._cm: Any = get_tracer("rag-chatbot.dialogue").start_as_current_span(name)
        # The CM's __enter__ yields the live span (and activates it).
        self._span: Any = self._cm.__enter__()
        for key in _KWARG_ATTRS:
            if kwargs.get(key) is not None:
                self._set(key, kwargs[key])

    def _set(self, key: str, value: Any) -> None:
        set_attribute = getattr(self._span, "set_attribute", None)
        if set_attribute is not None:
            set_attribute(key, value)

    def set_result_attrs(self, result: dict[str, Any]) -> None:
        for key in _RESULT_ATTRS:
            if result.get(key) is not None:
                self._set(key, result[key])

    def record_and_end(self, exc: BaseException) -> None:
        """Record the exception and close carrying it (for OTel status)."""
        record_exception = getattr(self._span, "record_exception", None)
        if record_exception is not None:
            record_exception(exc)
        self._cm.__exit__(type(exc), exc, exc.__traceback__)

    def end(self) -> None:
        self._cm.__exit__(None, None, None)


def traced_stage(name: str) -> Callable[[_F], _F]:
    """Identity-typed decorator wrapping a dialogue node in a span.

    The wrapper must stay a genuine coroutine function / async generator
    function: LangGraph probes with ``iscoroutinefunction``, which does
    NOT follow ``functools.wraps``' ``__wrapped__`` chain — a sync
    dispatch wrapper makes nodes look sync and their updates invalid.
    """

    def decorator(fn: _F) -> _F:
        if inspect.isasyncgenfunction(fn):

            @functools.wraps(fn)
            async def agen_wrapper(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
                agen = fn(*args, **kwargs)
                span = _StageSpan(name, kwargs)
                try:
                    async for item in agen:
                        yield item
                except BaseException as exc:
                    # BaseException: a consumer abandoning the stream
                    # throws GeneratorExit, which still ends the span.
                    span.record_and_end(exc)
                    raise
                else:
                    span.end()
                finally:
                    # Outer aclose() does not synchronously close the
                    # inner generator; close it deterministically here.
                    with contextlib.suppress(Exception):
                        await agen.aclose()

            return cast(_F, agen_wrapper)

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            span = _StageSpan(name, kwargs)
            try:
                result = await fn(*args, **kwargs)
            except Exception as exc:
                span.record_and_end(exc)
                raise
            else:
                if isinstance(result, dict):
                    span.set_result_attrs(result)
                span.end()
                return result

        return cast(_F, wrapper)

    return decorator
