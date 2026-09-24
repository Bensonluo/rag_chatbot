"""Full-pipeline stage tracing for the chat spine.

Wraps each dialogue-graph node (and the ChatService entrypoints) in an
OpenTelemetry span so one trace per request shows guardrail → intent →
slots → retrieval/FAQ/agent → generation → handoff as a span tree.
Legacy setup traced LLM calls only — a latency regression or error
hotspot was attributable to "the pipeline", not to a stage.
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

# Telemetry-safe attribute allowlists: identifiers and routing signals
# only — never message bodies or responses (PII discipline).
_KWARG_ATTRS = frozenset({"session_id", "user_id"})
_RESULT_ATTRS = frozenset({"intent", "confidence"})


# Identity-typed decorator (same pattern as functools.lru_cache): the
# wrapper preserves the decorated callable's kind and signature at
# runtime, and the static type is unchanged so callers of decorated
# methods keep their coroutine-vs-asyncgen narrowing.
_F = TypeVar("_F", bound=Callable[..., Any])


def traced_stage(name: str) -> Callable[[_F], _F]:
    """Decorator: wrap an async dialogue/chat method in an OTel span.

    The tracer is resolved at call time so a patched ``get_tracer``
    (tests) or a reconfigured provider (runtime) is always honored.
    ``functools.wraps`` keeps the original signature visible to
    LangGraph's config-injection introspection.
    """

    def decorator(fn: _F) -> _F:
        if inspect.isasyncgenfunction(fn):
            # The wrapper must itself be an async generator function:
            # LangGraph probes streaming entrypoints with
            # inspect.isasyncgenfunction, which does not follow the
            # __wrapped__ chain functools.wraps installs.
            @functools.wraps(fn)
            async def agen_wrapper(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
                agen = fn(*args, **kwargs)
                span = _start_span(name, kwargs)
                try:
                    async for item in agen:
                        yield item
                except Exception as exc:
                    if hasattr(span, "record_exception"):
                        span.record_exception(exc)
                    raise
                finally:
                    # Close the wrapped generator deterministically.
                    # Without this, closing the wrapper leaves the inner
                    # generator to the event loop's asyncgen finalizer,
                    # which runs a tick later — client-disconnect cleanup
                    # (task cancellation, partial-turn persistence) would
                    # observably lag the disconnect.
                    with contextlib.suppress(Exception):
                        await agen.aclose()
                    if hasattr(span, "end"):
                        span.end()

            return cast(_F, agen_wrapper)

        # Same for coroutine nodes: the wrapper must be a genuine
        # coroutine function — LangGraph's iscoroutinefunction probe
        # does not follow __wrapped__ either, and a sync wrapper makes
        # the node look sync so its coroutine return is never awaited.
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            span = _start_span(name, kwargs)
            try:
                result = await fn(*args, **kwargs)
                if isinstance(result, dict) and hasattr(span, "set_attribute"):
                    for key in _RESULT_ATTRS:
                        if result.get(key) is not None:
                            span.set_attribute(key, result[key])
                return result
            except Exception as exc:
                if hasattr(span, "record_exception"):
                    span.record_exception(exc)
                raise
            finally:
                if hasattr(span, "end"):
                    span.end()

        return cast(_F, wrapper)

    return decorator


def _start_span(name: str, kwargs: dict[str, Any]) -> Any:
    span = get_tracer("rag-chatbot.dialogue").start_as_current_span(name)
    if hasattr(span, "set_attribute"):
        for key in _KWARG_ATTRS:
            if kwargs.get(key) is not None:
                span.set_attribute(key, kwargs[key])
    return span
