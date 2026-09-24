"""Per-request LLM call budget.

At 800K-1M daily requests the cost risk is the pathological tail: a
single turn can fan out into intent detection, slot filling, agent
tool rounds, reranking, and generation — and LangGraph's recursion
limit only stops runaway graphs after dozens of expensive calls. The
budget caps LLM calls per request at the service boundary, BEFORE the
provider is hit (the industry reserve-then-call pattern), so every
call path — generate, generate_stream, generate_with_tools — counts
against the same per-request allowance.

Design:
- Scope rides a ContextVar: the composition root installs ONE
  ``BudgetedLLMService`` wrapper over the provider singleton, and
  ChatService opens a per-request scope. Concurrent requests each get
  an isolated budget (asyncio tasks copy the context; the state object
  is shared only within one request).
- Without an open scope the wrapper is a passthrough — background
  jobs (session compression) and tests are unaffected.
- Enforcement raises ``LLMBudgetExceeded`` before the provider call;
  callers that can degrade (the agent loop) catch it and fall back
  gracefully rather than failing the turn.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any

from app.services.llm.base import LLMMessage, LLMResponse, LLMServiceBase

logger = logging.getLogger(__name__)


class LLMBudgetExceeded(RuntimeError):
    """A provider call was refused: the request's LLM budget is spent."""


class _BudgetState:
    """Mutable counter shared by every call within one request scope."""

    __slots__ = ("max_calls", "used")

    def __init__(self, max_calls: int) -> None:
        self.max_calls = max_calls
        self.used = 0


_budget: ContextVar[_BudgetState | None] = ContextVar("llm_call_budget", default=None)


@asynccontextmanager
async def enter_llm_budget(max_calls: int) -> AsyncIterator[None]:
    """Open a per-request budget scope.

    ``max_calls <= 0`` disables budgeting (scope is a no-op), so a
    setting of 0 turns the cap off without unwiring the decorator.
    """
    if max_calls <= 0:
        yield
        return
    token = _budget.set(_BudgetState(max_calls))
    try:
        yield
    finally:
        _budget.reset(token)


def _reserve() -> None:
    """Reserve one call slot; raise before the provider is hit when spent."""
    state = _budget.get()
    if state is None:
        return
    if state.used >= state.max_calls:
        logger.warning(
            "LLM call budget exhausted (%d/%d); refusing call", state.used, state.max_calls
        )
        raise LLMBudgetExceeded(f"per-request LLM call budget exhausted ({state.max_calls} calls)")
    state.used += 1


class BudgetedLLMService(LLMServiceBase):
    """Delegating wrapper enforcing the active request budget.

    Installed once at the composition root over the provider
    singleton; every component that received the ``llm_service``
    (intent, slots, agent, generation) flows through here.
    """

    def __init__(self, inner: LLMServiceBase) -> None:
        super().__init__(
            api_key="budgeted-delegate",
            model=getattr(inner, "model", "inner"),
        )
        self._inner = inner

    async def generate(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        _reserve()
        return await self._inner.generate(
            messages, max_tokens=max_tokens, temperature=temperature, **kwargs
        )

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        # Reserve at iteration start (the asyncgen body's first tick),
        # mirroring the reserve-then-call pattern: an over-budget
        # stream never reaches the provider.
        _reserve()
        async for chunk in self._inner.generate_stream(
            messages, max_tokens=max_tokens, temperature=temperature, **kwargs
        ):
            yield chunk

    async def generate_with_tools(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        # Explicitly budgeted: providers override the base method, so
        # the wrapper must too — agent tool rounds must not bypass
        # the cap.
        _reserve()
        return await self._inner.generate_with_tools(
            messages, tools, max_tokens=max_tokens, temperature=temperature, **kwargs
        )

    def estimate_tokens(self, text: str) -> int:
        return self._inner.estimate_tokens(text)

    async def count_tokens(self, messages: list[LLMMessage]) -> int:
        return await self._inner.count_tokens(messages)
