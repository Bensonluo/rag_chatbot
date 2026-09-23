"""LLM call resilience: retry with backoff, circuit breaker, provider failover.

At 800K-1M daily requests every API replica shares the same upstream LLM
quota. A single provider hiccup (429, 503, timeout) would otherwise stall
in-flight requests across all replicas at once. This module wraps the
provider chain with:

- **Retry with exponential backoff** for transient failures
  (timeouts, 429/5xx), so brief blips self-heal.
- **Per-provider circuit breaker**: after N consecutive failures a
  provider is skipped for a recovery window instead of adding its
  latency to every request; one probe request re-admits it.
- **Failover**: providers are tried in configured priority order
  (GLM → OpenAI → Anthropic); a hard failure on the primary serves
  from the next configured provider.

Non-transient errors (4xx auth/validation) are never retried — they
fail over immediately, and if every provider is exhausted the last
error is raised unchanged.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator, Sequence
from typing import Any

from app.core.exceptions import ExternalServiceError
from app.services.llm.base import LLMMessage, LLMResponse, LLMServiceBase

logger = logging.getLogger(__name__)

# HTTP statuses worth retrying: request timeouts, rate limits, upstream blips.
RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})


def is_transient(exc: BaseException) -> bool:
    """Whether an exception is worth retrying on the same provider."""
    if isinstance(exc, ExternalServiceError):
        return exc.status_code is None or exc.status_code in RETRYABLE_STATUS_CODES
    return False


class CircuitBreaker:
    """Single-provider circuit breaker (closed → open → half-open).

    Not thread-safe by design: FastAPI runs one event loop per process
    and this class only has synchronous, non-blocking state updates.
    """

    def __init__(self, failure_threshold: int = 5, recovery_seconds: float = 30.0) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_seconds = recovery_seconds
        self._consecutive_failures = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> str:
        """One of "closed", "open", "half_open"."""
        if self._opened_at is None:
            return "closed"
        if time.monotonic() - self._opened_at >= self._recovery_seconds:
            return "half_open"
        return "open"

    def allow(self) -> bool:
        """Whether a call to this provider may be attempted now."""
        # closed or half_open: allow (half_open lets exactly the probe
        # through; a failure re-opens, a success closes)
        return self.state != "open"

    def record_success(self) -> None:
        """Reset the breaker after a successful call."""
        self._consecutive_failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        """Count a failed call; trips open at the threshold."""
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._failure_threshold:
            if self.state != "open":
                logger.warning(
                    "LLM circuit opened after %d consecutive failures", self._consecutive_failures
                )
            self._opened_at = time.monotonic()


class ResilientLLMService(LLMServiceBase):
    """Retrying, circuit-broken, failover-capable LLM service chain.

    Wraps a prioritized list of ``(name, service)`` providers. The first
    entry is the primary; the rest are failover targets in order.
    """

    def __init__(
        self,
        providers: Sequence[tuple[str, LLMServiceBase]],
        *,
        max_retries: int = 2,
        backoff_base: float = 0.5,
        circuit_failure_threshold: int = 5,
        circuit_recovery_seconds: float = 30.0,
        tracer: Any = None,
    ) -> None:
        if not providers:
            raise ValueError("ResilientLLMService requires at least one provider")
        self._providers: list[tuple[str, LLMServiceBase]] = list(providers)
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._breakers: dict[str, CircuitBreaker] = {
            name: CircuitBreaker(circuit_failure_threshold, circuit_recovery_seconds)
            for name, _ in self._providers
        }
        self._tracer = tracer
        # Mirror the primary's identity for prompt building / tracing.
        primary = self._providers[0][1]
        self.model = primary.model
        self.max_tokens = primary.max_tokens
        self.temperature = primary.temperature

    @property
    def provider_names(self) -> list[str]:
        """Configured provider names, priority order."""
        return [name for name, _ in self._providers]

    def breaker_state(self, name: str) -> str:
        """Expose breaker state for observability and tests."""
        return self._breakers[name].state

    async def generate(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        if self._tracer is None:
            return await self._generate_with_failover(
                messages, max_tokens=max_tokens, temperature=temperature, **kwargs
            )
        async with self._tracer.trace_generate(model=self.model):
            return await self._generate_with_failover(
                messages, max_tokens=max_tokens, temperature=temperature, **kwargs
            )

    async def _generate_with_failover(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        last_exc: BaseException = ExternalServiceError("llm-chain", "no provider available")
        for name, service in self._providers:
            breaker = self._breakers[name]
            if not breaker.allow():
                logger.info("Skipping provider %s: circuit open", name)
                continue
            for attempt in range(self._max_retries + 1):
                try:
                    response = await service.generate(
                        messages, max_tokens=max_tokens, temperature=temperature, **kwargs
                    )
                    breaker.record_success()
                    return response
                except ExternalServiceError as exc:
                    last_exc = exc
                    breaker.record_failure()
                    if not is_transient(exc) or attempt == self._max_retries:
                        logger.warning(
                            "Provider %s failed after %d attempt(s): %s",
                            name,
                            attempt + 1,
                            exc,
                        )
                        break  # try the next provider
                    await asyncio.sleep(self._backoff_base * (2**attempt))
                except Exception as exc:
                    # Unexpected errors are treated as non-transient: no
                    # retry, straight to the next provider.
                    last_exc = exc
                    breaker.record_failure()
                    logger.warning("Provider %s raised unexpected error: %s", name, exc)
                    break
        raise last_exc

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Stream with failover before the first token only.

        Once a chunk has been yielded the consumer owns partial output,
        so mid-stream failures propagate instead of restarting on
        another provider (which would duplicate the already-sent text).
        """
        if self._tracer is None:
            async for chunk in self._generate_stream_with_failover(
                messages, max_tokens=max_tokens, temperature=temperature, **kwargs
            ):
                yield chunk
            return
        async with self._tracer.trace_generate(model=self.model):
            async for chunk in self._generate_stream_with_failover(
                messages, max_tokens=max_tokens, temperature=temperature, **kwargs
            ):
                yield chunk

    async def _generate_stream_with_failover(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Failover/retry loop behind ``generate_stream`` (see its docstring)."""
        last_exc: BaseException = ExternalServiceError("llm-chain", "no provider available")
        for name, service in self._providers:
            breaker = self._breakers[name]
            if not breaker.allow():
                logger.info("Skipping provider %s: circuit open", name)
                continue
            for attempt in range(self._max_retries + 1):
                yielded = False
                try:
                    async for chunk in service.generate_stream(
                        messages, max_tokens=max_tokens, temperature=temperature, **kwargs
                    ):
                        yielded = True
                        yield chunk
                    breaker.record_success()
                    return
                except ExternalServiceError as exc:
                    breaker.record_failure()
                    if yielded:
                        raise
                    last_exc = exc
                    if not is_transient(exc) or attempt == self._max_retries:
                        logger.warning(
                            "Provider %s stream failed after %d attempt(s): %s",
                            name,
                            attempt + 1,
                            exc,
                        )
                        break  # next provider
                    await asyncio.sleep(self._backoff_base * (2**attempt))
                except Exception as exc:
                    breaker.record_failure()
                    if yielded:
                        raise
                    last_exc = exc
                    logger.warning("Provider %s stream raised unexpected error: %s", name, exc)
                    break
        raise last_exc

    def estimate_tokens(self, text: str) -> int:
        """Delegate to the primary provider (local heuristic, no I/O)."""
        return self._providers[0][1].estimate_tokens(text)

    async def count_tokens(self, messages: list[LLMMessage]) -> int:
        """Delegate to the primary provider."""
        return await self._providers[0][1].count_tokens(messages)
