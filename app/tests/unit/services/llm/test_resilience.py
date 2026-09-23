"""LLM resilience chain: retry, circuit breaker, provider failover.

At 800K-1M daily requests every replica shares the same upstream LLM
quota — one provider blip stalls requests fleet-wide without these
guards. These tests pin the contract with scripted fake providers:
transient failures self-heal via retry, hard failures fail over to
the next provider, trips open after consecutive failures, and streams
only fail over before the first token.
"""

from collections.abc import AsyncGenerator
from typing import Any

from app.core.exceptions import ExternalServiceError
from app.services.llm.base import LLMMessage, LLMResponse, LLMServiceBase
from app.services.llm.resilience import CircuitBreaker, ResilientLLMService


class _FakeProvider(LLMServiceBase):
    """Scripted provider: raises failures first, then succeeds."""

    def __init__(self, name: str, failures: int = 0, status_code: int | None = 503):
        super().__init__(api_key="test-key", model=f"model-{name}")
        self.name = name
        self._failures_left = failures
        self._status_code = status_code
        self.generate_calls = 0
        self.stream_calls = 0

    def _maybe_fail(self) -> None:
        if self._failures_left > 0:
            self._failures_left -= 1
            raise ExternalServiceError(
                service=self.name,
                message="scripted failure",
                status_code=self._status_code,
            )

    async def generate(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.generate_calls += 1
        self._maybe_fail()
        return LLMResponse(content=f"from-{self.name}", model=self.model)

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        self.stream_calls += 1
        self._maybe_fail()
        yield f"chunk-{self.name}-1"
        yield f"chunk-{self.name}-2"

    def estimate_tokens(self, text: str) -> int:
        return len(text)

    async def count_tokens(self, messages: list[LLMMessage]) -> int:
        return sum(len(m.content) for m in messages)


def _chain(*providers: _FakeProvider, **kwargs: Any) -> ResilientLLMService:
    defaults: dict[str, Any] = {"max_retries": 2, "backoff_base": 0.0}
    defaults.update(kwargs)
    return ResilientLLMService([(p.name, p) for p in providers], **defaults)


_MSGS = [LLMMessage(role="user", content="ping")]


class TestRetry:
    async def test_transient_failure_self_heals_on_retry(self):
        primary = _FakeProvider("glm", failures=1)
        service = _chain(primary)
        response = await service.generate(_MSGS)
        assert response.content == "from-glm"
        assert primary.generate_calls == 2  # one failure + one success

    async def test_retry_counts_honor_max_retries(self):
        primary = _FakeProvider("glm", failures=99)
        fallback = _FakeProvider("openai")
        service = _chain(primary, fallback)
        response = await service.generate(_MSGS)
        assert response.content == "from-openai"
        assert primary.generate_calls == 3  # initial + 2 retries

    async def test_non_transient_error_fails_over_without_retry(self):
        primary = _FakeProvider("glm", failures=1, status_code=401)
        fallback = _FakeProvider("openai")
        service = _chain(primary, fallback)
        response = await service.generate(_MSGS)
        assert response.content == "from-openai"
        assert primary.generate_calls == 1  # 401 is never retried

    async def test_all_providers_exhausted_raises_last_error(self):
        primary = _FakeProvider("glm", failures=99)
        fallback = _FakeProvider("openai", failures=99)
        service = _chain(primary, fallback)
        try:
            await service.generate(_MSGS)
            raise AssertionError("should have raised")
        except ExternalServiceError as exc:
            assert "openai" in str(exc)


class TestCircuitBreaker:
    def test_circuit_trips_after_threshold_and_recovers(self):
        breaker = CircuitBreaker(failure_threshold=3, recovery_seconds=0.05)
        for _ in range(3):
            breaker.record_failure()
        assert breaker.state == "open"
        assert breaker.allow() is False

        import time

        time.sleep(0.06)
        assert breaker.state == "half_open"
        assert breaker.allow() is True  # probe allowed
        breaker.record_success()
        assert breaker.state == "closed"

    async def test_open_circuit_skips_provider_entirely(self):
        primary = _FakeProvider("glm", failures=99)
        fallback = _FakeProvider("openai")
        service = _chain(
            primary,
            fallback,
            max_retries=0,
            circuit_failure_threshold=2,
            circuit_recovery_seconds=60.0,
        )
        # Calls one and two: primary is still allowed (threshold=2 not
        # yet reached), fails each time, fallback serves. Call two's
        # failure trips the circuit open.
        await service.generate(_MSGS)
        await service.generate(_MSGS)
        assert primary.generate_calls == 2
        assert service.breaker_state("glm") == "open"
        # Third call: circuit open — primary skipped without a request.
        assert (await service.generate(_MSGS)).content == "from-openai"
        assert primary.generate_calls == 2  # unchanged — skipped
        assert fallback.generate_calls == 3

    async def test_success_resets_failure_count(self):
        primary = _FakeProvider("glm", failures=1)
        service = _chain(
            primary,
            max_retries=2,
            circuit_failure_threshold=3,
            circuit_recovery_seconds=60.0,
        )
        await service.generate(_MSGS)  # 1 failure + 1 success
        await service.generate(_MSGS)  # clean
        assert service.breaker_state("glm") == "closed"


class TestStreamingFailover:
    async def test_immediate_stream_error_fails_over(self):
        primary = _FakeProvider("glm", failures=99)
        fallback = _FakeProvider("openai")
        service = _chain(primary, fallback)
        chunks = [c async for c in service.generate_stream(_MSGS)]
        assert chunks == ["chunk-openai-1", "chunk-openai-2"]

    async def test_midstream_failure_propagates_without_duplicate(self):
        """Once a token reached the consumer, failover must not restart."""

        class _MidstreamDropper(_FakeProvider):
            async def generate_stream(self, messages, max_tokens=None, temperature=None, **kw):
                self.stream_calls += 1
                yield "partial-token"
                raise ExternalServiceError(
                    service=self.name, message="dropped mid-stream", status_code=503
                )

        primary = _MidstreamDropper("glm")
        fallback = _FakeProvider("openai")
        service = _chain(primary, fallback)
        collected: list[str] = []
        try:
            async for chunk in service.generate_stream(_MSGS):
                collected.append(chunk)
            raise AssertionError("should have raised")
        except ExternalServiceError:
            pass
        assert collected == ["partial-token"]  # no duplicated/interleaved output
        assert fallback.stream_calls == 0  # never switched providers mid-stream

    async def test_stream_success_resets_breaker(self):
        primary = _FakeProvider("glm")
        service = _chain(primary)
        chunks = [c async for c in service.generate_stream(_MSGS)]
        assert chunks == ["chunk-glm-1", "chunk-glm-2"]
        assert service.breaker_state("glm") == "closed"


class TestDelegation:
    def test_estimate_tokens_delegates_to_primary(self):
        primary = _FakeProvider("glm")
        fallback = _FakeProvider("openai")
        service = _chain(primary, fallback)
        assert service.estimate_tokens("hello") == 5

    async def test_count_tokens_delegates_to_primary(self):
        primary = _FakeProvider("glm")
        service = _chain(primary)
        assert await service.count_tokens(_MSGS) == 4

    def test_primary_identity_is_mirrored(self):
        primary = _FakeProvider("glm")
        service = _chain(primary)
        assert service.model == "model-glm"

    def test_empty_chain_rejected(self):
        try:
            ResilientLLMService([])
            raise AssertionError("should have raised")
        except ValueError:
            pass
