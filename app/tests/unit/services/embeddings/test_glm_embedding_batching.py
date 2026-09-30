"""GLM embedding batching against the platform's 64-item input cap.

GLM's embedding API rejects input arrays larger than 64 items with
error 1214 ("input数组最大不得超过64条"). The shipped FAQ table
expands to 83 embed inputs (questions + variants), so one bulk call
fails the whole table build and disables the FAQ fast path per
process (observed live on the server, 2026-09-30). embed() must
chunk inputs at the cap and reassemble results in order.
"""

import asyncio
import json

import httpx
import pytest

from app.core.exceptions import ExternalServiceError
from app.services.embeddings.glm_embeddings import _MAX_BATCH_ITEMS, GLMEmbeddingService

# The platform cap GLM enforces (error 1214).
GLM_INPUT_CAP = 64


def _recording_service(calls: list[list[str]]) -> GLMEmbeddingService:
    """Build a GLMEmbeddingService whose transport records each batch."""

    def handler(request: httpx.Request) -> httpx.Response:
        batch: list[str] = json.loads(request.content)["input"]
        calls.append(batch)
        data = [{"embedding": [float(len(text)), 1.0], "index": i} for i, text in enumerate(batch)]
        return httpx.Response(
            200,
            json={"object": "list", "data": data, "usage": {"total_tokens": len(batch)}},
        )

    service = GLMEmbeddingService(api_key="id.secret")
    service.client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://glm.test/v4/"
    )
    return service


class TestGLMEmbeddingBatching:
    async def test_inputs_over_cap_are_chunked_in_order(self) -> None:
        calls: list[list[str]] = []
        service = _recording_service(calls)
        texts = [f"faq-{i}" for i in range(83)]  # shipped table size

        result = await service.embed(texts)

        assert [len(batch) for batch in calls] == [64, 19]
        # Reassembly preserves global order and per-item values.
        assert result.embeddings == [[float(len(t)), 1.0] for t in texts]
        assert result.tokens_used == 83
        await service.close()

    async def test_inputs_within_cap_stay_one_call(self) -> None:
        calls: list[list[str]] = []
        service = _recording_service(calls)

        await service.embed(["a", "b", "c"])

        assert calls == [["a", "b", "c"]]
        await service.close()

    async def test_short_batch_response_raises_instead_of_misaligning(self) -> None:
        # A batch that comes back with fewer vectors than inputs would
        # silently misalign owners<->vectors downstream; fail loudly.

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [{"embedding": [0.0], "index": 0}],
                    "usage": {"total_tokens": 1},
                },
            )

        service = GLMEmbeddingService(api_key="id.secret")
        service.client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="https://glm.test/v4/"
        )

        with pytest.raises(ExternalServiceError):
            await service.embed(["a", "b"])
        await service.close()


class _ConcurrencyTrackingTransport(httpx.AsyncBaseTransport):
    """Async transport that records batches and peak in-flight requests."""

    def __init__(self) -> None:
        self.batches: list[list[str]] = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.batches.append(json.loads(request.content)["input"])
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(0.01)  # widen the window for overlap
        self.in_flight -= 1
        data = [
            {"embedding": [float(len(text))], "index": i} for i, text in enumerate(self.batches[-1])
        ]
        return httpx.Response(
            200,
            json={"object": "list", "data": data, "usage": {"total_tokens": len(data)}},
        )


async def test_chunks_are_sent_sequentially_not_concurrently() -> None:
    # The Coding Plan package has a low concurrent-request ceiling;
    # chunked embeds must never overlap.
    transport = _ConcurrencyTrackingTransport()
    service = GLMEmbeddingService(api_key="id.secret")
    service.client = httpx.AsyncClient(transport=transport, base_url="https://glm.test/v4/")

    result = await service.embed([f"doc-{i}" for i in range(200)])

    assert [len(batch) for batch in transport.batches] == [64, 64, 64, 8]
    assert transport.max_in_flight == 1
    assert len(result.embeddings) == 200
    await service.close()


def test_platform_cap_constant_matches_observed_error() -> None:
    # The chunk size must track the platform limit from error 1214.
    assert _MAX_BATCH_ITEMS == GLM_INPUT_CAP
