"""Wider candidate pool before reranking (review #6, top_k slice).

Hybrid search used to be asked for top_k=3 and those same 3 were then
reranked — the reranker could only reorder the top 3, never rescue a
doc the fusion stage ranked 4th-10th. A precision stage that cannot
change the candidate set cannot improve recall, so the node now asks
for a wider pool (RETRIEVAL_CANDIDATE_POOL, default 10) and delivers
the best RETRIEVAL_TOP_K (default 3) after reranking.

The fake hybrid truncates to ``request.top_k`` exactly like the real
HybridSearchService, so the rescue test below fails for the same
reason production missed deep docs — not because the stub is
unfaithful.
"""

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

from app.models.enums.intent import Intent
from app.services.chat.chat_service import ChatService
from app.services.dialogue.graph import build_dialogue_graph
from app.services.llm.base import LLMMessage, LLMResponse, LLMServiceBase

RESCUE_ID = "pool-rescue-target"


class _StubLLM(LLMServiceBase):
    """Minimal generator: records calls, returns fixed content."""

    def __init__(self) -> None:
        super().__init__(api_key="test", model="test")

    async def generate(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        return LLMResponse(content="7-day no-reason returns.", model=self.model)

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        yield "7-day no-reason returns."

    def estimate_tokens(self, text: str) -> int:
        return len(text)

    async def count_tokens(self, messages: list[LLMMessage]) -> int:
        return sum(len(m.content) for m in messages)


class _FakeHit:
    """SearchResult-shaped object as hybrid_search returns."""

    def __init__(self, document_id: str, content: str, score: float) -> None:
        self.document_id = document_id
        self.content = content
        self.score = score
        self.metadata = None


def _truncating_hybrid(hits: list[_FakeHit]) -> Mock:
    """Fake HybridSearchService: serves hits[: request.top_k]."""
    hybrid = Mock()

    async def _search(request: Any) -> list[_FakeHit]:
        return hits[: request.top_k]

    hybrid.search = AsyncMock(side_effect=_search)
    return hybrid


class _RescueReranker:
    """Cross-encoder stand-in: promotes the marker doc, keeps top 5."""

    async def rerank(self, results: list[Any], request: Any) -> list[Any]:
        ranked = sorted(results, key=lambda r: (r.document_id != RESCUE_ID, -r.score))
        return ranked[:5]


def _ten_hits() -> list[_FakeHit]:
    """Hybrid order puts the target 5th — inside the pool, outside a
    top-3-only request."""
    hits = [_FakeHit(f"doc-{i}", f"filler {i}", 0.9 - i / 10) for i in range(10)]
    hits[4] = _FakeHit(RESCUE_ID, "the actual return policy", 0.55)
    return hits


def _chat(hybrid: Mock, reranker: Any = None) -> ChatService:
    pipeline: dict[str, Any] = {"hybrid_search": hybrid}
    if reranker is not None:
        pipeline["reranker"] = reranker
    detector = Mock()
    probe = Mock(intent=Intent.POLICY, confidence=0.9)
    detector.detect_with_confidence = AsyncMock(return_value=probe)
    return ChatService(
        graph=build_dialogue_graph(
            intent_detector=detector,
            slot_filler=None,
            tool_registry=Mock(),
            retrieval_pipeline=pipeline,
            llm_service=_StubLLM(),
        ),
    )


class TestCandidatePool:
    async def test_deep_doc_is_rescued_by_reranker(self) -> None:
        """The doc the reranker scores best sits 5th in hybrid order —
        unreachable when the pool is only as wide as the delivery
        count."""
        chat = _chat(_truncating_hybrid(_ten_hits()), _RescueReranker())

        response = await chat.process_message(1, "What is the return window?", 0)

        assert response.sources is not None
        assert RESCUE_ID in response.sources
        assert response.sources[0] == RESCUE_ID
        # Delivery stays capped: the pool widens recall, not context.
        assert len(response.sources) == 3

    async def test_pool_size_is_requested_not_delivery_count(self) -> None:
        """The hybrid request carries the candidate-pool width (10),
        not the final top_k (3)."""
        hybrid = _truncating_hybrid(_ten_hits())
        chat = _chat(hybrid, _RescueReranker())

        await chat.process_message(1, "What is the return window?", 0)

        request = hybrid.search.await_args.args[0]
        assert request.top_k == 10

    async def test_pool_setting_is_env_tunable(self) -> None:
        """RETRIEVAL_CANDIDATE_POOL overrides the pool width."""
        from app.config.settings import settings

        hybrid = _truncating_hybrid(_ten_hits())
        chat = _chat(hybrid, _RescueReranker())

        with patch.object(settings, "RETRIEVAL_CANDIDATE_POOL", 7, create=True):
            await chat.process_message(1, "What is the return window?", 0)

        request = hybrid.search.await_args.args[0]
        assert request.top_k == 7

    async def test_without_reranker_delivery_stays_capped(self) -> None:
        """No reranker in the pipeline (or a NoOp passthrough): the
        wider pool must not leak 10 docs into the generation context."""
        chat = _chat(_truncating_hybrid(_ten_hits()))

        response = await chat.process_message(1, "What is the return window?", 0)

        assert response.sources is not None
        assert len(response.sources) == 3
        assert RESCUE_ID not in response.sources
