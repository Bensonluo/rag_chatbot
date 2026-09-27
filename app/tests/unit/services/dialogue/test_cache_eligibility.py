"""History-derived answers never enter the shared caches.

Review finding (docs/reviews/agentic-rag-review-2026-09-26.md, #5):
generators fold session history into every prompt, but the L0 answer
cache and L1 semantic cache judged a turn "stateless" by user_id and
pending state alone. An anonymous session's history-derived answer was
written to a shared cache and replayed verbatim at any later visitor —
cross-session context leakage by construction.

The invariant: a shared-cache write requires a context-free generation.
When the prompt folded in prior turns (history provider returned any)
or cross-session user facts, the answer is personal to that dialogue
and must not be stored, whatever the other eligibility gates say.
"""

from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest

from app.models.enums.intent import Intent
from app.services.chat.chat_service import STREAM_ERROR, ChatService
from app.services.dialogue.graph import build_dialogue_graph
from app.services.llm.base import LLMMessage, LLMResponse, LLMServiceBase

PRIOR_TURN = [LLMMessage(role="user", content="我使用 A 款设备")]


class _RecordingLLM(LLMServiceBase):
    """Records the full message list per call — history folding is
    asserted on what the model actually saw, not on wiring."""

    def __init__(self) -> None:
        super().__init__(api_key="test", model="test")
        self.calls: list[list[LLMMessage]] = []

    def _answer(self, messages: list[LLMMessage]) -> str:
        self.calls.append(list(messages))
        prompt = messages[-1].content
        if "参考资料:" in prompt:
            return "请根据退货政策办理。"
        return "您好，有什么可以帮您？"

    async def generate(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        return LLMResponse(content=self._answer(messages), model=self.model)

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        yield self._answer(messages)

    def estimate_tokens(self, text: str) -> int:
        return len(text)

    async def count_tokens(self, messages: list[LLMMessage]) -> int:
        return sum(len(message.content) for message in messages)


@pytest.fixture(params=[False, True], ids=["response", "stream"])
def streaming(request: pytest.FixtureRequest) -> bool:
    return cast(bool, request.param)


@pytest.fixture
def llm() -> _RecordingLLM:
    return _RecordingLLM()


def _detector_returning(intent: Intent) -> Mock:
    probe = Mock()
    probe.intent = intent
    probe.confidence = 0.9
    detector = Mock()
    detector.detect_with_confidence = AsyncMock(return_value=probe)
    return detector


def _fake_hybrid_search() -> Mock:
    """One grounded doc per search — gives RAG turns real sources."""
    hit = SimpleNamespace(document_id="doc-1", content="退货政策全文", score=0.9, metadata=None)
    search = Mock()
    search.search = AsyncMock(return_value=[hit])
    return search


async def _turn(chat: ChatService, message: str, streaming: bool) -> str:
    if not streaming:
        return (await chat.process_message(1, message, 0)).content
    chunks = [chunk async for chunk in chat.process_message_stream(1, message, 0)]
    assert STREAM_ERROR not in chunks
    return "".join(chunk for chunk in chunks if isinstance(chunk, str))


# ── L1 semantic cache: direct tier ──────────────────────────────────────────


class TestSemanticCacheEligibility:
    async def test_history_folded_turn_is_not_cached(
        self, llm: _RecordingLLM, streaming: bool
    ) -> None:
        """Second direct-tier turn with history in the prompt must not
        write L1 — a later visitor would replay this dialogue's context.
        First turn (no history) is the positive control: it must write."""
        semantic_cache = Mock()
        semantic_cache.get = AsyncMock(return_value=None)
        semantic_cache.put = AsyncMock()
        history = AsyncMock(side_effect=[[], list(PRIOR_TURN)])

        chat = ChatService(
            graph=build_dialogue_graph(
                intent_detector=_detector_returning(Intent.CHITCHAT),
                slot_filler=None,
                tool_registry=Mock(),
                retrieval_pipeline={},
                llm_service=llm,
                semantic_cache=semantic_cache,
                history_provider=history,
            )
        )

        await _turn(chat, "你好呀", streaming)
        assert semantic_cache.put.await_count == 1

        await _turn(chat, "有意思的地方是哪里", streaming)

        # History really was in the second prompt: persona + prior + user.
        assert len(llm.calls[1]) == 3
        assert semantic_cache.put.await_count == 1  # not written again


# ── L0 answer cache: RAG tier ───────────────────────────────────────────────


class TestAnswerCacheEligibility:
    async def test_history_folded_rag_answer_is_not_cached(
        self, llm: _RecordingLLM, streaming: bool
    ) -> None:
        """A grounded RAG answer generated with session history in the
        prompt is personal to this dialogue — L0 must not store it. The
        no-history first turn is the positive control."""
        answer_cache = Mock()
        answer_cache.get = AsyncMock(return_value=None)
        answer_cache.put = AsyncMock()
        history = AsyncMock(side_effect=[[], list(PRIOR_TURN)])

        chat = ChatService(
            graph=build_dialogue_graph(
                intent_detector=_detector_returning(Intent.POLICY),
                slot_filler=None,
                tool_registry=Mock(),
                retrieval_pipeline={"hybrid_search": _fake_hybrid_search()},
                llm_service=llm,
                answer_cache=answer_cache,
                history_provider=history,
            ),
            answer_cache=answer_cache,
        )

        await _turn(chat, "退货政策是什么", streaming)
        assert answer_cache.put.await_count == 1

        await _turn(chat, "退款政策是什么", streaming)

        # Turn 2 added a retrieval-condense call (system prompt mentions
        # 改写); the generation call is the other one — and it is the one
        # that must still fold history into the RAG prompt.
        gen = next(m for m in llm.calls[1:] if "改写" not in m[0].content)
        assert len(gen) == 3
        assert answer_cache.put.await_count == 1  # history turn not cached

    async def test_anonymous_stateless_rag_answer_still_cached(self, llm: _RecordingLLM) -> None:
        """The eligibility tightening must not over-reach: a genuinely
        context-free grounded turn (anonymous, no history) still lands
        in L0 — that replay is the cache's entire purpose."""
        answer_cache = Mock()
        answer_cache.get = AsyncMock(return_value=None)
        answer_cache.put = AsyncMock()

        chat = ChatService(
            graph=build_dialogue_graph(
                intent_detector=_detector_returning(Intent.POLICY),
                slot_filler=None,
                tool_registry=Mock(),
                retrieval_pipeline={"hybrid_search": _fake_hybrid_search()},
                llm_service=llm,
                answer_cache=answer_cache,
                history_provider=AsyncMock(return_value=[]),
            ),
            answer_cache=answer_cache,
        )

        await _turn(chat, "退货政策是什么", False)

        answer_cache.put.assert_awaited_once()


# ── LLM view: the provider's emptiness decides, not wiring intent ───────────


class TestHistoryProviderContract:
    async def test_empty_history_does_not_penalize_caching(
        self, llm: _RecordingLLM, streaming: bool
    ) -> None:
        """A session whose provider returns no prior turns is
        context-free: the direct tier still writes L1 on every such
        turn (regression guard against gating on provider presence)."""
        semantic_cache = Mock()
        semantic_cache.get = AsyncMock(return_value=None)
        semantic_cache.put = AsyncMock()

        chat = ChatService(
            graph=build_dialogue_graph(
                intent_detector=_detector_returning(Intent.CHITCHAT),
                slot_filler=None,
                tool_registry=Mock(),
                retrieval_pipeline={},
                llm_service=llm,
                semantic_cache=semantic_cache,
                history_provider=AsyncMock(return_value=[]),
            )
        )

        await _turn(chat, "你好呀", streaming)
        await _turn(chat, "有意思的地方是哪里", streaming)

        assert semantic_cache.put.await_count == 2
