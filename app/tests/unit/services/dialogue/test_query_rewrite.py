"""Retrieval-side multi-turn query rewriting (review #6).

「那运费呢？」 arriving after a turn about iPhone 15 Pro used to hit
retrieval verbatim — both the vector and BM25 legs saw a query whose
subject only exists in the conversation, not in the string, because
history joined the pipeline at generation time, never before search.

Contract: when recent history exists, the retrieval query is first
condensed into a standalone question (one cheap LLM call, flag
controlled); without history, with the flag off, or on any rewrite
failure the pipeline searches the original message unchanged — the
rewrite is recall leverage, never a new failure mode.
"""

from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

from app.config.settings import settings
from app.models.enums.intent import Intent
from app.services.chat.chat_service import ChatService
from app.services.dialogue.graph import build_dialogue_graph
from app.services.llm.base import LLMMessage, LLMResponse, LLMServiceBase

HISTORY = [
    LLMMessage(role="user", content="iPhone 15 Pro 的屏幕怎么样？"),
    LLMMessage(role="assistant", content="6.1 英寸 OLED 直屏。"),
]


class _RewriteLLM(LLMServiceBase):
    """Serves the condense prompt (system message mentions 改写) from
    ``rewrite_output`` and every other generation from ``answer``.

    ``rewrite_output`` may be an Exception to exercise the fail-open
    contract without a broken stub.
    """

    def __init__(self, rewrite_output: str | Exception, answer: str) -> None:
        super().__init__(api_key="test", model="test")
        self._rewrite_output = rewrite_output
        self._answer = answer
        self.rewrite_calls = 0
        self.gen_calls = 0

    async def generate(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        first = messages[0].content if messages else ""
        if "改写" in first:
            self.rewrite_calls += 1
            if isinstance(self._rewrite_output, Exception):
                raise self._rewrite_output
            return LLMResponse(content=self._rewrite_output, model=self.model)
        self.gen_calls += 1
        return LLMResponse(content=self._answer, model=self.model)

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        yield self._answer

    def estimate_tokens(self, text: str) -> int:
        return len(text)

    async def count_tokens(self, messages: list[LLMMessage]) -> int:
        return sum(len(m.content) for m in messages)


def _detector_returning(intent: Intent) -> Mock:
    probe = Mock()
    probe.intent = intent
    probe.confidence = 0.9
    detector = Mock()
    detector.detect_with_confidence = AsyncMock(return_value=probe)
    return detector


def _hybrid() -> AsyncMock:
    search = AsyncMock(
        return_value=[
            SimpleNamespace(document_id="d1", content="iPhone 15 Pro 顺丰包邮", score=0.9)
        ]
    )
    return search


def _chat(llm: _RewriteLLM, hybrid: AsyncMock, history: list[LLMMessage]) -> ChatService:
    async def history_provider(session_id: int) -> list[LLMMessage]:
        return list(history)

    hybrid_mock = Mock()
    hybrid_mock.search = hybrid
    return ChatService(
        graph=build_dialogue_graph(
            intent_detector=_detector_returning(Intent.POLICY),
            slot_filler=None,
            tool_registry=Mock(),
            retrieval_pipeline={"hybrid_search": hybrid_mock},
            llm_service=llm,
            history_provider=history_provider,
        ),
    )


def _request_of(search: AsyncMock) -> Any:
    """The VectorSearchRequest the hybrid leg received (asserts it ran)."""
    assert search.await_args is not None, "hybrid search was never awaited"
    return search.await_args.args[0]


class TestQueryRewrite:
    async def test_followup_is_condensed_before_retrieval(self) -> None:
        """The review's repro: a follow-up whose subject lives in history
        must reach the hybrid leg as a standalone question, not as the
        bare fragment."""
        llm = _RewriteLLM("iPhone 15 Pro 的运费政策", "运费政策：顺丰包邮。")
        hybrid = _hybrid()
        chat = _chat(llm, hybrid, HISTORY)

        response = await chat.process_message(1, "那运费呢？", 0)

        hybrid.assert_awaited_once()
        request = _request_of(hybrid)
        assert request.query == "iPhone 15 Pro 的运费政策"
        assert llm.rewrite_calls == 1
        # The turn itself still completes through the normal RAG path.
        assert "顺丰包邮" in response.content
        assert llm.gen_calls >= 1

    async def test_first_turn_searches_verbatim(self) -> None:
        """No history → nothing to condense against: zero rewrite calls,
        the raw message is the query."""
        llm = _RewriteLLM("不应该被调用", "回答。")
        hybrid = _hybrid()
        chat = _chat(llm, hybrid, [])

        await chat.process_message(1, "退货政策是什么", 0)

        assert llm.rewrite_calls == 0
        request = _request_of(hybrid)
        assert request.query == "退货政策是什么"

    async def test_rewrite_failure_falls_back_to_original(self) -> None:
        """A condense-call failure is fail-open: the search runs with the
        original fragment and the turn still answers — the rewrite must
        not become a new outage path."""
        llm = _RewriteLLM(RuntimeError("llm down"), "按知识库的回答。")
        hybrid = _hybrid()
        chat = _chat(llm, hybrid, HISTORY)

        response = await chat.process_message(1, "那运费呢？", 0)

        assert llm.rewrite_calls == 1
        request = _request_of(hybrid)
        assert request.query == "那运费呢？"
        assert "知识库" in response.content

    async def test_empty_or_oversized_rewrite_keeps_original(self) -> None:
        """Empty and runaway rewrite outputs are useless as queries —
        both fall back to the original message."""
        for bad in ("", "  ", "废" * 400):
            llm = _RewriteLLM(bad, "回答。")
            hybrid = _hybrid()
            chat = _chat(llm, hybrid, HISTORY)

            await chat.process_message(1, "那运费呢？", 0)

            request = _request_of(hybrid)
            assert request.query == "那运费呢？", f"rewrite {bad[:10]!r} leaked into the query"

    async def test_disabled_flag_skips_the_condense_call(self, monkeypatch: Any) -> None:
        """QUERY_REWRITE_ENABLED=False turns the feature fully off: no
        LLM call, verbatim query."""
        monkeypatch.setattr(settings, "QUERY_REWRITE_ENABLED", False)
        llm = _RewriteLLM("不应该被调用", "回答。")
        hybrid = _hybrid()
        chat = _chat(llm, hybrid, HISTORY)

        await chat.process_message(1, "那运费呢？", 0)

        assert llm.rewrite_calls == 0
        request = _request_of(hybrid)
        assert request.query == "那运费呢？"
