"""PII never reaches the consumer ungated on any branch.

Review finding (docs/reviews/agentic-rag-review-2026-09-26.md, #4):
two leaks. (a) The direct tier's cache-miss generations returned
without the output guardrail — a non-stream direct answer carried a
raw email. (b) On token streams every other branch pushed claim-gated
but un-redacted chunks to the queue and sanitized only at finalize —
the consumer saw the PII, the persisted turn held ``[REDACTED]``.

The invariant: every byte the consumer sees has passed the same PII
redaction as the persisted body, whatever the branch or transport.
"""

from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest

from app.models.enums.intent import Intent
from app.services.chat.chat_service import STREAM_ERROR, ChatService
from app.services.dialogue.graph import build_dialogue_graph
from app.services.guardrails.base import GuardrailService
from app.services.guardrails.output_guard import DefaultOutputGuardrail
from app.services.guardrails.stream_redactor import PIIStreamRedactor
from app.services.llm.base import LLMMessage, LLMResponse, LLMServiceBase

EMAIL = "audit@example.invalid"


class _ChunkedLLM(LLMServiceBase):
    """Streams a fixed chunk sequence — chunk boundaries are the point."""

    def __init__(self, chunks: list[str]) -> None:
        super().__init__(api_key="test", model="test")
        self._chunks = chunks

    async def generate(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        return LLMResponse(content="".join(self._chunks), model=self.model)

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        for chunk in self._chunks:
            yield chunk

    def estimate_tokens(self, text: str) -> int:
        return len(text)

    async def count_tokens(self, messages: list[LLMMessage]) -> int:
        return sum(len(m.content) for m in messages)


@pytest.fixture(params=[False, True], ids=["response", "stream"])
def streaming(request: pytest.FixtureRequest) -> bool:
    return cast(bool, request.param)


def _detector_returning(intent: Intent) -> Mock:
    probe = Mock()
    probe.intent = intent
    probe.confidence = 0.9
    detector = Mock()
    detector.detect_with_confidence = AsyncMock(return_value=probe)
    return detector


def _fake_hybrid_search() -> Mock:
    hit = SimpleNamespace(document_id="doc-1", content="退货政策全文", score=0.9, metadata=None)
    search = Mock()
    search.search = AsyncMock(return_value=[hit])
    return search


def _guardrails() -> GuardrailService:
    return GuardrailService(output_guard=DefaultOutputGuardrail())


async def _turn(chat: ChatService, message: str, streaming: bool) -> tuple[str, list[str]]:
    if not streaming:
        return (await chat.process_message(1, message, 0)).content, []
    chunks = [c async for c in chat.process_message_stream(1, message, 0)]
    assert STREAM_ERROR not in chunks
    return "".join(c for c in chunks if isinstance(c, str)), chunks


async def _state_response(chat: ChatService) -> str:
    snapshot = await chat.graph.aget_state({"configurable": {"thread_id": "1"}})
    return cast(dict[str, Any], snapshot.values).get("response", "")


# ── Part A: direct tier cache-miss generations ──────────────────────────────


class TestDirectTierOutputGuardrail:
    async def test_direct_generation_is_redacted(self, streaming: bool) -> None:
        """The review's non-stream repro: a direct-tier answer carrying
        an email must come back with ``[REDACTED]``, not the raw PII."""
        llm = _ChunkedLLM([f"联络邮箱：{EMAIL}。"])
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
                guardrail_service=_guardrails(),
                semantic_cache=semantic_cache,
            )
        )

        content, _ = await _turn(chat, "客服邮箱多少", streaming)

        assert EMAIL not in content
        assert "[REDACTED]" in content
        # L1 stores the clean body — a cached replay must be redacted too.
        assert semantic_cache.put.await_count == 1
        assert EMAIL not in semantic_cache.put.await_args.args[1].response


# ── Part B: token streams ───────────────────────────────────────────────────


class TestStreamRedaction:
    async def test_rag_stream_never_emits_raw_pii(self) -> None:
        """The review's split-chunk repro: the email arrives whole in
        the second chunk. The consumer must never see it, and the
        persisted turn must equal what the consumer saw."""
        llm = _ChunkedLLM(["联络邮箱：", f"{EMAIL}。"])
        chat = ChatService(
            graph=build_dialogue_graph(
                intent_detector=_detector_returning(Intent.POLICY),
                slot_filler=None,
                tool_registry=Mock(),
                retrieval_pipeline={"hybrid_search": _fake_hybrid_search()},
                llm_service=llm,
                guardrail_service=_guardrails(),
            )
        )

        content, chunks = await _turn(chat, "退货政策是什么", True)

        assert EMAIL not in content
        assert "[REDACTED]" in content
        # Stream and persistence agree byte-for-byte.
        assert await _state_response(chat) == content

    async def test_chitchat_stream_split_phone_is_redacted(self) -> None:
        """No fact subgraph on this message, so the claim gate is not
        armed — the raw token path. A phone number split across chunk
        boundaries must still be caught by the stream redactor."""
        llm = _ChunkedLLM(["手机号 ", "138", "00138000", "。"])
        chat = ChatService(
            graph=build_dialogue_graph(
                intent_detector=_detector_returning(Intent.CHITCHAT),
                slot_filler=None,
                tool_registry=Mock(),
                retrieval_pipeline={},
                llm_service=llm,
                guardrail_service=_guardrails(),
            )
        )

        content, _ = await _turn(chat, "客服电话多少", True)

        assert "13800138000" not in content
        assert "[REDACTED]" in content
        assert await _state_response(chat) == content

    async def test_non_stream_rag_redaction_unchanged(self) -> None:
        """Regression pin: the non-stream RAG branch already sanitized
        at finalize — the fix must not regress it."""
        llm = _ChunkedLLM([f"联络邮箱：{EMAIL}。"])
        chat = ChatService(
            graph=build_dialogue_graph(
                intent_detector=_detector_returning(Intent.POLICY),
                slot_filler=None,
                tool_registry=Mock(),
                retrieval_pipeline={"hybrid_search": _fake_hybrid_search()},
                llm_service=llm,
                guardrail_service=_guardrails(),
            )
        )

        content, _ = await _turn(chat, "退货政策是什么", False)

        assert EMAIL not in content
        assert "[REDACTED]" in content


# ── PIIStreamRedactor unit contract ─────────────────────────────────────────


class TestPIIStreamRedactorUnit:
    def test_split_phone_across_feeds(self) -> None:
        """A phone split mid-number must be held until the sentence
        completes, then released redacted — never partially."""
        r = PIIStreamRedactor(_guardrails())
        assert r.feed("手机号 138") == ""
        out = r.feed("00138000。")
        assert "13800138000" not in out
        assert "[REDACTED]" in out
        assert r.flush() == ""

    def test_blocked_sentence_is_dropped(self) -> None:
        guard = Mock()
        result = Mock()
        result.was_blocked = True
        result.sanitized_content = ""
        guard.check_output = Mock(return_value=result)
        r = PIIStreamRedactor(guard)
        assert r.feed("违规句。") == ""
        assert r.flush() == ""

    def test_outage_releases_text_unchanged(self) -> None:
        guard = Mock()
        guard.check_output = Mock(side_effect=RuntimeError("boom"))
        r = PIIStreamRedactor(guard)
        assert r.feed("正常句子。") == "正常句子。"
