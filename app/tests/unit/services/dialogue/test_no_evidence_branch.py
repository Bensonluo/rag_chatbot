"""Deterministic no-evidence branch for knowledge questions (review #6).

When a knowledge-intent turn runs retrieval and every leg comes back
empty, the graph used to fall through to direct LLM generation with no
context — for a customer-service bot that is policy-hallucination
risk: the model invents return windows instead of admitting the KB has
nothing. And a retrieval leg that *raised* looked identical to an
honest empty result, so the user got a confident guess during an
outage.

Contracts under test: an empty retrieval yields a fixed no-evidence
response with zero LLM calls; a failed retrieval leg yields a
service-unavailable response (a different sentence, because the right
UX differs — retry/handoff vs rephrase); turns without retrieval keep
using the LLM; the flags reset between turns.
"""

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, Mock

from langgraph.checkpoint.memory import MemorySaver

from app.models.enums.intent import Intent
from app.services.chat.chat_service import ChatService
from app.services.dialogue.graph import build_dialogue_graph
from app.services.llm.base import LLMMessage, LLMResponse, LLMServiceBase
from app.services.retrieval.vector_base import VectorClientError

if TYPE_CHECKING:
    from app.services.graph.retrieval.graph_retrieval_service import GraphRetrievalService

NO_EVIDENCE_MARKER = "暂未在知识库中找到"
DEGRADED_MARKER = "检索服务暂时不可用"


class _CountingLLM(LLMServiceBase):
    """Records every generation call; streams a fixed chunk sequence."""

    def __init__(self, chunks: list[str]) -> None:
        super().__init__(api_key="test", model="test")
        self._chunks = chunks
        self.calls = 0

    async def generate(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.calls += 1
        return LLMResponse(content="".join(self._chunks), model=self.model)

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        self.calls += 1
        for chunk in self._chunks:
            yield chunk

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


def _hybrid(search: AsyncMock) -> Mock:
    hybrid = Mock()
    hybrid.search = search
    return hybrid


def _chat(llm: _CountingLLM, hybrid: Mock, **graph_kwargs: Any) -> ChatService:
    return ChatService(
        graph=build_dialogue_graph(
            intent_detector=_detector_returning(graph_kwargs.pop("intent", Intent.POLICY)),
            slot_filler=None,
            tool_registry=Mock(),
            retrieval_pipeline={"hybrid_search": hybrid, **graph_kwargs.pop("pipeline_extras", {})},
            llm_service=llm,
            **graph_kwargs,
        ),
    )


class TestNoEvidenceBranch:
    async def test_empty_retrieval_gets_deterministic_response(self) -> None:
        """KB has nothing for the question: the answer is fixed copy,
        not a free LLM generation — zero model calls on this turn."""
        llm = _CountingLLM(["七天无理由退货。"])
        chat = _chat(llm, _hybrid(AsyncMock(return_value=[])))

        response = await chat.process_message(1, "洗衣机保修几年", 0)

        assert NO_EVIDENCE_MARKER in response.content
        assert response.sources == []
        assert llm.calls == 0

    async def test_retrieval_crash_reports_service_failure(self) -> None:
        """A raised retrieval leg is an outage, not missing knowledge —
        the copy must say the service is unavailable, still without
        calling the LLM."""
        llm = _CountingLLM(["不应该出现。"])
        chat = _chat(llm, _hybrid(AsyncMock(side_effect=RuntimeError("qdrant down"))))

        response = await chat.process_message(1, "洗衣机保修几年", 0)

        assert DEGRADED_MARKER in response.content
        assert llm.calls == 0

    async def test_both_legs_down_flags_degraded(self) -> None:
        """HybridSearch raises VectorClientError when both legs fail;
        the node normalizes it to [] internally (filter-retry contract)
        — the empty outcome must still surface as degraded, not as
        'no knowledge'."""
        llm = _CountingLLM(["不应该出现。"])
        chat = _chat(llm, _hybrid(AsyncMock(side_effect=VectorClientError("both legs down"))))

        response = await chat.process_message(1, "洗衣机保修几年", 0)

        assert DEGRADED_MARKER in response.content
        assert llm.calls == 0

    async def test_graph_intent_empty_results_get_no_evidence(self) -> None:
        """Graph-intent turns with an empty graph answer take the same
        branch — the guarantee is per-turn, not per-leg."""
        llm = _CountingLLM(["编造的关系。"])
        graph_service = Mock()
        graph_service.search = AsyncMock(return_value=[])
        chat = ChatService(
            graph=build_dialogue_graph(
                intent_detector=_detector_returning(Intent.GLOBAL_SUMMARY),
                slot_filler=None,
                tool_registry=Mock(),
                retrieval_pipeline={},
                llm_service=llm,
                graph_retrieval_service=cast("GraphRetrievalService", graph_service),
            ),
        )

        response = await chat.process_message(1, "总结一下品牌之间的关系", 0)

        assert NO_EVIDENCE_MARKER in response.content
        assert llm.calls == 0


class TestBranchScoping:
    async def test_chitchat_turn_still_uses_llm(self) -> None:
        """The branch must not hijack turns that never ran retrieval:
        chitchat keeps its normal LLM answer."""
        llm = _CountingLLM(["您好，很高兴为您服务。"])
        chat = _chat(llm, _hybrid(AsyncMock(return_value=[])), intent=Intent.CHITCHAT)

        response = await chat.process_message(1, "你好呀", 0)

        assert NO_EVIDENCE_MARKER not in response.content
        assert llm.calls >= 1

    async def test_flags_reset_between_turns(self) -> None:
        """A no-evidence turn must not poison the next one: with a
        checkpointer the follow-up chitchat turn gets a fresh LLM
        answer (retrieval_ran reset by begin_turn)."""
        llm = _CountingLLM(["您好，很高兴为您服务。"])
        detector = Mock()
        policy_probe = Mock(intent=Intent.POLICY, confidence=0.9)
        chitchat_probe = Mock(intent=Intent.CHITCHAT, confidence=0.9)
        detector.detect_with_confidence = AsyncMock(side_effect=[policy_probe, chitchat_probe])
        chat = ChatService(
            graph=build_dialogue_graph(
                intent_detector=detector,
                slot_filler=None,
                tool_registry=Mock(),
                retrieval_pipeline={"hybrid_search": _hybrid(AsyncMock(return_value=[]))},
                llm_service=llm,
                checkpointer=MemorySaver(),
            ),
        )

        first = await chat.process_message(9, "洗衣机保修几年", 0)
        second = await chat.process_message(9, "你好呀", 0)

        assert NO_EVIDENCE_MARKER in first.content
        assert NO_EVIDENCE_MARKER not in second.content
        assert llm.calls == 1  # only the chitchat turn reached the model
