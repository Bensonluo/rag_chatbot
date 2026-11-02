"""General-knowledge lane: model answers for general-world questions.

A general-world question (头疼怎么办) used to land in the no-evidence
refusal: the KB only holds shop policy, so an honest shop-scoped miss
was also the only outcome for world knowledge. The lane added here
splits the two — the LLM intent leg positively classifies
general_knowledge, the direct tier answers from model knowledge under
a deterministic disclaimer prefix, and the shop-scoped miss path
(faq/policy with an empty KB slice) keeps the fixed no-evidence copy
(review 2026-09-26, #6) with no disclaimer.

Contracts under test: the disclaimer is deterministic zh/en copy
prepended to the body (never cached into it), the generation runs the
general persona (not the CS persona), chitchat turns carry no
disclaimer, a shop-policy miss still refuses with zero LLM calls, and
on streamed turns the disclaimer reaches the queue before any body
token while the persisted text equals the streamed text.
"""

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, Mock

from app.models.enums.intent import Intent
from app.services.chat.chat_service import ChatService
from app.services.dialogue.graph import build_dialogue_graph
from app.services.dialogue.i18n import GENERAL_KNOWLEDGE_DISCLAIMER
from app.services.dialogue.nodes import NodeFactory
from app.services.llm.base import LLMMessage, LLMResponse, LLMServiceBase

DISCLAIMER_ZH = GENERAL_KNOWLEDGE_DISCLAIMER["zh"]
DISCLAIMER_EN = GENERAL_KNOWLEDGE_DISCLAIMER["en"]
NO_EVIDENCE_MARKER = "暂未在知识库中找到"


class _RecordingLLM(LLMServiceBase):
    """Records every system prompt; returns a fixed body (stream or not)."""

    def __init__(self, body: str, tokens: list[str] | None = None) -> None:
        super().__init__(api_key="test", model="test")
        self._body = body
        self._tokens = tokens if tokens is not None else [body]
        self.calls = 0
        self.system_prompts: list[str] = []

    async def generate(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.calls += 1
        self.system_prompts.append(messages[0].content)
        return LLMResponse(content=self._body, model=self.model)

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        self.calls += 1
        self.system_prompts.append(messages[0].content)
        for token in self._tokens:
            yield token

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


def _chat(llm: _RecordingLLM, hybrid: Mock, intent: Intent) -> ChatService:
    return ChatService(
        graph=build_dialogue_graph(
            intent_detector=_detector_returning(intent),
            slot_filler=None,
            tool_registry=Mock(),
            retrieval_pipeline={"hybrid_search": hybrid},
            llm_service=llm,
        ),
    )


class TestGeneralKnowledgeLane:
    async def test_general_question_answered_with_disclaimer(self) -> None:
        """头疼怎么办: generated answer under the disclaimer, no refusal,
        and retrieval never runs (the lane skips the KB entirely)."""
        llm = _RecordingLLM("头疼时可以先休息、补水、避免长时间看屏幕；若持续或加重请及时就医。")
        search = AsyncMock(return_value=[])
        chat = _chat(llm, _hybrid(search), Intent.GENERAL_KNOWLEDGE)

        response = await chat.process_message(1, "头疼怎么办", 0)

        assert DISCLAIMER_ZH in response.content
        assert NO_EVIDENCE_MARKER not in response.content
        assert llm.calls == 1
        search.assert_not_called()

    async def test_general_persona_used_for_generation(self) -> None:
        """The lane's persona is the general-knowledge one — the CS
        persona's shop-grounding rules must not frame a world-knowledge
        answer."""
        llm = _RecordingLLM("多休息，必要时就医。")
        chat = _chat(llm, _hybrid(AsyncMock(return_value=[])), Intent.GENERAL_KNOWLEDGE)

        await chat.process_message(1, "头疼怎么办", 0)

        system = llm.system_prompts[0]
        assert "无关的通识问题" in system
        assert "只依据通用常识回答" in system

    async def test_english_turn_gets_english_disclaimer_and_persona(self) -> None:
        """Global audience: EN turn → EN disclaimer and EN persona."""
        llm = _RecordingLLM("Rest, hydrate, step away from screens; see a doctor if it persists.")
        chat = _chat(llm, _hybrid(AsyncMock(return_value=[])), Intent.GENERAL_KNOWLEDGE)

        response = await chat.process_message(1, "how do I deal with a headache", 0)

        assert DISCLAIMER_EN in response.content
        assert "unrelated to this store" in llm.system_prompts[0]

    async def test_chitchat_turn_has_no_disclaimer(self) -> None:
        """The plain direct tier keeps its CS persona and copy — the
        disclaimer marks ONLY general-knowledge answers."""
        llm = _RecordingLLM("哈哈，有什么可以帮您？")
        chat = _chat(llm, _hybrid(AsyncMock(return_value=[])), Intent.CHITCHAT)

        response = await chat.process_message(1, "哈哈", 0)

        assert DISCLAIMER_ZH not in response.content
        assert llm.calls == 1

    async def test_shop_policy_miss_keeps_no_evidence(self) -> None:
        """The guard rail is intact: a shop-scoped question with an empty
        KB still gets the fixed refusal (zero LLM calls), and the refusal
        must not wear the general-reference label."""
        llm = _RecordingLLM("不应该出现。")
        chat = _chat(llm, _hybrid(AsyncMock(return_value=[])), Intent.POLICY)

        response = await chat.process_message(1, "隐私政策是什么", 0)

        assert NO_EVIDENCE_MARKER in response.content
        assert DISCLAIMER_ZH not in response.content
        assert llm.calls == 0


class TestGeneralLaneStreaming:
    async def test_disclaimer_streams_before_body_and_matches_persisted(self) -> None:
        """Streamed turns see the disclaimer first, then the body tokens;
        the persisted response equals the streamed text concatenated."""
        import asyncio

        llm = _RecordingLLM("多休息，必要时就医。", tokens=["多休息，", "必要时就医。"])
        factory = NodeFactory(
            intent_detector=Mock(),
            slot_filler=None,
            tool_registry=Mock(),
            llm_service=llm,
        )
        queue: asyncio.Queue[Any] = asyncio.Queue()
        config: dict[str, Any] = {"configurable": {"thread_id": "t", "stream_queue": queue}}
        state: dict[str, Any] = {"message": "头疼怎么办", "intent": "general_knowledge"}

        updates = await factory.direct_response_node(state, config)  # type: ignore[arg-type]

        items: list[str] = []
        while not queue.empty():
            items.append(queue.get_nowait())
        assert items[0] == DISCLAIMER_ZH
        assert "".join(items) == updates["response"]
        assert updates["response"].startswith(DISCLAIMER_ZH)
