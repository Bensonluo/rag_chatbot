"""Language-following behavior for the demo's international visitors.

A live demo session showed the failure: "who are you?" was answered
in Chinese, "why not english" got a Chinese explanation that the
service defaults to Simplified Chinese, and only an explicit "english
please" switched the language. Root cause: both personas hard-instruct
「用简体中文回答」, and every deterministic branch (evidence-gap copy,
guardrail refusal, handoff acknowledgement, generation fallback) is
Chinese-only canned text.

Contracts under test: the personas instruct the model to answer in
the user's language; the deterministic branches pick their canned
copy by the same detection (Chinese default); slot-collection and
the irreversible-confirmation gate stay Chinese-only by design.
"""

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, Mock

from app.models.enums.intent import Intent
from app.services.agent.service import SYSTEM_PROMPT as AGENT_SYSTEM_PROMPT
from app.services.chat.chat_service import ChatService
from app.services.dialogue.graph import build_dialogue_graph
from app.services.llm.base import LLMMessage, LLMResponse, LLMServiceBase
from app.services.llm.prompt_templates import PromptTemplates


class _CountingLLM(LLMServiceBase):
    """Records generation calls; streams a fixed chunk."""

    def __init__(self) -> None:
        super().__init__(api_key="test", model="test")
        self.calls = 0

    async def generate(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.calls += 1
        return LLMResponse(content="回答。", model=self.model)

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        self.calls += 1
        yield "回答。"

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


def _hybrid_returning(results: list[Any]) -> Mock:
    hybrid = Mock()
    hybrid.search = AsyncMock(return_value=results)
    return hybrid


def _chat(llm: _CountingLLM, *, intent: Intent, pipeline: dict[str, Any]) -> ChatService:
    return ChatService(
        graph=build_dialogue_graph(
            intent_detector=_detector_returning(intent),
            slot_filler=None,
            tool_registry=Mock(),
            retrieval_pipeline=pipeline,
            llm_service=llm,
        ),
    )


class TestLanguageDetection:
    def test_english_message_detected(self) -> None:
        from app.services.dialogue.i18n import detect_language

        assert detect_language("what is your return policy?") == "en"
        assert detect_language("english please") == "en"

    def test_chinese_message_detected(self) -> None:
        from app.services.dialogue.i18n import detect_language

        assert detect_language("退货政策是什么") == "zh"

    def test_mixed_keeps_chinese(self) -> None:
        from app.services.dialogue.i18n import detect_language

        # A Chinese sentence naming an English product stays Chinese.
        assert detect_language("iPhone 15 Pro 怎么退货？") == "zh"

    def test_no_letters_defaults_to_english(self) -> None:
        from app.services.dialogue.i18n import detect_language

        # Global product: undetectable input falls back to English,
        # the international default — not to Chinese.
        assert detect_language("12345 ??") == "en"
        assert detect_language("") == "en"

    def test_other_alphabets_fall_back_to_english(self) -> None:
        from app.services.dialogue.i18n import detect_language

        # zh/en are the only canned pairs today — non-Latin scripts
        # that aren't Chinese (Cyrillic, kana) select English.
        assert detect_language("где мой заказ") == "en"
        assert detect_language("ですか") == "en"


class TestPersonaLanguageRule:
    def test_generation_persona_follows_user_language(self) -> None:
        prompt = PromptTemplates.get_cs_system_prompt()
        assert "与用户当前消息相同的语言" in prompt
        # Global product: the undetectable fallback is English.
        assert "无法判断时用英语" in prompt
        assert "用简体中文回答" not in prompt

    def test_agent_persona_follows_user_language(self) -> None:
        assert "与用户当前消息相同的语言" in AGENT_SYSTEM_PROMPT
        assert "无法判断时用英语" in AGENT_SYSTEM_PROMPT
        assert "用简体中文回答" not in AGENT_SYSTEM_PROMPT


class TestDeterministicBranchesFollowLanguage:
    async def test_english_no_evidence_copy(self) -> None:
        """An English knowledge question over an empty KB gets the
        English evidence-gap sentence — not Chinese canned text the
        visitor cannot read."""
        llm = _CountingLLM()
        chat = _chat(
            llm,
            intent=Intent.POLICY,
            pipeline={"hybrid_search": _hybrid_returning([])},
        )

        response = await chat.process_message(1, "what does the warranty cover?", 0)

        assert "knowledge base" in response.content
        assert "暂未在知识库中找到" not in response.content
        assert llm.calls == 0

    async def test_chinese_no_evidence_copy_unchanged(self) -> None:
        llm = _CountingLLM()
        chat = _chat(
            llm,
            intent=Intent.POLICY,
            pipeline={"hybrid_search": _hybrid_returning([])},
        )

        response = await chat.process_message(1, "洗衣机保修几年", 0)

        assert "暂未在知识库中找到" in response.content

    async def test_english_handoff_acknowledgement(self) -> None:
        """Handoff never touches the LLM — its fixed acknowledgement
        must follow the asking language too (queue info included)."""
        llm = _CountingLLM()
        handoff = Mock()
        handoff.create_ticket_for_session = AsyncMock(
            return_value={"ticket_id": 7, "queue_position": 2, "reused": False}
        )
        chat = ChatService(
            graph=build_dialogue_graph(
                intent_detector=_detector_returning(Intent.HANDOFF),
                slot_filler=None,
                tool_registry=Mock(),
                retrieval_pipeline={},
                llm_service=llm,
                handoff_service=handoff,
            ),
        )

        response = await chat.process_message(1, "human agent please", 0)

        assert "#7" in response.content
        assert "queue" in response.content.lower()
        assert "human agent" in response.content
        assert "转接人工客服" not in response.content
        assert llm.calls == 0

    async def test_chinese_handoff_acknowledgement_unchanged(self) -> None:
        llm = _CountingLLM()
        handoff = Mock()
        handoff.create_ticket_for_session = AsyncMock(
            return_value={"ticket_id": 7, "queue_position": 2, "reused": False}
        )
        chat = ChatService(
            graph=build_dialogue_graph(
                intent_detector=_detector_returning(Intent.HANDOFF),
                slot_filler=None,
                tool_registry=Mock(),
                retrieval_pipeline={},
                llm_service=llm,
                handoff_service=handoff,
            ),
        )

        response = await chat.process_message(1, "转人工", 0)

        assert "已为您转接人工客服（工单号 #7）" in response.content
