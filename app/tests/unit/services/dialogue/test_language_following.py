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

import json
import re
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


def _chat(llm: LLMServiceBase, *, intent: Intent, pipeline: dict[str, Any]) -> ChatService:
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


class _RecordingLLM(LLMServiceBase):
    """Captures the full message list of every generation call."""

    def __init__(self) -> None:
        super().__init__(api_key="test", model="test")
        self.captures: list[list[LLMMessage]] = []

    async def generate(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.captures.append(list(messages))
        return LLMResponse(content="Answer.", model=self.model)

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        self.captures.append(list(messages))
        yield "Answer."

    def estimate_tokens(self, text: str) -> int:
        return len(text)

    async def count_tokens(self, messages: list[LLMMessage]) -> int:
        return sum(len(m.content) for m in messages)


class TestGenerationLanguageDirective:
    """The reply language must be pinned per turn, not left to the model.

    Observed live (2026-09-30, benluo.art demo): an English conversation
    got full-Chinese replies twice — "Okay okay" confirming a complaint
    returned a Chinese ticket summary (投诉编号/处理专员 客服专员-小李/
    预计响应 24小时内), and "Who are you ... why" got a Chinese
    clarification. Both personas DO say「用与用户当前消息相同的语言回答」,
    but that one clause — written inside a Chinese persona — loses to the
    Chinese tool JSON and the Chinese prompt scaffolding (用户意图/工具执行
    结果) that dominate the composed prompt. Fix: an explicit,
    final-position directive keyed off the RAW user turn decides the
    reply language regardless of what the surrounding context drags
    toward.
    """

    @staticmethod
    def _factory_with(llm: _RecordingLLM) -> Any:
        from app.services.dialogue.nodes import NodeFactory
        from app.services.dialogue.tools import create_default_tool_registry

        detector = Mock()
        detector.detect_with_confidence = AsyncMock()
        return NodeFactory(
            intent_detector=detector,
            slot_filler=Mock(),
            tool_registry=create_default_tool_registry(),
            retrieval_pipeline={},
            llm_service=llm,
            guardrail_service=None,
            graph_retrieval_service=None,
        )

    async def test_english_tool_turn_gets_english_directive(self) -> None:
        """The live "Okay okay" shape: zh-scaffolded composed prompt +
        raw English turn. Drove via generate_response_node because a
        bare "where is my package" turn correctly stops at the
        order-number slot prompt before any generation."""
        from app.services.dialogue.state import DialogueState
        from app.services.dialogue.tools import mock_track_shipping

        llm = _RecordingLLM()
        factory = self._factory_with(llm)
        state: DialogueState = {
            "message": "okay okay, where is my package now?",
            "session_id": 1,
            "intent": "track_shipping",
            "tool_result": mock_track_shipping({"order_id": "ORD1001"}),
        }

        await factory.generate_response_node(state)

        assert llm.captures, "tool result present — generation must run"
        prompt = llm.captures[-1][-1].content
        assert "Reply in English only" in prompt
        assert "请只用中文回答" not in prompt
        # The composed context IS zh-scaffolded — the directive must key
        # off the raw English turn, not off the scaffolded prompt.
        assert "工具执行结果:" in prompt
        assert "where is my package now?" in prompt

    async def test_chinese_tool_turn_gets_chinese_directive(self) -> None:
        from app.services.dialogue.state import DialogueState
        from app.services.dialogue.tools import mock_track_shipping

        llm = _RecordingLLM()
        factory = self._factory_with(llm)
        state: DialogueState = {
            "message": "我的快递到哪了",
            "session_id": 1,
            "intent": "track_shipping",
            "tool_result": mock_track_shipping({"order_id": "ORD1001"}),
        }

        await factory.generate_response_node(state)

        assert llm.captures
        prompt = llm.captures[-1][-1].content
        assert "请只用中文回答" in prompt
        assert "Reply in English only" not in prompt

    async def test_english_direct_turn_gets_english_directive(self) -> None:
        """The "Who are you ... why" live failure: a direct-generation
        turn (no tool data at all) still needs the pin — the Chinese
        persona alone pulled the answer into Chinese."""
        llm = _RecordingLLM()
        chat = _chat(llm, intent=Intent.CHITCHAT, pipeline={})

        await chat.process_message(1, "who are you and why?", 0)

        assert llm.captures
        prompt = llm.captures[-1][-1].content
        assert "Reply in English only" in prompt
        assert "请只用中文回答" not in prompt


_CJK = re.compile(r"[一-鿿]")


class TestMockPayloadsAreLanguageNeutral:
    """Tool data must not decide the reply language.

    The mock tools returned Chinese display values (客服专员-小李,
    24小时内, 顺丰快递, 运输中) that the model quoted verbatim inside
    English replies (live 2026-09-30). Payloads now carry
    language-neutral / English field values; the generation directive
    owns the presentation language in both directions.
    """

    def test_complaint_payload_is_neutral(self) -> None:
        from app.services.dialogue.tools import mock_complaint

        payload = mock_complaint(
            {"category": "broken item", "description": "arrived broken", "order_id": "ORD1001"}
        )

        assert not _CJK.search(json.dumps(payload, ensure_ascii=False))

    def test_track_shipping_payload_is_neutral(self) -> None:
        from app.services.dialogue.tools import mock_track_shipping

        payload = mock_track_shipping({"order_id": "ORD1001"})

        assert not _CJK.search(json.dumps(payload, ensure_ascii=False))

    def test_recent_orders_statuses_are_neutral(self) -> None:
        from app.services.dialogue.tools import mock_get_recent_orders

        payload = mock_get_recent_orders({"user_id": 1})

        assert payload["count"] >= 1
        assert not _CJK.search(json.dumps(payload, ensure_ascii=False))

    def test_recent_orders_anonymous_message_is_english(self) -> None:
        from app.services.dialogue.tools import mock_get_recent_orders

        payload = mock_get_recent_orders({})

        assert "sign" in payload["message"]
        assert not _CJK.search(payload["message"])


# Finding ② (2026-09-30, live): with GLM down, the MiniMax fallback
# served a "who are you" turn and answered with its own training
# identity — "MiniMax-M3 developed by MiniMax". The assistant's
# identity must be stable no matter which provider serves the turn.
LIVE_SELF_ID_LEAK = "MiniMax-M3"


class TestPersonaIdentityAnchor:
    """Both personas must anchor identity and forbid model self-ID."""

    def test_generation_persona_anchors_identity(self) -> None:
        prompt = PromptTemplates.get_cs_system_prompt()

        # Identity: the platform's AI customer-service assistant.
        assert "智能客服助手" in prompt
        # Identity questions get the anchored answer, and the
        # underlying model / vendor is never disclosed — the assistant
        # presents as the store's assistant regardless of which
        # provider serves the turn.
        assert "你是谁" in prompt or "身份" in prompt
        assert "模型" in prompt

    def test_agent_persona_anchors_identity(self) -> None:
        assert "智能客服助手" in AGENT_SYSTEM_PROMPT
        assert "你是谁" in AGENT_SYSTEM_PROMPT or "身份" in AGENT_SYSTEM_PROMPT
        assert "模型" in AGENT_SYSTEM_PROMPT

    def test_anchor_names_the_assistant_role_not_a_model(self) -> None:
        """The anchor line itself must not leak a provider/model name."""
        for persona in (PromptTemplates.get_cs_system_prompt(), AGENT_SYSTEM_PROMPT):
            assert LIVE_SELF_ID_LEAK not in persona
            assert "MiniMax" not in persona
            assert "GLM" not in persona
