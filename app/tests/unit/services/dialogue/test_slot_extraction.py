"""Tests for LLM-assisted slot extraction in the task slot pipeline."""

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import Mock, patch

import pytest

from app.services.dialogue.nodes import NodeFactory
from app.services.dialogue.slot_extraction import (
    _build_prompt,
    llm_extract_slots,
    parse_slot_json,
)
from app.services.dialogue.state import DialogueState
from app.services.llm.base import LLMMessage, LLMResponse, LLMServiceBase


class _StubLLM(LLMServiceBase):
    """Minimal LLMServiceBase stand-in returning canned content."""

    def __init__(self, content: str = "{}") -> None:
        super().__init__(api_key="fake", model="fake")
        self.content = content
        self.calls: list[Any] = []

    async def generate(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.calls.append(messages)
        return LLMResponse(content=self.content, model="fake")

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        yield self.content

    def estimate_tokens(self, text: str) -> int:
        return len(text) // 4

    async def count_tokens(self, messages: list[LLMMessage]) -> int:
        return sum(len(m.content) for m in messages) // 4


class _RaisingLLM(_StubLLM):
    async def generate(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.calls.append(messages)
        raise RuntimeError("provider down")


def _factory(llm: LLMServiceBase | None) -> NodeFactory:
    return NodeFactory(
        intent_detector=Mock(),
        slot_filler=None,
        tool_registry=Mock(),
        llm_service=llm,
    )


# ── Module: prompt building ───────────────────────────────────────────────


class TestBuildPrompt:
    def test_prompt_lists_schema_slots_and_existing(self) -> None:
        prompt = _build_prompt("refund", "手机坏了", {"order_id": "A123"})

        assert prompt is not None
        assert "order_id" in prompt
        assert "reason" in prompt
        assert "A123" in prompt
        assert "手机坏了" in prompt

    def test_prompt_none_for_unknown_intent(self) -> None:
        assert _build_prompt("no_such_intent", "hi", {}) is None


# ── Module: response parsing ──────────────────────────────────────────────


class TestParseSlotJson:
    def test_plain_json(self) -> None:
        assert parse_slot_json('{"order_id": "12345"}', "refund") == {"order_id": "12345"}

    def test_code_fenced_json(self) -> None:
        content = '```json\n{"reason": "质量问题"}\n```'
        assert parse_slot_json(content, "refund") == {"reason": "质量问题"}

    def test_unknown_keys_dropped(self) -> None:
        assert parse_slot_json('{"order_id": "1", "evil": "x"}', "refund") == {"order_id": "1"}

    def test_number_coerced(self) -> None:
        assert parse_slot_json('{"amount": "59.9"}', "refund") == {"amount": 59.9}

    def test_bad_number_dropped(self) -> None:
        assert parse_slot_json('{"amount": "很多"}', "refund") == {}

    def test_empty_values_dropped(self) -> None:
        assert parse_slot_json('{"reason": "  "}', "refund") == {}

    def test_invalid_json_returns_empty(self) -> None:
        assert parse_slot_json("抱歉我无法", "refund") == {}

    def test_non_object_json_returns_empty(self) -> None:
        assert parse_slot_json('["order_id"]', "refund") == {}

    def test_values_capped(self) -> None:
        long_value = "x" * 500
        result = parse_slot_json(f'{{"reason": "{long_value}"}}', "refund")
        assert len(result["reason"]) == 200


# ── Module: extraction entry point ────────────────────────────────────────


class TestLLMExtractSlots:
    async def test_returns_only_new_slots(self) -> None:
        llm = _StubLLM('{"order_id": "A1", "reason": "质量问题"}')

        result = await llm_extract_slots("refund", "手机坏了", {"order_id": "A1"}, llm)

        assert result == {"reason": "质量问题"}

    async def test_failure_returns_empty(self) -> None:
        result = await llm_extract_slots("refund", "手机坏了", {}, _RaisingLLM())
        assert result == {}

    async def test_unknown_intent_skips_llm(self) -> None:
        llm = _StubLLM('{"order_id": "1"}')

        result = await llm_extract_slots("chitchat", "hi", {}, llm)

        assert result == {}
        assert llm.calls == []


# ── Node: LLM pass inside collect_slots_node ──────────────────────────────


class TestCollectSlotsNodeLLMPass:
    async def test_regex_hit_skips_llm(self) -> None:
        llm = _StubLLM('{"reason": "x"}')
        factory = _factory(llm)
        state = DialogueState(
            message="订单号12345",
            intent="refund",
            filled_slots={},
            pending_slots=["order_id", "reason"],
        )

        result = await factory.collect_slots_node(state)

        assert result["filled_slots"]["order_id"] == "12345"
        assert llm.calls == []

    async def test_regex_miss_llm_extracts(self) -> None:
        llm = _StubLLM('{"reason": "屏幕刮花"}')
        factory = _factory(llm)
        state = DialogueState(
            message="上周买的手机屏幕刮花了想退",
            intent="refund",
            filled_slots={"order_id": "A99"},
            pending_slots=["reason"],
        )

        result = await factory.collect_slots_node(state)

        # LLM value landed; the whole-message heuristic must not have
        # fired (it would have assigned the message to "reason").
        assert result["filled_slots"]["reason"] == "屏幕刮花"
        assert len(llm.calls) == 1

    async def test_llm_error_falls_back_to_heuristic(self) -> None:
        llm = _RaisingLLM()
        factory = _factory(llm)
        state = DialogueState(
            message="坏的",
            intent="refund",
            filled_slots={},
            pending_slots=["order_id", "reason"],
        )

        result = await factory.collect_slots_node(state)

        assert len(llm.calls) == 1  # LLM was attempted
        assert result["filled_slots"]["order_id"] == "坏的"  # heuristic fired

    async def test_disabled_flag_skips_llm(self) -> None:
        from app.config.settings import get_settings

        llm = _StubLLM('{"reason": "x"}')
        factory = _factory(llm)
        state = DialogueState(
            message="上周买的手机屏幕刮花了想退",
            intent="refund",
            filled_slots={"order_id": "A99"},
            pending_slots=["reason"],
        )

        with patch.object(get_settings(), "SLOT_LLM_EXTRACTION_ENABLED", False):
            result = await factory.collect_slots_node(state)

        assert llm.calls == []
        # Flag off restores pre-LLM behavior exactly: short message
        # without task keywords is assigned to the first missing slot.
        assert result["filled_slots"]["reason"] == "上周买的手机屏幕刮花了想退"

    async def test_no_pending_slots_skips_llm(self) -> None:
        llm = _StubLLM('{"reason": "x"}')
        factory = _factory(llm)
        state = DialogueState(
            message="手机坏了",
            intent="refund",
            filled_slots={},
            pending_slots=[],
        )

        await factory.collect_slots_node(state)

        assert llm.calls == []


# ── Finding ① (2026-09-30): rule-path resilience under extractor outage ───


# The exact message a live demo visitor sent while the extractor LLM leg
# was down (GLM 429 storm): the complaint flow re-prompted "What would
# you like to complain about?" forever because no rule path could fill
# the pattern-less category slot.
LIVE_EN_COMPLAINT = (
    "I want to file a complaint: product quality, the screen of my ORD1001 order arrived cracked"
)


class TestComplaintCategoryRulePatterns:
    """category must be regex-extractable from both languages."""

    def test_live_english_message_fills_category(self) -> None:
        from app.services.slot_filling.slot_types import extract_slots_from_message

        result = extract_slots_from_message("complaint", LIVE_EN_COMPLAINT, {})

        assert result.get("category")
        assert "quality" in result["category"].lower()

    def test_category_value_is_the_keyword_not_the_message(self) -> None:
        from app.services.slot_filling.slot_types import extract_slots_from_message

        result = extract_slots_from_message("complaint", LIVE_EN_COMPLAINT, {})

        # The slot value must stay a bounded keyword — the ticket needs
        # a category, not the 90-character message.
        assert len(result["category"]) <= 40

    @pytest.mark.parametrize(
        "message",
        ["我要投诉商品质量", "物流配送太慢了我要投诉", "服务态度太差了"],
    )
    def test_chinese_category_keywords_fill(self, message: str) -> None:
        from app.services.slot_filling.slot_types import extract_slots_from_message

        result = extract_slots_from_message("complaint", message, {})

        assert result.get("category")

    def test_complaint_order_id_requires_digits(self) -> None:
        from app.services.slot_filling.slot_types import extract_slots_from_message

        result = extract_slots_from_message("complaint", LIVE_EN_COMPLAINT, {})

        # A "(?:order)\s*([A-Za-z0-9]{3,})"-style pattern would capture
        # the word after "order" ("arrived") as the order id.
        assert result.get("order_id") in (None, "ORD1001")


class TestRefundReasonEnglishPatterns:
    """refund/return reason patterns were zh-only — under the same
    extractor outage an English "because it arrived broken" could never
    fill reason, looping the refund flow the same way."""

    def test_because_phrase(self) -> None:
        from app.services.slot_filling.slot_types import extract_slots_from_message

        result = extract_slots_from_message("refund", "because it arrived broken", {})

        assert result.get("reason") == "it arrived broken"

    def test_damage_keyword(self) -> None:
        from app.services.slot_filling.slot_types import extract_slots_from_message

        result = extract_slots_from_message("return", "item arrived damaged", {})

        assert result.get("reason")

    def test_chinese_reason_patterns_unchanged(self) -> None:
        from app.services.slot_filling.slot_types import extract_slots_from_message

        result = extract_slots_from_message("refund", "因为质量问题", {})

        assert result.get("reason") == "质量问题"


class TestComplaintDescriptionOutageTerminator:
    """With the LLM extractor unavailable, complaint collection must
    still terminate — the live loop re-prompted forever on messages the
    whole-message heuristic rejects (≥30 chars or task keywords)."""

    async def test_live_message_completes_both_slots_without_llm(self) -> None:
        factory = _factory(_StubLLM("{}"))
        state = DialogueState(
            message=LIVE_EN_COMPLAINT,
            intent="complaint",
            filled_slots={},
            pending_slots=["category", "description"],
        )

        result = await factory.collect_slots_node(state)

        assert result["pending_slots"] == []
        assert "quality" in result["filled_slots"]["category"].lower()
        assert "cracked" in result["filled_slots"]["description"]

    async def test_long_description_answer_under_llm_outage(self) -> None:
        """Turn 2 of the loop: category already filled, extractor LLM
        raising, answer longer than the heuristic's 30-char ceiling."""
        factory = _factory(_RaisingLLM())
        state = DialogueState(
            message="the screen is cracked and it arrived that way",
            intent="complaint",
            filled_slots={"category": "product quality"},
            pending_slots=["description"],
        )

        result = await factory.collect_slots_node(state)

        assert result["pending_slots"] == []
        assert "cracked" in result["filled_slots"]["description"]

    async def test_pure_order_id_turn_is_not_the_description(self) -> None:
        """A structured answer (order number) must not be swallowed as
        the free-form description."""
        factory = _factory(_StubLLM("{}"))
        state = DialogueState(
            message="my order is ORD1001",
            intent="complaint",
            filled_slots={"category": "product quality"},
            pending_slots=["description"],
        )

        result = await factory.collect_slots_node(state)

        assert "description" not in result["filled_slots"]
