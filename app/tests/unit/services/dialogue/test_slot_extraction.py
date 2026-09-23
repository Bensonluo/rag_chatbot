"""Tests for LLM-assisted slot extraction in the task slot pipeline."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.services.dialogue.nodes import NodeFactory
from app.services.dialogue.slot_extraction import (
    _build_prompt,
    llm_extract_slots,
    parse_slot_json,
)
from app.services.dialogue.state import DialogueState


class _StubLLM:
    """Minimal LLMServiceBase stand-in returning canned content."""

    def __init__(self, content: str = "{}") -> None:
        self.content = content
        self.calls = []

    async def generate(self, messages, **kwargs):  # noqa: ANN001, ANN002
        self.calls.append(messages)
        return SimpleNamespace(content=self.content)


class _RaisingLLM(_StubLLM):
    async def generate(self, messages, **kwargs):  # noqa: ANN001, ANN002
        self.calls.append(messages)
        raise RuntimeError("provider down")


def _factory(llm) -> NodeFactory:
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
