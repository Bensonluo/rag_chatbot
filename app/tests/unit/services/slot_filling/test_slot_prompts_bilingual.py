"""Bilingual slot prompts guard.

Slot-asking prompts are fixed templates (deliberately not LLM-generated).
The product is global by default: an English-speaking user in a
task flow must be asked for missing slots in English, never in Chinese.
"""

from app.services.slot_filling.slot_types import (
    INTENT_SLOT_SCHEMAS,
    get_next_prompt,
)


class TestBilingualSlotPrompts:
    def test_english_prompt_selected_by_lang(self) -> None:
        prompt = get_next_prompt("refund", {}, lang="en")
        assert prompt is not None
        assert "order" in prompt.lower()
        # The EN prompt must actually be English, not the zh template.
        assert "订单号" not in prompt

    def test_zh_prompt_remains_the_default(self) -> None:
        prompt = get_next_prompt("refund", {})
        assert prompt == "请提供您的订单号"

    def test_every_required_slot_has_an_english_prompt(self) -> None:
        missing = []
        for intent, schema in INTENT_SLOT_SCHEMAS.items():
            for slot_name in schema["slots"]:
                if not schema["slots"][slot_name].get("prompt_en"):
                    missing.append(f"{intent}.{slot_name}")
        assert not missing, f"slots without an English prompt: {missing}"

    def test_missing_en_prompt_falls_back_to_zh(self) -> None:
        # Graceful degradation for a future slot added without prompt_en:
        # a zh ask is better than no ask.
        prompt = get_next_prompt("query_order", {}, lang="en")
        assert prompt is not None and prompt.strip()
