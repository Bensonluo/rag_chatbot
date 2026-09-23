"""
Hybrid slot filler: rule-based first, LLM fallback.

Mirrors the HybridIntentDetector pattern — deterministic extraction
is preferred for speed and cost, with LLM as fallback for complex queries.
"""

from typing import Any

from app.models.enums.intent import Intent
from app.services.slot_filling.base import SlotFiller, SlotFillingResult
from app.services.slot_filling.llm_based import LLMSlotFiller
from app.services.slot_filling.rule_based import RuleBasedSlotFiller


class HybridSlotFiller(SlotFiller):
    """Rule-based first, LLM fallback when rules find nothing."""

    def __init__(
        self,
        rule_based: RuleBasedSlotFiller,
        llm_based: LLMSlotFiller | None = None,
        llm_fallback: bool = True,
    ) -> None:
        self._rule_based = rule_based
        self._llm_based = llm_based
        self._llm_fallback = llm_fallback

    async def fill_slots(
        self,
        query: str,
        intent: Intent | None = None,
        context: dict[str, Any] | None = None,
    ) -> SlotFillingResult:
        rule_result = await self._rule_based.fill_slots(query, intent, context)

        if rule_result.has_slots() or not self._llm_fallback or not self._llm_based:
            return rule_result

        llm_result = await self._llm_based.fill_slots(query, intent, context)
        metadata = llm_result.metadata or {}
        metadata["method"] = "llm_fallback"
        llm_result.metadata = metadata
        return llm_result
