"""
Rule-based slot filler using keyword matching and regex patterns.

Fast, deterministic extraction with zero LLM cost. Suitable for common
pharmaceutical domain entities with well-known names.
"""
import re
import logging
from typing import Optional, Dict, Any, List

from app.services.slot_filling.base import SlotFiller, ExtractedSlot, SlotFillingResult
from app.services.slot_filling.slot_types import SLOT_DEFINITIONS
from app.models.enums.intent import Intent

logger = logging.getLogger(__name__)

_RETRIEVAL_INTENTS = {
    Intent.QUESTION, Intent.HOW_TO, Intent.COMPARISON, Intent.DEFINITION,
    Intent.RECOMMENDATION, Intent.RELATIONSHIP_QUERY, Intent.GLOBAL_SUMMARY,
    Intent.ENTITY_LOOKUP,
}


class RuleBasedSlotFiller(SlotFiller):
    """Extract slots via keyword matching and regex patterns."""

    def __init__(
        self,
        slot_definitions: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        self._definitions = slot_definitions or SLOT_DEFINITIONS
        self._compiled_patterns = self._compile_patterns()

    def _compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        compiled: Dict[str, List[re.Pattern]] = {}
        for slot_type, definition in self._definitions.items():
            patterns = []
            for pattern_str in definition.get("patterns", []):
                try:
                    patterns.append(re.compile(pattern_str))
                except re.error as e:
                    logger.warning("Invalid pattern for %s: %s (%s)", slot_type, pattern_str, e)
            compiled[slot_type] = patterns
        return compiled

    async def fill_slots(
        self,
        query: str,
        intent: Optional[Intent] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> SlotFillingResult:
        if intent and intent not in _RETRIEVAL_INTENTS:
            return SlotFillingResult(
                slots=[], raw_query=query, metadata={"method": "skipped"},
            )

        slots: List[ExtractedSlot] = []
        matched_types: set = set()
        normalized = query.lower()

        for slot_type, definition in self._definitions.items():
            entity_type = definition["entity_type"]

            # 1. Keyword matching
            for variant, canonical in definition.get("keywords", {}).items():
                if variant.lower() in normalized:
                    slots.append(ExtractedSlot(
                        slot_type=slot_type,
                        entity_type=entity_type,
                        value=variant,
                        normalized_value=canonical,
                        confidence=0.9,
                        source="rule",
                    ))
                    matched_types.add(slot_type)
                    break

            # 2. Regex pattern matching (only if no keyword match for this type)
            if slot_type not in matched_types:
                for pattern in self._compiled_patterns.get(slot_type, []):
                    match = pattern.search(query)
                    if match:
                        for group_name, group_value in match.groupdict().items():
                            if group_value:
                                normalized_val = self._normalize_match(
                                    slot_type, group_name, group_value,
                                )
                                slots.append(ExtractedSlot(
                                    slot_type=slot_type,
                                    entity_type=entity_type,
                                    value=group_value,
                                    normalized_value=normalized_val,
                                    confidence=0.85,
                                    source="rule",
                                ))
                        matched_types.add(slot_type)
                        break

        return SlotFillingResult(
            slots=slots,
            raw_query=query,
            metadata={"method": "rule_based", "slot_count": len(slots)},
        )

    @staticmethod
    def _normalize_match(slot_type: str, group_name: str, value: str) -> str:
        if slot_type == "time_period":
            if group_name == "year":
                return value.strip()
            if group_name == "quarter":
                quarter_map = {"一": "Q1", "二": "Q2", "三": "Q3", "四": "Q4"}
                q = quarter_map.get(value, f"Q{value}")
                return q
            if group_name == "month":
                return f"{int(value):02d}"
            if group_name == "year_range":
                return value.strip()
        return value.strip()
