"""
LLM-based slot filler using structured extraction prompts.

Falls back gracefully on parse failures. Used when rule-based extraction
finds no slots for complex queries.
"""
import json
import logging
from typing import Optional, Dict, Any, List

from app.services.slot_filling.base import SlotFiller, ExtractedSlot, SlotFillingResult
from app.services.slot_filling.slot_types import SLOT_DEFINITIONS
from app.services.llm.base import LLMServiceBase, LLMMessage
from app.models.enums.intent import RAG_INTENTS, Intent

logger = logging.getLogger(__name__)

_SLOT_PROMPT = """Extract named entities from the user query as structured slots.

Entity types to extract:
{entity_types}

Rules:
- Only extract entities explicitly mentioned in the query.
- Return JSON: {{"slots": [{{"slot_type": "...", "value": "..."}}]}}
- If no entities found, return {{"slots": []}}
- slot_type must be one of: {slot_type_names}
- Normalize: "2024年" -> "2024", "Q1" -> "Q1", "电商" -> "线上"

Query: {query}"""


class LLMSlotFiller(SlotFiller):
    """Extract slots using an LLM with structured JSON output."""

    def __init__(
        self,
        llm_service: LLMServiceBase,
        slot_definitions: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        self._llm = llm_service
        self._definitions = slot_definitions or SLOT_DEFINITIONS

    async def fill_slots(
        self,
        query: str,
        intent: Optional[Intent] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> SlotFillingResult:
        if intent and intent.value not in RAG_INTENTS:
            return SlotFillingResult(
                slots=[], raw_query=query, metadata={"method": "skipped"},
            )

        entity_types_desc = self._format_entity_types()
        slot_type_names = ", ".join(self._definitions.keys())

        prompt = _SLOT_PROMPT.replace("{entity_types}", entity_types_desc)
        prompt = prompt.replace("{slot_type_names}", slot_type_names)
        prompt = prompt.replace("{query}", query)

        messages = [
            LLMMessage(role="system", content="Output only valid JSON."),
            LLMMessage(role="user", content=prompt),
        ]

        try:
            response = await self._llm.generate(
                messages=messages, max_tokens=256, temperature=0.0,
            )
            return self._parse_response(response.content, query)
        except Exception as e:
            logger.warning("LLM slot filling failed: %s", e)
            return SlotFillingResult(
                slots=[], raw_query=query, metadata={"method": "llm", "error": str(e)},
            )

    def _format_entity_types(self) -> str:
        parts: List[str] = []
        for name, definition in self._definitions.items():
            keywords = list(definition.get("keywords", {}).keys())[:5]
            examples = f" (e.g. {', '.join(keywords)})" if keywords else ""
            parts.append(f"  {name}: {definition['entity_type']}{examples}")
        return "\n".join(parts)

    def _parse_response(self, content: str, raw_query: str) -> SlotFillingResult:
        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:])
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse slot JSON: %s", e)
            return SlotFillingResult(
                slots=[], raw_query=raw_query, metadata={"method": "llm"},
            )

        raw_slots = data.get("slots", [])
        slots: List[ExtractedSlot] = []
        for s in raw_slots:
            if not isinstance(s, dict) or "slot_type" not in s or "value" not in s:
                continue
            slot_type = s["slot_type"]
            if slot_type not in self._definitions:
                continue
            definition = self._definitions[slot_type]
            slots.append(ExtractedSlot(
                slot_type=slot_type,
                entity_type=definition["entity_type"],
                value=s["value"],
                normalized_value=s.get("normalized_value", s["value"]),
                confidence=0.8,
                source="llm",
            ))

        return SlotFillingResult(
            slots=slots,
            raw_query=raw_query,
            metadata={"method": "llm", "slot_count": len(slots)},
        )
