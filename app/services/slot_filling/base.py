"""
Base classes and data models for slot filling.

Slot filling extracts structured entities from user queries after intent
detection, producing filters for vector search and entity hints for
graph retrieval (Text-to-Cypher).
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from app.models.enums.intent import Intent


@dataclass
class ExtractedSlot:
    """A single extracted entity slot from a user query."""

    slot_type: str
    entity_type: str
    value: str
    normalized_value: str
    confidence: float = 0.0
    source: str = "rule"

    def to_filter(self) -> Dict[str, str]:
        return {self.slot_type: self.normalized_value}

    def to_entity_hint(self) -> Dict[str, str]:
        return {"type": self.entity_type, "name": self.normalized_value}


@dataclass
class SlotFillingResult:
    """Result of slot extraction from a user query."""

    slots: List[ExtractedSlot] = field(default_factory=list)
    raw_query: str = ""
    metadata: Optional[Dict[str, Any]] = None

    def has_slots(self) -> bool:
        return len(self.slots) > 0

    def get_slot(self, slot_type: str) -> Optional[ExtractedSlot]:
        for s in self.slots:
            if s.slot_type == slot_type:
                return s
        return None

    def get_slot_value(self, slot_type: str) -> Optional[str]:
        slot = self.get_slot(slot_type)
        return slot.normalized_value if slot else None

    def to_filters(self) -> Dict[str, str]:
        filters: Dict[str, str] = {}
        for s in self.slots:
            filters[s.slot_type] = s.normalized_value
        return filters

    def to_entity_hints(self) -> List[Dict[str, str]]:
        return [s.to_entity_hint() for s in self.slots]

    def slot_types(self) -> set:
        return {s.slot_type for s in self.slots}


class SlotFiller(ABC):
    """Abstract base class for slot filling implementations."""

    @abstractmethod
    async def fill_slots(
        self,
        query: str,
        intent: Optional[Intent] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> SlotFillingResult:
        """Extract structured slots from a user query."""
        ...
