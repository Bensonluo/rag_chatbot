"""
Entity extraction base interface and data models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExtractionResult:
    """Result of entity/relationship extraction from text."""

    entities: list[dict[str, Any]] = field(default_factory=list)
    relations: list[dict[str, Any]] = field(default_factory=list)
    raw_text: str = ""


class EntityExtractor(ABC):
    """Abstract base class for entity extractors."""

    @abstractmethod
    async def extract(self, text: str, context: dict[str, Any] | None = None) -> ExtractionResult:
        """Extract entities and relations from text."""
        pass

    @abstractmethod
    async def extract_batch(
        self, texts: list[str], context: dict[str, Any] | None = None
    ) -> list[ExtractionResult]:
        """Extract entities and relations from multiple texts."""
        pass
