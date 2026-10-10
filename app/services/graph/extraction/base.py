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
    # Empty entities can be a valid answer. Keep failures explicit so ingestion
    # does not publish an apparently successful but incomplete graph revision.
    error: str | None = None


class EntityExtractor(ABC):
    """Abstract base class for entity extractors."""

    async def close(self) -> None:
        """Release extractor-owned resources, if any."""
        return None

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
