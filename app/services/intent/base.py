"""
Base interface for intent detection.

Provides abstract classes and data structures for intent classifiers.
"""

from abc import ABC, abstractmethod
from collections.abc import Coroutine
from dataclasses import dataclass, field
from typing import Any

from app.models.enums.intent import Intent


@dataclass
class IntentResult:
    """
    Result of intent detection.

    Attributes:
        intent: Detected intent category
        confidence: Confidence score (0.0 to 1.0)
        metadata: Additional metadata about detection
    """

    intent: Intent
    confidence: float = 0.0
    metadata: dict[str, Any] | None = None


@dataclass
class CoDetectedIntent:
    """An independent request, not an alternative label for the primary.

    ``query`` is a nonempty verbatim subspan of the original message.
    Distinct requests may share an intent label (for different orders).
    Queueing and execution decisions belong to the dialogue layer.
    """

    intent: Intent
    confidence: float = 0.0
    method: str = "rule_based"
    query: str = ""


@dataclass
class IntentDetection:
    """
    Full multi-intent detection result for one message.

    Attributes:
        primary: The dominant intent (same contract as IntentResult)
        co_detected: Independent requests in source order. No label-only deduplication.
        primary_query: Verbatim subspan for the primary request. Empty means
            the whole input, for legacy or unsegmented single-request results.

    Detectors do not truncate requests; consumers must report any queue overflow.
    """

    primary: IntentResult
    co_detected: list[CoDetectedIntent] = field(default_factory=list)
    primary_query: str = ""


class IntentDetector(ABC):
    """
    Abstract base class for intent detectors.

    All intent detection implementations must inherit from this class
    and implement the required methods.
    """

    @abstractmethod
    def detect(
        self, query: str, context: dict[str, Any] | None = None
    ) -> Coroutine[Any, Any, Intent]:
        """
        Detect the intent of a user query.

        Args:
            query: User's text query
            context: Optional context information (conversation history, etc.)

        Returns:
            Intent: Detected intent category

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("detect() must be implemented by subclass")

    @abstractmethod
    def detect_with_confidence(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> Coroutine[Any, Any, IntentResult]:
        """
        Detect intent and return confidence score.

        Args:
            query: User's text query
            context: Optional context information

        Returns:
            IntentResult: Detected intent with confidence score

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("detect_with_confidence() must be implemented by subclass")

    async def detect_multi(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> IntentDetection:
        """Detect the primary intent plus any co-occurring intents.

        Default implementation wraps the single-intent contract: the
        primary comes from detect_with_confidence() and no co-occurring
        signal is produced. Request-aware detectors classify independent
        source clauses or validate request-scoped LLM output and override
        this method. The empty co_detected list is the legacy single-result
        contract, not evidence that a model call succeeded.
        """
        primary = await self.detect_with_confidence(query, context)
        return IntentDetection(primary=primary, co_detected=[])

    def _normalize_query(self, query: str) -> str:
        """
        Normalize query text for processing.

        Args:
            query: Query text

        Returns:
            str: Normalized query
        """
        if not query:
            return ""

        # Convert to lowercase
        normalized = query.lower().strip()

        # Remove extra whitespace
        import re

        normalized = re.sub(r"\s+", " ", normalized)

        return normalized

    def _contains_any(self, text: str, keywords: list[str]) -> bool:
        """
        Check if text contains any of the keywords.

        Args:
            text: Text to search
            keywords: List of keywords to search for

        Returns:
            bool: True if any keyword is found
        """
        return any(keyword in text for keyword in keywords)

    def _count_matches(self, text: str, keywords: list[str]) -> int:
        """
        Count how many keywords appear in text.

        Args:
            text: Text to search
            keywords: List of keywords to search for

        Returns:
            int: Number of keyword matches
        """
        return sum(1 for keyword in keywords if keyword in text)
