"""
Hybrid intent detector combining rule-based and LLM approaches.

Uses fast rule-based detection when confident, falls back to LLM for complex queries.
"""

from typing import Any

from app.core.exceptions import ExternalServiceError
from app.models.enums.intent import Intent
from app.services.intent.base import (
    IntentDetection,
    IntentDetector,
    IntentResult,
)
from app.services.intent.llm_based import LLMIntentDetector
from app.services.intent.rule_based import RuleBasedIntentDetector


class HybridIntentDetector(IntentDetector):
    """
    Hybrid intent detector combining rule-based and LLM approaches.

    Strategy:
    1. Try rule-based detection first (fast, cheap)
    2. If confidence is below threshold, use LLM (accurate, expensive)
    3. Balance between speed and accuracy
    """

    def __init__(
        self,
        rule_based: RuleBasedIntentDetector,
        llm_based: LLMIntentDetector,
        confidence_threshold: float = 0.7,
    ) -> None:
        """
        Initialize the hybrid intent detector.

        Args:
            rule_based: Rule-based detector instance
            llm_based: LLM-based detector instance
            confidence_threshold: Threshold for using LLM fallback (0.0-1.0)
        """
        self.rule_based = rule_based
        self.llm_based = llm_based
        self.confidence_threshold = confidence_threshold

    async def detect(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> Intent:
        """
        Detect intent using hybrid approach.

        Args:
            query: User's text query
            context: Optional context

        Returns:
            Intent: Detected intent category
        """
        result = await self.detect_with_confidence(query, context)
        return result.intent

    async def detect_with_confidence(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> IntentResult:
        """
        Detect intent with confidence using hybrid approach.

        Args:
            query: User's text query
            context: Optional context

        Returns:
            IntentResult: Detected intent with confidence and metadata
        """
        # Try rule-based first
        rule_result = await self.rule_based.detect_with_confidence(query, context)

        # Use rule-based result if confidence is high enough
        if rule_result.confidence >= self.confidence_threshold:
            return IntentResult(
                intent=rule_result.intent,
                confidence=rule_result.confidence,
                metadata={
                    "method": "rule_based",
                    "threshold": self.confidence_threshold,
                    **(rule_result.metadata or {}),
                },
            )

        # Fall back to LLM for low confidence. Light-tier providers are
        # deliberately outside the resilient chain; an outage/empty reasoning
        # response must degrade to the deterministic rule signal, never fail
        # the whole dialogue turn.
        try:
            llm_result = await self.llm_based.detect_with_confidence(query, context)
        except ExternalServiceError:
            return IntentResult(
                intent=rule_result.intent,
                confidence=rule_result.confidence,
                metadata={
                    "method": "rule_based",
                    "fallback": "llm_unavailable",
                    "threshold": self.confidence_threshold,
                    "rule_intent": rule_result.intent.value,
                    **(rule_result.metadata or {}),
                },
            )

        # The LLM fallback is an upgrade attempt, not a blind override.
        # When it comes back weaker than the rule layer (e.g. a reasoning
        # model truncated its content to '' → unknown/0.0), keep the rule
        # result rather than discarding the rule signal entirely.
        if llm_result.confidence < rule_result.confidence:
            return IntentResult(
                intent=rule_result.intent,
                confidence=rule_result.confidence,
                metadata={
                    "method": "rule_based",
                    "fallback": "llm_below_rule_confidence",
                    "threshold": self.confidence_threshold,
                    "rule_intent": rule_result.intent.value,
                    "llm_intent": llm_result.intent.value,
                    "llm_confidence": llm_result.confidence,
                    **(rule_result.metadata or {}),
                },
            )

        return IntentResult(
            intent=llm_result.intent,
            confidence=llm_result.confidence,
            metadata={
                "method": "llm_fallback",
                "rule_confidence": rule_result.confidence,
                "threshold": self.confidence_threshold,
                "rule_intent": rule_result.intent.value,
                **(llm_result.metadata or {}),
            },
        )

    async def detect_multi(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> IntentDetection:
        """Arbitrate complete request-scoped verdicts, never union lexical labels.

        A strong primary does not short-circuit a plausibly mixed message.
        On invalid output or service outage, retain the conservative rule
        verdict, including its original request spans.
        """
        from app.services.intent.multi_intent import plausible_mixed_request

        rules = await self.rule_based.detect_multi(query, context)

        def annotate(detection: IntentDetection, **metadata: Any) -> IntentDetection:
            primary = detection.primary
            return IntentDetection(
                primary=IntentResult(
                    primary.intent, primary.confidence, {**(primary.metadata or {}), **metadata}
                ),
                primary_query=detection.primary_query,
                speech_act=detection.speech_act,
                entity_index=detection.entity_index,
                co_detected=detection.co_detected,
            )

        if rules.primary.confidence >= self.confidence_threshold and not plausible_mixed_request(
            query
        ):
            return annotate(rules, method="rule_based", threshold=self.confidence_threshold)
        try:
            llm = await self.llm_based.detect_multi(query, context)
        except ExternalServiceError:
            return annotate(rules, method="rule_based", fallback="llm_unavailable")

        if (llm.primary.metadata or {}).get("invalid_response"):
            return annotate(rules, method="rule_based", fallback="llm_invalid_response")
        mixed_upgrade = (
            plausible_mixed_request(query)
            and bool(llm.co_detected)
            and llm.primary.confidence >= self.confidence_threshold
        )
        if llm.primary.confidence < rules.primary.confidence and not mixed_upgrade:
            return annotate(rules, method="rule_based", fallback="llm_below_rule_confidence")
        return annotate(llm, method="llm_fallback", rule_confidence=rules.primary.confidence)
