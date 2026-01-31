"""
Rule-based intent detector using keyword matching and patterns.

Fast, lightweight intent detection using predefined rules.
"""
import re
from typing import Optional, dict, list

from app.services.intent.base import IntentDetector, IntentResult
from app.models.enums.intent import Intent


class RuleBasedIntentDetector(IntentDetector):
    """
    Rule-based intent detector using keyword matching.

    Fast and predictable intent detection using keyword patterns
    and regular expressions. Best for common queries with clear patterns.
    """

    def __init__(self) -> None:
        """Initialize the detector with default rules."""
        self.rules = self._build_rules()

    def _build_rules(self) -> dict[Intent, list[dict]]:
        """
        Build detection rules for each intent.

        Returns:
            dict: Mapping of intents to their detection rules
        """
        return {
            Intent.QUESTION: [
                {
                    "keywords": ["what", "where", "when", "who", "why", "which", "whose"],
                    "weight": 1.0,
                },
                {
                    "patterns": [r"can you tell me .+", r"do you know .+"],
                    "weight": 0.8,
                },
            ],
            Intent.COMPARISON: [
                {
                    "keywords": [
                        "vs", "versus", "compare", "difference",
                        "better", "worse", "between", "against"
                    ],
                    "weight": 1.0,
                },
                {
                    "patterns": [r".+ (than|versus|vs) .+", r".+ (better|worse) than .+"],
                    "weight": 1.0,
                },
            ],
            Intent.HOW_TO: [
                {
                    "keywords": ["how", "how to", "steps", "tutorial", "guide", "instructions"],
                    "weight": 1.0,
                },
                {
                    "patterns": [
                        r"how do (i|you|we) .+",
                        r"how to .+",
                        r"ways? to .+",
                        r"step by step .+"
                    ],
                    "weight": 1.0,
                },
            ],
            Intent.DEFINITION: [
                {
                    "keywords": [
                        "define", "definition", "what is", "what are",
                        "means", "refers to", "explain what", "describe"
                    ],
                    "weight": 1.0,
                },
                {
                    "patterns": [r"what (is|are) (an?|the) .+"],
                    "weight": 0.9,
                },
            ],
            Intent.SUMMARY: [
                {
                    "keywords": [
                        "summarize", "summary", "recap", "overview",
                        "brief", "condense", "key points", "main idea"
                    ],
                    "weight": 1.0,
                },
                {
                    "patterns": [r"give me (a )?(brief|short|quick) summary"],
                    "weight": 1.0,
                },
            ],
            Intent.CODE_HELP: [
                {
                    "keywords": [
                        "code", "function", "class", "debug", "error",
                        "bug", "syntax", "implement", "algorithm"
                    ],
                    "weight": 1.0,
                },
                {
                    "patterns": [
                        r"write .+ code",
                        r"create (a )?.+ function",
                        r"implement .+ (in|using) .+",
                        r"help me debug"
                    ],
                    "weight": 1.0,
                },
                # Programming language keywords (lower weight alone)
                {
                    "keywords": [
                        "python", "javascript", "java", "c\\+\\+", "go",
                        "rust", "typescript", "sql", "html", "css", "api"
                    ],
                    "weight": 0.6,
                },
            ],
            Intent.CREATIVE: [
                {
                    "keywords": [
                        "write", "story", "poem", "creative", "imagine",
                        "generate", "create content", "compose"
                    ],
                    "weight": 0.8,
                },
                {
                    "patterns": [
                        r"write (a |an )?(story|poem|article|essay)",
                        r"generate (a |an )?.+ (about|for)"
                    ],
                    "weight": 1.0,
                },
            ],
            Intent.CHITCHAT: [
                {
                    "keywords": [
                        "hello", "hi", "hey", "thanks", "thank you",
                        "bye", "goodbye", "good morning", "good night",
                        "how are you", "what's up"
                    ],
                    "weight": 1.0,
                },
                {
                    "patterns": [
                        r"^(hi|hello|hey)[!]*$",
                        r"how('s| is) it going",
                        r"what'?s up"
                    ],
                    "weight": 1.0,
                },
            ],
            Intent.TASK: [
                {
                    "keywords": [
                        "send", "create", "delete", "update", "schedule",
                        "remind", "book", "order", "buy", "search"
                    ],
                    "weight": 1.0,
                },
                {
                    "patterns": [
                        r"can you (send|create|delete|schedule|remind)",
                        r"i want to (buy|order|book)"
                    ],
                    "weight": 1.0,
                },
            ],
        }

    def detect(
        self,
        query: str,
        context: Optional[dict] = None,
    ) -> Intent:
        """
        Detect the intent of a user query.

        Args:
            query: User's text query
            context: Optional context (not used in rule-based)

        Returns:
            Intent: Detected intent category
        """
        result = self.detect_with_confidence(query, context)
        return result.intent

    def detect_with_confidence(
        self,
        query: str,
        context: Optional[dict] = None,
    ) -> IntentResult:
        """
        Detect intent and return confidence score.

        Args:
            query: User's text query
            context: Optional context

        Returns:
            IntentResult: Detected intent with confidence and metadata
        """
        if not query or not query.strip():
            return IntentResult(intent=Intent.UNKNOWN, confidence=0.0)

        # Normalize query
        normalized_query = self._normalize_query(query)

        # Score each intent
        scores = {intent: 0.0 for intent in Intent}

        # Track matched rules for metadata
        matched_rules = []

        for intent, rules in self.rules.items():
            for rule in rules:
                # Keyword matching
                if "keywords" in rule:
                    matched = self._contains_any(normalized_query, rule["keywords"])
                    if matched:
                        scores[intent] += rule["weight"]
                        matched_rules.append({
                            "intent": intent.value,
                            "rule_type": "keyword",
                            "weight": rule["weight"],
                        })

                # Pattern matching
                if "patterns" in rule:
                    for pattern in rule["patterns"]:
                        if re.search(pattern, normalized_query, re.IGNORECASE):
                            scores[intent] += rule["weight"]
                            matched_rules.append({
                                "intent": intent.value,
                                "rule_type": "pattern",
                                "pattern": pattern,
                                "weight": rule["weight"],
                            })

        # Get highest scoring intent
        max_intent = Intent.UNKNOWN
        max_score = 0.0

        for intent, score in scores.items():
            if score > max_score:
                max_score = score
                max_intent = intent

        # Normalize confidence to 0-1 range
        # Assume max reasonable score is around 3.0
        confidence = min(max_score / 3.0, 1.0) if max_score > 0 else 0.0

        return IntentResult(
            intent=max_intent,
            confidence=confidence,
            metadata={
                "matched_rules": matched_rules,
                "raw_score": max_score,
            } if matched_rules else None,
        )
