"""
LLM-based intent detector using language models.

Accurate intent detection by leveraging LLM understanding.
"""
import json
import re
from typing import Optional, Dict

from app.services.intent.base import IntentDetector, IntentResult
from app.models.enums.intent import Intent
from app.services.llm.base import LLMServiceBase, LLMMessage
from app.core.exceptions import ExternalServiceError


class LLMIntentDetector(IntentDetector):
    """
    LLM-based intent detector using language model understanding.

    More accurate than rule-based detection but slower and more expensive.
    Best for complex queries or when accuracy is critical.
    """

    def __init__(
        self,
        llm_service: LLMServiceBase,
        prompt_template: Optional[str] = None,
    ) -> None:
        """
        Initialize the LLM intent detector.

        Args:
            llm_service: LLM service to use for detection
            prompt_template: Optional custom prompt template
        """
        self.llm_service = llm_service
        self.prompt_template = prompt_template
        self.intents = [i.value for i in Intent]

    async def detect(
        self,
        query: str,
        context: Optional[dict] = None,
    ) -> Intent:
        """
        Detect the intent of a user query.

        Args:
            query: User's text query
            context: Optional context (conversation history, etc.)

        Returns:
            Intent: Detected intent category

        Raises:
            ExternalServiceError: If LLM API call fails
        """
        result = await self.detect_with_confidence(query, context)
        return result.intent

    async def detect_with_confidence(
        self,
        query: str,
        context: Optional[dict] = None,
    ) -> IntentResult:
        """
        Detect intent and return confidence score using LLM.

        Args:
            query: User's text query
            context: Optional context

        Returns:
            IntentResult: Detected intent with confidence score

        Raises:
            ExternalServiceError: If LLM API call fails
        """
        if not query or not query.strip():
            return IntentResult(intent=Intent.UNKNOWN, confidence=0.0)

        # Build prompt
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(query, context)

        # Create messages
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]

        try:
            # Call LLM
            response = await self.llm_service.generate(
                messages=messages,
                max_tokens=50,
                temperature=0.0,  # Deterministic for classification
            )

            # Parse response
            return self._parse_response(response.content, query)

        except ExternalServiceError:
            raise
        except Exception as e:
            raise ExternalServiceError(
                service="LLM Intent Detection",
                message=f"Failed to detect intent: {str(e)}",
            ) from e

    def _build_system_prompt(self) -> str:
        """
        Build system prompt for intent classification.

        Returns:
            str: System prompt
        """
        if self.prompt_template:
            return self.prompt_template

        intents_list = "\n".join(f"- {intent}" for intent in self.intents)

        return f"""You are an intent classifier. Classify user queries into one of the following intents:

{intents_list}

Guidelines:
- Respond with only the intent name (lowercase)
- Choose the most specific intent that matches
- If uncertain, choose 'unknown'
- Keep responses concise"""

    def _build_user_prompt(self, query: str, context: Optional[dict]) -> str:
        """
        Build user prompt with query and optional context.

        Args:
            query: User's query
            context: Optional context

        Returns:
            str: User prompt
        """
        prompt = f"Classify this query: {query}\n\nIntent:"

        # Add context if available
        if context:
            if "previous_messages" in context:
                messages = context["previous_messages"]
                if messages:
                    # Show last few messages for context
                    recent = messages[-2:] if len(messages) > 2 else messages
                    conversation = "\n".join(
                        f"{m.get('role', 'user')}: {m.get('content', '')}"
                        for m in recent
                    )
                    prompt = f"""Conversation context:
{conversation}

New query: {query}

Intent:"""

        return prompt

    def _parse_response(self, response: str, query: str) -> IntentResult:
        """
        Parse LLM response into IntentResult.

        Args:
            response: LLM response text
            query: Original query

        Returns:
            IntentResult: Parsed result
        """
        # Clean response
        cleaned = response.strip().lower()

        # Try to parse as JSON first
        if cleaned.startswith("{"):
            try:
                data = json.loads(cleaned)
                intent_str = data.get("intent", cleaned)
                confidence = float(data.get("confidence", 0.9))

                # Map to enum
                for intent in Intent:
                    if intent.value == intent_str:
                        return IntentResult(
                            intent=intent,
                            confidence=min(max(confidence, 0.0), 1.0),
                        )
            except json.JSONDecodeError:
                pass  # Fall through to string parsing

        # Parse as plain text
        # Remove common prefixes
        intent_str = re.sub(
            r"^(intent:|classification:|category:)\s*",
            "",
            cleaned,
            flags=re.IGNORECASE
        ).strip()

        # Map to enum
        for intent in Intent:
            if intent.value == intent_str or intent.value in intent_str:
                # LLM-based detection typically has high confidence
                return IntentResult(
                    intent=intent,
                    confidence=0.9,
                    metadata={"method": "llm", "raw_response": response},
                )

        # Fallback to unknown
        return IntentResult(
            intent=Intent.UNKNOWN,
            confidence=0.0,
            metadata={"method": "llm", "raw_response": response},
        )

    def _format_prompt(self, template: str, **kwargs) -> str:
        """
        Format a prompt template with variables.

        Args:
            template: Template string
            **kwargs: Variables to substitute

        Returns:
            str: Formatted template
        """
        return template.format(**kwargs)
