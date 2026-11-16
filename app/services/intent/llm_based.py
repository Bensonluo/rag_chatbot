"""
LLM-based intent detector for business intents.

Uses language model understanding to classify customer service queries
into business intent categories.
"""

import json
import re
from typing import Any

from app.core.exceptions import ExternalServiceError
from app.models.enums.intent import Intent
from app.services.intent.base import (
    CoDetectedIntent,
    IntentDetection,
    IntentDetector,
    IntentResult,
)
from app.services.llm.base import LLMMessage, LLMServiceBase

# Reasoning-capable GLM models (the deployment serves glm-5.3-flash)
# spend internal chain-of-thought tokens before emitting content: at a
# 50-token budget the same classification prompt at temperature 0
# intermittently returned EMPTY content (observed live 2026-09-30),
# degrading every low-rule-confidence query to unknown/0.0. The answer
# is one word, but the budget must cover the reasoning preamble.
_INTENT_MAX_TOKENS = 200

# The multi-intent verdict carries a primary label plus up to a few
# co-occurring labels in JSON — still far below the reasoning-preamble
# budget problem that fixed this constant, but the JSON envelope adds
# tokens on top of the single-word answer.
_MULTI_INTENT_MAX_TOKENS = 1200
_MULTI_RESPONSE_MAX_CHARS = 16000


class LLMIntentDetector(IntentDetector):
    """
    LLM-based intent detector for customer service business intents.

    Classifies queries into task-oriented (refund, return, order, shipping, complaint),
    knowledge (faq, policy), general-world knowledge (general_knowledge), dialogue
    (chitchat, greeting), and meta intents.
    """

    def __init__(
        self,
        llm_service: LLMServiceBase,
        prompt_template: str | None = None,
    ) -> None:
        self.llm_service = llm_service
        self.prompt_template = prompt_template
        self.intents = [i.value for i in Intent]

    async def detect(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> Intent:
        result = await self.detect_with_confidence(query, context)
        return result.intent

    async def detect_with_confidence(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> IntentResult:
        if not query or not query.strip():
            return IntentResult(intent=Intent.UNKNOWN, confidence=0.0)

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(query, context)

        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]

        try:
            response = await self.llm_service.generate(
                messages=messages,
                max_tokens=_INTENT_MAX_TOKENS,
                temperature=0.0,
            )
            # Reasoning providers (GLM 5.x, MiniMax M-series — both observed
            # live) inline <think>…</think> before the verdict; word-matching
            # the raw stream would read the deliberation ("return policy"
            # inside the think block) as the intent itself.
            from app.services.llm.reasoning_filter import strip_reasoning

            return self._parse_response(strip_reasoning(response.content), query)

        except ExternalServiceError:
            raise
        except Exception as e:
            raise ExternalServiceError(
                service="LLM Intent Detection",
                message=f"Failed to detect intent: {str(e)}",
            ) from e

    async def detect_multi(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> IntentDetection:
        """Use a separate request-scoped JSON protocol; legacy prompts stay unchanged.

        Invalid model output abstains atomically, without recovering executable
        labels from substrings or making a second, less constrained model call.
        """
        # Existing custom intent legs may override only the original
        # single-result contract. Preserve that extension point rather
        # than bypassing it with a new provider call. A subclass can opt
        # into decomposition explicitly by overriding detect_multi too.
        if type(self).detect_with_confidence is not LLMIntentDetector.detect_with_confidence:
            return await IntentDetector.detect_multi(self, query, context)
        if not query or not query.strip():
            return IntentDetection(primary=IntentResult(Intent.UNKNOWN, 0.0))

        messages = [
            LLMMessage(role="system", content=self._build_multi_intent_system_prompt()),
            LLMMessage(
                role="user",
                content="Conversation context (reference only):\n"
                + json.dumps(context or {}, ensure_ascii=False, default=str)
                + f"\n\nNew query: {query}\n\nReturn the request-scoped JSON object.",
            ),
        ]
        try:
            response = await self.llm_service.generate(
                messages=messages,
                max_tokens=_MULTI_INTENT_MAX_TOKENS,
                temperature=0.0,
            )
            from app.services.llm.reasoning_filter import strip_reasoning

            content = response.content
            if len(content) <= _MULTI_RESPONSE_MAX_CHARS:
                content = strip_reasoning(content)
            return self._parse_multi_response(content, query)
        except ExternalServiceError:
            raise
        except Exception as e:
            raise ExternalServiceError(
                service="LLM Multi-Intent Detection",
                message=f"Failed to detect intents: {str(e)}",
            ) from e

    def _build_multi_intent_system_prompt(self) -> str:
        labels = ", ".join(intent.value for intent in Intent)
        return (
            self._build_system_prompt()
            + f"""

For this request, replace the single-label output rule with this strict protocol.
Allowed labels: {labels}.
Return ONLY JSON, with exactly these fields:
{{"primary": "refund", "confidence": 0.9, "primary_query": "Refund order A",
  "speech_act": "action",
  "co_detected": [{{"intent": "refund", "confidence": 0.8, "query": "refund order B", "speech_act": "action"}}]}}
Each entry is an INDEPENDENT user request, not a competing label or topic keyword.
Every query must be a nonempty VERBATIM contiguous subspan of the new query;
never copy tasks from conversation context. Use disjoint spans. Preserve case.
Include the whole owning clause: never clip away its condition or history.
speech_act is one of: action (execute now), informational (policy/how-to
question), historical (something already done, e.g. "I already returned
ORD1001"), conditional (action wanted only if a condition holds),
alternative (asking which option is better, e.g. "should I return or
refund"). Never mark a current request as historical/alternative.
When one action applies to several orders ("refund orders A and B"),
return one entry for the entire shared span; the parser expands it once into
per-order frames. Never repeat that span in co_detected.
For an unsegmented single request primary_query may be empty (whole input).
A repeated label with different entities/clauses is still a distinct request.
Never turn a return policy or how-to question into an executable refund/return.
Ignore quoted, negated, deferred, and superseded/corrected requests.
Keep conditional and historical clauses with their non-action speech_act.
Unquoted third-party reports ("The email mentions a return") are informational,
not instructions. Definitions ("退款是什么意思") are informational too.
A correction replaces the earlier request rather than adding to it.
Use handoff as primary for an explicit human request; otherwise source order.
Use general_knowledge for unrelated world questions; preserve the other labels.
Confidence must be a finite JSON number between 0 and 1, never a string.
Do not truncate requests. If uncertain, use primary unknown, confidence 0,
primary_query empty, co_detected [].
"""
        )

    def _parse_multi_response(self, response: str, query: str = "") -> IntentDetection:
        """Validate the whole bounded JSON object and all source spans atomically."""
        from app.services.intent.multi_intent import (
            classify_speech_act,
            has_unquoted_action,
            is_blocked_request,
            is_informational_action,
            request_ranges,
            shared_target_count,
            visible_text,
        )

        valid_speech_acts = {
            "action",
            "informational",
            "historical",
            "conditional",
            "alternative",
        }

        def invalid() -> IntentDetection:
            return IntentDetection(
                primary=IntentResult(
                    Intent.UNKNOWN, 0.0, {"method": "llm", "invalid_response": True}
                )
            )

        def speech_act_of(item: dict[str, Any], span: str) -> str:
            # Missing is legacy-compatible; explicit null, containers and
            # unknown strings reject the entire verdict, including unknown.
            if "speech_act" not in item:
                return classify_speech_act(span)
            value = item["speech_act"]
            if not isinstance(value, str) or value not in valid_speech_acts:
                raise ValueError("Invalid speech_act")
            return value

        def unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate JSON key")
                result[key] = value
            return result

        def confidence(value: Any) -> float:
            if type(value) not in (int, float) or not 0 <= value <= 1:
                raise ValueError("Confidence must be a finite number in [0, 1]")
            return float(value)

        occupied: list[tuple[int, int]] = []
        safe_ranges = request_ranges(query)
        executable = {
            Intent.REFUND,
            Intent.RETURN,
            Intent.QUERY_ORDER,
            Intent.TRACK_SHIPPING,
            Intent.COMPLAINT,
            Intent.HANDOFF,
            Intent.CANCEL,
            Intent.CONFIRM,
        }

        def anchor(span: Any, intent: Intent, act: str = "action") -> str:
            if not isinstance(span, str) or not span.strip():
                raise ValueError("Missing request span")
            start = query.find(span)
            while start >= 0:
                end = start + len(span)
                if all(end <= left or start >= right for left, right in occupied):
                    if intent in executable and act == "action":
                        # Do not let a clipped positive substring escape the
                        # negation/quotation/correction of its owning clause.
                        allowed = False
                        for left, right in safe_ranges:
                            clause = query[left:right]
                            if (
                                not left <= start < end <= right
                                or is_blocked_request(clause)
                                or classify_speech_act(clause) != "action"
                                or not has_unquoted_action(clause)
                            ):
                                continue
                            if intent in {
                                Intent.REFUND,
                                Intent.RETURN,
                                Intent.CANCEL,
                            } and is_informational_action(clause):
                                continue
                            offset = start - left
                            visible = visible_text(clause)[offset : offset + len(span)]
                            if visible.strip():
                                allowed = True
                                break
                        if not allowed:
                            start = query.find(span, start + 1)
                            continue
                    occupied.append((start, end))
                    return span
                start = query.find(span, start + 1)
            raise ValueError("Unanchored or overlapping request span")

        try:
            if len(response) > _MULTI_RESPONSE_MAX_CHARS:
                return invalid()
            data = json.loads(response, object_pairs_hook=unique_keys)
            if not isinstance(data, dict) or not {
                "primary",
                "confidence",
                "primary_query",
                "co_detected",
            } <= set(data) <= {
                "primary",
                "confidence",
                "primary_query",
                "speech_act",
                "co_detected",
            }:
                return invalid()
            primary = Intent(data["primary"])
            score = confidence(data["confidence"])
            co = data["co_detected"]
            if not isinstance(co, list):
                return invalid()
            primary_query = data["primary_query"]
            if not isinstance(primary_query, str):
                return invalid()
            primary_act = speech_act_of(data, primary_query or query)
            if primary is Intent.UNKNOWN:
                if score != 0 or co or primary_query != "":
                    return invalid()
                return IntentDetection(primary=IntentResult(Intent.UNKNOWN, 0.0, {"method": "llm"}))
            if primary_query == "" and not co:
                anchor(query, primary, primary_act)
            else:
                anchor(primary_query, primary, primary_act)
            # Retain the actual validated occurrence, not query.find(span):
            # repeated text may first occur in a blocked or occupied clause.
            entries: list[tuple[Intent, float, str, int, str]] = [
                (primary, score, primary_query, occupied[-1][0], primary_act)
            ]
            for item in co:
                if not isinstance(item, dict) or not {
                    "intent",
                    "confidence",
                    "query",
                } <= set(item) <= {"intent", "confidence", "query", "speech_act"}:
                    return invalid()
                intent = Intent(item["intent"])
                if intent is Intent.UNKNOWN:
                    return invalid()
                item_score = confidence(item["confidence"])
                if not isinstance(item["query"], str):
                    return invalid()
                item_act = speech_act_of(item, item["query"])
                span = anchor(item["query"], intent, item_act)
                entries.append((intent, item_score, span, occupied[-1][0], item_act))
            # One action over several orders is N frames sharing one span.
            expanded: list[tuple[Intent, float, str, int, str, int]] = []
            for intent, item_score, span, offset, act in entries:
                for entity_index in range(shared_target_count(intent, span or query, act)):
                    expanded.append((intent, item_score, span, offset, act, entity_index))
            expanded.sort(key=lambda entry: entry[3])
            # Handoff is the only priority override; other requests retain
            # source order, including different orders sharing one label.
            primary_index = next(
                (index for index, entry in enumerate(expanded) if entry[0] is Intent.HANDOFF),
                0,
            )
            primary, score, primary_query, _, primary_act, primary_entity = expanded.pop(
                primary_index
            )
            return IntentDetection(
                primary=IntentResult(primary, score, {"method": "llm"}),
                primary_query=primary_query,
                speech_act=primary_act,
                entity_index=primary_entity,
                co_detected=[
                    CoDetectedIntent(
                        intent,
                        item_score,
                        "llm",
                        span or query,
                        speech_act=act,
                        entity_index=entity_index,
                    )
                    for intent, item_score, span, _, act, entity_index in expanded
                ],
            )
        except (ValueError, TypeError, KeyError, RecursionError):
            return invalid()

    def _build_system_prompt(self) -> str:
        if self.prompt_template:
            return self.prompt_template

        return """You are an intent classifier for a customer service chatbot.
Classify user queries into one of these business intents:

Task-oriented:
- refund: User wants a refund
- return: User wants to return/exchange an item
- query_order: User wants to check order status
- track_shipping: User wants to track shipping/delivery
- complaint: User wants to file a complaint

Knowledge:
- faq: General question about this shop's products, services, or processes
- policy: Question about this shop's policies (return policy, warranty, privacy, etc.)
- general_knowledge: General-world question clearly UNRELATED to this shop —
  health tips, science, geography, history, general tech concepts ("what is
  quantum entanglement", "how do I deal with a headache"). Anything phrased
  as "your policy on X" or asking about this store's products, orders,
  shipping, refunds, or payments is NEVER general_knowledge.

Dialogue:
- chitchat: Casual conversation, not service-related
- greeting: Greetings (hello, hi, good morning)

Meta:
- confirm: User confirms something (yes, correct, continue)
- deny: User denies something (no, not right)
- cancel: User wants to cancel current operation
- unknown: Cannot determine intent

Handoff:
- handoff: User explicitly asks for a human agent (transfer me to a human)

Rules:
- Respond with ONLY the intent name (lowercase)
- If user mentions order_id or order number with a service request, classify by the service type
- An explicit human-agent request is always "handoff", even if the message also contains a complaint or task keyword
- Shop-scoped how-things-work questions ("how do I return an item") are "faq"
- Use "general_knowledge" only when the question has nothing to do with this shop
- Default to "unknown" if truly uncertain"""

    def _build_user_prompt(self, query: str, context: dict[str, Any] | None) -> str:
        prompt = f"Classify this query: {query}\n\nIntent:"

        if context:
            if "previous_messages" in context:
                messages = context["previous_messages"]
                if messages:
                    recent = messages[-2:] if len(messages) > 2 else messages
                    conversation = "\n".join(
                        f"{m.get('role', 'user')}: {m.get('content', '')}" for m in recent
                    )
                    prompt = f"""Conversation context:
{conversation}

New query: {query}

Intent:"""
            if "current_intent" in context:
                prompt = f"""Current active task: {context["current_intent"]}
Filled slots: {context.get("filled_slots", {})}

New query: {query}

Intent:"""

        return prompt

    def _parse_response(self, response: str, query: str) -> IntentResult:  # noqa: ARG002  # mirrors rule-based parser signature
        cleaned = response.strip().lower()

        if cleaned.startswith("{"):
            try:
                data = json.loads(cleaned)
                intent_str = data.get("intent", cleaned)
                confidence = float(data.get("confidence", 0.9))

                for intent in Intent:
                    if intent.value == intent_str:
                        return IntentResult(
                            intent=intent,
                            confidence=min(max(confidence, 0.0), 1.0),
                        )
            except json.JSONDecodeError:
                pass

        intent_str = re.sub(
            r"^(intent:|classification:|category:)\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        # Providers occasionally emit "general knowledge" / "general-knowledge"
        # despite the label table; normalize so the underscore form matches.
        intent_str = re.sub(r"[\s\-]+", "_", intent_str)

        for intent in Intent:
            if intent.value == intent_str or intent.value in intent_str:
                return IntentResult(
                    intent=intent,
                    confidence=0.9,
                    metadata={"method": "llm", "raw_response": response},
                )

        return IntentResult(
            intent=Intent.UNKNOWN,
            confidence=0.0,
            metadata={"method": "llm", "raw_response": response},
        )
