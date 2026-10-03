"""
LangGraph dialogue nodes and conditional edge functions.

Each node is a function (state: DialogueState) -> dict that returns
only the fields it updates. The NodeFactory injects services via closure
so nodes stay pure with respect to the graph.

Terminal nodes accept an optional second ``config`` parameter: LangGraph
inspects the signature and injects the invoke-time RunnableConfig, whose
``configurable`` carries the per-request ``stream_queue`` used for true
token streaming (see ChatService.process_message_stream).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from app.services.agent.service import AgentService
    from app.services.chat.answer_cache import AnswerCacheService
    from app.services.chat.semantic_cache import SemanticCacheService
    from app.services.dialogue.tools import ToolDefinition, ToolRegistry
    from app.services.faq.store import FAQService
    from app.services.graph.retrieval.graph_retrieval_service import (
        GraphRetrievalService,
    )
    from app.services.guardrails.base import GuardrailService
    from app.services.handoff.service import HandoffService
    from app.services.intent.base import IntentDetector
    from app.services.llm.base import LLMMessage, LLMServiceBase
    from app.services.retrieval.retrieval_cache import RetrievalCacheService
    from app.services.retrieval.vector_base import SearchResult
    from app.services.slot_filling.base import SlotFiller, SlotFillingResult

from langchain_core.runnables import RunnableConfig

from app.config.settings import settings
from app.models.enums.intent import (
    DIRECT_INTENTS,
    GRAPH_INTENTS,
    HANDOFF_INTENTS,
    INTENT_DISPLAY_NAMES,
    META_INTENTS,
    RAG_INTENTS,
    TASK_INTENTS,
)
from app.services.agent.metrics import AGENT_LOOP_OUTCOMES, OUTCOME_FALLBACK
from app.services.chat.answer_cache import CachedAnswer
from app.services.dialogue.emotion import assess_emotion
from app.services.dialogue.funnel_metrics import (
    LAYER_AGENT_TOOL,
    LAYER_DIRECT,
    LAYER_FAQ,
    LAYER_HANDOFF,
    LAYER_L0_CACHE,
    LAYER_L1_SEMANTIC,
    LAYER_RAG,
    record_funnel_layer,
)
from app.services.dialogue.i18n import (
    CONFIRMATION_ACTION_NAMES,
    CONFIRMATION_ASK,
    CONFIRMATION_DETAIL_EMPTY,
    CONFIRMATION_DETAIL_JOIN,
    CONFIRMATION_EXPIRED,
    GENERATION_FAILED,
    GENERATION_LANG_DIRECTIVE,
    GUARDRAIL_INPUT_BLOCKED,
    GUARDRAIL_OUTPUT_BLOCKED,
    HANDOFF_ACK,
    HANDOFF_ALREADY_QUEUED,
    HANDOFF_CONTEXT_NOTE,
    HANDOFF_EMOTION_PREFIX,
    HANDOFF_NO_TICKET,
    HANDOFF_QUEUE,
    LANG_ZH,
    NO_EVIDENCE,
    RETRIEVAL_DEGRADED,
    detect_language,
)
from app.services.dialogue.query_rewriter import condense_for_retrieval
from app.services.dialogue.state import DialogueState
from app.services.facts.metrics import CLAIM_CHECKS, CLAIM_VIOLATIONS
from app.services.guardrails.base import GuardrailService
from app.services.guardrails.stream_redactor import PIIStreamRedactor
from app.services.handoff.service import (
    REASON_EMOTION,
    REASON_EXPLICIT,
)
from app.services.observability.pipeline_tracer import traced_stage
from app.services.observability.trace_events import emit_trace
from app.services.retrieval.metrics import (
    RETRIEVAL_FILTER_FALLBACKS,
    RETRIEVAL_FILTERED_SEARCHES,
)
from app.services.slot_filling.slot_types import (
    extract_slots_from_message,
    get_missing_slots,
    get_next_prompt,
    is_order_reference_only,
)

logger = logging.getLogger(__name__)

# Deterministic evidence-gap copy (review 2026-09-26, #6): a knowledge
# question with an empty KB slice must not fall through to free LLM
# generation — the model would invent policy. Two distinct sentences,
# because the correct next move differs: rephrase/handoff when the KB
# simply lacks the answer, retry/handoff when retrieval itself failed.
# Bilingual pairs live in i18n; these aliases keep the Chinese copy as
# the import anchor for existing pinned tests (language selection now
# happens at each branch, keyed by the turn's message).
NO_EVIDENCE_RESPONSE = NO_EVIDENCE[LANG_ZH]
RETRIEVAL_DEGRADED_RESPONSE = RETRIEVAL_DEGRADED[LANG_ZH]


def _turn_lang(state: DialogueState | None, fallback_text: str = "") -> str:
    """Language of the current turn, for canned-copy selection.

    Prefers the turn's user message; nodes without state (rare) fall
    back to whatever text they hold (e.g. the LLM prompt).
    """
    if state is not None and state.get("message"):
        return detect_language(state["message"])
    return detect_language(fallback_text)


def _stream_queue(config: Optional[RunnableConfig]) -> asyncio.Queue[Any] | None:
    """Extract the per-request stream queue from a LangGraph invoke config."""
    if config is None:
        return None
    queue = (config.get("configurable") or {}).get("stream_queue")
    return queue if isinstance(queue, asyncio.Queue) else None


def _used_history(config: Optional[RunnableConfig]) -> bool:
    """Whether this request's generation folded in history or user facts.

    ``_call_llm`` sets the flag on the per-request ``configurable`` when
    prior turns or cross-session facts entered the prompt. Write gates
    (L0 answer cache, L1 semantic cache) read it back through here: a
    history-derived answer is personal to this dialogue and must never
    be replayed at a different visitor (review 2026-09-26, finding #5).
    """
    if config is None:
        return False
    return bool((config.get("configurable") or {}).get("generation_used_history"))


def _emit_response(text: str, config: Optional[RunnableConfig]) -> None:
    """Push a complete response to the stream queue.

    Terminal nodes call this after producing their final response. LLM
    paths have already streamed token-by-token (flagged on the config),
    so only template/guardrail-generated responses go through here —
    which is what makes the stream interface uniform: every response the
    consumer sees came from the queue, never from node-name parsing.
    """
    queue = _stream_queue(config)
    if queue is None or not text:
        return
    configurable = config.get("configurable") if config else None
    if isinstance(configurable, dict) and configurable.get("streamed_response"):
        return  # tokens already streamed; pushing the full text would duplicate
    queue.put_nowait(text)


class NodeFactory:
    """Creates node functions with service dependencies injected via closure."""

    def __init__(
        self,
        intent_detector: IntentDetector,
        # Nullable: the API wiring passes None when slot filling is
        # disabled. Consumed by rag_lookup_node for retrieval
        # enrichment; task slots use extract_slots_from_message.
        slot_filler: SlotFiller | None,
        tool_registry: ToolRegistry,
        retrieval_pipeline: dict[str, Any] | None = None,
        llm_service: LLMServiceBase | None = None,
        guardrail_service: GuardrailService | None = None,
        graph_retrieval_service: GraphRetrievalService | None = None,
        handoff_service: HandoffService | None = None,
        agent_service: AgentService | None = None,
        faq_service: FAQService | None = None,
        # Recent-turn provider for LLM context: chronological prior
        # turns for the session (see chat/factory wiring — persister
        # backed, best-effort). None leaves generators single-message.
        history_provider: Callable[[int], Awaitable[list[LLMMessage]]] | None = None,
        # Persona system message prepended by every generator. None →
        # the built-in e-commerce CS persona; a custom string overrides
        # it (per-tenant voice) — see CHAT_SYSTEM_PROMPT setting.
        system_prompt: str | None = None,
        # Bounded cross-session user-facts recall (Phase B2): given a
        # user_id, returns that user's newest durable facts for prompt
        # context. None leaves generators persona-only; failures inside
        # the provider degrade to no personalization, never a failed chat.
        user_facts_provider: Callable[[int], Awaitable[list[str]]] | None = None,
        # L0 exact-answer cache (docs/cache-layering-plan.md): when
        # provided, a stateless grounded turn replays in ~5ms. None
        # leaves the lookup node as a pure pass-through — the cache is
        # an optimization, never a dependency.
        answer_cache: AnswerCacheService | None = None,
        # L1 semantic answer cache (plan layer 1): when provided, the
        # direct tier (greeting/chitchat) replays near-duplicate
        # smalltalk without the LLM call. Same fail-open doctrine.
        semantic_cache: SemanticCacheService | None = None,
        # L2 retrieval-result cache (plan layer 2): when provided,
        # repeated identical (query, filters) pairs replay their doc
        # set without re-running the Qdrant+BM25 legs.
        retrieval_cache: RetrievalCacheService | None = None,
    ) -> None:
        self._intent_detector = intent_detector
        self._history_provider = history_provider
        # One history fetch per turn: the retrieval-side condense and
        # generation read the same snapshot (turn_id-keyed single-slot
        # memo; interleaved turns just refetch on a key miss).
        self._turn_history_memo: tuple[str, list[Any]] | None = None
        self._system_prompt = system_prompt
        self._user_facts_provider = user_facts_provider
        self._slot_filler = slot_filler
        self._tool_registry = tool_registry
        self._retrieval_pipeline = retrieval_pipeline or {}
        self._llm_service = llm_service
        self._guardrail_service = guardrail_service
        self._graph_retrieval_service = graph_retrieval_service
        self._handoff_service = handoff_service
        self._agent_service = agent_service
        self._faq_service = faq_service
        self._answer_cache = answer_cache
        self._semantic_cache = semantic_cache
        self._retrieval_cache = retrieval_cache

    # ── Nodes ────────────────────────────────────────────────────────────────

    @traced_stage("cs.guardrail_input")
    async def guardrail_node(
        self, state: DialogueState, config: Optional[RunnableConfig] = None
    ) -> dict[str, Any]:
        """Check input message against guardrail rules.

        Passes through if no guardrail service is configured, the message
        is already blocked, or the check passes.  Otherwise blocks the
        conversation and returns a safe response.

        A block ends the turn here (see route_after_guardrail) — this
        node is the last stop, so it emits the refusal to the stream
        queue itself instead of relying on a generator node downstream.
        """
        if state.get("blocked"):
            return {}

        if self._guardrail_service is None:
            return {}

        message = state.get("message", "")
        result = self._guardrail_service.check_input(message)

        if result.was_blocked:
            response = GUARDRAIL_INPUT_BLOCKED[_turn_lang(state)]
            emit_trace("cs.guardrail_input", blocked=True, violations=result.violations[:3])
            _emit_response(response, config)
            return {
                "blocked": True,
                "blocked_reason": ", ".join(result.violations)
                if result.violations
                else "内容安全检查未通过",
                "response": response,
            }

        # Propagate sanitized content (PII redaction) into the dialogue state
        # so redacted text — not the raw input — flows to slots/LLM/retrieval.
        if result.sanitized_content and result.sanitized_content != message:
            emit_trace("cs.guardrail_input", redacted=True)
            return {"message": result.sanitized_content}

        return {}

    @traced_stage("cs.cache")
    async def answer_cache_lookup_node(
        self, state: DialogueState, config: Optional[RunnableConfig] = None
    ) -> dict[str, Any]:
        """L0 exact-answer cache lookup (docs/cache-layering-plan.md).

        Runs after the input guardrail so the lookup key uses sanitized
        text and blocked turns never hit. A miss routes into the normal
        pipeline unchanged; a hit skips intent → retrieval → generation
        entirely (~5s → ~5ms). The cached answer is not trusted
        blindly: the deterministic claim gate and output guardrail
        re-run on every serve — near-zero-cost freshness insurance this
        verify-everything skeleton gets for free.

        Turns with in-flight user state (pending slots, staged
        irreversible actions) skip the lookup: a cached answer cannot
        know about them.
        """
        if self._answer_cache is None or state.get("blocked"):
            return {"route_after_cache": "miss"}

        if state.get("pending_slots") or state.get("pending_confirmation"):
            return {"route_after_cache": "miss"}

        cached = await self._answer_cache.get(state.get("message", ""))
        if cached is None:
            return {"route_after_cache": "miss"}

        updates: dict[str, Any] = {
            "route_after_cache": "hit",
            "response": cached.response,
            "sources": cached.sources,
            "intent": cached.intent,
        }
        # Freshness insurance: re-verify with the deterministic gates
        # before the answer leaves the graph (same contract as the FAQ
        # fast path).
        updates = self._gate_claims(updates, message=state.get("message", ""), check_actions=True)
        response = updates["response"]
        if self._guardrail_service is not None:
            check = self._guardrail_service.check_output(response)
            if check.was_blocked:
                response = GUARDRAIL_OUTPUT_BLOCKED[_turn_lang(state)]
            elif check.sanitized_content and check.sanitized_content != response:
                response = check.sanitized_content
        updates["response"] = response

        emit_trace("cs.cache", hit=True, sources=cached.sources[:3])
        record_funnel_layer(LAYER_L0_CACHE)
        _emit_response(response, config)
        return updates

    @traced_stage("cs.intent")
    async def detect_intent_node(self, state: DialogueState) -> dict[str, Any]:
        """Detect user intent, then publish the verdict on the trace channel."""
        updates = await self._detect_intent_logic(state)
        emit_trace("cs.intent", intent=updates.get("intent"), confidence=updates.get("confidence"))
        return updates

    async def _detect_intent_logic(self, state: DialogueState) -> dict[str, Any]:
        """Detect user intent from the current message.

        Priority logic:
        1. Handoff / cancel / emotion escalation override everything
        2. Confirm/deny resolving a staged action become meta intents
        3. An affirmative with a suspended task resumes it; "unknown"
           never resumes — unrecognized input is answered, not hijacked
        4. A non-executed task intent captures slot answers and
           affirmatives; an executed task is terminal and captures
           nothing (see ``task_executed`` in state.py)
        """
        message = state.get("message", "")
        prev_intent = state.get("intent")

        result = await self._intent_detector.detect_with_confidence(message)
        detected_intent = result.intent.value
        confidence = result.confidence

        # An explicit human-agent request outranks everything, including
        # cancel — “不要了，给我转人工” means the user is abandoning the
        # task AND asking for a person; only the handoff answers that.
        if detected_intent == "handoff":
            return {
                "intent": "handoff",
                "prev_intent": prev_intent,
                "confidence": confidence,
                "handoff_reason": REASON_EXPLICIT,
            }

        # Cancel wins over everything else — user wants out.
        if detected_intent == "cancel":
            return {
                "intent": "cancel",
                "prev_intent": prev_intent,
                "confidence": confidence,
            }

        # Negative-emotion escalation outranks task resume: an angry user
        # mid-task must reach a human, not another slot prompt. (It does
        # NOT outrank cancel — “算了，太失望了” is the user leaving.)
        emotion = assess_emotion(message)
        if emotion.should_escalate:
            return {
                "intent": "handoff",
                "prev_intent": prev_intent,
                "confidence": confidence,
                "handoff_reason": REASON_EMOTION,
            }

        # A confirm/deny answering a staged irreversible action must resolve
        # to the meta intent (which executes or discards the staged action);
        # preserving the task intent here would re-enter the gate forever.
        if detected_intent in ("confirm", "deny") and state.get("pending_confirmation"):
            return {
                "intent": detected_intent,
                "prev_intent": prev_intent,
                "confidence": confidence,
            }

        # Resume: an affirmative (confirm/deny/cancel) with a task
        # suspended on the stack sets intent to that task so slot
        # collection continues. "unknown" deliberately does NOT resume —
        # an unrecognized message (e.g. an off-domain question) must be
        # answered, not silently folded back into the suspended task.
        if detected_intent in META_INTENTS:
            state_stack = state.get("state_stack") or []
            if state_stack:
                suspended_intent = state_stack[-1].get("intent", "")
                if suspended_intent in TASK_INTENTS:
                    return {
                        "intent": suspended_intent,
                        "prev_intent": prev_intent,
                        "confidence": confidence,
                    }

        # If prev is a task intent, check if message provides slot values.
        # An executed task is terminal: it no longer captures follow-ups
        # (slot answers, affirmatives, unknowns) — a "好的" landing after
        # a completed refund must never re-enter and re-stage it.
        if prev_intent and prev_intent in TASK_INTENTS and not state.get("task_executed"):
            existing = dict(state.get("filled_slots") or {})
            merged = extract_slots_from_message(prev_intent, message, existing)
            if len(merged) > len(existing):
                return {
                    "intent": prev_intent,
                    "prev_intent": prev_intent,
                    "confidence": confidence,
                }

            # Confirm/deny after a task preserves the task intent
            if detected_intent in META_INTENTS:
                return {
                    "intent": prev_intent,
                    "prev_intent": prev_intent,
                    "confidence": confidence,
                }

            # Unknown after a task with pending slots → user is answering a prompt
            if detected_intent == "unknown" and state.get("pending_slots"):
                return {
                    "intent": prev_intent,
                    "prev_intent": prev_intent,
                    "confidence": confidence,
                }

        return {
            "intent": detected_intent,
            "prev_intent": prev_intent,
            "confidence": confidence,
        }

    async def handle_switch_node(self, state: DialogueState) -> dict[str, Any]:
        """Detect and handle intent switching.

        When the user changes to a new task intent the current task state
        is pushed onto a stack so it can be resumed later.  If the user
        returns to a suspended task it is popped and restored.
        """
        intent = state.get("intent", "")
        prev_intent = state.get("prev_intent")

        # No previous context or same intent — nothing to do.
        if prev_intent is None or intent == prev_intent:
            return {}

        # Meta intents never trigger a switch.
        if intent in META_INTENTS:
            return {}

        # Only in-flight tasks are suspendable. The stack exists to hold
        # tasks a later "继续" can resume (see the resume check in
        # _detect_intent_logic and the pop scan below); pushing anything
        # else (greeting, chitchat, policy, handoff, unknown) would only
        # pollute the resume hint with an entry nothing can resume.
        if prev_intent not in TASK_INTENTS:
            return {}

        state_stack: list[dict[str, Any]] = list(state.get("state_stack") or [])

        # Check if the user is returning to a previously suspended task.
        if intent in TASK_INTENTS:
            for idx, suspended in enumerate(state_stack):
                if suspended.get("intent") == intent:
                    # Resume: pop and restore.
                    restored = state_stack.pop(idx)
                    return {
                        "state_stack": state_stack,
                        "filled_slots": restored.get("filled_slots", {}),
                        "pending_slots": restored.get("pending_slots", []),
                        # Anything restored here was suspended before it
                        # executed (executed tasks are never pushed), so a
                        # stale flag from the just-finished task must not
                        # leak into the resumed one.
                        "task_executed": False,
                    }

        # An executed task is terminal — never suspended, never resumable.
        # Resuming a task whose tool already ran could only re-run it
        # (risking a second irreversible action) or re-ask a dead
        # confirmation. The flag consumes exactly one switch.
        if state.get("task_executed"):
            return {
                "filled_slots": {},
                "pending_slots": [],
                "slot_prompt": "",
                "task_executed": False,
                "pending_confirmation": None,
            }

        # Suspend the current task. Its staged (unconfirmed) action is
        # discarded with it: a later bare "好的" must never execute a
        # refund the user walked away from. Resuming the task re-stages
        # and re-asks — the safe direction.
        suspended = {
            "intent": prev_intent,
            "filled_slots": state.get("filled_slots", {}),
            "pending_slots": state.get("pending_slots", []),
        }
        state_stack.append(suspended)

        return {
            "state_stack": state_stack,
            "filled_slots": {},
            "pending_slots": [],
            "slot_prompt": "",
            "task_executed": False,
            "pending_confirmation": None,
        }

    async def route_intent_node(self, state: DialogueState) -> dict[str, Any]:
        """Determine the processing route based on the detected intent."""
        intent = state.get("intent", "")

        if intent in TASK_INTENTS:
            # Agent mode: task intents go through the function-calling
            # loop instead of the slot pipeline. The agent node falls
            # back to the slot pipeline when the provider lacks tool
            # support, so availability never depends on agent mode.
            if self._agent_service is not None:
                return {"route": "agent"}
            return {"route": "task"}
        if intent in RAG_INTENTS:
            return {"route": "rag"}
        if intent in DIRECT_INTENTS:
            return {"route": "direct"}
        if intent in META_INTENTS:
            return {"route": "meta"}
        if intent in HANDOFF_INTENTS:
            return {"route": "handoff"}
        if intent in GRAPH_INTENTS:
            return {"route": "rag"}

        return {"route": "direct"}

    @traced_stage("cs.slots")
    async def collect_slots_node(self, state: DialogueState) -> dict[str, Any]:
        """Extract slot values from the user message and merge with existing.

        Falls back to treating the entire message as the value for the
        first missing required slot when regex extraction finds nothing new.
        """
        intent = state.get("intent", "")
        message = state.get("message", "")
        filled_slots: dict[str, Any] = dict(state.get("filled_slots") or {})

        merged = extract_slots_from_message(intent, message, filled_slots)

        # LLM pass: when regex found nothing new during an active
        # collection, let the LLM pull free-form answers into slots
        # before the whole-message heuristic guesses. Failures return
        # {} inside, so behavior degrades to the pre-LLM path.
        if (
            len(merged) == len(filled_slots)
            and state.get("pending_slots")
            and self._llm_service is not None
        ):
            from app.config.settings import get_settings

            if get_settings().SLOT_LLM_EXTRACTION_ENABLED:
                from app.services.dialogue.slot_extraction import llm_extract_slots

                extra = await llm_extract_slots(intent, message, filled_slots, self._llm_service)
                for key, value in extra.items():
                    merged.setdefault(key, value)

        # Fallback: when in a slot-collection flow and the message looks like
        # a direct answer (short, no task keywords), assign it to the first
        # missing required slot.
        if (
            len(merged) == len(filled_slots)
            and state.get("pending_slots")
            and len(message.strip()) > 1
            and len(message.strip()) < 30
            and not any(
                kw in message
                for kw in (
                    "退款",
                    "退货",
                    "订单",
                    "物流",
                    "投诉",
                    "查询",
                    "取消",
                    "refund",
                    "return",
                    "order",
                    "shipping",
                    "complaint",
                )
            )
        ):
            from app.services.slot_filling.slot_types import INTENT_SLOT_SCHEMAS

            schema = INTENT_SLOT_SCHEMAS.get(intent, {})
            required = schema.get("required", [])
            for slot_name in required:
                if slot_name not in merged:
                    merged[slot_name] = message.strip()
                    break

        # Outage terminator for the free-form complaint description
        # (finding ①, 2026-09-30): description carries no regex patterns
        # by nature, and the whole-message heuristic rejects long or
        # keyword-bearing messages — with the extractor LLM down (GLM
        # 429 storm) the complaint flow re-prompted "what would you like
        # to complain about?" forever. Once a category is known, any
        # message that is not merely an order reference IS the
        # description: the 10-char floor keeps one-word answers out and
        # is_order_reference_only keeps bare order numbers out.
        if (
            intent == "complaint"
            and "category" in merged
            and "description" not in merged
            and len(message.strip()) >= 10
            and not is_order_reference_only(message)
        ):
            merged["description"] = message.strip()[:200]

        pending = get_missing_slots(intent, merged)
        next_prompt = get_next_prompt(intent, merged, lang=_turn_lang(state))

        emit_trace("cs.slots", filled_slots=merged, pending_slots=pending)
        return {
            "filled_slots": merged,
            "pending_slots": pending,
            "slot_prompt": next_prompt or "",
        }

    @traced_stage("cs.tool")
    async def execute_tool_node(
        self, state: DialogueState, config: Optional[RunnableConfig] = None
    ) -> dict[str, Any]:
        """Execute the tool associated with the current intent.

        Irreversible tools (refund, return) are staged instead of run:
        the node stores the prepared action in ``pending_confirmation``
        and returns a fixed-template confirmation question. The action
        only executes when the user's next turn resolves to the
        ``confirm`` meta intent (see ``_handle_meta_intent``).
        """
        # Defense in depth: a blocked turn must never execute a tool,
        # even if a routing change lets it reach this node.
        if state.get("blocked"):
            return {}

        intent = state.get("intent", "")
        filled_slots = state.get("filled_slots") or {}
        user_id = state.get("user_id")

        tool = self._tool_registry.get_tool_for_intent(intent)

        if tool is not None and tool.requires_confirmation:
            # Gate every prepared irreversible action, overwriting any
            # stale pending confirmation from an earlier turn.
            summary = _build_confirmation_summary(tool, filled_slots, lang=_turn_lang(state))
            record_funnel_layer(LAYER_AGENT_TOOL)
            _emit_response(summary, config)
            return {
                "pending_confirmation": {
                    "intent": intent,
                    "args": dict(filled_slots),
                    "staged_at": time.time(),
                },
                "response": summary,
                # Staging re-opens the task: it is in flight again and
                # may be suspended/resumed until the action executes.
                "task_executed": False,
            }

        # Reversible tools run immediately; clear any stale pending
        # confirmation so it cannot gate a later action.
        result = await self._tool_registry.execute(intent, filled_slots, user_id=user_id)

        # Same audit-trail shape the meta-confirm branch (1525b69) and
        # the agent path record: a directly-run order query or complaint
        # submission invisible to the per-turn audit is unauditable
        # (review 2026-09-26 #9, same class — found by the routing eval:
        # task turns grounded answers in tool results while
        # executed_tools stayed empty). Failed attempts record ok=False.
        audit_entry = {
            "tool": tool.name if tool else intent,
            "ok": result.success,
            "args": dict(filled_slots),
            "summary": _safe_json(result.data if result.success else {"error": result.message})[
                :200
            ],
        }
        if result.success:
            record_funnel_layer(LAYER_AGENT_TOOL)
            return {
                "tool_result": result.data,
                "pending_confirmation": None,
                "task_executed": True,
                "executed_tools": [audit_entry],
            }

        logger.warning("Tool execution failed for intent %s: %s", intent, result.message)
        return {
            "tool_result": {"error": result.message},
            "pending_confirmation": None,
            # The tool ran and was refused — the task is terminal either
            # way; re-running it can only repeat the failure.
            "task_executed": True,
            "executed_tools": [audit_entry],
        }

    @traced_stage("cs.faq")
    async def faq_lookup_node(
        self, state: DialogueState, config: Optional[RunnableConfig] = None
    ) -> dict[str, Any]:
        """Serve a curated FAQ answer by semantic match, skipping RAG.

        The FAQ tier answers the hottest customer-service traffic with
        pre-approved, deterministic text — no retrieval, no LLM call,
        no hallucination surface on policy questions. Any failure
        (embedding outage, table build failure) routes as a miss into
        the full RAG pipeline: the fast path is an optimization, never
        a dependency. The output guardrail still runs on hits, and
        ``sources`` records ``faq:<id>`` provenance.
        """
        if self._faq_service is None:
            return {"route_after_faq": "miss"}

        try:
            entry = await self._faq_service.match(state.get("message", ""))
        except Exception:  # noqa: BLE001 - availability over fast path
            logger.exception("FAQ fast path failed; continuing with RAG")
            return {"route_after_faq": "miss"}

        if entry is None:
            return {"route_after_faq": "miss"}

        response = entry.answer
        if self._guardrail_service is not None:
            check = self._guardrail_service.check_output(response)
            if check.was_blocked:
                response = GUARDRAIL_OUTPUT_BLOCKED[_turn_lang(state)]
            elif check.sanitized_content and check.sanitized_content != response:
                response = check.sanitized_content

        emit_trace("cs.faq", hit=True, faq_id=entry.faq_id)
        record_funnel_layer(LAYER_FAQ)
        _emit_response(response, config)
        return {
            "route_after_faq": "hit",
            "response": response,
            "sources": [f"faq:{entry.faq_id}"],
        }

    @traced_stage("cs.retrieval")
    async def rag_lookup_node(self, state: DialogueState) -> dict[str, Any]:
        """Retrieve relevant documents via hybrid search.

        Entities extracted from the raw message by the slot filler
        enrich both retrieval paths: normalized slot values are
        appended to the search query (recall for BM25 + vector), and
        entity hints anchor graph retrieval's Cypher generation.
        Extraction failure degrades to searching the raw message.
        """
        message = state.get("message", "")
        intent = state.get("intent", "")
        retrieved_docs: list[dict[str, Any]] = []
        sources: list[str] = []
        retrieval_ran = False
        retrieval_degraded = False

        # Follow-up questions carry their subject in history, not in the
        # string (「那运费呢？」) — condense against recent turns before
        # search. Slot extraction still reads the raw message (order ids
        # and entities live there). Fail-open: None keeps the original.
        rewritten = await self._condense_query(message, state)
        search_message = rewritten if rewritten else message
        if rewritten:
            emit_trace("cs.query_rewrite", original=message, rewritten=rewritten)

        fill_result = await self._extract_query_entities(message)
        entity_hints = fill_result.to_entity_hints() if fill_result else []
        query = _enriched_query(search_message, fill_result)

        # Graph intents go through graph retrieval.
        if intent in GRAPH_INTENTS and self._graph_retrieval_service is not None:
            retrieval_ran = True
            try:
                # GraphRetrievalService exposes search() returning a fused,
                # ranked list — the old .query() call raised AttributeError
                # and silently degraded every graph-intent lookup to empty.
                graph_results = await self._graph_retrieval_service.search(
                    query, entity_hints=entity_hints or None
                )
                if graph_results:
                    retrieved_docs = [_graph_doc_to_dict(r) for r in graph_results]
                    sources = _extract_sources(retrieved_docs)
            except Exception:
                retrieval_degraded = True
                logger.exception("Graph retrieval failed for intent %s", intent)
            emit_trace("cs.retrieval", mode="graph", docs=len(retrieved_docs), sources=sources[:3])
            return {
                "retrieved_docs": retrieved_docs,
                "sources": sources,
                "retrieval_ran": retrieval_ran,
                "retrieval_degraded": retrieval_degraded,
            }

        # Standard hybrid search for RAG intents.
        hybrid_search = self._retrieval_pipeline.get("hybrid_search")
        if hybrid_search is not None:
            try:
                from app.services.retrieval.vector_base import (
                    VectorClientError,
                    VectorSearchRequest,
                    intersect_metadata_filters,
                    retrieval_acl_scope,
                )

                # Counted per _search_once call: a swallowed
                # VectorClientError means both legs raised. Ending the
                # turn with no docs AND swallowed failures is a service
                # outage, not an honest empty KB (review 2026-09-26, #6).
                leg_failures = 0

                async def _search_once(request: VectorSearchRequest) -> list[Any]:
                    nonlocal leg_failures
                    # HybridSearchService raises VectorClientError when both
                    # legs come back empty — a metadata-filter miss looks
                    # exactly like that. Normalize to [] so the unfiltered
                    # retry below stays reachable (the raw raise used to
                    # bypass it and zero recall on every filter miss).
                    try:
                        hits = await hybrid_search.search(request)
                        return list(hits)
                    except VectorClientError:
                        leg_failures += 1
                        return []

                filters = (
                    intersect_metadata_filters(fill_result.to_filters()) if fill_result else {}
                )
                # Server-side retrieval ACL (review 2026-09-26 #1, design
                # D5): the caller's scope is injected from server context
                # on its own request channel — it never passes through
                # slot extraction or the business-filter whitelist, so no
                # user-influenced condition can loosen it. Today the KB is
                # entirely public and the scope is the empty constraint;
                # private documents (design P2) swap in a per-caller
                # expression.
                acl_scope = retrieval_acl_scope()
                # Ask for the candidate pool, not the delivery count:
                # a reranker that only sees the top 3 can reorder them
                # but never rescue a doc the fusion stage ranked
                # 4th-Nth — recall is capped by the request width.
                pool = settings.RETRIEVAL_CANDIDATE_POOL
                search_req = VectorSearchRequest(
                    query=query, top_k=pool, filters=filters or None, acl_filters=acl_scope
                )
                # The L2 key folds the ACL in: entries are post-ACL doc
                # sets, so a scope-ineligible hit must never replay to a
                # caller. The identity merge keeps today's no-scope key
                # byte-identical to the pre-seam one.
                cache_key_filters = search_req.with_merged_filters().filters or {}
                # L2 retrieval cache: repeated identical (query, filters)
                # within this KB epoch replay without touching Qdrant —
                # the layer exists to protect it. Fail-open inside the
                # service, so an outage just means "search as usual".
                if self._retrieval_cache is not None:
                    cached_docs = await self._retrieval_cache.get(query, cache_key_filters)
                    if cached_docs is not None:
                        retrieved_docs = cached_docs
                        sources = _extract_sources(retrieved_docs)
                if not retrieved_docs:
                    if filters:
                        RETRIEVAL_FILTERED_SEARCHES.inc()
                    search_results = await _search_once(search_req)
                    if not search_results and filters:
                        RETRIEVAL_FILTER_FALLBACKS.inc()
                        # A metadata miss must not zero out recall: retry
                        # without the BUSINESS filters. The ACL channel is
                        # not a cancellable business condition — it
                        # survives the fallback (review #1).
                        search_results = await _search_once(
                            VectorSearchRequest(query=query, top_k=pool, acl_filters=acl_scope)
                        )
                    # Rerank before conversion and caching: the stored L2
                    # order IS the reranked order, so cache hits replay it
                    # without paying the rerank again.
                    search_results = await self._rerank_results(search_results, search_req)
                    # The pool exists to feed the reranker; delivery stays
                    # capped so a NoOp/absent reranker never dumps the
                    # whole pool into the generation context.
                    search_results = search_results[: settings.RETRIEVAL_TOP_K]
                    retrieved_docs = [_search_result_to_dict(r) for r in search_results]
                    sources = _extract_sources(retrieved_docs)
                    if retrieved_docs and self._retrieval_cache is not None:
                        await self._retrieval_cache.put(query, cache_key_filters, retrieved_docs)
                    if not retrieved_docs and leg_failures:
                        retrieval_degraded = True
            except Exception:
                retrieval_degraded = True
                logger.exception("Hybrid search failed")

        if hybrid_search is not None:
            retrieval_ran = True
        if retrieved_docs:
            record_funnel_layer(LAYER_RAG)
        emit_trace("cs.retrieval", mode="hybrid", docs=len(retrieved_docs), sources=sources[:3])
        return {
            "retrieved_docs": retrieved_docs,
            "sources": sources,
            "retrieval_ran": retrieval_ran,
            "retrieval_degraded": retrieval_degraded,
        }

    async def _rerank_results(self, results: list[Any], request: Any) -> list[Any]:
        """Rerank search results when the pipeline carries a reranker.

        The reranker is built and budget-wrapped in chat.py for exactly
        this call site; it stayed unconsumed for a long stretch (dead
        wiring — the documented Reranking stage silently never ran).
        Fail-open like every retrieval leg: a rerank failure degrades
        to the hybrid order, never a failed lookup.
        """
        reranker = self._retrieval_pipeline.get("reranker")
        if reranker is None or not results:
            return results
        try:
            reranked = await reranker.rerank(results, request)
            return list(reranked) if reranked else results
        except Exception:  # noqa: BLE001 - rerank is precision, not availability
            logger.exception("Reranking failed; keeping hybrid order")
            return results

    async def _turn_history(self, state: DialogueState) -> list[Any]:
        """Session history for this turn, fetched at most once.

        The retrieval-side condense and the generation prompt need the
        same snapshot; fetching twice would double the provider cost
        and could race a concurrent persist between the two reads.
        Raises whatever the provider raises — callers own the policy.
        """
        assert self._history_provider is not None
        turn_id = state.get("turn_id", "")
        if self._turn_history_memo is not None and self._turn_history_memo[0] == turn_id:
            return self._turn_history_memo[1]
        history = list(await self._history_provider(state.get("session_id", 0)))
        self._turn_history_memo = (turn_id, history)
        return history

    async def _condense_query(self, message: str, state: DialogueState) -> str | None:
        """Best-effort standalone-question rewrite for retrieval (review #6).

        Skipped when disabled, unwired, history-less, or on any failure —
        every path returns None and the raw message is searched instead.
        """
        if not settings.QUERY_REWRITE_ENABLED:
            return None
        if self._llm_service is None or self._history_provider is None:
            return None
        try:
            history = await self._turn_history(state)
        except Exception:  # noqa: BLE001 - history is best-effort context
            logger.warning("History fetch failed; retrieval searches the raw message")
            return None
        return await condense_for_retrieval(self._llm_service, message, history)

    async def _extract_query_entities(self, message: str) -> SlotFillingResult | None:
        """Run the slot filler over the raw message for retrieval enrichment.

        Best-effort: a disabled filler (None) or any extraction error
        degrades to None, and callers search the raw message unchanged.
        """
        if self._slot_filler is None:
            return None
        try:
            return await self._slot_filler.fill_slots(message, intent=None)
        except Exception:
            logger.warning("Slot extraction for retrieval enrichment failed", exc_info=True)
            return None

    def _apply_claim_gate(self, state: DialogueState, updates: dict[str, Any]) -> dict[str, Any]:
        """Main-path policy-claim verification (Phase A2).

        Wraps the shared gate for generate_response_node: fixed-format
        tool responses (state.tool_result) quote executed tool output
        verbatim — grounded by construction — so they skip entirely.
        See _gate_claims for the check itself and its limitations.
        """
        if state.get("blocked") or state.get("tool_result"):
            return updates
        return self._gate_claims(updates, message=state.get("message", ""), check_actions=True)

    def _gate_claims(
        self, updates: dict[str, Any], *, message: str, check_actions: bool
    ) -> dict[str, Any]:
        """Shared deterministic claim gate (Phase A2).

        The LLM proposes; this gate verifies — duration claims are
        checked against the message-anchored fact subgraph by range
        entailment with subject restriction; when ``check_actions`` is
        set, past-tense action assertions without an executed tool are
        softened to guidance. Violations are rewritten to grounded
        statements (downgrade before reject), so the user always leaves
        with the correct number. Skipped for empty subgraphs and when
        the setting is off.

        Known limitation: on the token-streamed path violating tokens
        may have already reached the consumer; the rewrite applies to
        the persisted/sync text (same contract as the output guardrail).
        """
        response = updates.get("response", "")
        if not response:
            return updates

        from app.config.settings import get_settings

        if not get_settings().FACT_CLAIM_CHECK_ENABLED:
            return updates

        from app.services.facts.claim_check import apply_violations, check_policy_claims
        from app.services.facts.fact_store import FactStore

        facts = FactStore.load_default().subgraph_for(message)
        if not facts:
            return updates

        CLAIM_CHECKS.inc()
        result = check_policy_claims(response, facts, check_actions=check_actions)
        for violation in result.violations:
            CLAIM_VIOLATIONS.labels(reason=violation.reason).inc()
            # Demo centerpiece: the exact triple the execution panel
            # renders — model's claim, why it fails, the grounded rewrite.
            emit_trace(
                "cs.claim_gate",
                action="rewrite",
                reason=violation.reason,
                clause=violation.clause,
                grounded=violation.grounded_statement,
            )
        if not result.violations:
            # A green check is only meaningful if the viewer knows the
            # gate actually ran, not that it was skipped.
            emit_trace("cs.claim_gate", action="pass", facts=len(facts))
        if result.violations:
            updates = {**updates, "response": apply_violations(response, result)}
        return updates

    @traced_stage("cs.generation")
    async def generate_response_node(
        self, state: DialogueState, config: Optional[RunnableConfig] = None
    ) -> dict[str, Any]:
        """Generate the final response, then run the output guardrail.

        The inner logic builds the response; this wrapper verifies
        policy claims against the curated fact table (Phase A2), applies
        the output-side safety check / PII redaction before the response
        leaves the graph, and emits the final text to the stream queue
        when it was not already streamed token-by-token.
        """
        updates = await self._generate_response_logic(state, config)
        # Shared-cache eligibility: when the generation folded in history
        # or user facts, flag it so the L0 write gate in ChatService can
        # refuse to store a dialogue-personal answer.
        if _used_history(config):
            updates["personalized"] = True
        updates = self._apply_claim_gate(state, updates)

        if self._guardrail_service is None or state.get("blocked"):
            _emit_response(updates.get("response", ""), config)
            return updates

        response = updates.get("response", "")
        if not response:
            return updates

        result = self._guardrail_service.check_output(response)
        if result.was_blocked:
            updates["response"] = GUARDRAIL_OUTPUT_BLOCKED[_turn_lang(state)]
        elif result.sanitized_content and result.sanitized_content != response:
            updates["response"] = result.sanitized_content
        _emit_response(updates.get("response", ""), config)
        return updates

    async def _generate_response_logic(
        self, state: DialogueState, config: Optional[RunnableConfig] = None
    ) -> dict[str, Any]:
        """Generate the final response based on the current state.

        Handles five cases:
        1. Blocked by guardrail — return the blocked response directly.
        2. Missing slots — return the slot prompt without calling the LLM.
        3. Tool result available — build a prompt with tool output.
        4. Retrieved docs available — build a RAG prompt.
        5. Meta intent — handle confirm / deny / cancel state actions.
        6. Fallback — direct LLM call.
        """
        # Case 1: blocked.
        if state.get("blocked"):
            return {"response": state.get("response", "抱歉，无法处理您的请求。")}

        intent = state.get("intent", "")
        message = state.get("message", "")

        # Case 2: meta intent (cancel/confirm/deny) — handle before slot prompt
        if intent in META_INTENTS:
            return await self._handle_meta_intent(state, config)

        # Case 3: slots still missing — prompt the user.
        slot_prompt = state.get("slot_prompt")
        if slot_prompt:
            return {"response": slot_prompt}

        # Case 4: tool result.
        tool_result = state.get("tool_result")
        if tool_result:
            return await self._generate_with_tool(intent, message, tool_result, state, config)

        # Case 4: RAG context.
        retrieved_docs = state.get("retrieved_docs")
        if retrieved_docs:
            return await self._generate_with_rag(intent, message, retrieved_docs, state, config)

        # Case 4b: retrieval ran and found nothing — deterministic
        # evidence gap. Free generation with no context is exactly how
        # a customer-service bot invents return policies; a failed leg
        # is a service outage and must not read as "no knowledge"
        # (review 2026-09-26, #6).
        if state.get("retrieval_ran") and not retrieved_docs:
            lang = detect_language(message)
            if state.get("retrieval_degraded"):
                return {"response": RETRIEVAL_DEGRADED[lang]}
            return {"response": NO_EVIDENCE[lang]}

        # Case 5: direct LLM call.
        return await self._generate_direct(message, config, state)

    @traced_stage("cs.direct")
    async def direct_response_node(
        self, state: DialogueState, config: Optional[RunnableConfig] = None
    ) -> dict[str, Any]:
        """Smalltalk tier: L1 semantic replay first, LLM call second.

        The L1 cache lives here — after intent detection, inside the
        only tier it serves — so the embedding lookup is paid on
        direct-tier turns rather than taxing every L0-miss turn. A hit
        re-runs the deterministic freshness insurance (claim gate +
        output guardrail) exactly like the L0 and FAQ fast paths: a
        cached serve is never trusted blindly.

        Lookup is intentionally NOT gated on pending task state: direct
        answers are persona + message only (anonymous writes only), so
        a replay cannot smuggle in another user's context. The write
        side is gated — see ``_maybe_put_semantic``.

        When a task is suspended, the answer carries a visible resume
        hint: the direct tier answers unrecognized input instead of
        silently resuming the suspended task (see
        ``_detect_intent_logic``). The hint is appended after the cache
        write/read so cached bodies stay state-free.
        """
        message = state.get("message", "")
        if self._semantic_cache is not None:
            cached = await self._semantic_cache.get(message)
            if cached is not None:
                updates: dict[str, Any] = {
                    "response": cached.response,
                    "sources": cached.sources,
                    "intent": cached.intent,
                }
                updates = self._gate_claims(updates, message=message, check_actions=True)
                response = updates["response"]
                if self._guardrail_service is not None:
                    check = self._guardrail_service.check_output(response)
                    if check.was_blocked:
                        response = GUARDRAIL_OUTPUT_BLOCKED[_turn_lang(state)]
                    elif check.sanitized_content and check.sanitized_content != response:
                        response = check.sanitized_content
                updates["response"] = response

                emit_trace("cs.semantic_cache", hit=True, intent=cached.intent)
                record_funnel_layer(LAYER_L1_SEMANTIC)
                response = _maybe_append_resume_hint(response, state)
                updates["response"] = response
                _emit_response(response, config)
                return updates

        record_funnel_layer(LAYER_DIRECT)
        response = await self._generate_direct(message, config, state)
        # Output guardrail (review 2026-09-26, #4): the direct tier's
        # fresh generations pass the same output check as every other
        # LLM branch — PII is redacted before the answer leaves the
        # node and before L1 stores a copy of the body.
        if self._guardrail_service is not None:
            raw = response.get("response", "")
            check = self._guardrail_service.check_output(raw)
            if check.was_blocked:
                response["response"] = GUARDRAIL_OUTPUT_BLOCKED[_turn_lang(state)]
            elif check.sanitized_content and check.sanitized_content != raw:
                response["response"] = check.sanitized_content
        # Cache the clean body; the resume hint is per-turn state.
        await self._maybe_put_semantic(
            state, message, response.get("response", ""), used_history=_used_history(config)
        )
        response["response"] = _maybe_append_resume_hint(response.get("response", ""), state)
        return response

    async def _maybe_put_semantic(
        self, state: DialogueState, message: str, response: str, *, used_history: bool = False
    ) -> None:
        """L1 write gate: only anonymous, stateless, context-free turns.

        Mirrors the L0 doctrine: personalized turns (identified users,
        whose generators may fold cross-session facts into the answer)
        are never written — a hit replays the stored text verbatim at
        any later visitor. Session history counts the same way: an
        anonymous session's follow-up answer still leaks that dialogue's
        context into a shared replay (review 2026-09-26, finding #5).
        The service's own intent allowlist and capacity cap apply on
        top; put() is fail-open.
        """
        if self._semantic_cache is None or not response:
            return
        if state.get("user_id") or state.get("pending_confirmation") or used_history:
            return
        await self._semantic_cache.put(
            message,
            CachedAnswer(response=response, sources=[], intent=state.get("intent", "")),
        )

    @traced_stage("cs.handoff")
    async def handle_handoff_node(
        self, state: DialogueState, config: Optional[RunnableConfig] = None
    ) -> dict[str, Any]:
        """Escalate the session to a human agent.

        Creates a handoff ticket carrying the dialogue context (intent,
        slots, current message) so the agent lands mid-conversation
        instead of starting from “您好，请问有什么可以帮您”. The response
        is a fixed template — the handoff path deliberately avoids LLM
        generation so a model outage can never block a user from
        reaching a human.

        Any staged irreversible action is discarded: the human agent
        owns that decision now, and a stale confirmation gate must not
        ambush a later turn.
        """
        reason = state.get("handoff_reason") or REASON_EXPLICIT
        # New turns clear live tool fields; retain the last tool-bearing
        # turn for the human without exposing it to generation/routing.
        tool_context = state.get("last_tool_execution")
        if state.get("tool_result") or state.get("executed_tools"):
            tool_context = {
                "turn_id": state.get("turn_id", ""),
                "tool_result": state.get("tool_result") or {},
                "executed_tools": state.get("executed_tools") or [],
            }
        context = {
            "trigger": reason,
            "user_message": state.get("message", ""),
            "intent": state.get("prev_intent") or state.get("intent", ""),
            "filled_slots": state.get("filled_slots") or {},
            "pending_slots": state.get("pending_slots") or [],
            # What the bot already tried, so the human agent does not
            # make the user repeat the story (industry-standard
            # context transfer on escalation).
            "bot_executed_tools": tool_context["executed_tools"] if tool_context else [],
            "last_tool_result": (tool_context["tool_result"] or None) if tool_context else None,
            "last_tool_turn_id": tool_context["turn_id"] if tool_context else None,
        }

        ticket: dict[str, Any] = {
            "ticket_id": None,
            "queue_position": None,
            "reused": False,
        }
        if self._handoff_service is not None:
            ticket = await self._handoff_service.create_ticket_for_session(
                session_id=state.get("session_id", 0),
                user_id=state.get("user_id"),
                reason=reason,
                context=context,
            )

        response = _build_handoff_response(reason, ticket, lang=_turn_lang(state))

        record_funnel_layer(LAYER_HANDOFF)
        emit_trace(
            "cs.handoff",
            reason=reason,
            ticket_id=ticket.get("ticket_id"),
            queue_position=ticket.get("queue_position"),
            reused=bool(ticket.get("reused")),
        )
        updates: dict[str, Any] = {
            "response": response,
            "intent": "handoff",
            "handoff_reason": reason,
            "pending_confirmation": None,
            "slot_prompt": "",
        }
        if ticket.get("ticket_id") is not None:
            updates["handoff_ticket_id"] = ticket["ticket_id"]

        _emit_response(response, config)
        return updates

    @traced_stage("cs.agent")
    async def handle_agent_node(
        self, state: DialogueState, config: Optional[RunnableConfig] = None
    ) -> dict[str, Any]:
        """Run the function-calling agent loop for a task intent.

        On success the agent's final answer (or confirmation question
        for a staged irreversible action) is the response and the turn
        ends here. On any failure — provider without function-calling
        support, LLM outage mid-loop — the node routes back into the
        deterministic slot pipeline so the user still gets served:
        agent mode is an upgrade path, never a dependency.

        The output guardrail runs on the LLM-generated answer (A6
        invariant), and a successful non-staging run clears any stale
        pending confirmation (same semantics as execute_tool_node).
        """
        # Defense in depth: a blocked turn must not run the agent loop
        # (same invariant as execute_tool_node). Ending the turn here
        # keeps the refusal the user already received.
        if state.get("blocked"):
            return {"route_after_agent": "agent_done"}

        if self._agent_service is None:
            AGENT_LOOP_OUTCOMES.labels(outcome=OUTCOME_FALLBACK).inc()
            return {"route_after_agent": "agent_fallback"}

        context_note = ""
        filled = state.get("filled_slots") or {}
        if filled:
            context_note = "对话中已知信息：" + "，".join(f"{k}={v}" for k, v in filled.items())
        facts = await self._recall_user_facts(state)
        if facts:
            block = "用户长期信息：" + "；".join(facts)
            context_note = f"{context_note}。{block}" if context_note else block

        try:
            history = None
            if self._history_provider is not None:
                try:
                    history = await self._turn_history(state)
                except Exception:  # noqa: BLE001 - history is best-effort
                    logger.warning("History fetch failed; agent runs without prior turns")
            result = await self._agent_service.run(
                user_message=state.get("message", ""),
                user_id=state.get("user_id"),
                context_note=context_note,
                history=history,
                # Server-truth conversation id: conversation-scoped agent
                # tools (human handoff) open tickets against the real
                # session, never a model-supplied one.
                session_id=state.get("session_id"),
            )
        except NotImplementedError:
            logger.warning(
                "Agent mode unavailable (provider lacks function calling); "
                "falling back to slot pipeline"
            )
            AGENT_LOOP_OUTCOMES.labels(outcome=OUTCOME_FALLBACK).inc()
            return {"route_after_agent": "agent_fallback"}
        except Exception:  # noqa: BLE001 - availability over agent mode
            logger.exception("Agent loop failed; falling back to slot pipeline")
            AGENT_LOOP_OUTCOMES.labels(outcome=OUTCOME_FALLBACK).inc()
            return {"route_after_agent": "agent_fallback"}

        # The agent loop is the funnel's deepest serving layer — its
        # traffic is the "real need" the funnel inversion optimizes for.
        record_funnel_layer(LAYER_AGENT_TOOL)
        updates: dict[str, Any] = {
            "route_after_agent": "agent_done",
            "response": result.response,
            # Structured audit trail of executed tools (empty list on
            # staged runs where nothing executed yet).
            "executed_tools": result.tool_trace,
            # None when this run staged nothing — clears stale gates.
            "pending_confirmation": result.pending_confirmation,
        }
        if result.tool_trace:
            # Tools actually ran — the task is terminal (task_executed).
            updates["task_executed"] = True
        emit_trace(
            "cs.agent",
            tools=[str(entry.get("tool", "")) for entry in result.tool_trace],
        )

        # Phase A2 claim gate: agent responses are LLM paraphrases of
        # tool output, not fixed formats — misstating tool numbers and
        # fabricating completions are documented agent hallucination
        # modes, so policy claims are verified here too. A completed-
        # action assertion is trusted only when tool_trace proves the
        # tool actually executed.
        updates = self._gate_claims(
            updates,
            message=state.get("message", ""),
            check_actions=not result.tool_trace,
        )

        response = updates.get("response", "")
        if self._guardrail_service is not None and response:
            check = self._guardrail_service.check_output(response)
            if check.was_blocked:
                updates["response"] = GUARDRAIL_OUTPUT_BLOCKED[_turn_lang(state)]
            elif check.sanitized_content and check.sanitized_content != response:
                updates["response"] = check.sanitized_content

        _emit_response(updates["response"], config)
        return updates

    @staticmethod
    def route_after_agent(state: DialogueState) -> str:
        """End the turn after a successful agent run; fall back to the
        slot pipeline otherwise."""
        return state.get("route_after_agent", "agent_fallback")

    @staticmethod
    def route_after_faq(state: DialogueState) -> str:
        """End the turn on a curated FAQ hit; continue into RAG on miss."""
        return state.get("route_after_faq", "miss")

    @staticmethod
    def route_after_guardrail(state: DialogueState) -> str:
        """End the turn immediately when input was blocked.

        A blocked turn must not reach the cache, intent detection, any
        tool, or LLM generation — the visible refusal has to equal the
        actual backend behavior (review 2026-09-26, finding #2).
        """
        return "blocked" if state.get("blocked") else "continue"

    @staticmethod
    def route_after_cache(state: DialogueState) -> str:
        """End the turn on an answer-cache hit; otherwise run the normal
        post-guardrail branching (the intent-skip logic)."""
        if state.get("route_after_cache") == "hit":
            return "hit"
        return NodeFactory.should_skip_intent(state)

    # ── Conditional edges ────────────────────────────────────────────────────

    @staticmethod
    def should_skip_intent(state: DialogueState) -> str:
        """Decide whether to skip intent detection.

        Returns "skip" when the user is clearly answering a slot prompt
        in an ongoing task flow — no need to re-run the full intent pipeline.
        Returns "full" otherwise.
        """
        intent = state.get("intent", "")
        pending = state.get("pending_slots") or []
        message = state.get("message", "")

        # Only skip when actively in a task with pending slots.
        if not pending or intent not in TASK_INTENTS:
            return "full"

        # Cancel / abort keywords always go through full detection.
        if any(
            kw in message
            for kw in (
                "取消",
                "算了",
                "不要了",
                "不了",
                "不想",
                "不想退",
                "cancel",
            )
        ):
            return "full"

        # Human-agent requests always go through full detection —
        # mid-slot-collection “转人工” must never be captured as a slot
        # value.
        if any(
            kw in message
            for kw in (
                "转人工",
                "人工客服",
                "人工服务",
                "找客服",
                "human agent",
                "human support",
            )
        ):
            return "full"

        # Greeting / chitchat go through full detection.
        if any(
            kw in message
            for kw in (
                "你好",
                "hello",
                "hi",
                "在吗",
                "谢谢",
                "再见",
            )
        ):
            return "full"

        # Question patterns — unlikely to be slot answers.
        if any(
            kw in message
            for kw in (
                "怎么",
                "什么",
                "为什么",
                "哪",
                "怎么样",
                "天气",
                "能不",
                "可以",
                "帮忙",
                "请问",
            )
        ):
            return "full"

        # Questions (contains ？ or ?) likely aren't slot answers.
        if "？" in message or "?" in message:
            return "full"

        # Task-switching keywords go through full detection.
        if any(
            kw in message
            for kw in (
                "退款",
                "退货",
                "订单",
                "物流",
                "投诉",
                "查询",
                "政策",
                "faq",
                "FAQ",
                "refund",
                "return",
                "order",
                "shipping",
                "complaint",
                "policy",
            )
        ):
            return "full"

        # Message is long (> 50 chars) — might be a complex query, detect fully.
        if len(message.strip()) > 50:
            return "full"

        return "skip"

    @staticmethod
    def check_slots(state: DialogueState) -> str:
        """Return 'complete' when all required slots are filled, else 'missing'."""
        pending = state.get("pending_slots") or []
        return "missing" if pending else "complete"

    @staticmethod
    def after_execute_tool(state: DialogueState) -> str:
        """Route after tool execution.

        Returns "confirm" when an irreversible action has been staged
        awaiting the user's explicit confirmation (the fixed-template
        question skips LLM generation so the wording cannot drift),
        otherwise "done" to generate the tool-based response.
        """
        return "confirm" if state.get("pending_confirmation") else "done"

    @staticmethod
    def route_by_intent(state: DialogueState) -> str:
        """Return the route determined by route_intent_node."""
        return state.get("route", "direct")

    # ── Private helpers ──────────────────────────────────────────────────────

    async def _generate_with_tool(
        self,
        intent: str,
        message: str,
        tool_result: dict[str, Any],
        state: DialogueState,
        config: Optional[RunnableConfig] = None,
    ) -> dict[str, Any]:
        """Generate response incorporating tool execution results."""
        display_name = INTENT_DISPLAY_NAMES.get(intent, intent)
        context = (
            f"用户意图: {display_name}\n"
            f"用户消息: {message}\n"
            f"工具执行结果: {_safe_json(tool_result)}\n"
            "请根据以上工具执行结果，用友好专业的语气回答用户。"
        )
        response_text = await self._call_llm(context, config, state)
        response_text = _maybe_append_resume_hint(response_text, state)
        return {"response": response_text}

    async def _generate_with_rag(
        self,
        intent: str,  # noqa: ARG002 - reserved for intent-conditioned prompts
        message: str,
        retrieved_docs: list[dict[str, Any]],
        state: DialogueState,
        config: Optional[RunnableConfig] = None,
    ) -> dict[str, Any]:
        """Generate response incorporating retrieved documents."""
        docs_text = "\n\n".join(
            f"[文档{i + 1}] {doc.get('content', '')}" for i, doc in enumerate(retrieved_docs)
        )
        context = (
            f"参考资料:\n{docs_text}\n\n"
            f"用户问题: {message}\n"
            "请根据以上参考资料回答用户的问题。如果资料中没有相关内容，请如实告知。"
        )
        response_text = await self._call_llm(context, config, state)
        response_text = _maybe_append_resume_hint(response_text, state)
        return {"response": response_text}

    async def _generate_direct(
        self,
        message: str,
        config: Optional[RunnableConfig] = None,
        state: DialogueState | None = None,
    ) -> dict[str, Any]:
        """Generate a direct response without additional context."""
        response_text = await self._call_llm(message, config, state)
        return {"response": response_text}

    async def _handle_meta_intent(
        self, state: DialogueState, config: Optional[RunnableConfig] = None
    ) -> dict[str, Any]:
        """Handle confirm / deny / cancel meta intents.

        ``confirm`` first checks for a staged irreversible action
        (``pending_confirmation``) and executes it with the caller's
        user_id; ``deny`` / ``cancel`` discard any staged action.
        """
        intent = state.get("intent", "")

        if intent == "cancel":
            state_stack: list[dict[str, Any]] = list(state.get("state_stack") or [])
            # Try to resume a suspended task after cancellation.
            if state_stack:
                restored = state_stack.pop()
                return {
                    "response": "已取消当前操作。返回到之前的任务。",
                    "state_stack": state_stack,
                    "intent": restored.get("intent", ""),
                    "filled_slots": restored.get("filled_slots", {}),
                    "pending_slots": restored.get("pending_slots", []),
                    "pending_confirmation": None,
                    # Restored tasks were suspended pre-execution; clear
                    # any stale flag from the cancelled turn (see the
                    # handle_switch resume note).
                    "task_executed": False,
                }
            return {
                "response": "已取消当前操作。",
                "filled_slots": {},
                "pending_slots": [],
                "pending_confirmation": None,
            }

        if intent == "confirm":
            pending = state.get("pending_confirmation")
            if pending and state.get("blocked"):
                # A blocked turn must not execute a staged irreversible
                # action; the staging stays intact for a clean retry.
                return {"response": GUARDRAIL_INPUT_BLOCKED[_turn_lang(state)]}
            if pending:
                # Review 2026-09-26 #9: a staged irreversible action must
                # not live forever — a refund staged yesterday must not
                # execute on a bare 确认 today. Discard stale actions and
                # ask the user to restage; TTL=0 disables (legacy).
                staged_at = float(pending.get("staged_at") or 0.0)
                if (
                    staged_at
                    and settings.CONFIRMATION_TTL_SECONDS > 0
                    and time.time() - staged_at > settings.CONFIRMATION_TTL_SECONDS
                ):
                    return {
                        "pending_confirmation": None,
                        "response": CONFIRMATION_EXPIRED[_turn_lang(state)],
                    }
                # Execute the staged irreversible action with the
                # caller's identity so ownership checks apply.
                pending_args = dict(pending.get("args") or {})
                result = await self._tool_registry.execute(
                    pending.get("intent", ""),
                    pending_args,
                    user_id=state.get("user_id"),
                )
                tool_result = result.data if result.success else {"error": result.message}
                # Same audit-trail shape the agent path records
                # (_execute_one in agent/service.py): the deterministic
                # confirm branch executes the most irreversible tools
                # in the product — an unrecorded refund execution is an
                # unauditable one (review 2026-09-26, #9). Summary cap
                # matches the agent path's _TRACE_SUMMARY_MAX.
                tool = self._tool_registry.get_tool_for_intent(pending.get("intent", ""))
                updates: dict[str, Any] = {
                    "pending_confirmation": None,
                    "intent": pending.get("intent", ""),
                    "filled_slots": pending_args,
                    "tool_result": tool_result,
                    # The staged action ran — the task is terminal now,
                    # success or failure (see task_executed in state.py).
                    "task_executed": True,
                    "executed_tools": [
                        {
                            "tool": tool.name if tool else pending.get("intent", ""),
                            "ok": result.success,
                            "args": pending_args,
                            "summary": _safe_json(tool_result)[:200],
                        }
                    ],
                }
                generated = await self._generate_with_tool(
                    pending.get("intent", ""),
                    state.get("message", ""),
                    tool_result,
                    state,
                    config,
                )
                updates.update(generated)
                return updates

            state_stack = list(state.get("state_stack") or [])
            if state_stack:
                restored = state_stack.pop()
                display = INTENT_DISPLAY_NAMES.get(restored.get("intent", ""), "")
                return {
                    "response": f"好的，继续为您处理{display}。"
                    if display
                    else "好的，继续为您处理。",
                    "state_stack": state_stack,
                    "intent": restored.get("intent", ""),
                    "filled_slots": restored.get("filled_slots", {}),
                    "pending_slots": restored.get("pending_slots", []),
                    # Restored tasks were suspended pre-execution; clear
                    # any stale flag from the previous turn (see the
                    # handle_switch resume note).
                    "task_executed": False,
                }
            return {"response": "好的，已确认。请稍等，我正在为您处理。"}

        if intent == "deny":
            if state.get("pending_confirmation"):
                return {
                    "pending_confirmation": None,
                    "response": "好的，已取消本次操作，未执行任何更改。请问还有什么可以帮您的？",
                }
            return {"response": "好的，已取消。请问还有什么可以帮您的？"}

        return {"response": "抱歉，我没有理解您的意思，请重新描述。"}

    async def _recall_user_facts(self, state: DialogueState | None) -> list[str]:
        """Newest durable facts for the current user, best-effort.

        Recall failures degrade to []: a memory outage must never fail
        the chat — the worst case is an unpersonalized answer.
        """
        if self._user_facts_provider is None or state is None:
            return []
        user_id = state.get("user_id")
        if not user_id:
            return []
        try:
            facts = [fact for fact in await self._user_facts_provider(int(user_id)) if fact]
        except Exception:  # noqa: BLE001 - availability over personalization
            logger.warning("User facts recall failed; continuing without them")
            return []
        emit_trace("cs.memory", facts=facts)
        return facts

    async def _call_llm(
        self,
        user_content: str,
        config: Optional[RunnableConfig] = None,
        state: DialogueState | None = None,
    ) -> str:
        """Call the LLM service with the recent turns plus this message.

        Follow-up questions ("那运费谁出？") dominate real CS traffic —
        without prior turns the model cannot resolve them. History comes
        from the provider (chronological, already bounded) and is purely
        best-effort: any provider failure degrades to the current
        message only.

        When the request carries a stream queue (token streaming), chunks
        are pushed to the queue as they arrive and the accumulated text is
        returned; the ``streamed_response`` flag tells terminal nodes not
        to re-emit the full response after the fact.
        """
        if self._llm_service is None:
            return user_content

        from app.services.llm.base import LLMMessage
        from app.services.llm.prompt_templates import PromptTemplates

        # Persona first: tone, grounding rules, and the handoff escape
        # hatch are policy, not per-turn context. Cross-session user
        # facts (Phase B2) extend the persona as bounded context.
        persona = self._system_prompt or PromptTemplates.get_cs_system_prompt()
        facts = await self._recall_user_facts(state)
        if facts:
            persona = f"{persona}\n\n已知用户信息（跨会话，仅供个性化参考）：{'；'.join(facts)}"
        messages: list[LLMMessage] = [LLMMessage(role="system", content=persona)]
        history_added = False
        if state is not None and self._history_provider is not None:
            try:
                history = await self._turn_history(state)
                messages.extend(history)
                history_added = bool(history)
            except Exception:  # noqa: BLE001 - history is best-effort
                logger.warning("History fetch failed; generating without prior turns")
        if (facts or history_added) and config is not None:
            # Shared-cache eligibility flag: this generation is personal
            # to the dialogue, so its output must not be stored in any
            # cross-session cache (see _used_history consumers).
            configurable = config.get("configurable")
            if isinstance(configurable, dict):
                configurable["generation_used_history"] = True
        # Reply-language pin, final position (see GENERATION_LANG_DIRECTIVE):
        # keyed off the RAW user turn — the composed prompt for tool/RAG
        # turns is zh-scaffolded (用户意图/工具执行结果) and would detect
        # zh for an English user. Suffixing the user message keeps the
        # directive the last thing the model reads.
        lang = _turn_lang(state, fallback_text=user_content)
        messages.append(
            LLMMessage(
                role="user",
                content=user_content + "\n\n" + GENERATION_LANG_DIRECTIVE[lang],
            )
        )
        queue = _stream_queue(config)

        if queue is not None:
            chunks: list[str] = []
            emitted: list[str] = []
            # Reasoning filter first: GLM inlines <think>…</think> in the
            # content stream; it must never reach the consumer or the gate
            # (deliberation text quoting policy numbers would false-trip it).
            from app.services.llm.reasoning_filter import ReasoningFilter

            rf = ReasoningFilter()
            # Sentence-buffered claim gating (A2 streaming close-out):
            # armed exactly when the turn is checkable, so violating
            # policy numbers can never reach the consumer ungated while
            # chitchat keeps raw per-token claim behavior. On top of it,
            # sentence-buffered PII redaction (review 2026-09-26, #4)
            # runs on every guarded turn — including chitchat — because
            # PII must never reach the consumer ahead of the finalize
            # pass. Everything put on the queue is already gated and
            # redacted, and the returned text is the emitted text —
            # stream and persistence cannot diverge.
            gate = None
            if state is not None and not state.get("blocked") and not state.get("tool_result"):
                from app.services.facts.stream_gate import make_stream_gate

                gate = make_stream_gate(state.get("message", ""))
            pii_redactor = (
                PIIStreamRedactor(self._guardrail_service)
                if self._guardrail_service is not None
                else None
            )

            def _release(text: str) -> None:
                # Single emission choke point: claim-gated text flows
                # through the PII redactor before it reaches the queue.
                if not text:
                    return
                if pii_redactor is not None:
                    text = pii_redactor.feed(text)
                if text:
                    emitted.append(text)
                    queue.put_nowait(text)

            try:
                async for chunk in self._llm_service.generate_stream(messages):
                    if not chunk:
                        continue
                    clean = rf.feed(chunk)
                    if not clean:
                        continue
                    chunks.append(clean)
                    _release(gate.feed(clean) if gate is not None else clean)
                rf_tail = rf.flush()
                if rf_tail:
                    _release(gate.feed(rf_tail) if gate is not None else rf_tail)
                if gate is not None:
                    _release(gate.flush())
                if pii_redactor is not None:
                    _release(pii_redactor.flush())
            except Exception:
                logger.exception("LLM streaming generation failed")
                fallback = GENERATION_FAILED[_turn_lang(state, user_content)]
                # Text may still be held mid-sentence in the gates; flush
                # both through so ungated, un-redacted remains never reach
                # the queue.
                held = gate.flush() if gate is not None else ""
                if pii_redactor is not None and held:
                    held = pii_redactor.feed(held)
                if emitted or held:
                    if held:
                        emitted.append(held)
                        queue.put_nowait(held)
                    if pii_redactor is not None:
                        pii_tail = pii_redactor.flush()
                        if pii_tail:
                            emitted.append(pii_tail)
                            queue.put_nowait(pii_tail)
                    # Partial tokens already reached the consumer; append a
                    # visible apology rather than silently truncating. The
                    # consumer has now seen the entire response (tokens +
                    # fallback), so flag it streamed to prevent the
                    # terminal-node full-text push from duplicating it.
                    queue.put_nowait(fallback)
                    configurable = config.get("configurable") if config else None
                    if isinstance(configurable, dict):
                        configurable["streamed_response"] = True
                    return "".join(emitted) + fallback
                return fallback
            configurable = config.get("configurable") if config else None
            if isinstance(configurable, dict):
                configurable["streamed_response"] = True
            return "".join(emitted)

        try:
            response = await self._llm_service.generate(messages)
            from app.services.llm.reasoning_filter import strip_reasoning

            return strip_reasoning(response.content)
        except Exception:
            logger.exception("LLM generation failed")
            return GENERATION_FAILED[_turn_lang(state, user_content)]


# ── Module-level helpers ─────────────────────────────────────────────────────


def _build_confirmation_summary(
    tool: ToolDefinition, filled_slots: dict[str, Any], lang: str = LANG_ZH
) -> str:
    """Fixed-template confirmation question for a staged irreversible action.

    Deliberately not LLM-generated: the wording of a gate that protects a
    money-moving action must be deterministic. The template follows the
    user's language (English fallback) — a confirmation the user cannot
    read is not a confirmation.
    """
    display = CONFIRMATION_ACTION_NAMES.get(tool.intent, {}).get(
        lang, INTENT_DISPLAY_NAMES.get(tool.intent, tool.description)
    )
    parts = [f"{slot}={value}" for slot, value in filled_slots.items()]
    detail = (
        CONFIRMATION_DETAIL_JOIN[lang].join(parts) if parts else CONFIRMATION_DETAIL_EMPTY[lang]
    )
    return CONFIRMATION_ASK[lang].format(action=display, detail=detail)


def _build_handoff_response(reason: str, ticket: dict[str, Any], lang: str = LANG_ZH) -> str:
    """Fixed-template handoff acknowledgement.

    Like the confirmation gate, the handoff path never touches the LLM:
    reaching a human must not depend on model availability. The
    acknowledgement follows the user's language (English fallback).
    """
    prefix = ""
    if reason == REASON_EMOTION:
        prefix = HANDOFF_EMOTION_PREFIX[lang]

    ticket_id = ticket.get("ticket_id")
    if ticket_id is None:
        return f"{prefix}{HANDOFF_NO_TICKET[lang]}"

    parts = [f"{prefix}{HANDOFF_ACK[lang].format(ticket_id=ticket_id)}"]
    queue_position = ticket.get("queue_position")
    if queue_position:
        parts.append(HANDOFF_QUEUE[lang].format(queue_position=queue_position))
    if ticket.get("reused"):
        parts.append(HANDOFF_ALREADY_QUEUED[lang])
    parts.append(HANDOFF_CONTEXT_NOTE[lang])
    if lang == LANG_ZH:
        return "，".join(parts) + "。"
    return ", ".join(parts) + "."


def _enriched_query(message: str, fill_result: SlotFillingResult | None) -> str:
    """Append normalized slot values not already present in the message.

    Slot values that already appear verbatim add nothing (the vector
    and BM25 scorers see them); appending them would only bloat the
    query. Capped at four extras to bound query length.
    """
    if fill_result is None:
        return message
    extras: list[str] = []
    for slot in fill_result.slots:
        value = slot.normalized_value.strip()
        if value and value not in message and value not in extras:
            extras.append(value)
    if not extras:
        return message
    return f"{message} {' '.join(extras[:4])}"


def _search_result_to_dict(result: SearchResult) -> dict[str, Any]:
    """Convert a SearchResult dataclass to a plain dict."""
    return {
        "document_id": getattr(result, "document_id", ""),
        "content": getattr(result, "content", ""),
        "score": getattr(result, "score", 0.0),
        "metadata": getattr(result, "metadata", None),
    }


def _graph_doc_to_dict(result: Any) -> dict[str, Any]:
    """Convert a graph retrieval result to a plain dict."""
    if isinstance(result, dict):
        return result
    return {
        "content": getattr(result, "content", str(result)),
        "metadata": getattr(result, "metadata", None),
    }


def _extract_sources(docs: list[dict[str, Any]]) -> list[str]:
    """Extract unique source identifiers from retrieved documents."""
    sources: list[str] = []
    for doc in docs:
        doc_id = doc.get("document_id")
        if doc_id and doc_id not in sources:
            sources.append(doc_id)
    return sources


def _safe_json(obj: Any) -> str:
    """Safely convert an object to a JSON-like string for prompts."""
    import json

    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(obj)


def _maybe_append_resume_hint(response: str, state: DialogueState) -> str:
    """Append a hint about a suspended task if one exists on the stack."""
    state_stack = state.get("state_stack")
    if state_stack:
        suspended = state_stack[-1]
        suspended_intent = suspended.get("intent", "")
        display = INTENT_DISPLAY_NAMES.get(suspended_intent, suspended_intent)
        if display:
            response += f"\n\n（提示：您还有一个进行中的{display}任务，随时可以继续。）"
    return response
