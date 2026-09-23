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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.services.agent.service import AgentService
    from app.services.dialogue.tools import ToolDefinition, ToolRegistry
    from app.services.faq.store import FAQService
    from app.services.graph.retrieval.graph_retrieval_service import (
        GraphRetrievalService,
    )
    from app.services.guardrails.base import GuardrailService
    from app.services.handoff.service import HandoffService
    from app.services.intent.base import IntentDetector
    from app.services.llm.base import LLMServiceBase
    from app.services.retrieval.vector_base import SearchResult
    from app.services.slot_filling.base import SlotFiller, SlotFillingResult

from langchain_core.runnables import RunnableConfig

from app.models.enums.intent import (
    DIRECT_INTENTS,
    GRAPH_INTENTS,
    HANDOFF_INTENTS,
    INTENT_DISPLAY_NAMES,
    META_INTENTS,
    RAG_INTENTS,
    TASK_INTENTS,
)
from app.services.dialogue.emotion import assess_emotion
from app.services.dialogue.state import DialogueState
from app.services.handoff.service import (
    REASON_EMOTION,
    REASON_EXPLICIT,
)
from app.services.slot_filling.slot_types import (
    extract_slots_from_message,
    get_missing_slots,
    get_next_prompt,
)

logger = logging.getLogger(__name__)


def _stream_queue(config: RunnableConfig | None) -> asyncio.Queue[Any] | None:
    """Extract the per-request stream queue from a LangGraph invoke config."""
    if config is None:
        return None
    queue = (config.get("configurable") or {}).get("stream_queue")
    return queue if isinstance(queue, asyncio.Queue) else None


def _emit_response(text: str, config: RunnableConfig | None) -> None:
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
    ) -> None:
        self._intent_detector = intent_detector
        self._slot_filler = slot_filler
        self._tool_registry = tool_registry
        self._retrieval_pipeline = retrieval_pipeline or {}
        self._llm_service = llm_service
        self._guardrail_service = guardrail_service
        self._graph_retrieval_service = graph_retrieval_service
        self._handoff_service = handoff_service
        self._agent_service = agent_service
        self._faq_service = faq_service

    # ── Nodes ────────────────────────────────────────────────────────────────

    async def guardrail_node(self, state: DialogueState) -> dict[str, Any]:
        """Check input message against guardrail rules.

        Passes through if no guardrail service is configured, the message
        is already blocked, or the check passes.  Otherwise blocks the
        conversation and returns a safe response.
        """
        if state.get("blocked"):
            return {}

        if self._guardrail_service is None:
            return {}

        message = state.get("message", "")
        result = self._guardrail_service.check_input(message)

        if result.was_blocked:
            return {
                "blocked": True,
                "blocked_reason": ", ".join(result.violations)
                if result.violations
                else "内容安全检查未通过",
                "response": "抱歉，您的消息未通过安全检查，请重新描述您的问题。",
            }

        # Propagate sanitized content (PII redaction) into the dialogue state
        # so redacted text — not the raw input — flows to slots/LLM/retrieval.
        if result.sanitized_content and result.sanitized_content != message:
            return {"message": result.sanitized_content}

        return {}

    async def detect_intent_node(self, state: DialogueState) -> dict[str, Any]:
        """Detect user intent from the current message.

        Priority logic:
        1. Cancel always overrides — user wants to abort current task
        2. If prev intent is a task and the message provides slot values
           for that task, keep the task intent (user is answering a prompt)
        3. Confirm/deny after a task preserves the task intent
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

        # Resume: if confirm/unknown and a task is suspended on the stack,
        # set intent to the suspended task so slot collection continues.
        if detected_intent in META_INTENTS or detected_intent == "unknown":
            state_stack = state.get("state_stack") or []
            if state_stack:
                suspended_intent = state_stack[-1].get("intent", "")
                if suspended_intent in TASK_INTENTS:
                    return {
                        "intent": suspended_intent,
                        "prev_intent": prev_intent,
                        "confidence": confidence,
                    }

        # If prev is a task intent, check if message provides slot values
        if prev_intent and prev_intent in TASK_INTENTS:
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

        # Direct/social intents do not push state.
        if prev_intent in DIRECT_INTENTS or prev_intent == "unknown":
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
                    }

        # Suspend the current task.
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

        pending = get_missing_slots(intent, merged)
        next_prompt = get_next_prompt(intent, merged)

        return {
            "filled_slots": merged,
            "pending_slots": pending,
            "slot_prompt": next_prompt or "",
        }

    async def execute_tool_node(
        self, state: DialogueState, config: RunnableConfig | None = None
    ) -> dict[str, Any]:
        """Execute the tool associated with the current intent.

        Irreversible tools (refund, return) are staged instead of run:
        the node stores the prepared action in ``pending_confirmation``
        and returns a fixed-template confirmation question. The action
        only executes when the user's next turn resolves to the
        ``confirm`` meta intent (see ``_handle_meta_intent``).
        """
        intent = state.get("intent", "")
        filled_slots = state.get("filled_slots") or {}
        user_id = state.get("user_id")

        tool = self._tool_registry.get_tool_for_intent(intent)

        if tool is not None and tool.requires_confirmation:
            # Gate every prepared irreversible action, overwriting any
            # stale pending confirmation from an earlier turn.
            summary = _build_confirmation_summary(tool, filled_slots)
            _emit_response(summary, config)
            return {
                "pending_confirmation": {"intent": intent, "args": dict(filled_slots)},
                "response": summary,
            }

        # Reversible tools run immediately; clear any stale pending
        # confirmation so it cannot gate a later action.
        result = await self._tool_registry.execute(intent, filled_slots, user_id=user_id)

        if result.success:
            return {"tool_result": result.data, "pending_confirmation": None}

        logger.warning("Tool execution failed for intent %s: %s", intent, result.message)
        return {"tool_result": {"error": result.message}, "pending_confirmation": None}

    async def faq_lookup_node(
        self, state: DialogueState, config: RunnableConfig | None = None
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
                response = "抱歉，该回复未能通过安全检查，请重新提问。"
            elif check.sanitized_content and check.sanitized_content != response:
                response = check.sanitized_content

        _emit_response(response, config)
        return {
            "route_after_faq": "hit",
            "response": response,
            "sources": [f"faq:{entry.faq_id}"],
        }

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

        fill_result = await self._extract_query_entities(message)
        entity_hints = fill_result.to_entity_hints() if fill_result else []
        query = _enriched_query(message, fill_result)

        # Graph intents go through graph retrieval.
        if intent in GRAPH_INTENTS and self._graph_retrieval_service is not None:
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
                logger.exception("Graph retrieval failed for intent %s", intent)
            return {"retrieved_docs": retrieved_docs, "sources": sources}

        # Standard hybrid search for RAG intents.
        hybrid_search = self._retrieval_pipeline.get("hybrid_search")
        if hybrid_search is not None:
            try:
                from app.services.retrieval.vector_base import VectorSearchRequest

                search_req = VectorSearchRequest(query=query, top_k=3)
                search_results = await hybrid_search.search(search_req)
                retrieved_docs = [_search_result_to_dict(r) for r in search_results]
                sources = _extract_sources(retrieved_docs)
            except Exception:
                logger.exception("Hybrid search failed")

        return {"retrieved_docs": retrieved_docs, "sources": sources}

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

    async def generate_response_node(
        self, state: DialogueState, config: RunnableConfig | None = None
    ) -> dict[str, Any]:
        """Generate the final response, then run the output guardrail.

        The inner logic builds the response; this wrapper applies the
        output-side safety check / PII redaction before the response
        leaves the graph, and emits the final text to the stream queue
        when it was not already streamed token-by-token.
        """
        updates = await self._generate_response_logic(state, config)

        if self._guardrail_service is None or state.get("blocked"):
            _emit_response(updates.get("response", ""), config)
            return updates

        response = updates.get("response", "")
        if not response:
            return updates

        result = self._guardrail_service.check_output(response)
        if result.was_blocked:
            updates["response"] = "抱歉，该回复未能通过安全检查，请重新提问。"
        elif result.sanitized_content and result.sanitized_content != response:
            updates["response"] = result.sanitized_content
        _emit_response(updates.get("response", ""), config)
        return updates

    async def _generate_response_logic(
        self, state: DialogueState, config: RunnableConfig | None = None
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

        # Case 5: direct LLM call.
        return await self._generate_direct(message, config)

    async def direct_response_node(
        self, state: DialogueState, config: RunnableConfig | None = None
    ) -> dict[str, Any]:
        """Simple LLM call for chitchat / greeting."""
        message = state.get("message", "")
        response = await self._generate_direct(message, config)
        return response

    async def handle_handoff_node(
        self, state: DialogueState, config: RunnableConfig | None = None
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
        context = {
            "trigger": reason,
            "user_message": state.get("message", ""),
            "intent": state.get("prev_intent") or state.get("intent", ""),
            "filled_slots": state.get("filled_slots") or {},
            "pending_slots": state.get("pending_slots") or [],
            # What the bot already tried, so the human agent does not
            # make the user repeat the story (industry-standard
            # context transfer on escalation).
            "bot_executed_tools": state.get("executed_tools") or [],
            "last_tool_result": state.get("tool_result") or None,
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

        response = _build_handoff_response(reason, ticket)

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

    async def handle_agent_node(
        self, state: DialogueState, config: RunnableConfig | None = None
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
        if self._agent_service is None:
            return {"route_after_agent": "agent_fallback"}

        context_note = ""
        filled = state.get("filled_slots") or {}
        if filled:
            context_note = "对话中已知信息：" + "，".join(f"{k}={v}" for k, v in filled.items())

        try:
            result = await self._agent_service.run(
                user_message=state.get("message", ""),
                user_id=state.get("user_id"),
                context_note=context_note,
            )
        except NotImplementedError:
            logger.warning(
                "Agent mode unavailable (provider lacks function calling); "
                "falling back to slot pipeline"
            )
            return {"route_after_agent": "agent_fallback"}
        except Exception:  # noqa: BLE001 - availability over agent mode
            logger.exception("Agent loop failed; falling back to slot pipeline")
            return {"route_after_agent": "agent_fallback"}

        updates: dict[str, Any] = {
            "route_after_agent": "agent_done",
            "response": result.response,
            # Structured audit trail of executed tools (empty list on
            # staged runs where nothing executed yet).
            "executed_tools": result.tool_trace,
            # None when this run staged nothing — clears stale gates.
            "pending_confirmation": result.pending_confirmation,
        }

        if self._guardrail_service is not None and result.response:
            check = self._guardrail_service.check_output(result.response)
            if check.was_blocked:
                updates["response"] = "抱歉，该回复未能通过安全检查，请重新提问。"
            elif check.sanitized_content and check.sanitized_content != result.response:
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
        config: RunnableConfig | None = None,
    ) -> dict[str, Any]:
        """Generate response incorporating tool execution results."""
        display_name = INTENT_DISPLAY_NAMES.get(intent, intent)
        context = (
            f"用户意图: {display_name}\n"
            f"用户消息: {message}\n"
            f"工具执行结果: {_safe_json(tool_result)}\n"
            "请根据以上工具执行结果，用友好专业的语气回答用户。"
        )
        response_text = await self._call_llm(context, config)
        response_text = _maybe_append_resume_hint(response_text, state)
        return {"response": response_text}

    async def _generate_with_rag(
        self,
        intent: str,  # noqa: ARG002 - reserved for intent-conditioned prompts
        message: str,
        retrieved_docs: list[dict[str, Any]],
        state: DialogueState,
        config: RunnableConfig | None = None,
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
        response_text = await self._call_llm(context, config)
        response_text = _maybe_append_resume_hint(response_text, state)
        return {"response": response_text}

    async def _generate_direct(
        self, message: str, config: RunnableConfig | None = None
    ) -> dict[str, Any]:
        """Generate a direct response without additional context."""
        response_text = await self._call_llm(message, config)
        return {"response": response_text}

    async def _handle_meta_intent(
        self, state: DialogueState, config: RunnableConfig | None = None
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
                }
            return {
                "response": "已取消当前操作。",
                "filled_slots": {},
                "pending_slots": [],
                "pending_confirmation": None,
            }

        if intent == "confirm":
            pending = state.get("pending_confirmation")
            if pending:
                # Execute the staged irreversible action with the
                # caller's identity so ownership checks apply.
                result = await self._tool_registry.execute(
                    pending.get("intent", ""),
                    pending.get("args") or {},
                    user_id=state.get("user_id"),
                )
                tool_result = result.data if result.success else {"error": result.message}
                updates: dict[str, Any] = {
                    "pending_confirmation": None,
                    "intent": pending.get("intent", ""),
                    "filled_slots": dict(pending.get("args") or {}),
                    "tool_result": tool_result,
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

    async def _call_llm(self, user_content: str, config: RunnableConfig | None = None) -> str:
        """Call the LLM service with a single user message and return text.

        When the request carries a stream queue (token streaming), chunks
        are pushed to the queue as they arrive and the accumulated text is
        returned; the ``streamed_response`` flag tells terminal nodes not
        to re-emit the full response after the fact.
        """
        if self._llm_service is None:
            return user_content

        from app.services.llm.base import LLMMessage

        messages = [LLMMessage(role="user", content=user_content)]
        queue = _stream_queue(config)

        if queue is not None:
            chunks: list[str] = []
            try:
                async for chunk in self._llm_service.generate_stream(messages):
                    if not chunk:
                        continue
                    chunks.append(chunk)
                    queue.put_nowait(chunk)
            except Exception:
                logger.exception("LLM streaming generation failed")
                fallback = "抱歉，生成回复时出现错误，请稍后重试。"
                if chunks:
                    # Partial tokens already reached the consumer; append a
                    # visible apology rather than silently truncating. The
                    # consumer has now seen the entire response (tokens +
                    # fallback), so flag it streamed to prevent the
                    # terminal-node full-text push from duplicating it.
                    queue.put_nowait(fallback)
                    configurable = config.get("configurable") if config else None
                    if isinstance(configurable, dict):
                        configurable["streamed_response"] = True
                    return "".join(chunks) + fallback
                return fallback
            configurable = config.get("configurable") if config else None
            if isinstance(configurable, dict):
                configurable["streamed_response"] = True
            return "".join(chunks)

        try:
            response = await self._llm_service.generate(messages)
            return response.content
        except Exception:
            logger.exception("LLM generation failed")
            return "抱歉，生成回复时出现错误，请稍后重试。"


# ── Module-level helpers ─────────────────────────────────────────────────────


def _build_confirmation_summary(tool: ToolDefinition, filled_slots: dict[str, Any]) -> str:
    """Fixed-template confirmation question for a staged irreversible action.

    Deliberately not LLM-generated: the wording of a gate that protects a
    money-moving action must be deterministic.
    """
    display = INTENT_DISPLAY_NAMES.get(tool.intent, tool.description)
    parts = [f"{slot}={value}" for slot, value in filled_slots.items()]
    detail = "，".join(parts) if parts else "（无附加信息）"
    return (
        f"⚠️ 即将为您执行「{display}」：{detail}。\n"
        "该操作不可自动撤销。请回复「确认」执行，或回复「取消」放弃。"
    )


def _build_handoff_response(reason: str, ticket: dict[str, Any]) -> str:
    """Fixed-template handoff acknowledgement.

    Like the confirmation gate, the handoff path never touches the LLM:
    reaching a human must not depend on model availability.
    """
    prefix = ""
    if reason == REASON_EMOTION:
        prefix = "非常抱歉给您带来了不好的体验，"

    ticket_id = ticket.get("ticket_id")
    if ticket_id is None:
        return f"{prefix}正在为您转接人工客服，请稍候。"

    parts = [f"{prefix}已为您转接人工客服（工单号 #{ticket_id}）"]
    queue_position = ticket.get("queue_position")
    if queue_position:
        parts.append(f"当前排队人数：{queue_position} 人")
    if ticket.get("reused"):
        parts.append("您已在排队中，请耐心等待")
    parts.append("人工客服可查看本次会话的完整上下文，请稍候")
    return "，".join(parts) + "。"


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
