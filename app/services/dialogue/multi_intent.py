"""Bounded FIFO requests sharing the existing LIFO interruption stack.

Queue entries are distinguished by provenance and request identity, never by
intent kind. Only the serial completion boundary may remove a queued request.
"""

from typing import Any
from uuid import uuid4

from app.models.enums.intent import GRAPH_INTENTS, RAG_INTENTS, TASK_INTENTS
from app.services.chat.citations import normalize_citations
from app.services.dialogue.state import DialogueState

MULTI_INTENT_QUEUE_CAP = 2
QUEUEABLE_INTENTS = TASK_INTENTS | RAG_INTENTS | GRAPH_INTENTS | {"general_knowledge", "chitchat"}


def pack_detection(detection: Any) -> dict[str, Any]:
    """Keep checkpoint values primitive, including compatibility detector fakes."""
    return {
        "intent": detection.primary.intent.value,
        "confidence": detection.primary.confidence,
        "primary_query": getattr(detection, "primary_query", "") or "",
        "co_detected": [
            {
                "intent": item.intent.value,
                "confidence": item.confidence,
                "query": getattr(item, "query", "") or "",
            }
            for item in detection.co_detected
        ],
    }


def discard_queue(state: DialogueState) -> dict[str, Any]:
    """Remove queued requests without changing interrupted-task lifecycle."""
    stack = state.get("state_stack") or []
    kept = [item for item in stack if item.get("queued_from") != "multi_intent"]
    return {"state_stack": kept} if len(kept) != len(stack) else {}


def terminate_queue(state: DialogueState) -> dict[str, Any]:
    """Invalidate a failed/expired chain instead of resuming it on a later answer."""
    if not state.get("multi_intent_turn"):
        return {}
    return {
        **discard_queue(state),
        "active_request_id": "",
        "active_request_query": "",
        "multi_intent_stop": True,
        "filled_slots": {},
        "pending_slots": [],
    }


def pop_queued(stack: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """FIFO among queue entries, independent of LIFO suspended tasks."""
    for index, item in enumerate(stack):
        if item.get("queued_from") == "multi_intent":
            return item, [*stack[:index], *stack[index + 1 :]]
    return None, list(stack)


def enqueue_requests(state: DialogueState, detection: Any) -> dict[str, Any]:
    """Append at most two requests in detector order; report every overflow.

    Missing secondary subqueries are not executable: replaying the entire
    compound message risks extracting another order's slots. Report them as
    unqueued rather than inventing a split or silently dropping the request.
    """
    packed = detection if isinstance(detection, dict) else pack_detection(detection)
    candidates = [item for item in packed["co_detected"] if item["intent"] in QUEUEABLE_INTENTS]
    primary_query = packed["primary_query"]
    # Filtering negated/quoted/superseded clauses can leave one accepted
    # request. Its source span must still own slot extraction and output
    # restoration even when there is no secondary work to queue.
    isolated_primary = bool(primary_query and primary_query != state.get("message", ""))
    if not candidates and not isolated_primary:
        return {}
    stack = list(state.get("state_stack") or [])
    count = sum(item.get("queued_from") == "multi_intent" for item in stack)
    overflow = list(state.get("multi_intent_overflow") or [])
    for candidate in candidates:
        query = candidate["query"]
        if count >= MULTI_INTENT_QUEUE_CAP or not query:
            overflow.append(query or candidate["intent"])
            continue
        stack.append(
            {
                "intent": candidate["intent"],
                "confidence": candidate["confidence"],
                "queued_from": "multi_intent",
                "request_id": uuid4().hex,
                "query": query,
                "filled_slots": {},
                "pending_slots": [],
            }
        )
        count += 1
    return {
        "state_stack": stack,
        "multi_intent_turn": True,
        "multi_intent_original_message": state.get("message", ""),
        "multi_intent_overflow": overflow,
        "shared_cache_blocked": True,
        "active_request_id": state.get("active_request_id") or uuid4().hex,
        "active_request_query": packed["primary_query"] or state.get("message", ""),
        "message": packed["primary_query"] or state.get("message", ""),
    }


def aggregate_outputs(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    """Join already gated answers without asking a model to rewrite outcomes."""
    return {
        "response": "\n\n".join(output["response"] for output in outputs if output["response"]),
        "action_execution": next(
            (
                output["action_execution"]
                for output in reversed(outputs)
                if output["action_execution"]
            ),
            {},
        ),
        "sources": list(
            dict.fromkeys(source for output in outputs for source in output["sources"])
        ),
        "citations": normalize_citations(
            [citation for output in outputs for citation in output["citations"]]
        ),
        "executed_tools": [entry for output in outputs for entry in output["executed_tools"]],
        "retrieved_docs": [doc for output in outputs for doc in output["retrieved_docs"]],
        "retrieval_ran": any(output["retrieval_ran"] for output in outputs),
        "retrieval_degraded": any(output["retrieval_degraded"] for output in outputs),
    }
