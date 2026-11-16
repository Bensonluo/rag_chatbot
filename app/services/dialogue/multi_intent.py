"""Bounded FIFO requests sharing the existing LIFO interruption stack.

Queue entries are distinguished by provenance and request identity, never by
intent kind. Only the serial completion boundary may remove a queued request.
"""

import re
from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from app.models.enums.intent import GRAPH_INTENTS, RAG_INTENTS, TASK_INTENTS
from app.services.chat.citations import normalize_citations
from app.services.dialogue.state import DialogueState

MULTI_INTENT_QUEUE_CAP = 2
QUEUEABLE_INTENTS = TASK_INTENTS | RAG_INTENTS | GRAPH_INTENTS | {"general_knowledge", "chitchat"}
WRITE_INTENTS = {"refund", "return", "cancel"}
# Degraded (rule-fallback) write admission floor: only an explicit
# imperative pattern may stage a write when the model leg was unavailable
# or invalid. History/consultation phrasings must re-ask instead.
_STRICT_IMPERATIVE = re.compile(
    r"^(?:please\s+)?(?:i\s+(?:want|would\s+like|need)\s+(?:to\s+)?|help\s+me\s+|"
    r"can\s+you\s+|could\s+you\s+|i(')?d\s+like\s+to\s+)?"
    r"(?:refund|return|cancel|exchange)\b"
    r"|^(?:请|我要|帮我|麻烦|想)(?:退款|退货|换货|取消)",
    re.IGNORECASE,
)


def admit_request(intent: str, speech_act: str, span: str, *, degraded: bool = False) -> str:
    """Admission verdict for one request frame: task | clarify_* | knowledge.

    Reads and knowledge intents stay permissive (idempotent, low cost).
    A write task additionally requires current-action evidence: a plain
    action verdict (strict imperative when detection is degraded), never a
    history mention, an unverified condition or an option comparison.
    """
    if speech_act == "historical" and intent in TASK_INTENTS | WRITE_INTENTS:
        return "clarify_history"
    if speech_act == "informational" and intent in TASK_INTENTS | WRITE_INTENTS:
        return "knowledge"
    if intent not in WRITE_INTENTS:
        return "task"
    if speech_act == "action":
        if degraded and not _STRICT_IMPERATIVE.match(span.strip()):
            return "clarify_degraded"
        return "task"
    if speech_act == "conditional":
        return "clarify_condition"
    if speech_act == "historical":
        return "clarify_history"
    return "clarify_alternative"


def normalize_admission(detection: dict[str, Any], message: str) -> dict[str, Any]:
    """Apply the same admission contract before overrides, slots and queues.

    Knowledge frames use the existing guarded FAQ lane. Historical frames
    are context, not executable slot answers. Deferred clarification frames
    retain their source evidence and are served after current requests.
    """
    frames = [
        {
            "intent": detection["intent"],
            "confidence": detection["confidence"],
            "query": detection.get("primary_query") or message,
            "speech_act": detection.get("speech_act", "action"),
            "entity_index": detection.get("entity_index", 0),
        },
        *detection["co_detected"],
    ]
    accepted: list[dict[str, Any]] = []
    clarifications: list[dict[str, Any]] = []
    for frame in frames:
        verdict = admit_request(
            frame["intent"],
            frame.get("speech_act", "action"),
            frame.get("query", ""),
            degraded=bool(detection.get("degraded")),
        )
        if verdict == "knowledge":
            accepted.append({**frame, "intent": "faq", "entity_index": 0})
        elif verdict == "task":
            accepted.append(frame)
        else:
            clarifications.append({**frame, "admission_verdict": verdict})
    # A history-only turn asks what is wanted; a history accompanying a
    # current request adds context but no additional prompt or task.
    if accepted:
        clarifications = [f for f in clarifications if f["admission_verdict"] != "clarify_history"]
    ordered = accepted + clarifications
    first, *rest = ordered
    return {
        **detection,
        "intent": first["intent"],
        "confidence": first["confidence"],
        "primary_query": first.get("query", ""),
        "speech_act": first.get("speech_act", "action"),
        "entity_index": first.get("entity_index", 0),
        "admission_verdict": first.get("admission_verdict", "task"),
        "co_detected": rest,
    }


def pack_detection(detection: Any) -> dict[str, Any]:
    """Keep checkpoint values primitive, including compatibility detector fakes."""
    return {
        "intent": detection.primary.intent.value,
        "confidence": detection.primary.confidence,
        "primary_query": getattr(detection, "primary_query", "") or "",
        "speech_act": getattr(detection, "speech_act", "action") or "action",
        "entity_index": getattr(detection, "entity_index", 0) or 0,
        "degraded": bool((getattr(detection.primary, "metadata", None) or {}).get("fallback")),
        "co_detected": [
            {
                "intent": item.intent.value,
                "confidence": item.confidence,
                "query": getattr(item, "query", "") or "",
                "speech_act": getattr(item, "speech_act", "action") or "action",
                "entity_index": getattr(item, "entity_index", 0) or 0,
            }
            for item in detection.co_detected
        ],
    }


def make_request_policy(
    state: DialogueState, *, read_only: bool = False, request_id: str = ""
) -> dict[str, Any]:
    """Mint server-owned authority without changing the scheduler's membership."""
    return {
        "version": 1,
        "request_id": request_id or state.get("active_request_id") or uuid4().hex,
        "mode": "read_only" if read_only else "normal",
        "user_id": state.get("user_id"),
        "intents": ["query_order"] if read_only else [],
        "target_order_id": None,
    }


def valid_request_policy(policy: Any) -> bool:
    """Validate persisted primitives; a malformed scope is never a normal scope."""
    if not isinstance(policy, dict):
        return False
    user = policy.get("user_id")
    target = policy.get("target_order_id")
    return (
        type(policy.get("version")) is int
        and policy["version"] == 1
        and isinstance(policy.get("request_id"), str)
        and bool(policy["request_id"])
        and "user_id" in policy
        and (user is None or type(user) is int)
        and "target_order_id" in policy
        and (target is None or isinstance(target, str) and bool(target))
        and (
            policy.get("mode") == "read_only"
            and policy.get("intents") == ["query_order"]
            or policy.get("mode") == "normal"
            and policy.get("intents") == []
            and target is None
        )
    )


def policy_fields(frame: Mapping[str, Any]) -> dict[str, Any]:
    """Move permission and its independent presence marker with task progress."""
    return {
        "request_policy": frame.get("request_policy"),
        "request_policy_required": frame.get("request_policy_required", False),
    }


def policy_denies(
    state: DialogueState,
    intent: str,
    args: dict[str, Any],
    *,
    executing: bool = False,
    agent: bool = False,
) -> bool:
    """Complete mediation for task/user/intent/target, including legacy absence."""
    policy = state.get("request_policy")
    if policy is None:
        return state.get("request_policy_required", False) is not False
    if not valid_request_policy(policy):
        return True
    if policy["user_id"] != state.get("user_id"):
        return True
    owner = state.get("active_request_id")
    if owner and owner != policy["request_id"]:
        return True
    if policy["mode"] == "normal":
        return False
    if agent or intent != "query_order":
        return True
    target = policy["target_order_id"]
    return bool(
        executing
        and (not target or args.get("order_id") != target)
        or target
        and args.get("order_id")
        and args["order_id"] != target
    )


def discard_queue(state: DialogueState) -> dict[str, Any]:
    """Remove queued requests without changing interrupted-task lifecycle."""
    stack = state.get("state_stack") or []
    kept = [item for item in stack if item.get("queued_from") != "multi_intent"]
    return {"state_stack": kept} if len(kept) != len(stack) else {}


def terminate_queue(state: DialogueState) -> dict[str, Any]:
    """Invalidate a failed/expired chain instead of resuming it on a later answer."""
    cleanup = {
        "request_policy": None,
        "request_policy_required": False,
        "pending_admission_request": None,
    }
    if not state.get("multi_intent_turn"):
        return cleanup
    return {
        **cleanup,
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
    degraded = bool(packed.get("degraded"))
    candidates: list[dict[str, Any]] = []
    for item in packed["co_detected"]:
        verdict = item.get("admission_verdict") or admit_request(
            item["intent"],
            item.get("speech_act", "action"),
            item.get("query", ""),
            degraded=degraded,
        )
        if verdict == "knowledge":
            candidates.append({**item, "intent": "faq", "entity_index": 0})
        elif verdict.startswith("clarify_"):
            if verdict != "clarify_history":
                candidates.append({**item, "admission_verdict": verdict})
        elif item["intent"] in QUEUEABLE_INTENTS:
            candidates.append(item)
    primary_query = packed["primary_query"]
    # Filtering negated/quoted/superseded clauses can leave one accepted
    # request. Its source span must still own slot extraction and output
    # restoration even when there is no secondary work to queue.
    isolated_primary = bool(primary_query and primary_query != state.get("message", ""))
    if not candidates and not isolated_primary:
        return {"entity_index": packed.get("entity_index", 0)}
    stack = list(state.get("state_stack") or [])
    count = sum(item.get("queued_from") == "multi_intent" for item in stack)
    overflow = list(state.get("multi_intent_overflow") or [])
    for candidate in candidates:
        query = candidate["query"]
        if count >= MULTI_INTENT_QUEUE_CAP or not query:
            overflow.append(query or candidate["intent"])
            continue
        request_id = uuid4().hex
        stack.append(
            {
                "request_policy": make_request_policy(state, request_id=request_id),
                "request_policy_required": True,
                "intent": candidate["intent"],
                "confidence": candidate["confidence"],
                "queued_from": "multi_intent",
                "request_id": request_id,
                "query": query,
                "speech_act": candidate.get("speech_act", "action"),
                "entity_index": candidate.get("entity_index", 0),
                "admission_verdict": candidate.get("admission_verdict", ""),
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
        "active_request_id": state.get("active_request_id")
        or (state.get("request_policy") or {}).get("request_id")
        or uuid4().hex,
        "active_request_query": packed["primary_query"] or state.get("message", ""),
        "entity_index": packed.get("entity_index", 0),
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
