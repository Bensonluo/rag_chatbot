"""Offline, synthetic request-level evaluation; never calls a hosted model.

Label recall keeps multiplicity (two refunds are two requests). Exact
request coverage additionally checks the source subquery, so assigning
both orders to the same task cannot masquerade as successful routing.
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, cast

from langchain_core.runnables import RunnableConfig

from app.models.enums.intent import DIRECT_INTENTS, GRAPH_INTENTS, RAG_INTENTS, TASK_INTENTS, Intent
from app.services.intent.base import IntentDetector

GOLDEN_PATH = Path(__file__).with_name("multi_intent_golden_set.json")
NEGATIVE_PATH = Path(__file__).with_name("multi_intent_negative_set.json")
WRITE_INTENTS = {"refund", "return", "complaint"}
Request = tuple[str, str]


def load_cases(path: Path = GOLDEN_PATH) -> list[dict[str, Any]]:
    """Reject invalid datasets instead of silently shrinking the denominator."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("Corpus must be a nonempty list")
    ids: set[str] = set()
    known = {intent.value for intent in Intent}
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Every case must be an object")
        for name in ("id", "query", "combo"):
            if not isinstance(item.get(name), str) or not item[name].strip():
                raise ValueError(f"Case needs nonempty {name}")
        if item["id"] in ids:
            raise ValueError("duplicate case id")
        ids.add(item["id"])
        requests = item.get("expect_requests")
        if not isinstance(requests, list) or not requests:
            raise ValueError("Case needs nonempty expect_requests")
        for request in requests:
            if (
                not isinstance(request, dict)
                or not isinstance(request.get("intent"), str)
                or request["intent"] not in known
            ):
                raise ValueError("Invalid expected intent")
            query = request.get("query")
            if not isinstance(query, str) or not query.strip() or query not in item["query"]:
                raise ValueError("Expected query must be a source span")
    return raw


def score_requests(predicted: list[Request], expected: list[Request]) -> dict[str, Any]:
    """Use multiset matching; neither deduplication nor repeats earn free credit."""
    actual_labels = Counter(intent for intent, _ in predicted)
    expected_labels = Counter(intent for intent, _ in expected)
    false_positives = actual_labels - expected_labels
    return {
        "expected": len(expected),
        "predicted": len(predicted),
        "matched_labels": sum((actual_labels & expected_labels).values()),
        "matched_requests": sum((Counter(predicted) & Counter(expected)).values()),
        "exact_requests": Counter(predicted) == Counter(expected),
        "primary_correct": bool(predicted and predicted[0][0] == expected[0][0]),
        "false_positive_writes": sum(
            count for intent, count in false_positives.items() if intent in WRITE_INTENTS
        ),
    }


def summarize(scores: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        key: sum(score[key] for score in scores)
        for key in (
            "expected",
            "predicted",
            "matched_labels",
            "matched_requests",
            "false_positive_writes",
            "primary_correct",
            "exact_requests",
        )
    }
    return {
        "cases": len(scores),
        **totals,
        "label_recall": totals["matched_labels"] / max(totals["expected"], 1),
        "label_precision": totals["matched_labels"] / max(totals["predicted"], 1),
        "request_recall": totals["matched_requests"] / max(totals["expected"], 1),
        "exact_request_rate": totals["exact_requests"] / max(len(scores), 1),
    }


async def run_evaluation(
    detector: IntentDetector, cases: list[dict[str, Any]], *, multi: bool
) -> dict[str, Any]:
    """Compare frozen single-label API to multi API using the same corpus."""
    rows: list[dict[str, Any]] = []
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        if multi:
            result = await detector.detect_multi(case["query"])
            predicted = [(result.primary.intent.value, result.primary_query or case["query"])]
            predicted.extend((item.intent.value, item.query) for item in result.co_detected)
        else:
            single = await detector.detect_with_confidence(case["query"])
            predicted = [(single.intent.value, case["query"])]
        expected = [(item["intent"], item["query"]) for item in case["expect_requests"]]
        score = score_requests(predicted, expected)
        buckets[case["combo"]].append(score)
        rows.append(
            {
                "id": case["id"],
                "predicted_requests": predicted,
                "expected_requests": expected,
                **score,
            }
        )
    return {
        "mode": "multi" if multi else "single",
        "corpus": "synthetic; detection only; not production prevalence or task completion",
        "overall": summarize(rows),
        "by_combo": {name: summarize(values) for name, values in sorted(buckets.items())},
        "cases": rows,
    }


def load_negative_cases(path: Path = NEGATIVE_PATH) -> list[dict[str, Any]]:
    """Validate every annotation; malformed values never shrink populations."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("Negative corpus must be a nonempty list")
    known = TASK_INTENTS | RAG_INTENTS | GRAPH_INTENTS | DIRECT_INTENTS
    ids: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Every negative case must be an object")
        for name in ("id", "query", "category"):
            if not isinstance(item.get(name), str) or not item[name].strip():
                raise ValueError(f"Negative case needs nonempty {name}")
        if item["id"] in ids:
            raise ValueError("duplicate negative case id")
        ids.add(item["id"])
        for name, allowed in (
            ("forbidden_writes", WRITE_INTENTS | {"cancel"}),
            ("expected_tasks", known),
        ):
            values = item.get(name)
            if not isinstance(values, list) or any(
                not isinstance(value, str) or value not in allowed for value in values
            ):
                raise ValueError(f"{item['id']}: invalid {name}")
        requests = item.get("expected_requests")
        if not isinstance(requests, list):
            raise ValueError("expected_requests must be a list")
        for request in requests:
            if not isinstance(request, dict):
                raise ValueError("Every expected request must be an object")
            intent, query = request.get("intent"), request.get("query")
            if not isinstance(intent, str) or intent not in known:
                raise ValueError("Invalid expected request intent")
            if not isinstance(query, str) or not query.strip() or query not in item["query"]:
                raise ValueError("Expected query must be a source span")
            entity = request.get("order_id", "")
            if not isinstance(entity, str) or (entity and entity not in item["query"]):
                raise ValueError("Expected order_id must occur in source")
            if intent in {"refund", "return", "query_order", "track_shipping"} and not entity:
                raise ValueError("Order-scoped expected requests need order_id")
        if [r["intent"] for r in requests] != item["expected_tasks"]:
            raise ValueError("expected_tasks must agree with expected_requests")
        if type(item.get("clarification_expected")) is not bool:
            raise ValueError("clarification_expected must be boolean")
        language = item.get("language", "unannotated")
        if not isinstance(language, str) or language not in {"en", "zh", "unannotated"}:
            raise ValueError("Invalid language annotation")
        outcome = item.get("expected_outcome")
        if not isinstance(outcome, str) or outcome not in {
            "requests",
            "no_business_action",
            "cancel_conversation",
        }:
            raise ValueError("Invalid expected_outcome")
        if (outcome == "requests") != bool(requests):
            raise ValueError("Expected outcome disagrees with requests")
    return raw


# CANCEL is dialogue control, NOT a commerce write. Complaint handlers do
# create business records and must be included in side-effect accounting.
CANDIDATE_INTENTS = TASK_INTENTS | RAG_INTENTS | GRAPH_INTENTS | DIRECT_INTENTS
ORDER_PATTERN = re.compile(r"\bORD\d+\b", re.IGNORECASE)
FrameKey = tuple[str, str, str]


def _entity(query: str, index: int) -> str:
    entities = list(dict.fromkeys(ORDER_PATTERN.findall(query)))
    return entities[index] if 0 <= index < len(entities) else ""


def _key(request: dict[str, Any], *, source: bool) -> FrameKey:
    intent = request["intent"]
    # Knowledge routing intentionally maps informational business frames to
    # FAQ. Compare knowledge requests by source, not the classifier's label.
    if intent in RAG_INTENTS:
        intent = "knowledge"
    span = request.get("query", "") if source or intent == "knowledge" else ""
    return intent, span, "" if intent == "knowledge" else request.get("order_id", "")


def _metric(rows: list[dict[str, Any]], flag: str, population: str) -> dict[str, Any]:
    numerator = sum(bool(row[flag]) for row in rows)
    return {
        "numerator": numerator,
        "denominator": len(rows),
        "population": population,
        "population_case_ids": [row["id"] for row in rows],
        "rate": numerator / len(rows) if rows else None,
    }


async def run_negative_evaluation(
    detector: IntentDetector, cases: list[dict[str, Any]]
) -> dict[str, Any]:
    """Score candidate admission only, never call an admission label STAGED.

    Source span, entity index and fallback metadata are preserved per frame.
    Multi-request credit requires the complete multiset, not label recall.
    """
    from app.services.dialogue.multi_intent import admit_request, pack_detection

    rows: list[dict[str, Any]] = []
    for case in cases:
        packed = pack_detection(await detector.detect_multi(case["query"]))
        frames = [
            {
                "intent": packed["intent"],
                "query": packed["primary_query"] or case["query"],
                "speech_act": packed["speech_act"],
                "entity_index": packed["entity_index"],
            },
            *packed["co_detected"],
        ]
        admitted = []
        for frame in frames:
            span = frame["query"]
            frame["degraded"] = packed["degraded"]
            frame["verdict"] = admit_request(
                frame["intent"], frame["speech_act"], span, degraded=packed["degraded"]
            )
            frame["anchored"] = bool(span and span in case["query"])
            frame["order_id"] = _entity(span, frame.get("entity_index", 0))
            if frame["anchored"] and frame["intent"] in CANDIDATE_INTENTS:
                if frame["verdict"] == "knowledge":
                    admitted.append({**frame, "intent": "faq"})
                elif frame["verdict"] == "task":
                    admitted.append(frame)
        expected = Counter(_key(r, source=True) for r in case["expected_requests"])
        actual = Counter(_key(r, source=True) for r in admitted)
        extra = actual - expected
        rows.append(
            {
                "id": case["id"],
                "category": case["category"],
                "frames": frames,
                "admitted_tasks": [r["intent"] for r in admitted],
                "admitted_requests": admitted,
                "expected_requests": case["expected_requests"],
                "single": len(case["expected_requests"]) == 1,
                "multiple": len(case["expected_requests"]) > 1,
                "no_write": not any(
                    r["intent"] in WRITE_INTENTS for r in case["expected_requests"]
                ),
                "contaminated": len(admitted) > len(case["expected_requests"]),
                "request_identity_mismatch": actual != expected,
                "false_write": any(k[0] in WRITE_INTENTS for k in extra),
                "recovered": actual == expected,
                "missing_requests": list((expected - actual).elements()),
                "extra_requests": list(extra.elements()),
            }
        )
    return {
        "mode": "candidate-admission",
        "cases": len(rows),
        "corpus": "synthetic challenge; candidate verdicts, NOT staged writes or completion",
        "metrics": {
            "candidate_single_request_contamination": _metric(
                [r for r in rows if r["single"]],
                "contaminated",
                "exactly one expected request; numerator is an added request, not source/binding mismatch",
            ),
            "candidate_false_write_admission": _metric(
                [r for r in rows if r["no_write"]],
                "false_write",
                "no current business write requested",
            ),
            "candidate_multi_request_recovery": _metric(
                [r for r in rows if r["multiple"]],
                "recovered",
                "two or more expected requests; span+entity multiset",
            ),
        },
        "not_measured": ["STAGED writes", "task completion", "actual clarification responses"],
        "case_rows": rows,
    }


class _OfflineFAQ:
    """Deterministic knowledge-lane probe, not a policy correctness oracle."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def match(self, query: str) -> Any:
        from types import SimpleNamespace

        self.calls.append(query)
        return SimpleNamespace(faq_id=query, question=query, answer="Offline knowledge: " + query)


async def _probe_graph_case(detector: IntentDetector, case: dict[str, Any]) -> dict[str, Any]:
    from langgraph.checkpoint.memory import MemorySaver

    from app.services.dialogue.graph import build_dialogue_graph
    from app.services.dialogue.tools import ToolDefinition, ToolRegistry
    from app.services.faq.store import FAQService

    calls: list[dict[str, Any]] = []
    registry = ToolRegistry()
    for intent in sorted(TASK_INTENTS):

        def handler(args: dict[str, Any], intent: str = intent) -> dict[str, Any]:
            calls.append(
                {"intent": intent, "order_id": args.get("order_id", ""), "args": dict(args)}
            )
            return {"order_id": args.get("order_id", ""), "status": "offline_done"}

        registry.register(
            ToolDefinition(
                name="offline_" + intent,
                intent=intent,
                description="in-memory evaluation recorder",
                required_slots=["order_id"],
                handler=handler,
                requires_confirmation=intent in WRITE_INTENTS,
            )
        )
    faq = _OfflineFAQ()
    graph = build_dialogue_graph(
        intent_detector=detector,
        slot_filler=None,
        tool_registry=registry,
        retrieval_pipeline={},
        llm_service=None,
        faq_service=cast(FAQService, faq),
        checkpointer=MemorySaver(),
    )
    config: RunnableConfig = {"configurable": {"thread_id": case["id"]}, "recursion_limit": 100}
    stages: dict[str, dict[str, Any]] = {}
    observed: dict[str, dict[str, Any]] = {}
    snapshots: list[dict[str, Any]] = []
    error = ""

    async def turn(message: str) -> dict[str, Any]:
        final: dict[str, Any] = {}
        async for value in graph.astream(
            {"message": message, "session_id": 1, "user_id": 1}, config, stream_mode="values"
        ):
            final = dict(value)
            snapshots.append(final)
            pending = final.get("pending_confirmation") or {}
            if pending:
                stages[pending["action_id"]] = {
                    "intent": pending["intent"],
                    "order_id": pending["args"].get("order_id", ""),
                    "action_id": pending["action_id"],
                    "args": dict(pending["args"]),
                }
            for queued in final.get("state_stack") or []:
                if (
                    queued.get("queued_from") == "multi_intent"
                    and queued["intent"] in TASK_INTENTS
                    and not queued.get("admission_verdict", "").startswith("clarify_")
                ):
                    observed[queued["request_id"]] = {
                        "intent": queued["intent"],
                        "order_id": _entity(queued["query"], queued.get("entity_index", 0)),
                    }
            # Identify current tasks by request id, so a queued duplicate of
            # the primary counts twice, while repeated checkpoints count once.
            if (
                final.get("intent") in TASK_INTENTS
                and not final.get("admission_clarify")
                and not final.get("task_executed")
                and (final.get("pending_slots") or final.get("filled_slots") or pending)
            ):
                args = pending.get("args") or final.get("filled_slots") or {}
                observed[final.get("active_request_id") or "primary"] = {
                    "intent": final["intent"],
                    "order_id": args.get("order_id", ""),
                }
        return final

    try:
        first = await turn(case["query"])
        current = first
        writes_expected = any(r["intent"] in WRITE_INTENTS for r in case["expected_requests"])
        remaining_writes = Counter(
            _key(r, source=False) for r in case["expected_requests"] if r["intent"] in WRITE_INTENTS
        )
        # Only acknowledge a stage whose intent AND entity are still requested.
        for _ in range(sum(remaining_writes.values())):
            pending = current.get("pending_confirmation") or {}
            if not pending:
                break
            stage_key = _key(
                {"intent": pending["intent"], "order_id": pending["args"].get("order_id", "")},
                source=False,
            )
            if remaining_writes[stage_key] <= 0:
                break
            remaining_writes[stage_key] -= 1
            current = await turn("yes")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        first = snapshots[-1] if snapshots else {}
        current = first
        writes_expected = any(r["intent"] in WRITE_INTENTS for r in case["expected_requests"])
    expected = Counter(_key(r, source=False) for r in case["expected_requests"])
    completed = [*calls, *({"intent": "faq", "query": q} for q in faq.calls)]
    actual = Counter(_key(r, source=False) for r in completed)
    # Request IDs deduplicate checkpoints without collapsing two requests for
    # the same entity. Completed evidence also covers tasks with no slot prompt.
    scheduled = Counter(_key(r, source=False) for r in observed.values())
    staged = Counter(_key(r, source=False) for r in stages.values())
    tasks_seen = scheduled | actual
    extra = tasks_seen - expected
    clarification = bool(first.get("admission_clarify") or first.get("pending_slots"))
    false_stages = bool(
        staged - Counter({k: v for k, v in expected.items() if k[0] in WRITE_INTENTS})
    )
    false_executions = any(k[0] in WRITE_INTENTS for k in actual - expected)
    cancel_ok = (
        case["expected_outcome"] == "cancel_conversation"
        and first.get("intent") == "cancel"
        and not stages
        and not calls
        and not first.get("pending_confirmation")
        and not first.get("state_stack")
    )
    return {
        "id": case["id"],
        "category": case["category"],
        "expected_requests": case["expected_requests"],
        "expected_outcome": case["expected_outcome"],
        "single": len(case["expected_requests"]) == 1,
        "multiple": len(case["expected_requests"]) > 1,
        "no_write": not writes_expected,
        "clarification_not_needed": not case["clarification_expected"],
        "contaminated": sum(tasks_seen.values()) > len(case["expected_requests"]),
        "request_identity_mismatch": actual != expected,
        "false_staged": false_stages,
        "false_executed": false_executions,
        "recovered": not error
        and actual == expected
        and not current.get("pending_confirmation")
        and not current.get("pending_slots")
        and not current.get("state_stack"),
        "unnecessary_clarification": clarification and not case["clarification_expected"],
        "clarification_observed": clarification,
        "staged_writes": list(stages.values()),
        "executed_tools": calls,
        "knowledge_queries": faq.calls,
        "observed_task_requests": list(observed.values()),
        "extra_requests": list(extra.elements()),
        "missing_completed_requests": list((expected - actual).elements()),
        "cancel_outcome_correct": cancel_ok
        if case["expected_outcome"] == "cancel_conversation"
        else None,
        "response": first.get("response", ""),
        "error": error,
    }


async def run_graph_evaluation(
    detector: IntentDetector, cases: list[dict[str, Any]]
) -> dict[str, Any]:
    """Execute real compiled graph with in-memory checkpoints and fake tools.

    No model, retrieval, database, network or business provider is constructed.
    Only actual pending_confirmation snapshots count as STAGED. The harness
    acknowledges stages in cases with requested writes; it never acknowledges
    a false stage, invents missing slots, or confirms conditional verification.
    This measures offline lifecycle behavior, not real-world business success.
    """
    rows = [await _probe_graph_case(detector, case) for case in cases]
    metrics = {
        "single_request_contamination_rate": _metric(
            [r for r in rows if r["single"]],
            "contaminated",
            "exactly one expected request; numerator is added lifecycle requests (queued/slot/staged/executed/knowledge), not source/binding mismatch",
        ),
        "false_staged_write_rate": _metric(
            [r for r in rows if r["no_write"]],
            "false_staged",
            "no current business write requested; actual pending_confirmation snapshots",
        ),
        "multi_request_complete_recovery": _metric(
            [r for r in rows if r["multiple"]],
            "recovered",
            "two or more supported current requests; exact completed intent+entity multiset after separate stage acknowledgments, no invented slots",
        ),
        "unnecessary_clarification_rate": _metric(
            [r for r in rows if r["clarification_not_needed"]],
            "unnecessary_clarification",
            "annotations say no clarification required; initial graph admission_clarify or pending_slots",
        ),
        "false_executed_write_rate": _metric(
            [r for r in rows if r["no_write"]],
            "false_executed",
            "no current business write requested; fake handler invocations",
        ),
    }
    return {
        "mode": "compiled-graph-offline",
        "cases": len(rows),
        "corpus": "synthetic challenge; real compiled lifecycle, fake in-memory business handlers; not production prevalence",
        "metrics": metrics,
        "case_rows": rows,
        "by_category": dict(Counter(r["category"] for r in rows)),
        "language_counts": dict(Counter(c.get("language", "unannotated") for c in cases)),
        "safety_gate_passed": not any(
            r["false_staged"] or r["false_executed"] or r["error"] for r in rows
        ),
        "limitations": [
            "No real business or model correctness measured; only offline graph behavior.",
            "Contamination observes task lifecycle evidence, not hidden provider-side records.",
            "Clarification uses explicit admission and slot signals, not free-text response judgment.",
            "Missing slots are not fabricated; those cases may fail complete recovery.",
            "CANCEL is meta conversation cancellation, never two order cancellations.",
        ],
        "approved_metrics_not_measured": [],
    }
