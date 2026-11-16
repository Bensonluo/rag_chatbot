"""Offline, synthetic request-level evaluation; never calls a hosted model.

Label recall keeps multiplicity (two refunds are two requests). Exact
request coverage additionally checks the source subquery, so assigning
both orders to the same task cannot masquerade as successful routing.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from app.models.enums.intent import Intent
from app.services.intent.base import IntentDetector

GOLDEN_PATH = Path(__file__).with_name("multi_intent_golden_set.json")
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
            if not isinstance(request, dict) or request.get("intent") not in known:
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
