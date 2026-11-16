"""Accounting regressions: recognition labels are not lifecycle evidence."""

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from app.models.enums.intent import Intent
from app.services.evaluation.multi_intent_eval import (
    load_negative_cases,
    run_negative_evaluation,
)
from app.services.intent.base import CoDetectedIntent, IntentDetection, IntentResult
from app.services.intent.rule_based import RuleBasedIntentDetector


async def run_graph_evaluation(*args: Any) -> dict[str, Any]:
    from app.services.evaluation.multi_intent_eval import run_graph_evaluation as evaluate

    return await evaluate(*args)


class FixedDetector(RuleBasedIntentDetector):
    def __init__(self, detection: IntentDetection) -> None:
        self.detection = detection

    async def detect_multi(
        self, query: str, context: dict[str, Any] | None = None
    ) -> IntentDetection:
        if query == "yes":
            return await super().detect_multi(query, context)
        return self.detection


def case(query: str, *requests: dict[str, str]) -> dict[str, Any]:
    return {
        "id": "probe",
        "query": query,
        "category": "probe",
        "expected_requests": list(requests),
        "expected_tasks": [r["intent"] for r in requests],
        "forbidden_writes": [],
        "clarification_expected": False,
        "expected_outcome": "requests" if requests else "no_business_action",
    }


@pytest.mark.parametrize("payload", [[None], [1], ["bad"], [True], [{}]])
def test_negative_loader_rejects_nonobjects(tmp_path: Path, payload: object) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_negative_cases(path)


@pytest.mark.parametrize("field", ["expected_tasks", "forbidden_writes"])
@pytest.mark.parametrize("value", [[{}], [[]], [1], [None]])
def test_negative_loader_rejects_unhashable_and_invalid_members(
    tmp_path: Path, field: str, value: object
) -> None:
    item = case("hello")
    item[field] = value
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([item]))
    with pytest.raises(ValueError):
        load_negative_cases(path)


@pytest.mark.parametrize("language", [[], {}, True, "fr"])
def test_negative_loader_rejects_malformed_language(tmp_path: Path, language: object) -> None:
    item = case("hello")
    item["language"] = language
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([item]))
    with pytest.raises(ValueError, match="language"):
        load_negative_cases(path)


def test_challenge_population_and_cancel_semantics_are_explicit() -> None:
    cases = load_negative_cases()
    assert len(cases) == 32
    assert Counter(c["category"] for c in cases) == {
        "historical": 8,
        "negation": 4,
        "conditional": 4,
        "alternative": 4,
        "quoted": 4,
        "consultation": 4,
        "shared": 4,
    }
    cancel = next(c for c in cases if c["id"] == "neg-shr-3")
    assert cancel["expected_outcome"] == "cancel_conversation"
    assert cancel["expected_requests"] == []


async def test_unknown_is_not_counted_as_an_admitted_task() -> None:
    detector = FixedDetector(IntentDetection(primary=IntentResult(Intent.UNKNOWN, 0.1)))
    report = await run_negative_evaluation(detector, [case("unclear")])
    assert report["case_rows"][0]["admitted_tasks"] == []
    assert report["metrics"]["candidate_false_write_admission"]["denominator"] == 1
    assert "false_staged_write_rate" not in report["metrics"]


async def test_degraded_admission_uses_each_frame_span_not_whole_message() -> None:
    query = "query order ORD1001; a refund for ORD1002 would be great"
    detector = FixedDetector(
        IntentDetection(
            primary=IntentResult(Intent.QUERY_ORDER, 0.9, metadata={"fallback": "llm_error"}),
            primary_query="query order ORD1001",
            co_detected=[
                CoDetectedIntent(
                    Intent.REFUND,
                    0.9,
                    query="a refund for ORD1002 would be great",
                    speech_act="action",
                )
            ],
        )
    )
    report = await run_negative_evaluation(
        detector,
        [
            case(
                query,
                {"intent": "query_order", "query": "query order ORD1001", "order_id": "ORD1001"},
            )
        ],
    )
    row = report["case_rows"][0]
    assert row["admitted_tasks"] == ["query_order"]
    assert row["frames"][1]["verdict"] == "clarify_degraded"
    assert row["frames"][1]["degraded"] is True


async def test_degraded_secondary_imperative_keeps_its_own_source_span() -> None:
    query = "query order ORD1001; Refund order ORD1002 because damaged"
    detector = FixedDetector(
        IntentDetection(
            primary=IntentResult(Intent.QUERY_ORDER, 0.9, metadata={"fallback": "llm_error"}),
            primary_query="query order ORD1001",
            co_detected=[
                CoDetectedIntent(
                    Intent.REFUND,
                    0.9,
                    query="Refund order ORD1002 because damaged",
                    speech_act="action",
                )
            ],
        )
    )
    report = await run_negative_evaluation(
        detector,
        [
            case(
                query,
                *[
                    {"intent": intent, "query": span, "order_id": order}
                    for intent, span, order in (
                        ("query_order", "query order ORD1001", "ORD1001"),
                        ("refund", "Refund order ORD1002 because damaged", "ORD1002"),
                    )
                ],
            )
        ],
    )
    assert report["case_rows"][0]["admitted_tasks"] == ["query_order", "refund"]
    assert report["metrics"]["candidate_multi_request_recovery"]["numerator"] == 1


async def test_same_label_and_span_need_distinct_entities_for_recovery() -> None:
    query = "Refund orders ORD1001 and ORD1002 because damaged"
    detector = FixedDetector(
        IntentDetection(
            primary=IntentResult(Intent.REFUND, 0.9),
            primary_query=query,
            co_detected=[CoDetectedIntent(Intent.REFUND, 0.9, query=query, entity_index=0)],
        )
    )
    report = await run_negative_evaluation(
        detector,
        [
            case(
                query,
                *[
                    {"intent": "refund", "query": query, "order_id": order}
                    for order in ("ORD1001", "ORD1002")
                ],
            )
        ],
    )
    metric = report["metrics"]["candidate_multi_request_recovery"]
    assert metric["numerator"] == 0 and metric["denominator"] == 1


async def test_actual_graph_stages_are_not_detector_labels() -> None:
    query = "refund order ORD1001 because damaged"
    detector = FixedDetector(
        IntentDetection(
            primary=IntentResult(Intent.REFUND, 0.9),
            primary_query=query,
        )
    )
    report = await run_graph_evaluation(detector, [case(query)])
    metric = report["metrics"]["false_staged_write_rate"]
    assert metric["numerator"] == 1 and metric["denominator"] == 1
    assert report["case_rows"][0]["staged_writes"][0]["intent"] == "refund"
    assert report["case_rows"][0]["executed_tools"] == []
    assert report["safety_gate_passed"] is False


@pytest.mark.parametrize(
    "query,intents",
    [
        ("Refund orders ORD1001 and ORD1002 because damaged", ["refund", "refund"]),
        (
            "return order ORD1001 because damaged; refund order ORD1002 because defective",
            ["return", "refund"],
        ),
        ("query order ORD1001; track my package ORD1002", ["query_order", "track_shipping"]),
    ],
)
async def test_model_free_graph_recovers_legal_requests(query: str, intents: list[str]) -> None:
    requests = [
        {"intent": intent, "query": query, "order_id": order}
        for intent, order in zip(intents, ("ORD1001", "ORD1002"), strict=True)
    ]
    report = await run_graph_evaluation(RuleBasedIntentDetector(), [case(query, *requests)])
    row = report["case_rows"][0]
    assert [(r["intent"], r["order_id"]) for r in row["executed_tools"]] == list(
        zip(intents, ("ORD1001", "ORD1002"), strict=True)
    )
    assert report["metrics"]["multi_request_complete_recovery"]["numerator"] == 1
    assert report["safety_gate_passed"]


async def test_graph_counts_any_false_side_effect_not_only_named_forbidden_intent() -> None:
    query = "file complaint ORD1001 because damaged"
    detector = FixedDetector(
        IntentDetection(
            primary=IntentResult(Intent.COMPLAINT, 0.9),
            primary_query=query,
        )
    )
    item = case(query)
    item["forbidden_writes"] = ["refund"]
    report = await run_graph_evaluation(detector, [item])
    assert report["metrics"]["false_staged_write_rate"]["numerator"] == 1


async def test_graph_reports_actual_unnecessary_slot_clarification() -> None:
    query = "ORD1001 和 ORD1002 都要退款"
    item = case(
        query,
        *[
            {"intent": "refund", "query": query, "order_id": order}
            for order in ("ORD1001", "ORD1002")
        ],
    )
    # Corpus annotates this prompt as necessary: no reason supplied. This
    # deliberate probe flips the expectation to test the metric, not the model.
    report = await run_graph_evaluation(RuleBasedIntentDetector(), [item])
    metric = report["metrics"]["unnecessary_clarification_rate"]
    assert metric["numerator"] == 1 and metric["denominator"] == 1
    assert report["case_rows"][0]["clarification_observed"]


async def test_full_challenge_graph_never_connects_to_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import socket

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Offline graph attempted network access")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    report = await run_graph_evaluation(RuleBasedIntentDetector(), load_negative_cases())
    assert report["cases"] == 32
    assert len(report["case_rows"]) == 32
    assert not any(row["error"] for row in report["case_rows"])
    assert report["safety_gate_passed"]
    assert report["metrics"]["false_staged_write_rate"]["denominator"] == 30
    assert report["metrics"]["multi_request_complete_recovery"]["denominator"] == 3


@pytest.mark.parametrize(
    "query,stages",
    [
        ("Refund order ORD1001 because damaged", 1),
        ("Refund order ORD1001", 0),
    ],
)
def test_cli_keeps_detector_comparison_and_writes_failure_artifact(
    tmp_path: Path, query: str, stages: int
) -> None:
    path = tmp_path / "challenge.json"
    output = tmp_path / "report.json"
    # A deliberately mislabeled oracle validates the gate, not detector safety.
    path.write_text(json.dumps([case(query)]))
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_multi_intent_eval.py",
            "--negative-cases",
            str(path),
            "--json",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parents[3],
    )
    assert result.returncode == 1, result.stderr
    report = json.loads(output.read_text())
    assert [r["overall"]["cases"] for r in report["detector_comparison"]] == [40, 40]
    assert (
        report["admission_challenge"]["compiled_graph"]["metrics"]["false_staged_write_rate"][
            "numerator"
        ]
        == stages
    )
    assert report["safety_gate_passed"] is False


def test_cli_bad_challenge_is_usage_error_not_traceback(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("[null]")
    result = subprocess.run(
        [sys.executable, "scripts/run_multi_intent_eval.py", "--negative-cases", str(path)],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parents[3],
    )
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert "must be an object" in result.stderr


async def test_single_request_source_mismatch_is_not_an_added_task() -> None:
    query = "I already returned ORD1001; what is your policy?"
    item = case(query, {"intent": "policy", "query": "what is your policy?"})
    detector = FixedDetector(
        IntentDetection(
            primary=IntentResult(Intent.POLICY, 0.9),
            primary_query=query,
        )
    )
    for evaluate, name in (
        (run_negative_evaluation, "candidate_single_request_contamination"),
        (run_graph_evaluation, "single_request_contamination_rate"),
    ):
        report = await evaluate(detector, [item])
        assert report["metrics"][name]["numerator"] == 0
        assert report["case_rows"][0]["extra_requests"]


async def test_duplicate_queued_write_is_contamination_before_second_stage() -> None:
    query = "Refund order ORD1001 because damaged"
    item = case(query, {"intent": "refund", "query": query, "order_id": "ORD1001"})
    detector = FixedDetector(
        IntentDetection(
            primary=IntentResult(Intent.REFUND, 0.9),
            primary_query=query,
            co_detected=[CoDetectedIntent(Intent.REFUND, 0.9, query=query)],
        )
    )
    report = await run_graph_evaluation(detector, [item])
    assert report["metrics"]["single_request_contamination_rate"]["numerator"] == 1


async def test_isolated_primary_terminal_id_reset_is_not_a_second_task() -> None:
    query = "不要退款 ORD1001，帮我查物流 ORD1002"
    item = case(
        query, {"intent": "track_shipping", "query": "帮我查物流 ORD1002", "order_id": "ORD1002"}
    )
    report = await run_graph_evaluation(RuleBasedIntentDetector(), [item])
    assert report["metrics"]["single_request_contamination_rate"]["numerator"] == 0
    assert len(report["case_rows"][0]["executed_tools"]) == 1


async def test_knowledge_candidate_does_not_bind_incidental_order_entity() -> None:
    query = "What does refund of ORD1001 mean?"
    detector = FixedDetector(
        IntentDetection(
            primary=IntentResult(Intent.REFUND, 0.9),
            primary_query=query,
            speech_act="informational",
        )
    )
    report = await run_negative_evaluation(
        detector, [case(query, {"intent": "faq", "query": query})]
    )
    assert report["case_rows"][0]["missing_requests"] == []
    assert report["case_rows"][0]["extra_requests"] == []
