"""Request-level accounting for the offline multi-intent experiment."""

import json
from pathlib import Path
from typing import Any

import pytest

from app.services.evaluation.multi_intent_eval import load_cases, score_requests


def test_repeated_labels_are_distinct_requests() -> None:
    score = score_requests(
        [("refund", "refund order A")],
        [("refund", "refund order A"), ("refund", "refund order B")],
    )
    assert score["matched_labels"] == 1
    assert score["expected"] == 2
    assert score["matched_requests"] == 1
    assert not score["exact_requests"]


def test_extra_write_is_a_false_positive_not_a_recall_improvement() -> None:
    score = score_requests(
        [("policy", "return policy"), ("return", "return policy")],
        [("policy", "return policy")],
    )
    assert score["matched_labels"] == 1
    assert score["predicted"] == 2
    assert score["false_positive_writes"] == 1
    assert not score["exact_requests"]


def test_label_coverage_does_not_claim_query_isolation() -> None:
    score = score_requests(
        [("refund", "refund A and track B"), ("track_shipping", "refund A and track B")],
        [("refund", "refund A"), ("track_shipping", "track B")],
    )
    assert score["matched_labels"] == 2
    assert score["matched_requests"] == 0


def test_duplicate_output_cannot_inflate_coverage() -> None:
    score = score_requests(
        [("faq", "shipping fee"), ("faq", "shipping fee")], [("faq", "shipping fee")]
    )
    assert score["matched_labels"] == 1
    assert score["matched_requests"] == 1
    assert not score["exact_requests"]


@pytest.mark.parametrize("payload", [[], {}, [{"id": "a"}], [None]])
def test_loader_fails_closed_on_bad_corpus(tmp_path: Path, payload: object) -> None:
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_cases(path)


def test_loader_rejects_duplicate_ids_and_unanchored_queries(tmp_path: Path) -> None:
    item: dict[str, Any] = {
        "id": "one",
        "query": "refund A",
        "combo": "single",
        "expect_requests": [{"intent": "refund", "query": "invented"}],
    }
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([item]))
    with pytest.raises(ValueError, match="span"):
        load_cases(path)
    item["expect_requests"][0]["query"] = "refund A"
    path.write_text(json.dumps([item, item]))
    with pytest.raises(ValueError, match="duplicate"):
        load_cases(path)


@pytest.mark.parametrize("intent", [[], {}, 1, None, True])
def test_golden_loader_malformed_intent_types_fail_closed(tmp_path: Path, intent: object) -> None:
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "bad-type",
                    "query": "refund A",
                    "combo": "single",
                    "expect_requests": [{"intent": intent, "query": "refund A"}],
                }
            ]
        )
    )
    with pytest.raises(ValueError, match="intent"):
        load_cases(path)
