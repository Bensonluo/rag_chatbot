"""Evaluation runner: loading, route buckets, scoring math."""

import json
from unittest.mock import Mock

from app.models.enums.intent import Intent
from app.services.evaluation import (
    EvalReport,
    GoldenCase,
    load_golden_cases,
    route_bucket,
    run_intent_eval,
)
from app.services.evaluation.runner import GOLDEN_SET_FILE


def _result(intent: Intent):
    mock = Mock()
    mock.intent = Mock(value=intent.value)
    return mock


class SyncFakeDetector:
    """Sync detector (rule-based shape)."""

    def __init__(self, mapping: dict[str, Intent]) -> None:
        self.mapping = mapping

    def detect_with_confidence(self, query: str):
        return _result(self.mapping.get(query, Intent.UNKNOWN))


class AsyncFakeDetector:
    """Async detector (hybrid/LLM shape) — the runner must accept both."""

    def __init__(self, mapping: dict[str, Intent]) -> None:
        self.mapping = mapping

    async def detect_with_confidence(self, query: str):
        return _result(self.mapping.get(query, Intent.UNKNOWN))


class TestRouteBucket:
    def test_buckets_mirror_graph_routing(self):
        assert route_bucket("refund") == "task"
        assert route_bucket("complaint") == "task"
        assert route_bucket("faq") == "rag"
        assert route_bucket("policy") == "rag"
        assert route_bucket("greeting") == "direct"
        assert route_bucket("chitchat") == "direct"
        assert route_bucket("confirm") == "meta"
        assert route_bucket("deny") == "meta"
        assert route_bucket("cancel") == "meta"
        assert route_bucket("handoff") == "handoff"
        assert route_bucket("entity_lookup") == "graph"

    def test_unknown_falls_to_direct(self):
        assert route_bucket("unknown") == "direct"
        assert route_bucket("not-a-real-intent") == "direct"


class TestLoading:
    def test_bundled_set_loads_with_expected_families(self):
        cases = load_golden_cases()
        families = {case.expect_intent for case in cases}
        assert {"refund", "return", "handoff", "faq", "policy"} <= families
        assert len(cases) >= 40

    def test_malformed_json_degrades_to_empty(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        assert load_golden_cases(path) == []

    def test_missing_file_degrades_to_empty(self, tmp_path):
        assert load_golden_cases(tmp_path / "nope.json") == []

    def test_malformed_cases_skipped(self, tmp_path):
        path = tmp_path / "mixed.json"
        path.write_text(
            json.dumps(
                [
                    {"id": "ok", "query": "q", "expect_intent": "faq"},
                    {"id": "no_query", "expect_intent": "faq"},
                    "not-even-a-dict",
                ]
            ),
            encoding="utf-8",
        )
        cases = load_golden_cases(path)
        assert [case.id for case in cases] == ["ok"]


class TestRunIntentEval:
    async def test_scoring_math_and_failures(self):
        mapping = {"我要退款": Intent.REFUND, "你好": Intent.GREETING}
        cases = [
            GoldenCase(id="a", query="我要退款", expect_intent="refund"),
            GoldenCase(id="b", query="你好", expect_intent="chitchat"),  # wrong bucket too
            GoldenCase(id="c", query="随便说点什么", expect_intent="unknown"),
        ]

        report = await run_intent_eval(SyncFakeDetector(mapping), cases)

        assert report.total == 3
        assert report.passed == 2
        assert report.accuracy == 2 / 3
        assert report.route_passed == 3  # b: greeting/chitchat share "direct"
        assert report.route_accuracy == 1.0
        assert [f.case.id for f in report.failures] == ["b"]
        assert report.by_intent["chitchat"] == {"total": 1, "passed": 0}

    async def test_accepts_async_detector(self):
        mapping = {"我要退款": Intent.REFUND}
        cases = [GoldenCase(id="a", query="我要退款", expect_intent="refund")]

        report = await run_intent_eval(AsyncFakeDetector(mapping), cases)

        assert report.accuracy == 1.0

    async def test_empty_cases_yield_zeroed_report(self):
        report = await run_intent_eval(SyncFakeDetector({}), [])
        assert report.total == 0
        assert report.accuracy == 0.0  # no divide-by-zero on empty sets


class TestEvalReportDefaults:
    def test_default_report_has_zeroed_ratios(self):
        report = EvalReport()
        assert report.total == 0
        assert report.accuracy == 0.0
        assert report.route_accuracy == 0.0
        assert report.failures == []

    def test_golden_set_file_lives_beside_runner(self):
        assert GOLDEN_SET_FILE.exists()
        assert GOLDEN_SET_FILE.name == "golden_set.json"
