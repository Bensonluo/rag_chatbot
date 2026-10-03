"""Routing golden-set eval: the deterministic skeleton has a CI gate.

Review 2026-09-26 #10's answer-eval recommendation ("回答评测调用实际
ChatService") had only its retrieval half closed (ff7748d); the routing
half stayed untested offline. The evidence for needing it: three live
routing bugs reached production unnoticed — EN policy misroute (283064d),
reversed zh how-to phrasing (20201e9), complaint tie arbitration
(f723874). These tests pin the harness that runs golden cases through
the REAL compiled graph (keyless, offline, deterministic) so that class
of regression fails CI instead of a live user.
"""

from typing import Any

from app.services.evaluation.routing_eval import (
    format_routing_table,
    load_routing_cases,
    run_routing_baseline,
)


class TestGoldenSetSchema:
    def test_cases_load_with_unique_ids(self) -> None:
        cases = load_routing_cases()
        assert cases, "routing golden set must not be empty"
        ids = [case.id for case in cases]
        assert len(ids) == len(set(ids)), "case ids must be unique"

    def test_every_turn_declares_an_outcome(self) -> None:
        cases = load_routing_cases()
        for case in cases:
            assert case.turns, f"{case.id}: at least one turn"
            assert case.category, f"{case.id}: category required"
            for turn in case.turns:
                assert turn.utterance.strip()
                # An expectation must assert something beyond intent:
                # grounding, tool identity, LLM budget, or copy anchors.
                expect = turn.expect
                assert any(
                    [
                        expect.executed_tools is not None,
                        expect.sources_contain,
                        expect.content_contain,
                        expect.llm_calls is not None,
                        expect.tickets_created is not None,
                        expect.prompt_contain,
                    ]
                ), f"{case.id}: turn {turn.utterance!r} asserts no outcome"


class TestLiveBugRegressionPins:
    """Each live-found routing bug gets its pin at the graph level."""

    async def test_en_return_policy_is_not_a_return_task(self) -> None:
        """283064d: "What is your return policy" was misrouted to RETURN
        by keyword rules and the bot demanded an order number."""
        failures = _failures_of(await _full_report(), "en-policy-return")
        assert not failures, failures

    async def test_reversed_howto_is_policy_not_return(self) -> None:
        """20201e9: "7天无理由退货怎么算" classified RETURN and demanded
        an order number instead of explaining the policy."""
        failures = _failures_of(await _full_report(), "zh-howto-reversed")
        assert not failures, failures

    async def test_complaint_verb_wins_arbitration(self) -> None:
        """f723874: "我要投诉物流配送…" tied at 1.5 with QUERY_ORDER and
        the enum order crowned query_order; the complaint never landed."""
        failures = _failures_of(await _full_report(), "zh-complaint-logistics")
        assert not failures, failures

    async def test_embedded_threat_is_refund_not_complaint(self) -> None:
        """f723874 guard: 再不退款我就投诉 carries 投诉 as a threat, not
        an explicit complaint verb — the refund flow must keep it."""
        failures = _failures_of(await _full_report(), "zh-embedded-threat")
        assert not failures, failures

    async def test_hi_inside_thing_is_not_greeting(self) -> None:
        """283064d: the "hi" substring inside "thing" classified a
        broken-item complaint as GREETING."""
        failures = _failures_of(await _full_report(), "en-broken-item-word-boundary")
        assert not failures, failures


class TestBranchCoverage:
    async def test_faq_fastpath_zero_llm(self) -> None:
        """Exact FAQ variant serves canonical copy with no LLM call —
        the fast path's whole reason to exist."""
        failures = _failures_of(await _full_report(), "zh-faq-exact-variant")
        assert not failures, failures

    async def test_tool_turn_grounded_in_tool_result(self) -> None:
        failures = _failures_of(await _full_report(), "zh-query-order-tool")
        assert not failures, failures

    async def test_injection_blocked_zero_tools_zero_llm(self) -> None:
        """Review #2's acceptance at eval scale: blocked input executes
        nothing and generates nothing."""
        failures = _failures_of(await _full_report(), "zh-injection-blocked")
        assert not failures, failures

    async def test_no_evidence_fixed_copy(self) -> None:
        """9ada2c1: empty retrieval ends in the deterministic copy, not
        free generation."""
        failures = _failures_of(await _full_report(), "zh-no-evidence")
        assert not failures, failures

    async def test_confirmation_lifecycle(self) -> None:
        """Stage never executes; a confirm executes once; a repeat
        confirm does not re-execute (1525b69 idempotency)."""
        failures = _failures_of(await _full_report(), "zh-refund-confirmation")
        assert not failures, failures

    async def test_handoff_template_never_touches_llm(self) -> None:
        failures = _failures_of(await _full_report(), "zh-handoff-explicit")
        assert not failures, failures


class TestFullBaseline:
    async def test_entire_golden_set_passes(self) -> None:
        """The golden set is the contract: against current code it must
        be all green. A failure here is either a regression (the point)
        or a stale expectation to re-derive from the pinned semantics."""
        report = await run_routing_baseline()
        assert report["failed_turns"] == 0, report["failures"]
        assert report["total_turns"] >= 20, "golden set must stay broad"
        assert report["pass_rate"] == 1.0

    async def test_report_shape_and_table(self) -> None:
        report = await run_routing_baseline()
        for key in (
            "total_cases",
            "total_turns",
            "passed_turns",
            "failed_turns",
            "pass_rate",
            "failures",
            "categories",
        ):
            assert key in report
        table = format_routing_table(report)
        assert "routing" in table.lower()
        assert str(report["total_cases"]) in table

    async def test_deterministic_across_runs(self) -> None:
        first = await run_routing_baseline()
        second = await run_routing_baseline()
        assert first["pass_rate"] == second["pass_rate"]
        assert first["total_turns"] == second["total_turns"]


_REPORT_CACHE: dict[str, dict[str, Any]] = {}


async def _full_report() -> dict[str, Any]:
    if "report" not in _REPORT_CACHE:
        _REPORT_CACHE["report"] = await run_routing_baseline()
    return _REPORT_CACHE["report"]


def _failures_of(report: dict[str, Any], case_id: str) -> list[dict[str, Any]]:
    return [f for f in report["failures"] if f["case"] == case_id]
