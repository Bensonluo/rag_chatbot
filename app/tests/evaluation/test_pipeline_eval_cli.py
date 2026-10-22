"""CLI orchestration gates with local fixtures only; no provider network access."""

import asyncio
import json
import logging
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from app.services.evaluation.groundedness_eval import GroundednessVerdict
from app.services.evaluation.judge_calibration import calibration_samples
from app.services.evaluation.pipeline_eval import (
    PipelineEvalReport,
    PipelineObservation,
    PipelineTurnResult,
    RecordedPipelineLLM,
    load_pipeline_cases,
    run_pipeline_eval,
)
from app.services.llm.base import LLMServiceBase
from app.services.llm.factory import LLMFactory
from scripts import run_pipeline_eval as cli


@pytest.fixture(autouse=True)
def isolate_models_and_logging(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    previous = logging.root.manager.disable
    monkeypatch.setattr(LLMFactory, "create", Mock(side_effect=AssertionError("network forbidden")))
    monkeypatch.setattr(
        LLMFactory, "create_from_settings", Mock(side_effect=AssertionError("network forbidden"))
    )
    try:
        yield
    finally:
        logging.disable(previous)


def _verdict(passed: bool = True, *, error: str | None = None) -> GroundednessVerdict:
    return GroundednessVerdict(
        grounded=passed,
        action_claims_supported=True,
        abstention_correct=True,
        promises_supported=True,
        points=[passed],
        error=error,
        model="fixture-model",
    )


def _pipeline() -> PipelineEvalReport:
    observation = PipelineObservation(
        case_id="fixture-case",
        turn_index=1,
        category="policy",
        language="en",
        utterance="private-query-fixture",
        answer="private-answer-fixture",
        intent="policy",
        evidence=[{"content": "private-evidence-fixture"}],
        citations=[],
        metadata={},
        persisted_messages=[],
        history_messages=[],
        expected_history=[],
        handoff_outcome=None,
        ticket_ids=[],
        ticket_records=[],
        session_id=17,
        user_id=23,
        executed_tools=[],
        tool_result={},
        retrieval_ran=True,
        required_points=["required point"],
        forbidden_promises=[],
        abstention=False,
    )
    return PipelineEvalReport(
        [PipelineTurnResult(observation, [], [])],
        model_mode="live_fixture",
    )


def _live(
    monkeypatch: pytest.MonkeyPatch, final: GroundednessVerdict | None = None
) -> tuple[Mock, AsyncMock]:
    judge = cli.ObservedLLM(RecordedPipelineLLM())
    generator = cli.ObservedLLM(RecordedPipelineLLM())
    judge.models.add("same-fixture-model")
    generator.models.add("same-fixture-model")
    providers = Mock(side_effect=[judge, generator])
    monkeypatch.setattr(cli, "_provider", providers)
    verdicts = [_verdict(expected) for _, _, expected in calibration_samples()]
    grading = AsyncMock(side_effect=[*verdicts, final or _verdict()])
    monkeypatch.setattr(cli, "judge_groundedness", grading)
    return providers, grading


@pytest.mark.parametrize("threshold", ["nan", "inf", "-inf", "-0.1", "1.01"])
async def test_invalid_threshold_stops_before_provider_or_pipeline(
    monkeypatch: pytest.MonkeyPatch, threshold: str
) -> None:
    provider, pipeline = Mock(), AsyncMock()
    monkeypatch.setattr(cli, "_provider", provider)
    monkeypatch.setattr(cli, "run_pipeline_eval", pipeline)
    with pytest.raises(SystemExit) as error:
        await cli.main(["--mode=live", f"--threshold={threshold}"])
    assert error.value.code == 2
    provider.assert_not_called()
    pipeline.assert_not_awaited()


async def test_recorded_runs_real_graph_without_creating_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    provider = Mock(side_effect=AssertionError("recorded must not create providers"))
    grading = AsyncMock(side_effect=AssertionError("recorded must not grade semantics"))
    monkeypatch.setattr(cli, "_provider", provider)
    monkeypatch.setattr(cli, "judge_groundedness", grading)
    output = tmp_path / "recorded.json"
    assert await cli.main(["--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["contract_passed"] is True
    assert report["turns"] > 0
    assert report["semantic_quality"] == "not_evaluated"
    assert "semantic_results" not in report
    assert "persisted_messages" not in capsys.readouterr().out
    provider.assert_not_called()
    grading.assert_not_awaited()


@pytest.mark.parametrize("kind", ["wrong_verdict", "judge_error"])
async def test_calibration_failure_prevents_generator_and_pipeline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, kind: str
) -> None:
    providers, grading = _live(monkeypatch)
    verdicts = [_verdict(expected) for _, _, expected in calibration_samples()]
    verdicts[0] = _verdict(False, error="TimeoutError" if kind == "judge_error" else None)
    grading.side_effect = verdicts
    pipeline = AsyncMock()
    monkeypatch.setattr(cli, "run_pipeline_eval", pipeline)
    output = tmp_path / "failed-controls.json"
    assert await cli.main(["--mode=live", "--threshold=0", "--output", str(output)]) == 1
    report = json.loads(output.read_text())
    assert report["judge_controls_passed"] is False
    assert report["semantic_quality"] == "not_evaluated"
    assert "pipeline" not in report
    assert providers.call_count == 1
    pipeline.assert_not_awaited()


async def test_empty_calibration_cannot_authorize_live_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    providers, _ = _live(monkeypatch)
    monkeypatch.setattr(cli, "calibration_samples", lambda: [])
    pipeline = AsyncMock(return_value=_pipeline())
    monkeypatch.setattr(cli, "run_pipeline_eval", pipeline)
    assert await cli.main(["--mode=live"]) == 1
    assert providers.call_count == 1
    pipeline.assert_not_awaited()


async def test_calibration_only_makes_no_pipeline_quality_claim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    providers, _ = _live(monkeypatch)
    pipeline = AsyncMock()
    monkeypatch.setattr(cli, "run_pipeline_eval", pipeline)
    output = tmp_path / "calibration.json"
    assert await cli.main(["--mode=live", "--calibration-only", "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["judge_controls_passed"] is True
    assert report["scope"] == "judge_controls_only; no pipeline quality claim"
    assert report["semantic_quality"] == "not_evaluated"
    assert providers.call_count == 1
    pipeline.assert_not_awaited()


async def test_semantic_success_cannot_override_pipeline_artifact_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _live(monkeypatch)
    case = load_pipeline_cases()[0]
    case = replace(case, turns=[replace(case.turns[0], expected_intent="deliberately_wrong")])
    real_report = await run_pipeline_eval(cases=[case])
    assert not real_report.passed
    assert "intent_mismatch" in real_report.results[0].routing_failures
    monkeypatch.setattr(cli, "run_pipeline_eval", AsyncMock(return_value=real_report))
    output = tmp_path / "contract-failed.json"
    assert await cli.main(["--mode=live", "--output", str(output)]) == 1
    report = json.loads(output.read_text())
    assert report["contract_passed"] is False
    assert report["semantic_pass_rate"] == 1.0
    assert report["exit_code"] == 1


async def test_semantic_error_retains_points_and_cannot_pass_zero_threshold(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _live(monkeypatch, _verdict(False, error="TimeoutError"))
    monkeypatch.setattr(cli, "run_pipeline_eval", AsyncMock(return_value=_pipeline()))
    output = tmp_path / "judge-failed.json"
    assert await cli.main(["--mode=live", "--threshold=0", "--output", str(output)]) == 1
    report = json.loads(output.read_text())
    assert report["point_total"] == 1
    assert report["point_passed"] == 0
    assert report["semantic_results"][0]["verdict"]["error"] == "TimeoutError"


@pytest.mark.parametrize("failure", [RuntimeError("private-key"), TimeoutError("private-key")])
async def test_pipeline_exception_produces_safe_nonzero_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    failure: Exception,
) -> None:
    monkeypatch.setattr(cli, "run_pipeline_eval", AsyncMock(side_effect=failure))
    output = tmp_path / "error.json"
    assert await cli.main(["--output", str(output)]) == 1
    report = json.loads(output.read_text())
    assert report["error"] == type(failure).__name__
    captured = capsys.readouterr()
    assert "private-key" not in output.read_text() + captured.out + captured.err


async def test_private_report_and_same_model_limit_are_explicit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _live(monkeypatch)
    monkeypatch.setattr(cli, "run_pipeline_eval", AsyncMock(return_value=_pipeline()))
    output = tmp_path / "private.json"
    output.write_text("previous")
    output.chmod(0o644)
    assert await cli.main(["--mode=live", "--output", str(output)]) == 0
    assert output.stat().st_mode & 0o777 == 0o600
    report = json.loads(output.read_text())
    assert report["same_model_judging"] is True
    assert "fallible" in report["semantic_quality"]
    assert "synthetic fixtures only" in report["semantic_quality"]
    assert report["pipeline"]["results"][0]["observation"]["answer"] == "private-answer-fixture"
    captured = capsys.readouterr()
    for private in ("private-query-fixture", "private-answer-fixture", "private-evidence-fixture"):
        assert private not in captured.out + captured.err


@pytest.mark.parametrize("mode", ["recorded", "live"])
async def test_cli_restores_callers_logging_level(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    logging.disable(logging.ERROR)
    monkeypatch.setattr(cli, "run_pipeline_eval", AsyncMock(side_effect=RuntimeError("fixture")))
    if mode == "live":
        _live(monkeypatch)
    await cli.main([f"--mode={mode}"])
    assert logging.root.manager.disable == logging.ERROR


async def test_observed_llm_timeout_cancels_request_and_counts_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancelled = asyncio.Event()

    async def hang(*args: object, **kwargs: object) -> None:
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    inner = AsyncMock(spec=LLMServiceBase)
    inner.generate.side_effect = hang
    observed = cli.ObservedLLM(inner)
    wait_for = asyncio.wait_for

    async def fast_timeout(awaitable, *, timeout):
        assert timeout == 90
        return await wait_for(awaitable, timeout=0.001)

    monkeypatch.setattr(asyncio, "wait_for", fast_timeout)
    with pytest.raises(TimeoutError):
        await observed.generate([])
    assert cancelled.is_set()
    assert observed.calls == 1 and observed.errors == 1
    assert observed.models == set()


async def test_recovered_generation_error_cannot_hide_behind_successful_semantics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    providers, _ = _live(monkeypatch)
    inner = AsyncMock(spec=LLMServiceBase)
    inner.generate.side_effect = RuntimeError("private-generation-failure")
    generator = cli.ObservedLLM(inner)
    providers.side_effect = [cli.ObservedLLM(RecordedPipelineLLM()), generator]

    async def recovered_pipeline(model, cases):
        # Production may catch a provider error and return a safe fallback. The
        # CLI must still report that the measured model run was incomplete.
        assert model is generator and cases
        with suppress(RuntimeError):
            await model.generate([])
        return _pipeline()

    monkeypatch.setattr(cli, "run_pipeline_eval", recovered_pipeline)
    output = tmp_path / "generation-failed.json"
    assert await cli.main(["--mode=live", "--output", str(output)]) == 1
    report = json.loads(output.read_text())
    assert report["contract_passed"] is True
    assert report["semantic_pass_rate"] == 1.0
    assert report["generator"]["provider_errors"] == 1
    assert "private-generation-failure" not in output.read_text()


async def test_semantic_actions_preserve_ticket_ownership_and_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, grading = _live(monkeypatch)
    pipeline = _pipeline()
    obs = pipeline.results[0].observation
    obs.handoff_outcome = "queued"
    obs.ticket_ids = [41]
    obs.ticket_records = [{"id": 41, "session_id": 17, "user_id": 23, "status": "claimed"}]
    monkeypatch.setattr(cli, "run_pipeline_eval", AsyncMock(return_value=pipeline))

    assert await cli.main(["--mode=live"]) == 0

    sample = grading.await_args_list[-1].args[0]
    assert sample.actions == {
        "handoff_outcome": "queued",
        "ticket_ids": [41],
        "ticket_records": obs.ticket_records,
        "session_id": 17,
        "user_id": 23,
        "executed_tools": [],
        "tool_result": {},
    }
