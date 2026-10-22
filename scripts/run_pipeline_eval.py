"""Run isolated production-graph contracts, optionally with live model grading.

Default recorded mode makes no model calls. Live mode runs synthetic judge controls
before evaluating graph answers. It never uses production databases or tools.
"""

import argparse
import asyncio
import json
import logging
import math
import os
import sys
from collections.abc import AsyncGenerator
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.evaluation.groundedness_eval import (  # noqa: E402
    GroundednessInput,
    judge_groundedness,
)
from app.services.evaluation.judge_calibration import calibration_samples  # noqa: E402
from app.services.evaluation.pipeline_eval import (  # noqa: E402
    load_pipeline_cases,
    run_pipeline_eval,
)
from app.services.llm.base import LLMMessage, LLMResponse, LLMServiceBase  # noqa: E402
from app.services.llm.factory import LLMFactory  # noqa: E402


class ObservedLLM(LLMServiceBase):
    """Bound calls and record actual model identities, never credentials/prompts."""

    def __init__(self, inner: LLMServiceBase) -> None:
        super().__init__(api_key="evaluation-wrapper", model="observed")
        self.inner = inner
        self.models: set[str] = set()
        self.calls = 0
        self.errors = 0

    async def generate(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.calls += 1
        try:
            result = await asyncio.wait_for(
                self.inner.generate(
                    messages, max_tokens=max_tokens or 2048, temperature=temperature, **kwargs
                ),
                timeout=90,
            )
        except Exception:
            self.errors += 1
            raise
        self.models.add(result.model)
        return result

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        yield (await self.generate(messages, max_tokens, temperature, **kwargs)).content

    async def count_tokens(self, messages: list[LLMMessage]) -> int:
        return await self.inner.count_tokens(messages)

    def estimate_tokens(self, text: str) -> int:
        return self.inner.estimate_tokens(text)


def _threshold(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("threshold must be finite and between 0 and 1")
    return number


def _provider(name: str) -> ObservedLLM:
    return ObservedLLM(
        LLMFactory.create_from_settings()
        if name == "settings"
        else LLMFactory.create(provider=name)
    )


def write_report(path: str, report: dict[str, Any]) -> None:
    """Reports contain synthetic prompts/answers; nevertheless keep them private."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        os.fchmod(handle.fileno(), 0o600)
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


async def main(argv: list[str] | None = None) -> int:
    previous_logging_level = logging.root.manager.disable
    try:
        return await _main(argv)
    finally:
        logging.disable(previous_logging_level)


async def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["recorded", "live"], default="recorded")
    parser.add_argument(
        "--provider", choices=["settings", *LLMFactory.SUPPORTED_PROVIDERS], default="settings"
    )
    parser.add_argument(
        "--judge-provider",
        choices=["settings", *LLMFactory.SUPPORTED_PROVIDERS],
        default="settings",
    )
    parser.add_argument("--threshold", type=_threshold, default=1.0)
    parser.add_argument("--cases", help="pipeline fixture JSON")
    parser.add_argument("--output", help="private JSON report path; includes synthetic transcripts")
    parser.add_argument("--calibration-only", action="store_true", help="live judge controls only")
    args = parser.parse_args(argv)
    if args.calibration_only and args.mode != "live":
        parser.error("--calibration-only requires --mode live")
    # Providers may log raw error payloads; emit only our own safe status/report.
    logging.disable(logging.CRITICAL)
    report: dict[str, Any] = {
        "mode": args.mode,
        "semantic_quality": "not_evaluated",
        "rubric_version": "2-language-consistency",
    }
    generator: ObservedLLM | None = None
    judge: ObservedLLM | None = None
    code = 1
    try:
        cases = load_pipeline_cases(args.cases) if args.cases else load_pipeline_cases()
        if args.mode == "live":
            judge = _provider(args.judge_provider)
            controls: list[dict[str, Any]] = []
            for control in calibration_samples():
                name = control.name
                verdict = await judge_groundedness(control.sample, judge)
                controls.append(
                    {
                        "id": name,
                        "expected_pass": control.expected_pass,
                        "expected_language_consistent": control.expected_language_consistent,
                        "verdict": asdict(verdict),
                        "matched": (
                            verdict.error is None
                            and verdict.passed == control.expected_pass
                            and verdict.language_consistent == control.expected_language_consistent
                        ),
                    }
                )
                print(
                    f"Judge control {name}: {'PASS' if controls[-1]['matched'] else 'FAIL'}",
                    file=sys.stderr,
                )
            report["judge_controls"] = controls
            report["judge_controls_passed"] = bool(controls) and all(
                item["matched"] for item in controls
            )
            if not report["judge_controls_passed"]:
                report["error"] = "judge_controls_failed; pipeline semantic gate not run"
                return _finish(args.output, report, generator, judge, 1)
            if args.calibration_only:
                report["scope"] = "judge_controls_only; no pipeline quality claim"
                return _finish(args.output, report, generator, judge, 0)
            generator = _provider(args.provider)
        pipeline = await asyncio.wait_for(run_pipeline_eval(generator, cases), timeout=600)
        report["pipeline"] = asdict(pipeline)
        report["contract_passed"] = pipeline.passed
        report["turns"] = pipeline.total_turns
        code = 0 if pipeline.passed else 1
        if judge is not None:
            semantics: list[dict[str, Any]] = []
            for result in pipeline.results:
                obs = result.observation
                sample = GroundednessInput(
                    query=obs.utterance,
                    answer=obs.answer,
                    evidence=obs.evidence,
                    actions={
                        "handoff_outcome": obs.handoff_outcome,
                        "ticket_ids": obs.ticket_ids,
                        "ticket_records": obs.ticket_records,
                        "session_id": obs.session_id,
                        "user_id": obs.user_id,
                        "executed_tools": obs.executed_tools,
                        "tool_result": obs.tool_result,
                    },
                    required_points=obs.required_points,
                    forbidden_promises=obs.forbidden_promises,
                    expect_abstention=obs.abstention,
                    expected_language=obs.language,
                )
                verdict = await judge_groundedness(sample, judge)
                semantics.append(
                    {
                        "case_id": obs.case_id,
                        "turn_index": obs.turn_index,
                        "verdict": asdict(verdict),
                        "passed": verdict.passed,
                    }
                )
                print(
                    f"Semantic {obs.case_id}/{obs.turn_index}: {'PASS' if verdict.passed else 'FAIL'}",
                    file=sys.stderr,
                )
            report["semantic_quality"] = "model_judged; fallible, synthetic fixtures only"
            report["semantic_results"] = semantics
            rate = sum(item["passed"] for item in semantics) / len(semantics) if semantics else 0
            report["semantic_pass_rate"] = rate
            report["point_total"] = sum(len(item["verdict"]["points"]) for item in semantics)
            report["point_passed"] = sum(sum(item["verdict"]["points"]) for item in semantics)
            if (
                not semantics
                or any(item["verdict"]["error"] for item in semantics)
                or rate < args.threshold
            ):
                code = 1
            if generator is not None and generator.errors:
                code = 1
    except Exception as exc:
        report["error"] = type(exc).__name__
        code = 1
    return _finish(args.output, report, generator, judge, code)


def _finish(
    output: str | None,
    report: dict[str, Any],
    generator: ObservedLLM | None,
    judge: ObservedLLM | None,
    code: int,
) -> int:
    report["exit_code"] = code
    for role, model in (("generator", generator), ("judge", judge)):
        if model is not None:
            report[role] = {
                "actual_models": sorted(model.models),
                "calls": model.calls,
                "provider_errors": model.errors,
            }
    if generator and judge:
        report["same_model_judging"] = bool(generator.models & judge.models)
    if output:
        write_report(output, report)
    summary = {
        key: value
        for key, value in report.items()
        if key not in {"pipeline", "semantic_results", "judge_controls"}
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
