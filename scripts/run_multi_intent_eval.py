"""Compare detection separately from the offline compiled admission challenge.

    .venv/bin/python scripts/run_multi_intent_eval.py --json report.json

No credentials, model or network; graph tools are in-memory recorders only.
Exit 1 on any false write admission, stage, execution or graph error, retaining
JSON evidence. Recovery and clarification scores are not safety proxies.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.evaluation.multi_intent_eval import (  # noqa: E402
    GOLDEN_PATH,
    NEGATIVE_PATH,
    load_cases,
    load_negative_cases,
    run_evaluation,
    run_graph_evaluation,
    run_negative_evaluation,
)
from app.services.intent.rule_based import RuleBasedIntentDetector  # noqa: E402


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=GOLDEN_PATH)
    parser.add_argument("--negative-cases", type=Path, default=NEGATIVE_PATH)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--mode", choices=("single", "multi", "both"), default="both")
    args = parser.parse_args()
    try:
        cases = load_cases(args.cases)
        challenge = load_negative_cases(args.negative_cases)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    detector = RuleBasedIntentDetector()
    reports = []
    modes = [False, True] if args.mode == "both" else [args.mode == "multi"]
    for multi in modes:
        report = await run_evaluation(detector, cases, multi=multi)
        reports.append(report)
        stats = report["overall"]
        print(f"{report['mode']}: {stats['cases']} synthetic cases")
        print(f"  Label recall: {stats['label_recall']:.2%}")
        print(f"  Label precision: {stats['label_precision']:.2%}")
        print(f"  Source-query request recall: {stats['request_recall']:.2%}")
        print(f"  Exact request sets: {stats['exact_request_rate']:.2%}")
        print(f"  Extra write-intent predictions: {stats['false_positive_writes']}")
    admission = await run_negative_evaluation(detector, challenge)
    lifecycle = await run_graph_evaluation(detector, challenge)
    gate = (
        lifecycle["safety_gate_passed"]
        and admission["metrics"]["candidate_false_write_admission"]["numerator"] == 0
    )
    artifact = {
        "schema_version": 2,
        "detector_comparison": reports,
        "admission_challenge": {"candidate_admission": admission, "compiled_graph": lifecycle},
        "safety_gate_passed": gate,
    }
    print(f"Compiled graph challenge: {lifecycle['cases']} synthetic cases")
    for name, metric in lifecycle["metrics"].items():
        rate = "not applicable" if metric["rate"] is None else f"{metric['rate']:.2%}"
        print(f"  {name}: {metric['numerator']}/{metric['denominator']} ({rate})")
    print(f"Safety gate: {'PASS' if gate else 'FAIL'}")
    if args.json:
        args.json.write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Saved: {args.json}")
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
