"""Compare single- and multi-intent detection on a synthetic, offline corpus.

    .venv/bin/python scripts/run_multi_intent_eval.py --json report.json

No credential lookup or network request is performed. These are request-level
classification metrics, not production prevalence or completed business actions.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.evaluation.multi_intent_eval import (  # noqa: E402
    GOLDEN_PATH,
    load_cases,
    run_evaluation,
)
from app.services.intent.rule_based import RuleBasedIntentDetector  # noqa: E402


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=GOLDEN_PATH)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--mode", choices=("single", "multi", "both"), default="both")
    args = parser.parse_args()
    try:
        cases = load_cases(args.cases)
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
    if args.json:
        args.json.write_text(
            json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Saved: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
