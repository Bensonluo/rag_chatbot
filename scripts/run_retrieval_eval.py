"""Run the retrieval golden-set baseline (review 2026-09-26, #6/#10).

Offline, deterministic, keyless — the counterpart of the LLM-judged
answer eval. Grades the real BM25 keyword leg and the real RRF fusion
(with a lexical stand-in vector leg) over the seeded demo corpus, and
prints the tokenizer A/B and candidate-pool-width comparison the review
asked to quantify. Exit code 1 when a --threshold is given and the
fused arm's recall@3 falls below it, so it can gate a release run.

Usage:
    python scripts/run_retrieval_eval.py [--threshold 0.9] \
        [--cases PATH] [--json PATH]
"""

import argparse
import asyncio
import json
import logging
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.evaluation.retrieval_eval import (  # noqa: E402
    format_baseline_table,
    load_retrieval_cases,
    run_retrieval_baseline,
)

logger = logging.getLogger(__name__)


def _threshold(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("threshold must be finite and between 0 and 1")
    return number


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run the retrieval golden-set baseline")
    parser.add_argument(
        "--threshold",
        type=_threshold,
        default=None,
        help="minimum fused-arm recall@3 to pass; omit for report-only",
    )
    parser.add_argument("--cases", type=str, default=None, help="golden set path")
    parser.add_argument("--json", type=str, default=None, help="dump the report as JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        # Load defaults explicitly so an empty bundled set cannot bypass the gate.
        cases = load_retrieval_cases(args.cases)
        if not cases:
            raise ValueError("Empty retrieval cases")
    except (OSError, ValueError, KeyError, TypeError):
        logger.error("Invalid or empty retrieval golden set; no evaluation run")
        return 2
    try:
        report = await run_retrieval_baseline(cases=cases)
        fused = report["arms"]["hybrid_rrf_lexical_vector"]["overall"]["recall@3"]
        if (
            report["cases"] != len(cases)
            or type(fused) not in (int, float)
            or not math.isfinite(fused)
            or not 0 <= fused <= 1
        ):
            raise ValueError("Invalid retrieval report")
        rendered = format_baseline_table(report)
        if args.json:
            Path(args.json).write_text(
                json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
            )
        print(rendered)
    except Exception:
        # Never echo exception strings: downstream failures may carry credentials.
        logger.error("Retrieval evaluation failed or produced an invalid report")
        return 1

    if args.threshold is not None and fused < args.threshold:
        logger.error("fused recall@3 %.3f below threshold %.3f", fused, args.threshold)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
