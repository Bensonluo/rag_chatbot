"""Run the routing golden-set baseline (review 2026-09-26, #10).

Offline, deterministic, keyless — runs the golden dialogue cases through
the REAL compiled graph (recording LLM, lexical hybrid stand-in, hashed
FAQ embeddings; see routing_eval.py for the stand-in doctrine). Exit
code 1 when a --threshold is given and the pass rate falls below it, so
it can gate a release run alongside the retrieval baseline.

Usage:
    python scripts/run_routing_eval.py [--threshold 1.0] \
        [--cases PATH] [--json PATH]
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.evaluation.routing_eval import (  # noqa: E402
    format_routing_table,
    load_routing_cases,
    run_routing_baseline,
)

logger = logging.getLogger(__name__)


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run the routing golden-set baseline")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="minimum golden-set pass rate to pass; omit for report-only",
    )
    parser.add_argument("--cases", type=str, default=None, help="golden set path")
    parser.add_argument("--json", type=str, default=None, help="dump the report as JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cases = load_routing_cases(args.cases) if args.cases else None
    if cases is not None and not cases:
        logger.error("No routing cases loaded; nothing to run")
        return 2

    report = await run_routing_baseline(cases=cases)
    print(format_routing_table(report))

    if args.json:
        Path(args.json).write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Report written to %s", args.json)

    if args.threshold is not None and report["pass_rate"] < args.threshold:
        logger.error("pass rate %.3f below threshold %.3f", report["pass_rate"], args.threshold)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
