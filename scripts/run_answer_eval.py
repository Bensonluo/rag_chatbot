"""Run the LLM answer-quality golden set against a configured LLM provider.

CI-optional tier: needs a real LLM key (set OPENAI_API_KEY/OPENAI_MODEL/
OPENAI_BASE_URL — any OpenAI-compatible provider, e.g. MiniMax via
``OPENAI_BASE_URL=https://api.minimaxi.com/v1``). Exit code 1 when the
pass rate is below the threshold, so it can gate a release run.

System under test: FAQ-context-grounded answer generation — the same
"retrieved policy text + LLM composition" shape the chat pipeline uses,
wired directly so no database or vector store is required.

Usage:
    python scripts/run_answer_eval.py [--threshold 0.8] [--cases PATH]
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.evaluation.answer_eval import (  # noqa: E402
    AnswerCase,
    load_answer_cases,
    run_answer_eval,
)
from app.services.faq.store import DEFAULT_FAQ_FILE, load_faq_entries  # noqa: E402
from app.services.llm.base import LLMMessage  # noqa: E402
from app.services.llm.factory import LLMFactory  # noqa: E402

logger = logging.getLogger(__name__)

GENERATION_MAX_TOKENS = 2048  # reasoning models spend part of this on <think>

SYSTEM_PROMPT = (
    "你是资深电商客服，用简体中文回答。要求：礼貌、简洁、可操作；"
    "只依据给定的政策内容回答，不编造政策条款、时限或承诺；"
    "政策未覆盖的内容如实说明并建议联系人工客服；不要提及你是AI或模型。"
)


def build_context_block(case: AnswerCase, faq_map: dict[str, str]) -> str:
    """Render the ground-truth policy context for a case."""
    blocks = []
    for faq_id in case.faq_ids:
        answer = faq_map.get(faq_id)
        if answer:
            blocks.append(answer)
        else:
            logger.warning("Case %s references unknown faq %s", case.id, faq_id)
    return "\n\n".join(blocks)


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run the LLM answer-quality gate")
    parser.add_argument("--threshold", type=float, default=0.8, help="pass-rate gate")
    parser.add_argument("--cases", type=str, default=None, help="golden set path")
    parser.add_argument("--provider", type=str, default="openai", help="LLM provider")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cases = load_answer_cases(args.cases) if args.cases else load_answer_cases()
    if not cases:
        logger.error("No answer cases loaded; nothing to run")
        return 2

    faq_map = {e.faq_id: e.answer for e in load_faq_entries(DEFAULT_FAQ_FILE)}
    llm = LLMFactory.create(provider=args.provider)

    async def answer_fn(case: AnswerCase) -> str:
        context = build_context_block(case, faq_map)
        prompt = f"顾客提问：{case.query}"
        if context:
            prompt += f"\n\n可依据的平台政策：\n{context}"
        else:
            prompt += "\n\n（本问题无政策条款，请按客服礼仪回答）"
        response = await llm.generate(
            messages=[
                LLMMessage(role="system", content=SYSTEM_PROMPT),
                LLMMessage(role="user", content=prompt),
            ],
            max_tokens=GENERATION_MAX_TOKENS,
            temperature=0.1,
        )
        return response.content

    report = await run_answer_eval(answer_fn, judge_llm=llm, cases=cases)

    print(
        f"\n=== Answer-quality gate: {report.passed}/{report.total} passed "
        f"(pass_rate={report.pass_rate:.1%}, point_coverage={report.point_coverage:.1%}, "
        f"judge_errors={report.judge_errors}) ==="
    )
    for failure in report.failures:
        print(f"\n[FAIL] {failure.case.id} ({failure.case.category})")
        print(f"  query: {failure.case.query}")
        if failure.points_missed:
            print(f"  missed points: {failure.points_missed}")
        if failure.forbidden_hits:
            print(f"  forbidden hits: {failure.forbidden_hits}")
        if failure.judge_error:
            print("  judge error: verdict unparseable")
        print(f"  answer: {failure.answer[:200]}{'…' if len(failure.answer) > 200 else ''}")

    if report.pass_rate < args.threshold:
        print(f"\nBELOW THRESHOLD: {report.pass_rate:.1%} < {args.threshold:.0%}")
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
