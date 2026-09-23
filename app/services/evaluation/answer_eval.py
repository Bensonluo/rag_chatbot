"""LLM-as-judge answer-quality golden set (CI-optional quality tier).

The intent golden set (``runner.py``) gates *routing* deterministically.
This tier gates the answer itself: given a customer query and the correct
retrieved context, does the assistant produce a complete, policy-accurate
reply free of fabricated promises? That is the industry's "resolution
quality" gate (Zendesk/Intercom Fin resolution benchmarks, JD Yanhu and
Alibaba Xiaomi's quality-inspection pipelines) and it needs a judge LLM,
so it runs on demand with a configured key rather than in keyless CI.

Scoring model:

- ``required_points``: facts the answer must cover; judged by an LLM
  referee returning strict JSON, because point coverage is semantic.
- ``forbidden_patterns``: substrings that must never appear (fabricated
  SLAs, meta leaks like "as an AI"); checked deterministically, no judge
  needed — a substring hit is unambiguous.
- Fail-closed judge: if the judge output cannot be parsed, the case
  fails with ``judge_error`` set rather than silently passing.

Reasoning models (e.g. MiniMax-M2.7) emit ``<think>...</think>`` blocks
inside ``content``; ``strip_think`` removes them before any text is
scored, on both the answer and judge sides.
"""

import json
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.services.llm.base import LLMMessage

logger = logging.getLogger(__name__)

ANSWER_GOLDEN_SET_FILE = Path(__file__).parent / "answer_golden_set.json"

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

# Bound judge output so a runaway reasoning model cannot stall the gate.
# Reasoning models inline <think> blocks that count toward the budget,
# so this must cover chain-of-thought plus the small verdict JSON.
_JUDGE_MAX_TOKENS = 2048


def strip_think(text: str) -> str:
    """Remove ``<think>...</think>`` reasoning blocks a provider may emit.

    Providers like MiniMax-M2.7 inline chain-of-thought in ``content``;
    scoring must see only the final answer text.
    """
    return _THINK_RE.sub("", text).strip()


@dataclass
class AnswerCase:
    """One query with the facts a good answer must (not) contain.

    ``faq_ids`` optionally names curated FAQ entries whose answers are
    the ground-truth context, letting a runner fetch them by id instead
    of re-testing retrieval (retrieval has its own tier).
    """

    id: str
    query: str
    category: str
    required_points: list[str]
    forbidden_patterns: list[str] = field(default_factory=list)
    faq_ids: list[str] = field(default_factory=list)


@dataclass
class AnswerCaseResult:
    case: AnswerCase
    answer: str
    points_passed: list[str] = field(default_factory=list)
    points_missed: list[str] = field(default_factory=list)
    forbidden_hits: list[str] = field(default_factory=list)
    judge_error: bool = False

    @property
    def passed(self) -> bool:
        return not self.points_missed and not self.forbidden_hits and not self.judge_error


@dataclass
class AnswerEvalReport:
    """Aggregate answer-quality scores over a golden set run."""

    total: int = 0
    passed: int = 0
    judge_errors: int = 0
    point_total: int = 0
    point_passed: int = 0
    failures: list[AnswerCaseResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def point_coverage(self) -> float:
        return self.point_passed / self.point_total if self.point_total else 0.0


def load_answer_cases(path: str | Path = ANSWER_GOLDEN_SET_FILE) -> list[AnswerCase]:
    """Load answer cases, degrading to empty on any read/shape failure."""
    file_path = Path(path)
    try:
        raw: list[dict[str, Any]] = json.loads(file_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Answer golden set unreadable (%s): %s", file_path, exc)
        return []
    cases: list[AnswerCase] = []
    for index, item in enumerate(raw):
        try:
            points = [str(p) for p in item["required_points"]]
            if not points:
                raise ValueError("required_points is empty")
            raw_faqs = item.get("faq_ids", [])
            if item.get("faq_id") is not None:
                raw_faqs = [item["faq_id"], *raw_faqs]
            cases.append(
                AnswerCase(
                    id=str(item["id"]),
                    query=str(item["query"]),
                    category=str(item["category"]),
                    required_points=points,
                    forbidden_patterns=[str(p) for p in item.get("forbidden_patterns", [])],
                    faq_ids=[str(f) for f in raw_faqs],
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Skipping malformed answer case #%d: %s", index, exc)
    return cases


def build_judge_messages(case: AnswerCase, answer: str) -> list[LLMMessage]:
    """Build the referee prompt demanding strict per-point JSON verdicts."""
    lines = [
        "你是电商客服质检员。判断客服回复是否覆盖了每个必需要点。",
        "",
        f"【用户问题】{case.query}",
        "",
        "【政策要点】回答必须覆盖以下每一条:",
    ]
    for i, point in enumerate(case.required_points):
        lines.append(f"[{i}] {point}")
    lines += [
        "",
        "【客服回复】",
        answer,
        "",
        "判断标准: 要点被明确表达或语义等价表述即为覆盖; 未提及或表述矛盾为未覆盖。",
        '只输出 JSON 对象，不要输出任何其他文字: {"0": true, "1": false, ...}',
    ]
    return [LLMMessage(role="user", content="\n".join(lines))]


def parse_judge_json(text: str, point_count: int) -> list[bool] | None:
    """Parse the judge's verdict into per-point booleans.

    Returns None when no verdict can be recovered — the caller treats
    that as a judge error rather than guessing.
    """
    cleaned = strip_think(text)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        payload: dict[str, Any] = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None

    verdicts: list[bool] = []
    for i in range(point_count):
        value = payload.get(str(i))
        if not isinstance(value, bool):
            return None
        verdicts.append(value)
    return verdicts


async def run_answer_eval(
    answer_fn: Callable[[AnswerCase], Awaitable[str]],
    judge_llm: object,
    cases: list[AnswerCase] | None = None,
) -> AnswerEvalReport:
    """Score an answer-producing pipeline over the answer golden set.

    Args:
        answer_fn: Produces the assistant reply for a case (the system
            under test — full pipeline, or retrieval+LLM composition).
        judge_llm: LLM service exposing ``generate(messages=..., ...)``
            whose response carries ``.content`` (``LLMServiceBase``).
        cases: Override the loaded golden set (tests / subsets).
    """
    if cases is None:
        cases = load_answer_cases()

    report = AnswerEvalReport(total=len(cases))
    for case in cases:
        answer = strip_think(await answer_fn(case))

        forbidden_hits = [p for p in case.forbidden_patterns if p in answer]

        response = await judge_llm.generate(  # type: ignore[attr-defined]
            messages=build_judge_messages(case, answer),
            max_tokens=_JUDGE_MAX_TOKENS,
            temperature=0.0,
        )
        verdicts = parse_judge_json(response.content, len(case.required_points))

        if verdicts is None:
            logger.warning("Judge verdict unparseable for case %s", case.id)
            result = AnswerCaseResult(
                case=case, answer=answer, forbidden_hits=forbidden_hits, judge_error=True
            )
            report.judge_errors += 1
        else:
            result = AnswerCaseResult(
                case=case,
                answer=answer,
                forbidden_hits=forbidden_hits,
                points_passed=[
                    p for p, ok in zip(case.required_points, verdicts, strict=True) if ok
                ],
                points_missed=[
                    p for p, ok in zip(case.required_points, verdicts, strict=True) if not ok
                ],
            )
            report.point_total += len(case.required_points)
            report.point_passed += len(result.points_passed)

        report.passed += int(result.passed)
        if not result.passed:
            report.failures.append(result)
    return report
