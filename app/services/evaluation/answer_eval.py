"""Given-context answer evaluation using semantic point coverage and forbidden text.

This evaluates generated replies against listed expected points. Point coverage
is not a faithfulness, retrieval-quality, resolution, or production-chat metric:
the judge does not check every claim against retrieved source evidence. The CLI
supplies curated FAQ context directly; callers may supply their own answer_fn.

Malformed verdicts and provider failures fail their case and retain every
required point in the denominator. Completed provider <think> blocks are removed
before scoring; all remaining verdict text must follow the strict JSON contract.
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
    generation_error: bool = False

    @property
    def passed(self) -> bool:
        return not (
            self.points_missed or self.forbidden_hits or self.judge_error or self.generation_error
        )


@dataclass
class AnswerEvalReport:
    """Aggregate answer-quality scores over a golden set run."""

    total: int = 0
    passed: int = 0
    judge_errors: int = 0
    point_total: int = 0
    point_passed: int = 0
    failures: list[AnswerCaseResult] = field(default_factory=list)
    generation_errors: int = 0

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def point_coverage(self) -> float:
        return self.point_passed / self.point_total if self.point_total else 0.0


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError("Duplicate JSON object key")
        payload[key] = value
    return payload


def _text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Expected a nonempty string")
    return value.strip()


def _texts(value: Any, *, required: bool = False) -> list[str]:
    if not isinstance(value, list) or (required and not value):
        raise ValueError("Expected a list of strings")
    strings = [_text(item) for item in value]
    if len(set(strings)) != len(strings):
        raise ValueError("Duplicate list entry")
    return strings


def _validate_cases(cases: list[AnswerCase]) -> None:
    if not isinstance(cases, list) or not cases:
        raise ValueError("Answer golden set must be a nonempty list")
    seen: set[str] = set()
    for index, case in enumerate(cases):
        try:
            if not isinstance(case, AnswerCase):
                raise ValueError("Expected AnswerCase")
            identifier = _text(case.id)
            if identifier in seen:
                raise ValueError("Duplicate case ID")
            seen.add(identifier)
            _text(case.query)
            _text(case.category)
            _texts(case.required_points, required=True)
            _texts(case.forbidden_patterns)
            _texts(case.faq_ids)
        except ValueError:
            raise ValueError(f"Invalid or duplicate answer case at index {index}") from None


def load_answer_cases(path: str | Path = ANSWER_GOLDEN_SET_FILE) -> list[AnswerCase]:
    """Load the entire validated set or raise ValueError; never skip failed rows.

    The legacy singular faq_id is accepted alongside faq_ids. No other scalar
    coercion or schema guessing is performed, and duplicate case IDs are errors.
    """
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, ValueError):
        raise ValueError("Answer golden set is unreadable or not valid JSON") from None
    if not isinstance(raw, list) or not raw:
        raise ValueError("Answer golden set must be a nonempty list")
    cases: list[AnswerCase] = []
    allowed = {
        "id",
        "query",
        "category",
        "required_points",
        "forbidden_patterns",
        "faq_ids",
        "faq_id",
    }
    for index, item in enumerate(raw):
        try:
            if not isinstance(item, dict) or set(item) - allowed:
                raise ValueError("Invalid case object")
            faqs = _texts(item.get("faq_ids", []))
            if "faq_id" in item:
                faqs = [_text(item["faq_id"]), *faqs]
            cases.append(
                AnswerCase(
                    id=_text(item["id"]),
                    query=_text(item["query"]),
                    category=_text(item["category"]),
                    required_points=_texts(item["required_points"], required=True),
                    forbidden_patterns=_texts(item.get("forbidden_patterns", [])),
                    faq_ids=faqs,
                )
            )
        except (KeyError, ValueError):
            raise ValueError(f"Invalid answer case at index {index}") from None
    _validate_cases(cases)
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
    """Accept a JSON object with exactly one boolean per expected point index.

    Completed provider reasoning blocks are tolerated for compatibility; prose,
    fences, arrays, duplicate/extra keys and nonboolean values are rejected.
    """
    if not isinstance(text, str) or type(point_count) is not int or point_count < 1:
        return None
    try:
        payload = json.loads(strip_think(text), object_pairs_hook=_unique_object)
    except ValueError:
        return None
    keys = {str(i) for i in range(point_count)}
    if not isinstance(payload, dict) or set(payload) != keys:
        return None
    if any(type(payload[key]) is not bool for key in keys):
        return None
    return [payload[str(i)] for i in range(point_count)]


async def run_answer_eval(
    answer_fn: Callable[[AnswerCase], Awaitable[str]],
    judge_llm: object,
    cases: list[AnswerCase] | None = None,
) -> AnswerEvalReport:
    """Score an answer-producing pipeline over the answer golden set.

    Args:
        answer_fn: Produces a reply; this evaluator scores listed point
            coverage, not evidence faithfulness or retrieval correctness.
        judge_llm: LLM service exposing ``generate(messages=..., ...)``
            whose response carries ``.content`` (``LLMServiceBase``).
        cases: Override the loaded golden set (tests / subsets).
    """
    if cases is None:
        cases = load_answer_cases()

    _validate_cases(cases)
    report = AnswerEvalReport(
        total=len(cases), point_total=sum(len(case.required_points) for case in cases)
    )
    for index, case in enumerate(cases):
        result = AnswerCaseResult(case=case, answer="", points_missed=list(case.required_points))
        try:
            raw_answer = await answer_fn(case)
            if not isinstance(raw_answer, str):
                raise ValueError("Invalid answer type")
            answer = strip_think(raw_answer)
            if not answer or "<think>" in answer or "</think>" in answer:
                raise ValueError("Empty or unfinished answer")
            result.answer = answer
        except Exception:
            # Provider exceptions may contain credentials or request payloads.
            # Cancellation/KeyboardInterrupt are BaseExceptions and propagate.
            logger.warning("Answer generation failed for case index %d", index)
            result.generation_error = True
            report.generation_errors += 1
        else:
            result.forbidden_hits = [p for p in case.forbidden_patterns if p in answer]
            try:
                response = await judge_llm.generate(  # type: ignore[attr-defined]
                    messages=build_judge_messages(case, answer),
                    max_tokens=_JUDGE_MAX_TOKENS,
                    temperature=0.0,
                )
                verdicts = parse_judge_json(response.content, len(case.required_points))
            except Exception:
                verdicts = None
            if verdicts is None:
                logger.warning(
                    "Judge failed or returned an invalid verdict for case index %d", index
                )
                result.judge_error = True
                report.judge_errors += 1
            else:
                result.points_passed = [
                    p for p, ok in zip(case.required_points, verdicts, strict=True) if ok
                ]
                result.points_missed = [
                    p for p, ok in zip(case.required_points, verdicts, strict=True) if not ok
                ]
                report.point_passed += len(result.points_passed)
        report.passed += int(result.passed)
        if not result.passed:
            report.failures.append(result)
    return report
