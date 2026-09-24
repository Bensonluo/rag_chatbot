"""Deterministic policy-claim verification (Phase A2).

The LLM proposes; this gate verifies — never the reverse. Three failure
modes from the Air Canada / Klarna postmortems are checked mechanically:

1. **numeric_mismatch (durations)** — a duration claim (SLA days,
   shipping hours, return windows) that no eligible fact can entail.
2. **numeric_mismatch (money)** — a monetary claim (运费险 payout caps,
   auto-refund thresholds, 基础运费 ranges) no eligible fact can back.
   The Air Canada ruling made hallucinated compensation amounts a legal
   liability the company must honor, so money claims get the same
   deterministic gate; cap facts are enforced one-sidedly (only
   over-promising violates — under-claiming a cap is conservative and
   passes).
3. **unexecuted_action** — past-tense action assertions ("已为您办理退
   款") asserted without an executed tool behind them. This is the #1
   trust killer in commerce bots; the clause is softened to guidance.

Both numeric checks share one binding discipline: claims are checked per
comma-clause so mixed-subject sentences ("普通订单 48 小时发货，大促
72 小时") let each number pair with its own subject; subject keywords
restrict which channel's numbers apply (支付宝 1-3 vs 银行卡 3-7), and a
clause naming no subject is checked against the topic's default subject
only. A claim passes when ANY eligible topic group entails it; nothing
checkable → conservative pass (never reject what the table cannot
disprove) — order-specific amounts ("您的订单 200 元") name no money
topic and are never touched.

Violation handling follows "downgrade before you reject": the violating
clause is rewritten to the fact's grounded statement, so the user always
leaves with the correct number, never an apology dead-end.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.services.facts.fact_store import PolicyFact

# Sentences first (。！？；), then comma-clauses within them — each
# number is judged against its nearest subject context.
_SENTENCE_RE = re.compile(r"[^。！？!?；;\n]+[。！？!?；;\n]?")
_CLAUSE_RE = re.compile(r"[^，,、]+[，,、]?")

# "3 个工作日", "1-3 个工作日", "10—30 分钟", "7 到 15 天", "48小时".
# 工作日 must be tried before 日 in the alternation.
_CLAIM_RE = re.compile(r"(\d+)(?:\s*[-–~到至]\s*(\d+))?\s*(?:个)?\s*(工作日|小时|分钟|天|日)")

# "25 元", "6-12 元", "赔付 24.5 元", "10 块钱" — decimal amounts are
# first-class money claims (9.9-元 promos), so both bounds parse as floats.
_MONEY_CLAIM_RE = re.compile(r"(\d+(?:\.\d+)?)(?:\s*[-–~到至]\s*(\d+(?:\.\d+)?))?\s*(?:元|块)")

_UNIT_ALIASES = {"日": "天"}

# Past-tense completion assertions without an executed action behind
# them. Future phrasing ("确认后将为您办理退款") must NOT match.
_ACTION_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"已(?:经)?(?:为您)?(?:办理|操作|提交|完成|处理)(?:了)?(?:退款|退货|取消|赔付|换货)",
        r"(?:退款|退货|取消|赔付|换货)(?:申请)?已(?:经)?(?:为您)?(?:完成|办理|通过|处理|提交)",
        r"已(?:经)?(?:为您)?(?:退款|赔付)",
    )
)

ACTION_FALLBACK = (
    "相关业务操作请以订单页面的实际处理进度为准；如需办理，我可在确认后为您处理或转接人工客服。"
)


@dataclass(frozen=True)
class ClaimViolation:
    """One caught violation with its grounded replacement text."""

    clause: str
    reason: str  # numeric_mismatch | unexecuted_action
    grounded_statement: str


@dataclass(frozen=True)
class ClaimCheckResult:
    passed: bool
    violations: list[ClaimViolation] = field(default_factory=list)


def check_policy_claims(
    text: str, facts: list[PolicyFact], *, check_actions: bool = True
) -> ClaimCheckResult:
    """Verify every checkable claim in ``text`` against the subgraph.

    ``check_actions=False`` skips the unexecuted-action check — for
    callers that hold execution proof (e.g. the agent path's tool_trace)
    while still wanting numeric policy claims verified.
    """
    violations: list[ClaimViolation] = []
    for sentence_match in _SENTENCE_RE.finditer(text):
        sentence = sentence_match.group(0)
        for clause_match in _CLAUSE_RE.finditer(sentence):
            _check_clause(
                clause_match.group(0), sentence, facts, violations, check_actions=check_actions
            )
    return ClaimCheckResult(passed=not violations, violations=violations)


def apply_violations(text: str, result: ClaimCheckResult) -> str:
    """Rewrite violating clauses to their grounded statements (in place
    by string replacement, first occurrence each — clauses are unique
    enough in practice; a repeated clause keeps its later copies)."""
    rewritten = text
    for violation in result.violations:
        rewritten = rewritten.replace(violation.clause, violation.grounded_statement, 1)
    return rewritten


def _check_clause(
    clause: str,
    sentence: str,
    facts: list[PolicyFact],
    violations: list[ClaimViolation],
    *,
    check_actions: bool = True,
) -> None:
    for match in _CLAIM_RE.finditer(clause):
        lo = int(match.group(1))
        hi = int(match.group(2) or match.group(1))
        unit = _UNIT_ALIASES.get(match.group(3), match.group(3))
        _check_numeric_claim(clause, sentence, lo, hi, unit, facts, violations)
    for match in _MONEY_CLAIM_RE.finditer(clause):
        amount_lo = float(match.group(1))
        amount_hi = float(match.group(2) or match.group(1))
        _check_numeric_claim(clause, sentence, amount_lo, amount_hi, "元", facts, violations)
    if check_actions and any(pattern.search(clause) for pattern in _ACTION_PATTERNS):
        violations.append(
            ClaimViolation(
                clause=clause, reason="unexecuted_action", grounded_statement=ACTION_FALLBACK
            )
        )


def _check_numeric_claim(
    clause: str,
    sentence: str,
    lo: float,
    hi: float,
    unit: str,
    facts: list[PolicyFact],
    violations: list[ClaimViolation],
) -> None:
    """Check one numeric claim (duration or money) against eligible topic groups.

    Topic binding is clause-first: the number pairs with the keywords in
    its own comma-clause. When the clause carries no topic keyword (LLMs
    often front-load context: "已经安排发货啦，24 小时内发出"), binding
    falls back to the whole sentence — subject restriction stays at
    clause level so "大促…72 小时" still pairs with the peak subject.
    """

    def eligible_topics(scope: str) -> dict[str, list[PolicyFact]]:
        groups: dict[str, list[PolicyFact]] = {}
        for fact in facts:
            if fact.unit != unit or not fact.value:
                continue
            if not any(keyword in scope for keyword in fact.topic_keywords):
                continue
            groups.setdefault(fact.topic, []).append(fact)
        return groups

    groups = eligible_topics(clause) or eligible_topics(sentence)
    if not groups:
        return  # no eligible topic — cannot disprove, conservative pass

    checked_any = False
    for group in groups.values():
        subject_matched = [
            f for f in group if f.subject_keywords and any(k in clause for k in f.subject_keywords)
        ]
        candidates = subject_matched or [f for f in group if f.default_subject]
        if not candidates:
            continue
        checked_any = True
        if any(_entails(lo, hi, f) for f in candidates):
            return  # backed by an eligible subject — the claim stands

    if not checked_any:
        return  # nothing checkable after subject restriction

    # Ground the correction with every subject variant of the offending
    # topic so the user sees all channel options, not just one.
    topic = next(iter(groups))
    grounded = " ".join(f.statement for f in facts if f.topic == topic and f.statement)
    violations.append(
        ClaimViolation(clause=clause, reason="numeric_mismatch", grounded_statement=grounded)
    )


def _entails(lo: float, hi: float, fact: PolicyFact) -> bool:
    """True when the fact's policy semantics back the claim's range.

    Cap facts are one-sided: only the upper bound is enforced, because
    over-promising (运费险赔 80 元 when the cap is 25) is the Air Canada
    liability direction while under-claiming is merely conservative.
    """
    bounds = fact.value_bounds()
    if bounds is None:
        return False
    fact_lo, fact_hi = bounds
    if fact.kind == "cap":
        return hi <= fact_hi
    return fact_lo <= lo and hi <= fact_hi
