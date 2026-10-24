"""Live invariant probe — catch routing drift before users do.

Every live-found bug so far (2026-09-30 mid-collection handoff swallow,
2026-10-03 polite-split refund, EN zh-leak, detector tie families) was a
*deterministic* routing/vocabulary regression, each reproducible with a
handful of fixed turns. This script replays those families against a
running server (local dev or production) and asserts the invariants:

  1. mid-collection handoff: 我要投诉 → slot prompt; 那我要人工 → ticket
     (+ EN mirror, whose reply must be CJK-free)
  2. handoff phrasing battery: every HANDOFF vocabulary phrase creates
     a ticket, alone and under politeness perturbations
  3. polite-split refund: 麻烦帮我退一下款 enters the refund slot flow,
     never chitchat and never an unprompted ticket
  4. task guard: 我要退款 asks for the order, not a ticket
  5. knowledge turns answer from the KB, EN mirror stays CJK-free

Demo-mode chat accepts anonymous turns with a caller-chosen positive
session_id (see app/api/deps/authorization.py), so each case mints its
own session and cases stay isolated regardless of order. Turns run
strictly sequentially — the GLM free tier rejects concurrency.

Usage:
    python scripts/probe_live.py                    # http://localhost:8000
    python scripts/probe_live.py --base-url https://host

Exit code 0 = all green, 1 = any FAIL (CI/cron friendly).
"""

import argparse
import asyncio
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_BASE_URL = "http://localhost:8000"
TURN_TIMEOUT_SECONDS = 120.0  # LLM generation can take a while cold

# ── reply predicates ────────────────────────────────────────────────────────

# Same CJK window the server's detect_language uses (i18n.py [一-鿿]).
_CJK = re.compile(r"[一-鿿]")
_TICKET = re.compile(r"工单号\s*#|ticket\s*#", re.IGNORECASE)
_ORDER_ASK = re.compile(r"订单号|订单编号|order\s*(number|id|no)", re.IGNORECASE)


def has_cjk(text: str) -> bool:
    return bool(_CJK.search(text))


def has_ticket(text: str) -> bool:
    return bool(_TICKET.search(text))


def asks_for_order(text: str) -> bool:
    return bool(_ORDER_ASK.search(text))


# KB evidence anchors, from app/data/demo_kb/demo_corpus.json:
# demo_returns_policy_zh ("签收后 7 天内无理由退货") and its EN twin
# ("within 7 days of delivery"). Grounding these checks keeps a
# chitchat-shaped non-answer from passing on shape alone.
_ZH_POLICY_EVIDENCE = re.compile(r"7\s*天|七天")
_EN_POLICY_EVIDENCE = re.compile(r"7\s*days?|seven\s+days?", re.IGNORECASE)

# A check receives the full ChatResponse body and returns a failure
# reason, or None when the invariant holds.
Check = Callable[[dict[str, Any]], str | None]


def check(cond: bool, reason: str) -> str | None:
    return None if cond else reason


def is_ticket(lang_ok_cjk_free: bool = False) -> Check:
    """Reply must confirm a ticket (and optionally stay CJK-free)."""

    def _check(reply: dict[str, Any]) -> str | None:
        content = str(reply.get("content") or "")
        failure = check(has_ticket(content), f"expected ticket ack, got: {content[:120]!r}")
        if failure:
            return failure
        if lang_ok_cjk_free:
            return check(not has_cjk(content), f"CJK leaked into EN reply: {content[:120]!r}")
        return None

    return _check


def is_slot_prompt() -> Check:
    """Task turn must collect slots: asks for the order, no ticket yet."""

    def _check(reply: dict[str, Any]) -> str | None:
        content = str(reply.get("content") or "")
        if has_ticket(content):
            return f"ticket created on a slot-collection turn: {content[:120]!r}"
        if not asks_for_order(content):
            return f"expected an order-id slot prompt, got: {content[:120]!r}"
        return None

    return _check


def is_knowledge_answer(evidence: re.Pattern[str] | None = None) -> Check:
    """Knowledge turn must answer substantively.

    ``evidence`` anchors the answer to the seeded KB (e.g. the 7-day
    window in the returns policy) so a chitchat-shaped non-answer
    cannot pass on shape alone.
    """

    def _check(reply: dict[str, Any]) -> str | None:
        content = str(reply.get("content") or "")
        if not content.strip():
            return "empty answer"
        if asks_for_order(content):
            return f"knowledge question answered with a slot prompt: {content[:120]!r}"
        if evidence is not None and not evidence.search(content):
            return f"answer lacks KB evidence {evidence.pattern!r}: {content[:120]!r}"
        return None

    return _check


def no_cjk() -> Check:
    def _check(reply: dict[str, Any]) -> str | None:
        content = str(reply.get("content") or "")
        return check(not has_cjk(content), f"CJK leaked into EN reply: {content[:120]!r}")

    return _check


# ── probe model ─────────────────────────────────────────────────────────────


@dataclass
class Turn:
    send: str
    check: Check
    label: str = ""


@dataclass
class ProbeCase:
    name: str
    turns: list[Turn]
    failures: list[str] = field(default_factory=list)


HANDOFF_PHRASES_ZH = [
    "转人工",
    "请帮我转接人工客服",
    "找人工",
    "要人工",
    "给我人工",
    "真人客服在哪里",
]
HANDOFF_PHRASES_EN = [
    "I want to talk to a real person",
    "please connect me to an agent",
    "human agent please",
]
HANDOFF_PERTURBED = [
    "麻烦转人工吧",
    "转人工呢",
    "帮我转人工好吗",
    "谢谢，我要人工",
    "human agent please 😊",
    "can I talk to a real person please",
]


def build_cases() -> list[ProbeCase]:
    """The invariant families, each in its own fresh session."""
    cases: list[ProbeCase] = [
        ProbeCase(
            name="mid-collection-handoff-zh",
            turns=[
                Turn("我要投诉", is_slot_prompt(), "complaint opens slot collection"),
                Turn("那我要人工", is_ticket(), "handoff escapes the gate"),
            ],
        ),
        ProbeCase(
            name="mid-collection-handoff-en",
            turns=[
                Turn("I want to file a complaint", is_slot_prompt(), "complaint opens slots"),
                Turn(
                    "then get me a human agent",
                    is_ticket(lang_ok_cjk_free=True),
                    "handoff ticket, EN copy",
                ),
            ],
        ),
        ProbeCase(
            # The 2026-10-03 residual this repo just closed: the polite
            # split must reach the money-moving task flow deterministically.
            name="polite-split-refund",
            turns=[Turn("麻烦帮我退一下款", is_slot_prompt(), "polite split reaches refund")],
        ),
        ProbeCase(
            name="task-refund-guard",
            turns=[Turn("我要退款", is_slot_prompt(), "plain refund collects slots")],
        ),
        ProbeCase(
            name="knowledge-zh",
            turns=[
                Turn(
                    "退货政策是什么",
                    is_knowledge_answer(_ZH_POLICY_EVIDENCE),
                    "policy answered from KB",
                )
            ],
        ),
        ProbeCase(
            name="knowledge-en",
            turns=[
                Turn(
                    "what is the return policy",
                    is_knowledge_answer(_EN_POLICY_EVIDENCE),
                    "policy answered from KB",
                ),
                Turn("and how long do refunds take", no_cjk(), "EN follow-up stays EN"),
            ],
        ),
    ]
    for phrase in HANDOFF_PHRASES_ZH:
        cases.append(
            ProbeCase(
                name=f"handoff-zh:{phrase}",
                turns=[Turn(phrase, is_ticket(), "handoff phrasing battery")],
            )
        )
    for phrase in HANDOFF_PHRASES_EN:
        cases.append(
            ProbeCase(
                name=f"handoff-en:{phrase}",
                turns=[Turn(phrase, is_ticket(lang_ok_cjk_free=True), "handoff phrasing battery")],
            )
        )
    for phrase in HANDOFF_PERTURBED:
        cjk_free = not has_cjk(phrase)
        cases.append(
            ProbeCase(
                name=f"handoff-perturbed:{phrase}",
                turns=[Turn(phrase, is_ticket(cjk_free), "politeness perturbation")],
            )
        )
    return cases


async def run_case(client: httpx.AsyncClient, case: ProbeCase, session_id: int) -> None:
    for i, turn in enumerate(case.turns):
        try:
            response = await client.post(
                "/api/v1/chat",
                json={"message": turn.send, "session_id": session_id},
            )
        except httpx.HTTPError as e:
            case.failures.append(f"turn {i} ({turn.label}): transport error: {e}")
            return
        if response.status_code != 200:
            case.failures.append(
                f"turn {i} ({turn.label}): HTTP {response.status_code}: {response.text[:200]}"
            )
            return
        failure = turn.check(response.json())
        if failure:
            case.failures.append(f"turn {i} ({turn.label}): {failure}")
            return  # later turns depend on earlier state; stop the case


async def run_probes(client: httpx.AsyncClient) -> list[ProbeCase]:
    """Sequential execution: the GLM free tier forbids concurrency."""
    cases = build_cases()
    # Distinct positive session ids per run; demo mode never verifies
    # ownership, so a time-based id is enough for isolation.
    base_session = int(time.time())
    for idx, case in enumerate(cases):
        await run_case(client, case, base_session + idx)
    return cases


def format_report(cases: list[ProbeCase]) -> str:
    lines = []
    for case in cases:
        if case.failures:
            for failure in case.failures:
                lines.append(f"FAIL [{case.name}] {failure}")
        else:
            lines.append(f"PASS [{case.name}]")
    failed = sum(1 for c in cases if c.failures)
    lines.append("")
    lines.append(f"{len(cases) - failed}/{len(cases)} cases passed")
    return "\n".join(lines)


async def main(argv: list[str] | None = None, client: httpx.AsyncClient | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=float, default=TURN_TIMEOUT_SECONDS)
    args = parser.parse_args(argv)

    if client is not None:
        cases = await run_probes(client)
    else:
        async with httpx.AsyncClient(base_url=args.base_url, timeout=args.timeout) as owned:
            cases = await run_probes(owned)
    print(format_report(cases))
    return 1 if any(c.failures for c in cases) else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
