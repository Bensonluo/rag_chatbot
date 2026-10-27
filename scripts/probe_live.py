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

The default anonymous demo mode uses random positive 31-bit IDs. This is
best-effort isolation, not a reservation in the server database. For owned,
server-allocated sessions and owner-history readback, use --auth-token-env.
An optional --ticket-token-env enables a separate read-only staff queue
observation. No probe claims or resolves tickets. Successful reply contracts,
history readback and observed tickets are reported separately; none proves
that a human accepted the request or that every policy claim is correct.

Probes create real chat messages and handoff tickets on the target server.
Use a designated probe account/instance. No cleanup deletes customer records.

Usage:
    python scripts/probe_live.py                    # http://localhost:8000
    python scripts/probe_live.py --base-url https://host

Exit code 0 = all green, 1 = any FAIL (CI/cron friendly).
"""

import argparse
import asyncio
import contextlib
import math
import os
import re
import secrets
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, TypeGuard

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_BASE_URL = "http://localhost:8000"
TURN_TIMEOUT_SECONDS = 120.0  # LLM generation can take a while cold

# ── reply predicates ────────────────────────────────────────────────────────

# Same CJK window the server's detect_language uses (i18n.py [一-鿿]).
_CJK = re.compile(r"[一-鿿]")
_TICKET = re.compile(r"(?:工单号\s*#|ticket\s*#)\s*([0-9]+)", re.IGNORECASE)
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
_EN_REFUND_EVIDENCE = re.compile(
    r"1\s*[-–—]\s*3\s+business\s+days|one\s+to\s+three\s+business\s+days", re.IGNORECASE
)

# A check receives the full ChatResponse body and returns a failure
# reason, or None when the invariant holds.
Check = Callable[[dict[str, Any]], str | None]


def check(cond: bool, reason: str) -> str | None:
    return None if cond else reason


def _positive_id(value: object) -> TypeGuard[int]:
    return type(value) is int and 0 < value <= 2_147_483_647


def _content(reply: dict[str, Any]) -> str:
    value = reply.get("content")
    return value if isinstance(value, str) else ""


def _metadata(reply: dict[str, Any]) -> dict[str, Any]:
    value = reply.get("metadata")
    return value if isinstance(value, dict) else {}


def _ticket_numbers(content: str) -> list[int]:
    # Never convert an unbounded server-controlled digit string to an integer.
    return [int(n) if len(n) <= 10 else 0 for n in _TICKET.findall(content)]


def is_ticket(lang_ok_cjk_free: bool = False) -> Check:
    """Require a consistent server-reported queued ticket, not just its copy."""

    def _check(reply: dict[str, Any]) -> str | None:
        content, metadata = _content(reply), _metadata(reply)
        if lang_ok_cjk_free and has_cjk(content):
            return "CJK leaked into EN reply"
        if reply.get("intent") != "handoff" or metadata.get("handoff_outcome") != "queued":
            return "expected handoff intent and queued outcome"
        ticket_id = metadata.get("handoff_ticket_id")
        if not _positive_id(ticket_id):
            return "expected positive integer handoff ticket ID"
        numbers = _ticket_numbers(content)
        if not numbers or any(number != ticket_id for number in numbers):
            return "ticket acknowledgement ID does not match metadata"
        return None

    return _check


def _slot_contract(reply: dict[str, Any], intent: str, first_slot: str) -> str | None:
    metadata = _metadata(reply)
    if (
        has_ticket(_content(reply))
        or metadata.get("handoff_outcome")
        or metadata.get("handoff_ticket_id") is not None
    ):
        return "ticket reported on a slot-collection turn"
    pending = metadata.get("pending_slots")
    if (
        reply.get("intent") != intent
        or not isinstance(pending, list)
        or not pending
        or pending[0] != first_slot
        or not all(isinstance(slot, str) and slot for slot in pending)
    ):
        return "expected task intent and pending slot metadata"
    return None


def is_order_slot_prompt() -> Check:
    def _check(reply: dict[str, Any]) -> str | None:
        failure = _slot_contract(reply, "refund", "order_id")
        if failure:
            return failure
        return check(asks_for_order(_content(reply)), "expected an order-id slot prompt")

    return _check


def is_category_slot_prompt() -> Check:
    def _check(reply: dict[str, Any]) -> str | None:
        failure = _slot_contract(reply, "complaint", "category")
        if failure:
            return failure
        content = _content(reply)
        anchored = bool(re.search(r"投诉|complain", content, re.IGNORECASE))
        return check(
            bool(anchored and re.search(r"[？?]", content)),
            "expected a complaint-category slot prompt",
        )

    return _check


def _citation_shape(citation: dict[str, Any]) -> bool:
    """Check the public citation fields without importing the server runtime."""
    if citation.get("source_type") not in ("faq", "document", "graph"):
        return False
    for name in ("document_id", "version_id", "chunk_id", "faq_id", "title"):
        value = citation.get(name)
        if value is not None and (
            not isinstance(value, str) or len(value) > (500 if name == "title" else 512)
        ):
            return False
    excerpt = citation.get("excerpt")
    if not isinstance(excerpt, str) or not excerpt.strip() or len(excerpt) > 800:
        return False
    for name in ("page_start", "page_end", "start_pos", "end_pos"):
        value = citation.get(name)
        if value is not None and (
            type(value) is not int or value < (1 if name.startswith("page") else 0)
        ):
            return False
    for start, end in (("page_start", "page_end"), ("start_pos", "end_pos")):
        if (
            citation.get(start) is not None
            and citation.get(end) is not None
            and citation[end] < citation[start]
        ):
            return False
    for name in ("section_path", "source_document_ids", "source_version_ids", "source_chunk_ids"):
        values = citation.get(name, [])
        if (
            not isinstance(values, list)
            or len(values) > (20 if name == "section_path" else 50)
            or not all(isinstance(value, str) and len(value) <= 512 for value in values)
        ):
            return False
    return True


def _citation_sources(citation: dict[str, Any]) -> list[str]:
    kind = citation.get("source_type")
    if not isinstance(kind, str):
        return []
    version = citation.get("version_id")
    if kind in {"faq", "document"}:
        identity = citation.get("faq_id" if kind == "faq" else "document_id")
        if (
            isinstance(identity, str)
            and identity.strip()
            and isinstance(version, str)
            and version.strip()
        ):
            return [f"faq:{identity}" if kind == "faq" else identity]
    if kind == "graph":
        ids, versions = citation.get("source_document_ids"), citation.get("source_version_ids")
        if (
            isinstance(ids, list)
            and ids
            and isinstance(versions, list)
            and versions
            and all(isinstance(v, str) and v.strip() for v in [*ids, *versions])
        ):
            return ids
    return []


def is_knowledge_answer(
    evidence: re.Pattern[str] | None = None, *, cjk_free: bool = False
) -> Check:
    """Check declared evidence/anchors, not arbitrary natural-language entailment."""

    def _check(reply: dict[str, Any]) -> str | None:
        content = _content(reply)
        if not content.strip():
            return "empty answer"
        if cjk_free and has_cjk(content):
            return "CJK leaked into EN reply"
        if reply.get("intent") not in {"faq", "policy"}:
            return "expected knowledge intent"
        if asks_for_order(content) and not evidence:
            return "knowledge question answered with a slot prompt"
        if (
            _metadata(reply).get("handoff_outcome")
            or _metadata(reply).get("handoff_ticket_id") is not None
        ):
            return "unexpected handoff on knowledge turn"
        if evidence is not None and not evidence.search(content):
            return "answer lacks expected policy anchor"
        sources, citations = reply.get("sources"), reply.get("citations")
        if (
            not isinstance(sources, list)
            or not sources
            or not all(isinstance(v, str) and v.strip() for v in sources)
        ):
            return "knowledge answer lacks source identities"
        if not isinstance(citations, list) or not citations:
            return "knowledge answer lacks citations"
        anchored = False
        for citation in citations:
            if not isinstance(citation, dict) or not _citation_shape(citation):
                return "malformed knowledge citation"
            excerpt = citation.get("excerpt")
            identities = _citation_sources(citation)
            if (
                not isinstance(excerpt, str)
                or not excerpt.strip()
                or not identities
                or not any(identity in sources for identity in identities)
            ):
                return "citation lacks usable identity, version, excerpt or matching source"
            if evidence is None or evidence.search(excerpt):
                anchored = True
        return check(anchored, "citation excerpt lacks expected policy anchor")

    return _check


def no_cjk() -> Check:
    def _check(reply: dict[str, Any]) -> str | None:
        content = _content(reply)
        return check(
            bool(content.strip()) and not has_cjk(content), "empty or CJK-containing EN reply"
        )

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
    session_id: int | None = None
    authenticated_session: bool = False
    owner_id: int | None = None
    history_verified: bool = False
    ticket_ids: list[int] = field(default_factory=list)
    verified_ticket_ids: list[int] = field(default_factory=list)
    ticket_evidence_requested: bool = False


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
                Turn("我要投诉", is_category_slot_prompt(), "complaint opens slot collection"),
                Turn("那我要人工", is_ticket(), "handoff escapes the gate"),
            ],
        ),
        ProbeCase(
            name="mid-collection-handoff-en",
            turns=[
                Turn(
                    "I want to file a complaint", is_category_slot_prompt(), "complaint opens slots"
                ),
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
            turns=[Turn("麻烦帮我退一下款", is_order_slot_prompt(), "polite split reaches refund")],
        ),
        ProbeCase(
            name="task-refund-guard",
            turns=[Turn("我要退款", is_order_slot_prompt(), "plain refund collects slots")],
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
                    is_knowledge_answer(_EN_POLICY_EVIDENCE, cjk_free=True),
                    "policy answered from KB",
                ),
                Turn(
                    "and how long do refunds take",
                    is_knowledge_answer(_EN_REFUND_EVIDENCE, cjk_free=True),
                    "refund timing has evidence and stays EN",
                ),
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


RATE_LIMIT_RETRY_SECONDS = 15.0
RATE_LIMIT_MAX_RETRIES = 6
MAX_RETRY_AFTER_SECONDS = 60.0
TICKET_PAGE_SIZE = 200
TICKET_MAX_PAGES = 10


def _retry_delay(response: httpx.Response) -> float:
    value = response.headers.get("Retry-After", "")
    try:
        seconds = float(value)
    except ValueError:
        try:
            when = parsedate_to_datetime(value)
            if when.tzinfo is None:
                when = when.replace(tzinfo=UTC)
            seconds = (when - datetime.now(UTC)).total_seconds()
        except (ValueError, TypeError, OverflowError):
            seconds = RATE_LIMIT_RETRY_SECONDS
    if not math.isfinite(seconds) or seconds < 0:
        seconds = RATE_LIMIT_RETRY_SECONDS
    return min(seconds, MAX_RETRY_AFTER_SECONDS)


async def _post_turn(
    client: httpx.AsyncClient, turn_payload: dict[str, Any]
) -> httpx.Response | str:
    """Only retry explicit 429 rejections; other POST failures may have committed."""
    for attempt in range(RATE_LIMIT_MAX_RETRIES + 1):
        try:
            response = await client.post("/api/v1/chat", json=turn_payload)
        except httpx.HTTPError:
            return "transport error sending chat turn"
        if response.status_code != 429:
            return response
        if attempt < RATE_LIMIT_MAX_RETRIES:
            await asyncio.sleep(_retry_delay(response))
    return f"rate-limited (429) after {RATE_LIMIT_MAX_RETRIES} retries"


def _json_object(response: httpx.Response) -> dict[str, Any] | None:
    try:
        value = response.json()
    except (ValueError, UnicodeError):
        return None
    return value if isinstance(value, dict) else None


def _reply_contract(reply: dict[str, Any] | None, session_id: int) -> str | None:
    if reply is None:
        return "expected JSON object response"
    if not _positive_id(reply.get("session_id")) or reply["session_id"] != session_id:
        return "response session ID does not match request"
    if not isinstance(reply.get("content"), str) or not reply["content"].strip():
        return "expected nonempty string content"
    if not isinstance(reply.get("intent"), str) or not reply["intent"].strip():
        return "expected intent string"
    if not isinstance(reply.get("metadata"), dict):
        return "expected metadata object"
    return None


async def _request_object(
    client: httpx.AsyncClient, method: str, path: str, **kwargs: Any
) -> dict[str, Any] | str:
    try:
        response = await client.request(method, path, **kwargs)
    except httpx.HTTPError:
        return "transport error"
    if response.status_code != 200:
        return f"HTTP {response.status_code}"
    return _json_object(response) or "expected nonempty JSON object"


async def run_case(client: httpx.AsyncClient, case: ProbeCase, session_id: int) -> None:
    case.session_id = session_id
    expected_history = []
    for i, turn in enumerate(case.turns):
        outcome = await _post_turn(client, {"message": turn.send, "session_id": session_id})
        prefix = f"turn {i} ({turn.label})"
        if isinstance(outcome, str):
            case.failures.append(f"{prefix}: {outcome}")
            return
        if outcome.status_code != 200:
            case.failures.append(f"{prefix}: HTTP {outcome.status_code}")
            return
        reply = _json_object(outcome)
        failure = _reply_contract(reply, session_id)
        if failure:
            case.failures.append(f"{prefix}: {failure}")
            return
        assert reply is not None
        failure = turn.check(reply)
        if failure:
            case.failures.append(f"{prefix}: {failure}")
            return
        ticket_id = _metadata(reply).get("handoff_ticket_id")
        if _positive_id(ticket_id):
            case.ticket_ids.append(ticket_id)
        expected_history.extend(
            [
                ("user", turn.send, [], []),
                (
                    "assistant",
                    reply["content"],
                    reply.get("sources") or [],
                    reply.get("citations") or [],
                ),
            ]
        )
    if case.authenticated_session:
        history = await _request_object(
            client, "GET", "/api/v1/chat/history", params={"session_id": session_id, "limit": 100}
        )
        if isinstance(history, str):
            case.failures.append(f"owner history readback: {history}")
            return
        messages = history.get("messages")
        if (
            not _positive_id(history.get("session_id"))
            or history["session_id"] != session_id
            or not isinstance(messages, list)
            or not all(isinstance(message, dict) for message in messages)
        ):
            case.failures.append("owner history readback: malformed response or wrong session")
            return
        observed = [
            (
                message.get("role"),
                message.get("content"),
                message.get("sources") or [],
                message.get("citations") or [],
            )
            for message in messages
        ]
        if observed != expected_history:
            case.failures.append(
                "owner history readback: conversation or citations differ from sent/received turns"
            )
            return
        case.history_verified = True


async def _observe_tickets(client: httpx.AsyncClient, cases: list[ProbeCase]) -> None:
    """Bounded separate API read; no claim/resolve, and no absence assertion on 404."""
    owners = {
        (ticket_id, case.session_id): case.owner_id
        for case in cases
        for ticket_id in case.ticket_ids
    }
    pending = set(owners)
    observed: set[tuple[int, int | None]] = set()
    error = "ticket not observed within bounded staff queue scan"
    for status in ("open", "claimed", "resolved"):
        for page in range(TICKET_MAX_PAGES):
            if not pending:
                break
            result = await _request_object(
                client,
                "GET",
                "/api/v1/handoff/tickets",
                params={
                    "status": status,
                    "skip": page * TICKET_PAGE_SIZE,
                    "limit": TICKET_PAGE_SIZE,
                },
            )
            if isinstance(result, str):
                error = f"staff ticket observation: {result}"
                break
            rows = result.get("tickets")
            if (
                not isinstance(rows, list)
                or type(result.get("count")) is not int
                or result["count"] != len(rows)
                or len(rows) > TICKET_PAGE_SIZE
                or not all(
                    isinstance(row, dict)
                    and _positive_id(row.get("id"))
                    and _positive_id(row.get("session_id"))
                    and row.get("status") == status
                    for row in rows
                )
            ):
                error = "staff ticket observation: malformed queue page"
                break
            for row in rows:
                key = (row["id"], row["session_id"])
                if key in pending:
                    owner_id = owners[key]
                    if owner_id is not None and (
                        not _positive_id(row.get("user_id")) or row["user_id"] != owner_id
                    ):
                        error = "staff ticket observation: wrong ticket owner"
                        break
                    observed.add(key)
                    pending.remove(key)
            if error.startswith("staff ticket observation:"):
                break
            if len(rows) < TICKET_PAGE_SIZE:
                break
        else:
            continue
        if error.startswith("staff ticket observation:"):
            break
    for case in cases:
        case.verified_ticket_ids = [
            ticket_id for ticket_id in case.ticket_ids if (ticket_id, case.session_id) in observed
        ]
        if any((ticket_id, case.session_id) in pending for ticket_id in case.ticket_ids):
            case.failures.append(error)


async def run_probes(
    client: httpx.AsyncClient,
    turn_pace_seconds: float = 0.0,
    *,
    authenticated_sessions: bool = False,
    ticket_client: httpx.AsyncClient | None = None,
) -> list[ProbeCase]:
    """Sequential replay; authenticated sessions are allocated by the server."""
    cases = build_cases()
    for case in cases:
        case.authenticated_session = authenticated_sessions
        case.ticket_evidence_requested = ticket_client is not None
    owner_id = None
    if authenticated_sessions:
        owner = await _request_object(client, "GET", "/api/v1/auth/me")
        if isinstance(owner, str) or not _positive_id(owner.get("id")):
            for case in cases:
                case.failures.append("authenticated identity preflight failed")
            return cases
        owner_id = owner["id"]
    for case in cases:
        case.owner_id = owner_id
    used_ids: set[int] = set()
    for idx, case in enumerate(cases):
        if idx and turn_pace_seconds:
            await asyncio.sleep(turn_pace_seconds)
        if authenticated_sessions:
            session = await _request_object(
                client, "POST", "/api/v1/sessions", json={"title": f"Live probe: {case.name}"}
            )
            if isinstance(session, str):
                case.failures.append(f"session allocation: {session}")
                continue
            session_id = session.get("id")
            if (
                not _positive_id(session_id)
                or not _positive_id(session.get("user_id"))
                or session["user_id"] != owner_id
                or session_id in used_ids
            ):
                case.failures.append("session allocation: invalid ID, reused ID or wrong owner")
                continue
        else:
            session_id = secrets.randbelow(2_147_483_647) + 1
            while session_id in used_ids:
                session_id = secrets.randbelow(2_147_483_647) + 1
        used_ids.add(session_id)
        await run_case(client, case, session_id)
    if ticket_client is not None:
        await _observe_tickets(ticket_client, cases)
    return cases


def format_report(cases: list[ProbeCase]) -> str:
    lines = []
    for case in cases:
        if case.failures:
            for failure in case.failures:
                lines.append(f"FAIL [{case.name}] {failure}")
        else:
            lines.append(f"PASS [{case.name}]")
    failed = sum(1 for case in cases if case.failures)
    lines.extend(
        [
            "",
            f"{len(cases) - failed}/{len(cases)} cases passed",
            "PASS scope: reply contracts and seeded policy anchors; not full semantic correctness or human acceptance.",
        ]
    )
    if any(case.authenticated_session for case in cases):
        lines.append(
            f"Owned session history readback: {sum(case.history_verified for case in cases)}/{len(cases)} verified"
        )
    else:
        lines.append(
            "Anonymous demo IDs: best-effort isolation; server reservation/ownership NOT VERIFIED."
        )
    if any(case.ticket_evidence_requested for case in cases):
        total = sum(len(case.ticket_ids) for case in cases)
        verified = sum(len(case.verified_ticket_ids) for case in cases)
        lines.append(
            f"Read-only staff ticket observation: {verified}/{total} ticket/session pairs verified"
        )
    else:
        lines.append("Ticket persistence: NOT VERIFIED (no separate staff observer configured).")
    return "\n".join(lines)


def _finite_positive(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return number


def _finite_nonnegative(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return number


def _env_token(parser: argparse.ArgumentParser, name: str | None) -> str | None:
    if name is None:
        return None
    value = os.environ.get(name)
    if not value or not value.strip() or any(ord(char) < 33 or ord(char) > 126 for char in value):
        parser.error("token environment variable is missing, empty or invalid")
    return value


async def main(
    argv: list[str] | None = None,
    client: httpx.AsyncClient | None = None,
    *,
    ticket_client: httpx.AsyncClient | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=_finite_positive, default=TURN_TIMEOUT_SECONDS)
    parser.add_argument(
        "--pace", type=_finite_nonnegative, default=0.0, help="seconds between cases"
    )
    parser.add_argument(
        "--auth-token-env",
        help="environment variable containing a probe account access token; enables owned sessions and history readback",
    )
    parser.add_argument(
        "--ticket-token-env",
        help="environment variable containing a staff access token for separate read-only ticket observation",
    )
    parser.add_argument(
        "--require-ticket-evidence",
        action="store_true",
        help="fail configuration unless a separate staff observer is supplied",
    )
    args = parser.parse_args(argv)
    if client is not None and client is ticket_client:
        parser.error("chat and staff observer must use separate clients")
    auth_token = _env_token(parser, args.auth_token_env)
    staff_token = _env_token(parser, args.ticket_token_env)
    if args.require_ticket_evidence and staff_token is None and ticket_client is None:
        parser.error(
            "--require-ticket-evidence requires --ticket-token-env or an injected ticket_client"
        )
    try:
        url = httpx.URL(args.base_url)
        valid_url = (
            url.scheme in {"http", "https"}
            and bool(url.host)
            and not url.userinfo
            and not url.query
            and not url.fragment
        )
    except (httpx.InvalidURL, ValueError):
        valid_url = False
    if not valid_url:
        parser.error("base URL must be HTTP(S), with no credentials, query or fragment")
    async with contextlib.AsyncExitStack() as stack:
        if client is None:
            client = await stack.enter_async_context(
                httpx.AsyncClient(base_url=args.base_url, timeout=args.timeout)
            )
        if ticket_client is None and staff_token is not None:
            ticket_client = await stack.enter_async_context(
                httpx.AsyncClient(base_url=args.base_url, timeout=args.timeout)
            )
        # Restore headers on injected clients; credentials never enter reports.
        previous = [
            (c, c.headers.get("Authorization")) for c in (client, ticket_client) if c is not None
        ]
        try:
            if auth_token is not None:
                client.headers["Authorization"] = f"Bearer {auth_token}"
            if staff_token is not None and ticket_client is not None:
                ticket_client.headers["Authorization"] = f"Bearer {staff_token}"
            cases = await run_probes(
                client,
                turn_pace_seconds=args.pace,
                authenticated_sessions=bool(client.headers.get("Authorization")),
                ticket_client=ticket_client,
            )
        finally:
            for configured, old in previous:
                if old is None:
                    configured.headers.pop("Authorization", None)
                else:
                    configured.headers["Authorization"] = old
    print(format_report(cases))
    return 1 if any(case.failures for case in cases) else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
