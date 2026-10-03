"""Deterministic routing golden-set eval over the REAL compiled graph.

Review 2026-09-26 #10 (answer-eval realism) recommended 回答评测调用实际
ChatService; the retrieval half closed offline (ff7748d, retrieval_eval),
but routing had no offline gate. The evidence that it needed one: three
routing bugs reached users live — EN "return policy" misrouted to the
RETURN task (283064d), reversed zh how-to phrasing misrouted to RETURN
(20201e9), and complaint intent lost the keyword tie to QUERY_ORDER by
enum order (f723874). Unit tests pin each rule; what stayed unpinned is
the COMPOSITION: intent arbitration + FAQ fast path + slot collection +
branch routing + grounding, end to end. This eval runs golden cases
through the real compiled graph so that class of regression fails CI
instead of a live user. Keyless, offline, deterministic — runnable in
CI exactly like the retrieval baseline.

What is REAL here: the compiled dialogue graph, a MemorySaver
checkpointer, rule-based intent detection, rule-based slot extraction,
the default demo tool registry, the default input guardrail, and the
FAQ fast path (a real FAQService over the shipped faqs.json).

What is a STAND-IN (same doctrine as retrieval_eval):

- LLM: a recording stub answering from the prompt scaffolding markers
  (工具执行结果 / 参考资料 / chitchat). Generation quality is a model
  value; the eval asserts grounding at the prompt layer — which context
  the real model would have been given.
- hybrid search: a lexical proxy over a seeded mini-corpus (production
  bigram tokenizer, overlap scoring). Vector-leg recall is a model
  value; the eval pins that RAG turns anchor on 参考资料 and the right
  seeded source.
- FAQ embeddings: hashed bigram bags. Exact variants score cosine 1.0,
  different topics stay far under threshold — hit/miss on exact
  variants is deterministic; semantic-match ranking is a model value.

Boundaries (each has its own test suite): query rewriting is disabled
here (LLM-dependent); L0/L1/L2 caches are not wired (eligibility gates
live in test_cache_eligibility.py); the hybrid intent leg is not wired
(the deterministic skeleton is exactly the layer the three live bugs
lived in); streaming transport equivalence is pinned by the real-graph
integration tests.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from langgraph.checkpoint.memory import MemorySaver

from app.config.settings import settings
from app.services.chat.chat_service import ChatService
from app.services.dialogue.graph import build_dialogue_graph
from app.services.dialogue.tools import create_default_tool_registry
from app.services.embeddings.base import EmbeddingResult
from app.services.faq.store import create_faq_service
from app.services.guardrails.base import GuardrailService
from app.services.guardrails.input_guard import DefaultInputGuardrail
from app.services.handoff.service import HandoffService
from app.services.intent.rule_based import RuleBasedIntentDetector
from app.services.llm.base import LLMMessage, LLMResponse, LLMServiceBase
from app.services.retrieval.hybrid_search import KeywordSearch

logger = logging.getLogger(__name__)

ROUTING_GOLDEN_SET_PATH = Path(__file__).parent / "routing_golden_set.json"

_HASH_DIMS = 4096

# Seeded mini-corpus for the lexical hybrid stand-in. Every RAG golden
# case anchors on one of these ids; retrieval realism itself is the
# retrieval baseline's job (retrieval_eval.py).
# Doc wording deliberately avoids the 政策/policy token: the no-evidence
# golden queries ("隐私政策是什么", "...policy on quantum entanglement")
# share exactly that token with a policy-titled doc, and the lexical
# stand-in would surface it — turning an honest KB miss into a fake hit.
_EVAL_CORPUS: list[dict[str, Any]] = [
    {
        "id": "demo_returns_policy_zh",
        "content": "退换须知：支持七天无理由退货，签收后 7 天内均可申请。",
    },
    {
        "id": "demo_shipping_policy_zh",
        "content": "运费与配送：普通订单满 99 元包邮，偏远地区不支持配送。",
    },
    {
        "id": "demo_member_points_zh",
        "content": "会员积分：购物 1 元积 1 分，积分可在下单时抵扣现金。",
    },
    {
        "id": "demo_returns_policy_en",
        "content": "Returns and exchanges: items may be returned within 7 days of receipt.",
    },
]

_EXPECTATION_KEYS = frozenset(
    {
        "intent",
        "executed_tools",
        "sources_contain",
        "content_contain",
        "content_not_contain",
        "llm_calls",
        "prompt_contain",
        "prompt_not_contain",
        "tickets_created",
        "pending_confirmation",
    }
)


@dataclass
class RoutingExpectation:
    """What one golden turn must produce. Every field is optional; an
    expectation must assert at least one outcome (loader enforces)."""

    intent: str | None = None
    executed_tools: list[str] | None = None
    sources_contain: list[str] = field(default_factory=list)
    content_contain: list[str] = field(default_factory=list)
    content_not_contain: list[str] = field(default_factory=list)
    llm_calls: int | None = None
    prompt_contain: list[str] = field(default_factory=list)
    prompt_not_contain: list[str] = field(default_factory=list)
    tickets_created: int | None = None
    pending_confirmation: bool | None = None

    def asserts_anything(self) -> bool:
        """True when at least one non-default outcome is pinned."""
        return any(
            [
                self.intent is not None,
                self.executed_tools is not None,
                self.sources_contain,
                self.content_contain,
                self.content_not_contain,
                self.llm_calls is not None,
                self.prompt_contain,
                self.prompt_not_contain,
                self.tickets_created is not None,
                self.pending_confirmation is not None,
            ]
        )


@dataclass
class RoutingTurn:
    """One user utterance and the outcome it must produce."""

    utterance: str
    expect: RoutingExpectation


@dataclass
class RoutingCase:
    """A golden scenario: one session (fresh thread), N turns."""

    id: str
    category: str
    turns: list[RoutingTurn]


def load_routing_cases(path: str | Path | None = None) -> list[RoutingCase]:
    """Load golden cases; strict schema — this file is code-reviewed."""
    file_path = Path(path) if path else ROUTING_GOLDEN_SET_PATH
    raw: list[dict[str, Any]] = json.loads(file_path.read_text(encoding="utf-8"))
    cases: list[RoutingCase] = []
    for item in raw:
        case_id = str(item["id"])
        turns: list[RoutingTurn] = []
        for turn_item in item["turns"]:
            expect_raw = dict(turn_item["expect"])
            unknown = set(expect_raw) - _EXPECTATION_KEYS
            if unknown:
                raise ValueError(f"{case_id}: unknown expectation keys {sorted(unknown)}")
            expect = RoutingExpectation(
                intent=expect_raw.get("intent"),
                executed_tools=expect_raw.get("executed_tools"),
                sources_contain=list(expect_raw.get("sources_contain", [])),
                content_contain=list(expect_raw.get("content_contain", [])),
                content_not_contain=list(expect_raw.get("content_not_contain", [])),
                llm_calls=expect_raw.get("llm_calls"),
                prompt_contain=list(expect_raw.get("prompt_contain", [])),
                prompt_not_contain=list(expect_raw.get("prompt_not_contain", [])),
                tickets_created=expect_raw.get("tickets_created"),
                pending_confirmation=expect_raw.get("pending_confirmation"),
            )
            if not expect.asserts_anything():
                raise ValueError(f"{case_id}: turn {turn_item['utterance']!r} asserts no outcome")
            turns.append(RoutingTurn(utterance=str(turn_item["utterance"]), expect=expect))
        cases.append(RoutingCase(id=case_id, category=str(item["category"]), turns=turns))
    return cases


class _RecordingLLM(LLMServiceBase):
    """Context-aware stub: records every generation prompt so the eval
    can assert what actually reached the model, and answers per context
    the same way the real prompt scaffolding distinguishes them."""

    def __init__(self) -> None:
        super().__init__(api_key="test", model="test")
        self.prompts: list[str] = []

    def _answer(self, messages: list[LLMMessage]) -> str:
        prompt = messages[-1].content
        self.prompts.append(prompt)
        if "工具执行结果:" in prompt:
            return "您的订单已发出，预计明日送达。"
        if "参考资料:" in prompt:
            return "根据平台政策：支持七天无理由退货。"
        return "您好，有什么可以帮您？"

    async def generate(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,  # noqa: ARG002  # interface conformance
        temperature: float | None = None,  # noqa: ARG002  # interface conformance
        **kwargs: Any,  # noqa: ARG002  # interface conformance
    ) -> LLMResponse:
        return LLMResponse(content=self._answer(messages), model=self.model)

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,  # noqa: ARG002  # interface conformance
        temperature: float | None = None,  # noqa: ARG002  # interface conformance
        **kwargs: Any,  # noqa: ARG002  # interface conformance
    ) -> AsyncGenerator[str, None]:
        yield self._answer(messages)

    def estimate_tokens(self, text: str) -> int:
        return len(text)

    async def count_tokens(self, messages: list[LLMMessage]) -> int:
        return sum(len(m.content) for m in messages)


class _HashedBagEmbeddings:
    """Deterministic embedding stand-in for the FAQ matcher.

    Bigram-bag vectors hashed to a fixed width (blake2b, not Python's
    salted hash(): vectors must be reproducible across processes).
    Identical strings score cosine 1.0 — the eval only asserts FAQ
    hit/miss on exact variants, never the semantic ranker's ordering.
    """

    async def embed(self, texts: list[str]) -> EmbeddingResult:
        return EmbeddingResult(
            embeddings=[self._vector(text) for text in texts],
            model="hash-bigram-bag",
            dimensions=_HASH_DIMS,
            tokens_used=sum(len(text) for text in texts),
        )

    async def embed_single(self, text: str) -> list[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        terms = KeywordSearch()._extract_terms(text)
        vector = [0.0] * _HASH_DIMS
        for term, tf in terms.items():
            digest = hashlib.blake2b(term.encode("utf-8"), digest_size=2).digest()
            vector[int.from_bytes(digest, "big") % _HASH_DIMS] += float(tf)
        return vector


class _LexicalHybridSearch:
    """Single deterministic leg for the hybrid-search seam.

    Production bigram tokenizer, normalized term-overlap scoring, ties
    broken by document id — the lexical-stand-in doctrine of
    retrieval_eval.LexicalVectorClient, shaped as the object the
    dialogue graph actually calls. No hits means the no-evidence branch
    (9ada2c1) gets exercised with an empty, non-degraded retrieval.
    """

    def __init__(self, corpus: list[dict[str, Any]]) -> None:
        tokenizer = KeywordSearch()
        self._doc_terms = {
            doc["id"]: set(tokenizer._extract_terms(doc["content"])) for doc in corpus
        }
        self._contents = {doc["id"]: doc["content"] for doc in corpus}

    async def search(self, request: Any) -> list[SimpleNamespace]:
        tokenizer = KeywordSearch()
        query_terms = set(tokenizer._extract_terms(getattr(request, "query", "")))
        if not query_terms:
            return []
        scored = sorted(
            (
                (len(query_terms & terms), doc_id)
                for doc_id, terms in self._doc_terms.items()
                if query_terms & terms
            ),
            key=lambda pair: (-pair[0], pair[1]),
        )
        return [
            SimpleNamespace(
                document_id=doc_id,
                content=self._contents[doc_id],
                score=float(overlap),
                metadata=None,
            )
            for overlap, doc_id in scored[: getattr(request, "top_k", 3)]
        ]


class _RecordingHandoffService(HandoffService):
    """Deterministic ticket creator that records escalation context.

    Subclasses the real service (typed seam) but overrides the entire
    chat-path method — no session_maker call ever happens, so the
    placeholder factory below is never used.
    """

    def __init__(self) -> None:
        super().__init__(session_maker=_never_called_session_factory)
        self.calls: list[dict[str, Any]] = []

    async def create_ticket_for_session(
        self,
        session_id: int,
        user_id: int | None,
        reason: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "session_id": session_id,
                "user_id": user_id,
                "reason": reason,
                "context": context,
            }
        )
        return {
            "ticket_id": f"CP-EVAL-{len(self.calls)}",
            "queue_position": 3,
            "reused": False,
        }


def _never_called_session_factory() -> Any:
    raise AssertionError("the recording handoff stand-in never opens a DB session")


def build_routing_harness() -> tuple[ChatService, _RecordingLLM, _RecordingHandoffService]:
    """Real graph; only network-facing seams stand-in'd (see module docstring)."""
    llm = _RecordingLLM()
    handoff = _RecordingHandoffService()
    faq_service = create_faq_service(_HashedBagEmbeddings())
    graph = build_dialogue_graph(
        intent_detector=RuleBasedIntentDetector(),
        slot_filler=None,
        tool_registry=create_default_tool_registry(),
        retrieval_pipeline={"hybrid_search": _LexicalHybridSearch(_EVAL_CORPUS)},
        llm_service=llm,
        guardrail_service=GuardrailService(input_guard=DefaultInputGuardrail()),
        checkpointer=MemorySaver(),
        faq_service=faq_service,
        handoff_service=handoff,
    )
    return ChatService(graph=graph), llm, handoff


@dataclass
class _TurnObservation:
    """Everything a turn's expectation is checked against."""

    intent: str
    content: str
    sources: list[str]
    llm_calls: int
    executed_tool_names: list[str]
    tickets_created: int
    last_prompt: str | None
    pending_confirmation: bool


def _check_expectation(
    case_id: str, turn_index: int, utterance: str, expect: RoutingExpectation, obs: _TurnObservation
) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []

    def fail(check: str, expected: str, observed: str) -> None:
        failures.append(
            {
                "case": case_id,
                "turn": f"#{turn_index} {utterance!r}",
                "check": check,
                "expected": expected,
                "observed": observed,
            }
        )

    if expect.intent is not None and obs.intent != expect.intent:
        fail("intent", expect.intent, obs.intent)
    if expect.executed_tools is not None and obs.executed_tool_names != expect.executed_tools:
        fail("executed_tools", str(expect.executed_tools), str(obs.executed_tool_names))
    for anchor in expect.sources_contain:
        if not any(anchor in source for source in obs.sources):
            fail("sources_contain", anchor, str(obs.sources))
    for anchor in expect.content_contain:
        if anchor not in obs.content:
            fail("content_contain", anchor, obs.content[:200])
    for anchor in expect.content_not_contain:
        if anchor in obs.content:
            fail("content_not_contain", anchor, obs.content[:200])
    if expect.llm_calls is not None and obs.llm_calls != expect.llm_calls:
        fail("llm_calls", str(expect.llm_calls), str(obs.llm_calls))
    for anchor in expect.prompt_contain:
        if obs.last_prompt is None or anchor not in obs.last_prompt:
            fail("prompt_contain", anchor, (obs.last_prompt or "<no generation>")[:200])
    for anchor in expect.prompt_not_contain:
        if obs.last_prompt is not None and anchor in obs.last_prompt:
            fail("prompt_not_contain", anchor, obs.last_prompt[:200])
    if expect.tickets_created is not None and obs.tickets_created != expect.tickets_created:
        fail("tickets_created", str(expect.tickets_created), str(obs.tickets_created))
    if (
        expect.pending_confirmation is not None
        and obs.pending_confirmation != expect.pending_confirmation
    ):
        fail(
            "pending_confirmation",
            str(expect.pending_confirmation),
            str(obs.pending_confirmation),
        )
    return failures


async def run_routing_baseline(cases: list[RoutingCase] | None = None) -> dict[str, Any]:
    """Run every golden case through the real graph; report pass/fail."""
    if cases is None:
        cases = load_routing_cases()

    # Query rewriting is an LLM-dependent unit-tested behavior; routing
    # asserts the deterministic skeleton, so the rewriter stays off.
    previous_rewrite_flag = settings.QUERY_REWRITE_ENABLED
    settings.QUERY_REWRITE_ENABLED = False
    try:
        chat, llm, handoff = build_routing_harness()

        failures: list[dict[str, str]] = []
        categories: dict[str, dict[str, int]] = {}
        total_turns = 0
        for case_index, case in enumerate(cases, start=1):
            session_id = 10_000 + case_index  # fresh checkpointer thread per case
            category_stat = categories.setdefault(case.category, {"turns": 0, "passed": 0})
            for turn_index, turn in enumerate(case.turns, start=1):
                total_turns += 1
                category_stat["turns"] += 1

                prompts_before = len(llm.prompts)
                tickets_before = len(handoff.calls)
                response = await chat.process_message(session_id, turn.utterance, 1)
                snapshot = await chat.graph.aget_state(
                    {"configurable": {"thread_id": str(session_id)}}
                )
                state = cast(dict[str, Any], snapshot.values)
                prompts_this_turn = llm.prompts[prompts_before:]

                executed_entries = state.get("executed_tools") or []
                executed_names = [str(entry.get("tool")) for entry in executed_entries]
                obs = _TurnObservation(
                    intent=response.intent,
                    content=response.content,
                    sources=list(response.sources or []),
                    llm_calls=len(prompts_this_turn),
                    executed_tool_names=executed_names,
                    tickets_created=len(handoff.calls) - tickets_before,
                    last_prompt=prompts_this_turn[-1] if prompts_this_turn else None,
                    pending_confirmation=state.get("pending_confirmation") is not None,
                )
                turn_failures = _check_expectation(
                    case.id, turn_index, turn.utterance, turn.expect, obs
                )
                if turn_failures:
                    failures.extend(turn_failures)
                else:
                    category_stat["passed"] += 1
    finally:
        settings.QUERY_REWRITE_ENABLED = previous_rewrite_flag

    passed_turns = total_turns - len(failures)
    return {
        "total_cases": len(cases),
        "total_turns": total_turns,
        "passed_turns": passed_turns,
        "failed_turns": len(failures),
        "pass_rate": round(passed_turns / total_turns, 4) if total_turns else 0.0,
        "failures": failures,
        "categories": categories,
    }


def format_routing_table(report: dict[str, Any]) -> str:
    """One-screen summary, mirroring the retrieval baseline's shape."""
    lines = [
        "Routing golden-set baseline (real graph, deterministic stubs)",
        f"cases {report['total_cases']}  turns {report['total_turns']}  "
        f"passed {report['passed_turns']}  failed {report['failed_turns']}  "
        f"pass_rate {report['pass_rate']:.3f}",
    ]
    for category, stat in sorted(report["categories"].items()):
        lines.append(f"  {category:<20} {stat['passed']}/{stat['turns']}")
    for failure in report["failures"]:
        lines.append(
            f"FAIL {failure['case']} {failure['turn']}: {failure['check']} "
            f"expected={failure['expected']!r} observed={failure['observed']!r}"
        )
    return "\n".join(lines)
