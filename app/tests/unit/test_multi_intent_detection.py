"""Request-scoped detection regressions; no live model or commerce calls."""

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from app.core.exceptions import ExternalServiceError
from app.models.enums.intent import Intent
from app.services.intent.base import IntentDetection, IntentDetector, IntentResult
from app.services.intent.hybrid import HybridIntentDetector
from app.services.intent.llm_based import LLMIntentDetector
from app.services.intent.rule_based import RuleBasedIntentDetector
from app.services.llm.base import LLMServiceBase


def model(content: str) -> Mock:
    service = Mock(spec=LLMServiceBase)
    service.generate = AsyncMock(return_value=SimpleNamespace(content=content))
    return service


def verdict(
    primary: str = "refund",
    primary_query: str = "Refund order A",
    co: list[object] | None = None,
) -> str:
    return json.dumps(
        {
            "primary": primary,
            "confidence": 0.9,
            "primary_query": primary_query,
            "co_detected": co or [],
        }
    )


def requests(result: IntentDetection, original: str) -> list[tuple[Intent, str]]:
    return [(result.primary.intent, result.primary_query or original)] + [
        (item.intent, item.query) for item in result.co_detected
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query,expected",
    [
        ("return policy", Intent.POLICY),
        ("how do I refund", Intent.FAQ),
        ("退款多久到账", Intent.FAQ),
        ("退货政策是什么", Intent.POLICY),
    ],
)
async def test_keyword_overlap_is_not_a_second_request(query: str, expected: Intent) -> None:
    result = await RuleBasedIntentDetector().detect_multi(query)
    assert result.primary.intent is expected
    assert result.co_detected == []
    assert result.primary_query == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query,expected",
    [
        ("Refund order A and return my shoes", [Intent.REFUND, Intent.RETURN]),
        ("Refund order A; refund order B", [Intent.REFUND, Intent.REFUND]),
        ("我要退款，另外帮我查物流", [Intent.REFUND, Intent.TRACK_SHIPPING]),
        ("我要退货，顺便问下运费怎么算", [Intent.RETURN, Intent.FAQ]),
        ("退货商品A，退货商品B", [Intent.RETURN, Intent.RETURN]),
    ],
)
async def test_independent_requests_keep_original_subspans(
    query: str, expected: list[Intent]
) -> None:
    result = await RuleBasedIntentDetector().detect_multi(query)
    found = requests(result, query)
    assert [intent for intent, _ in found] == expected
    assert all(span and span in query for _, span in found)
    assert len({span for _, span in found}) == len(found)
    assert result.primary_query != query


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        "Do not refund order A",
        "Don't return my shoes",
        "不要退款",
        "暂时不要退货",
        "Maybe refund order A later",
        'He said "refund order A"',
        "Explain the phrase 'return my shoes'",
        "如果可以，明天再退款",
    ],
)
async def test_negated_deferred_and_quoted_requests_are_not_executable(query: str) -> None:
    result = await RuleBasedIntentDetector().detect_multi(query)
    assert result.primary.intent not in {Intent.REFUND, Intent.RETURN, Intent.CANCEL}
    assert not result.co_detected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        "Refund order A, actually return my shoes",
        "我要退款，不对，我要退货",
    ],
)
async def test_correction_replaces_the_earlier_request(query: str) -> None:
    result = await RuleBasedIntentDetector().detect_multi(query)
    assert result.primary.intent is Intent.RETURN
    assert result.co_detected == []
    assert result.primary_query and result.primary_query in query


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        "订单 ORD1001 有质量问题，我要退款",
        "The order ORD1001 is damaged, refund please",
    ],
)
async def test_entity_context_before_action_stays_one_request(query: str) -> None:
    result = await RuleBasedIntentDetector().detect_multi(query)
    assert result.primary.intent is Intent.REFUND
    assert not result.co_detected
    assert (result.primary_query or query) == query


@pytest.mark.asyncio
async def test_unknown_slot_answer_is_retained_before_new_request() -> None:
    query = "damaged; also query order ORD1002"
    result = await RuleBasedIntentDetector().detect_multi(
        query,
        {
            "current_intent": "refund",
            "filled_slots": {"order_id": "ORD1001"},
            "pending_slots": ["reason"],
            "task_executed": False,
        },
    )
    assert result.primary.intent is Intent.UNKNOWN
    assert result.primary_query == "damaged"
    assert [(item.intent, item.query) for item in result.co_detected] == [
        (Intent.QUERY_ORDER, "query order ORD1002")
    ]


@pytest.mark.asyncio
async def test_entity_comma_is_not_a_clause_boundary() -> None:
    query = "Return my shoes, size large"
    result = await RuleBasedIntentDetector().detect_multi(query)
    assert result.primary.intent is Intent.RETURN
    assert result.primary_query == ""
    assert result.co_detected == []


@pytest.mark.asyncio
async def test_rule_clause_classifier_receives_context(monkeypatch: pytest.MonkeyPatch) -> None:
    detector = RuleBasedIntentDetector()
    original = detector.detect_with_confidence
    spy = AsyncMock(side_effect=original)
    monkeypatch.setattr(detector, "detect_with_confidence", spy)
    context = {"current_intent": "refund", "filled_slots": {"order_id": "A"}}
    await detector.detect_multi("Refund order A; return my shoes", context)
    assert spy.await_count >= 2
    assert all(call.args[1] is context for call in spy.await_args_list)


@pytest.mark.asyncio
async def test_default_detector_remains_single_compatible() -> None:
    class LegacyDetector(IntentDetector):
        async def detect(self, query: str, context: dict[str, Any] | None = None) -> Intent:
            return Intent.GENERAL_KNOWLEDGE

        async def detect_with_confidence(
            self, query: str, context: dict[str, Any] | None = None
        ) -> IntentResult:
            return IntentResult(Intent.GENERAL_KNOWLEDGE, 0.8)

    result = await LegacyDetector().detect_multi("What is gravity?")
    assert result.primary.intent is Intent.GENERAL_KNOWLEDGE
    assert result.primary_query == ""
    assert not result.co_detected


@pytest.mark.asyncio
async def test_detector_does_not_silently_truncate_requests() -> None:
    query = "; ".join(f"Refund order {letter}" for letter in "ABCDE")
    result = await RuleBasedIntentDetector().detect_multi(query)
    assert len(requests(result, query)) == 5


@pytest.mark.asyncio
async def test_llm_accepts_anchored_same_intent_requests_and_context() -> None:
    query = "Refund order A; refund order B"
    service = model(
        verdict(co=[{"intent": "refund", "query": "refund order B", "confidence": 0.8}])
    )
    detector = LLMIntentDetector(service)
    context = {"current_intent": "return", "filled_slots": {"order_id": "B"}}
    result = await detector.detect_multi(query, context)
    assert requests(result, query) == [
        (Intent.REFUND, "Refund order A"),
        (Intent.REFUND, "refund order B"),
    ]
    assert result.co_detected[0].method == "llm"
    assert service.generate.call_args is not None
    prompt = service.generate.call_args.kwargs["messages"][-1].content
    assert "return" in prompt and "order_id" in prompt and query in prompt


INVALID_RESPONSES = [
    "refund",
    '{"primary":"refund",',
    'prefix {"primary":"refund"}',
    "[]",
    "null",
    "42",
    '"refund"',
    verdict(primary="invented"),
    verdict(primary="unknown"),
    verdict(primary_query="Refund order Z"),
    verdict(co=["return"]),
    verdict(co=[{"intent": "invented", "query": "return my shoes", "confidence": 0.8}]),
    verdict(co=[{"intent": "return", "query": "", "confidence": 0.8}]),
    verdict(co=[{"intent": "return", "query": "RETURN MY SHOES", "confidence": 0.8}]),
    verdict(co=[{"intent": "refund", "query": "Refund order A", "confidence": 0.8}]),
    verdict().replace("0.9", "NaN"),
    verdict().replace("0.9", "Infinity"),
    verdict().replace("0.9", "true"),
    verdict().replace("0.9", '"0.9"'),
    verdict().replace("0.9", "1.5"),
    verdict().replace("0.9", "-0.1"),
    verdict() + " " * 20000,
    verdict().replace('"confidence": 0.9', '"confidence": 0.9, "confidence": 0.1'),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("content", INVALID_RESPONSES)
async def test_llm_invalid_json_fails_closed_without_substring_fallback(
    content: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    detector = LLMIntentDetector(model(content))
    monkeypatch.setattr(
        detector,
        "_parse_response",
        Mock(side_effect=AssertionError("legacy substring parser called")),
    )
    result = await detector.detect_multi("Refund order A; return my shoes")
    assert result.primary.intent is Intent.UNKNOWN
    assert result.primary.confidence == 0
    assert result.co_detected == []
    assert result.primary_query == ""
    assert result.primary.metadata is not None
    assert result.primary.metadata["invalid_response"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "label", [intent.value for intent in Intent if intent is not Intent.UNKNOWN]
)
async def test_llm_multi_preserves_all_enum_categories(label: str) -> None:
    query = {
        "refund": "Refund order A",
        "return": "Return my shoes",
        "query_order": "Check order A",
        "track_shipping": "Track my parcel",
        "complaint": "File a complaint",
        "confirm": "Confirm",
        "cancel": "Cancel",
        "handoff": "Connect me to a human",
    }.get(label, "What is gravity?")
    result = await LLMIntentDetector(model(verdict(label, query))).detect_multi(query)
    assert result.primary.intent.value == label


@pytest.mark.asyncio
async def test_llm_prompt_includes_general_world_knowledge() -> None:
    detector = LLMIntentDetector(model(verdict("general_knowledge", "What is gravity?")))
    assert "general_knowledge" in detector._build_multi_intent_system_prompt()
    result = await detector.detect_multi("What is gravity?")
    assert result.primary.intent is Intent.GENERAL_KNOWLEDGE


@pytest.mark.asyncio
async def test_hybrid_checks_mixed_request_despite_high_rule_confidence() -> None:
    query = "我要退款; what is gravity?"
    service = model(
        verdict(
            "refund",
            "我要退款",
            [{"intent": "general_knowledge", "query": "what is gravity?", "confidence": 0.9}],
        )
    )
    hybrid = HybridIntentDetector(RuleBasedIntentDetector(), LLMIntentDetector(service))
    context = {"current_intent": "refund"}
    result = await hybrid.detect_multi(query, context)
    service.generate.assert_awaited_once()
    assert requests(result, query) == [
        (Intent.REFUND, "我要退款"),
        (Intent.GENERAL_KNOWLEDGE, "what is gravity?"),
    ]


@pytest.mark.asyncio
async def test_hybrid_uses_one_complete_verdict_not_union() -> None:
    rules = RuleBasedIntentDetector()
    service = model(verdict("faq", "how do I refund"))
    result = await HybridIntentDetector(rules, LLMIntentDetector(service), 1.1).detect_multi(
        "how do I refund"
    )
    assert result.primary.intent is Intent.FAQ
    assert not result.co_detected


@pytest.mark.asyncio
async def test_hybrid_invalid_llm_cannot_restore_whole_query_write() -> None:
    service = model('{"primary":"refund",')
    result = await HybridIntentDetector(
        RuleBasedIntentDetector(), LLMIntentDetector(service)
    ).detect_multi("Do not refund order A")
    assert result.primary.intent not in {Intent.REFUND, Intent.RETURN, Intent.CANCEL}
    assert not result.co_detected


@pytest.mark.asyncio
async def test_hybrid_outage_preserves_scoped_rule_requests() -> None:
    service = model("")
    service.generate.side_effect = ExternalServiceError(service="test", message="offline")
    result = await HybridIntentDetector(
        RuleBasedIntentDetector(), LLMIntentDetector(service)
    ).detect_multi("我要退款; return my shoes")
    assert result.primary_query == "我要退款"
    assert [(item.intent, item.query) for item in result.co_detected] == [
        (Intent.RETURN, "return my shoes")
    ]
    assert result.primary.metadata is not None
    assert result.primary.metadata["fallback"] == "llm_unavailable"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query,span",
    [
        ("Do not refund order A", "refund order A"),
        ('He said "refund order A"', "refund order A"),
        ("Refund order A, actually return my shoes", "Refund order A"),
        ("return policy", "return"),
        ("how do I refund", "refund"),
    ],
)
async def test_llm_cannot_clip_negative_quoted_corrected_or_informational_scope(
    query: str, span: str
) -> None:
    result = await LLMIntentDetector(model(verdict("refund", span))).detect_multi(query)
    assert result.primary.intent is Intent.UNKNOWN
    assert not result.co_detected


@pytest.mark.asyncio
async def test_llm_anchor_uses_occurrence_not_same_text_in_another_clause() -> None:
    query = "Do not refund order A; refund order A"
    content = verdict(
        "refund",
        "refund order A",
        [{"intent": "refund", "query": "refund order A", "confidence": 0.8}],
    )
    result = await LLMIntentDetector(model(content)).detect_multi(query)
    assert result.primary.intent is Intent.UNKNOWN


@pytest.mark.asyncio
async def test_negation_covers_coordinated_actions() -> None:
    result = await RuleBasedIntentDetector().detect_multi(
        "Do not refund order A and return my shoes"
    )
    assert result.primary.intent is Intent.UNKNOWN
    assert not result.co_detected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        'He said "refund order A"',
        'Explain the phrase "return my shoes"',
        "She wrote “cancel my order”",
    ],
)
async def test_llm_full_quoted_mention_is_not_an_action(query: str) -> None:
    result = await LLMIntentDetector(model(verdict("refund", query))).detect_multi(query)
    assert result.primary.intent is Intent.UNKNOWN


@pytest.mark.asyncio
async def test_llm_all_context_fields_are_forwarded() -> None:
    service = model(verdict("faq", "How do I refund?"))
    context = {
        "current_intent": "return",
        "filled_slots": {"order_id": "A"},
        "previous_messages": [{"role": "user", "content": "Earlier distinct context"}],
    }
    await LLMIntentDetector(service).detect_multi("How do I refund?", context)
    assert service.generate.call_args is not None
    user_prompt = service.generate.call_args.kwargs["messages"][-1].content
    assert "Earlier distinct context" in user_prompt
    assert "order_id" in user_prompt


@pytest.mark.asyncio
async def test_hybrid_uses_confident_mixed_verdict_not_primary_score_alone() -> None:
    service = model(
        json.dumps(
            {
                "primary": "refund",
                "confidence": 0.8,
                "primary_query": "我要退款",
                "co_detected": [
                    {"intent": "general_knowledge", "query": "what is gravity?", "confidence": 0.8}
                ],
            }
        )
    )
    result = await HybridIntentDetector(
        RuleBasedIntentDetector(), LLMIntentDetector(service)
    ).detect_multi("我要退款; what is gravity?")
    assert [item.intent for item in result.co_detected] == [Intent.GENERAL_KNOWLEDGE]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query,expected",
    [
        ("我要退款并且查物流", [Intent.REFUND, Intent.TRACK_SHIPPING]),
        ("Refund order A, no, return my shoes", [Intent.RETURN]),
        ("Refund order A. return my shoes", [Intent.REFUND, Intent.RETURN]),
    ],
)
async def test_explicit_connectors_correction_and_lowercase_sentence(
    query: str, expected: list[Intent]
) -> None:
    result = await RuleBasedIntentDetector().detect_multi(query)
    assert [intent for intent, _ in requests(result, query)] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "clause",
    [
        "I am not asking for a refund for ORD1001 because damaged",
        "I'm not requesting a refund for ORD1001 because damaged",
        "I do not want a refund for ORD1001 because damaged",
        "Refund order ORD1001 because damaged unless it has shipped",
        "Refund order ORD1001 provided that it has not shipped",
        "What does refund order mean?",
        "What is a refund?",
        "Explain what refund means",
    ],
)
async def test_ordinary_nonrequests_are_guarded_by_both_detectors(clause: str) -> None:
    query = clause + "; what is your policy?"
    rule = await RuleBasedIntentDetector().detect_multi(query)
    assert all(
        intent not in {Intent.REFUND, Intent.RETURN, Intent.CANCEL}
        for intent, _ in requests(rule, query)
    )
    assert any(intent is Intent.POLICY for intent, _ in requests(rule, query))
    # A clipped action mention must inherit the owning clause's guards too.
    span = "refund" if "refund" in clause else "Refund"
    llm = await LLMIntentDetector(model(verdict("refund", span))).detect_multi(query)
    assert llm.primary.intent is Intent.UNKNOWN


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        "Refund order ORD1001 because it was not delivered",
        "Please refund order ORD1001 because damaged",
        "Can you refund order ORD1001 because damaged?",
        "I am asking for a refund for ORD1001 because damaged",
    ],
)
async def test_legitimate_refund_requests_survive_shared_guards(query: str) -> None:
    rule = await RuleBasedIntentDetector().detect_multi(query)
    assert rule.primary.intent is Intent.REFUND
    assert rule.primary_query == ""
    llm = await LLMIntentDetector(model(verdict("refund", query))).detect_multi(query)
    assert llm.primary.intent is Intent.REFUND
    assert llm.primary_query == query


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "first,second,first_label,second_label",
    [
        ("Check order ORD1001", "refund order ORD1002 because damaged", "query_order", "refund"),
        ("Refund order ORD1001", "refund order ORD1002", "refund", "refund"),
        ("Refund order ORD1001", "Connect me to a human", "refund", "handoff"),
    ],
)
async def test_llm_canonicalizes_reversed_model_order(
    first: str, second: str, first_label: str, second_label: str
) -> None:
    query = first + "; " + second
    result = await LLMIntentDetector(
        model(
            verdict(
                second_label,
                second,
                [{"intent": first_label, "query": first, "confidence": 0.8}],
            )
        )
    ).detect_multi(query)
    expected = [(Intent(first_label), first), (Intent(second_label), second)]
    if second_label == "handoff":
        expected.reverse()
    assert requests(result, query) == expected
    assert result.primary.confidence == (0.9 if second_label == "handoff" else 0.8)


@pytest.mark.asyncio
async def test_llm_identical_repeated_spans_use_disjoint_occurrences() -> None:
    span = "refund order ORD1001"
    query = span + "; " + span
    result = await LLMIntentDetector(
        model(
            verdict(
                "refund",
                span,
                [{"intent": "refund", "query": span, "confidence": 0.8}],
            )
        )
    ).detect_multi(query)
    assert requests(result, query) == [(Intent.REFUND, span), (Intent.REFUND, span)]


@pytest.mark.asyncio
async def test_llm_sort_uses_validated_occurrence_after_blocked_duplicate() -> None:
    query = "Do not refund order A; Check order B; refund order A"
    result = await LLMIntentDetector(
        model(
            verdict(
                "refund",
                "refund order A",
                [
                    {"intent": "query_order", "query": "Check order B", "confidence": 0.8},
                ],
            )
        )
    ).detect_multi(query)
    assert requests(result, query) == [
        (Intent.QUERY_ORDER, "Check order B"),
        (Intent.REFUND, "refund order A"),
    ]


@pytest.mark.asyncio
async def test_llm_promotes_explicit_handoff_from_secondary() -> None:
    query = "Refund order A; Connect me to a human; Check order B"
    result = await LLMIntentDetector(
        model(
            verdict(
                "refund",
                "Refund order A",
                [
                    {"intent": "query_order", "query": "Check order B", "confidence": 0.8},
                    {"intent": "handoff", "query": "Connect me to a human", "confidence": 0.7},
                ],
            )
        )
    ).detect_multi(query)
    assert requests(result, query) == [
        (Intent.HANDOFF, "Connect me to a human"),
        (Intent.REFUND, "Refund order A"),
        (Intent.QUERY_ORDER, "Check order B"),
    ]
    assert result.primary.confidence == 0.7
