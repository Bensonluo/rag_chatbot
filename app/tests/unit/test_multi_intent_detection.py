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
from app.services.intent.multi_intent import classify_speech_act
from app.services.intent.rule_based import RuleBasedIntentDetector
from app.services.llm.base import LLMServiceBase
from app.services.slot_filling.slot_types import find_order_ids


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
        "不用退款订单A了，帮我看下物流",
        "无需退款，只查物流",
        "There is no need to refund order A",
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
    write_intents = {Intent.REFUND, Intent.RETURN, Intent.CANCEL}
    frames = [(rule.primary.intent, rule.speech_act)] + [
        (item.intent, item.speech_act) for item in rule.co_detected
    ]
    # A conditional clause may keep its write frame for admission (B3), but
    # it must never be presented as a current action request.
    assert all(
        speech_act == "conditional" for intent, speech_act in frames if intent in write_intents
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
        "特别想退款订单A",
        "别人帮我下的订单A要退款",
        "级别最高的会员也想退款订单B",
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


# ---------------------------------------------------------------------------
# Speech-act classification (admission layer P1)
# ---------------------------------------------------------------------------


class TestClassifySpeechAct:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("I already returned order ORD1001 because damaged", "historical"),
            ("我已经退过订单 ORD1001 了", "historical"),
            ("I refunded ORD1001 last week", "historical"),
            ("我已经申请过退款了", "historical"),
            ("Should I return or refund order ORD1001 because damaged?", "alternative"),
            ("退货还是换货哪个更划算", "alternative"),
            ("If the item is damaged, refund order ORD1001", "conditional"),
            ("如果符合条件就退款 ORD1001", "conditional"),
            ("What is your return policy?", "informational"),
            ("退款政策是什么", "informational"),
            ("Refund order ORD1001 because damaged", "action"),
            ("帮我退款 ORD1001", "action"),
            ("Track my package ORD1002", "action"),
        ],
    )
    def test_classify_speech_act(self, text: str, expected: str) -> None:
        assert classify_speech_act(text) == expected


@pytest.mark.asyncio
async def test_detection_carries_speech_act() -> None:
    result = await RuleBasedIntentDetector().detect_multi(
        "I already returned order ORD1001 because damaged; what is your return policy?"
    )
    assert result.speech_act == "historical"
    acts = {item.query: item.speech_act for item in result.co_detected}
    assert acts.get("what is your return policy?") == "informational"


# ---------------------------------------------------------------------------
# Shared-target expansion (admission layer P2)
# ---------------------------------------------------------------------------


def test_find_order_ids_returns_every_instance_in_order() -> None:
    assert find_order_ids("Refund orders ORD1001 and ORD1002 because damaged") == [
        "ORD1001",
        "ORD1002",
    ]
    assert find_order_ids("ORD1001 和 ORD1002 都要退款") == ["ORD1001", "ORD1002"]
    assert find_order_ids("refund ORD1001") == ["ORD1001"]


def test_bare_quantity_after_order_word_is_not_an_order_entity() -> None:
    # "order 3 items" captures "3": a quantity must not expand the
    # request into a second confirmation-gated write frame.
    assert find_order_ids("order 3 items and refund ORD1001") == ["ORD1001"]
    assert find_order_ids("I want to return order 12") == []
    # Purely numeric ids stay valid when long enough to be ids, and the
    # explicit 订单号 marker keeps accepting any form.
    assert find_order_ids("订单 12345 怎么退款") == ["12345"]
    assert find_order_ids("订单号 3 的退款进度") == ["3"]


@pytest.mark.asyncio
async def test_shared_verb_two_orders_expand_to_two_frames() -> None:
    query = "Refund orders ORD1001 and ORD1002 because damaged"
    result = await RuleBasedIntentDetector().detect_multi(query)
    frames = [(result.primary.intent.value, result.entity_index)] + [
        (item.intent.value, item.entity_index) for item in result.co_detected
    ]
    assert frames == [("refund", 0), ("refund", 1)]


# ---------------------------------------------------------------------------
# LLM speech-act passthrough (admission layer P3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_historical_frame_anchored_in_blocked_clause() -> None:
    query = "I already returned order ORD1001 because damaged; what is your return policy?"
    content = json.dumps(
        {
            "primary": "return",
            "confidence": 0.9,
            "primary_query": "I already returned order ORD1001 because damaged",
            "speech_act": "historical",
            "co_detected": [
                {
                    "intent": "policy",
                    "confidence": 0.9,
                    "query": "what is your return policy?",
                    "speech_act": "informational",
                }
            ],
        }
    )
    result = await LLMIntentDetector(model(content)).detect_multi(query)
    assert result.primary.intent is Intent.RETURN
    assert result.speech_act == "historical"
    assert result.co_detected[0].speech_act == "informational"


@pytest.mark.asyncio
async def test_llm_action_frame_expands_shared_orders() -> None:
    query = "Refund orders ORD1001 and ORD1002 because damaged"
    content = json.dumps(
        {
            "primary": "refund",
            "confidence": 0.9,
            "primary_query": query,
            "co_detected": [],
        }
    )
    result = await LLMIntentDetector(model(content)).detect_multi(query)
    assert result.entity_index == 0
    assert [(item.intent.value, item.entity_index) for item in result.co_detected] == [
        ("refund", 1)
    ]


@pytest.mark.asyncio
async def test_llm_invalid_speech_act_rejects_whole_verdict() -> None:
    content = json.dumps(
        {
            "primary": "refund",
            "confidence": 0.9,
            "primary_query": "refund order A",
            "speech_act": "wishful-thinking",
            "co_detected": [],
        }
    )
    result = await LLMIntentDetector(model(content)).detect_multi("refund order A")
    assert result.primary.intent is Intent.UNKNOWN
    assert (result.primary.metadata or {}).get("invalid_response") is True


@pytest.mark.parametrize(
    "query,expected",
    [
        ("退款是什么意思", "informational"),
        ("解释一下退货的含义", "informational"),
        ("取消是什么含义", "informational"),
        ("The email mentions a return for ORD1001", "informational"),
        ("客服说订单 ORD1001 可以退款", "informational"),
        ("Refund order ORD1001 that I bought last week", "action"),
        ("Please return the item I received yesterday", "action"),
        ("Refund ORD1001; I already have the receipt", "action"),
        ("我上周买的订单 ORD1001，请退款", "action"),
        ("我已经收到货了，请退款 ORD1001", "action"),
        ("If ORD1001 has already shipped, refund it", "conditional"),
        ("如果订单已经发货，请退款 ORD1001", "conditional"),
        ("I tracked order ORD1001 last week", "historical"),
        ("I already checked order ORD1001", "historical"),
        ("我上周查过物流，还没到", "historical"),
    ],
)
def test_admission_speech_act_depends_on_request_not_incidental_time(
    query: str, expected: str
) -> None:
    assert classify_speech_act(query) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        "退款是什么意思",
        "退货的含义是什么",
        "The email mentions a return for ORD1001; I never asked for it",
        "The agent suggested a refund for ORD1001",
    ],
)
async def test_definitions_and_unquoted_reports_never_become_current_writes(query: str) -> None:
    result = await RuleBasedIntentDetector().detect_multi(query)
    frames = [(result.primary.intent, result.speech_act)] + [
        (item.intent, item.speech_act) for item in result.co_detected
    ]
    assert all(
        act != "action"
        for intent, act in frames
        if intent in {Intent.REFUND, Intent.RETURN, Intent.CANCEL}
    )
    assert result.speech_act == "informational"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        "Do not refund order ORD1001; just track it",
        "Do not refund order ORD1001; track orders ORD1002 and ORD1003",
        "Track orders ORD1001 and ORD1002",
        "Track packages ORD1001 and ORD1002",
        "Track my parcels ORD1001 and ORD1002",
        "Check orders ORD1001 and ORD1002",
    ],
)
async def test_tracking_and_order_plurals_keep_read_tasks(query: str) -> None:
    result = await RuleBasedIntentDetector().detect_multi(query)
    expected = Intent.QUERY_ORDER if query.startswith("Check") else Intent.TRACK_SHIPPING
    count = 1 if query.endswith("track it") else 2
    assert [intent for intent, _ in requests(result, query)] == [expected] * count
    assert [result.entity_index, *(item.entity_index for item in result.co_detected)] == list(
        range(count)
    )
    assert result.speech_act == "action"


@pytest.mark.asyncio
@pytest.mark.parametrize("intent", ["policy", "faq", "complaint", "handoff", "cancel"])
async def test_shared_entities_expand_only_order_bound_tasks(intent: str) -> None:
    query = {
        "policy": "Return policy for ORD1001 and ORD1002",
        "faq": "What do ORD1001 and ORD1002 mean?",
        "complaint": "File a complaint about ORD1001 and ORD1002",
        "handoff": "Connect me to a human about ORD1001 and ORD1002",
        "cancel": "Cancel ORD1001 and ORD1002",
    }[intent]
    rule = await RuleBasedIntentDetector().detect_multi(query)
    llm = await LLMIntentDetector(model(verdict(intent, query))).detect_multi(query)
    assert rule.co_detected == []
    assert llm.co_detected == []
    assert llm.primary.intent.value == intent


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize(
    "query,span",
    [
        ("I already returned order ORD1001", "returned order ORD1001"),
        ("I refunded order ORD1001 last week", "refunded order ORD1001"),
        ("I already checked order ORD1001", "checked order ORD1001"),
        ("If ORD1001 has already shipped, refund it", "refund it"),
        ("If a refund is possible, refund ORD1001", "refund ORD1001"),
        ("如果可以退款，请退款 ORD1001", "退款 ORD1001"),
        ("The email mentions a return for ORD1001", "return for ORD1001"),
        ("退款是什么意思", "退款"),
    ],
)
async def test_model_action_cannot_escape_nonaction_owner_clause(
    query: str, span: str, explicit: bool
) -> None:
    data = json.loads(verdict("query_order" if "checked" in span else "refund", span))
    if explicit:
        data["speech_act"] = "action"
    result = await LLMIntentDetector(model(json.dumps(data))).detect_multi(query)
    # Legacy labels may safely inherit the non-action classification. An
    # explicitly declared action must never override the source evidence.
    assert result.primary.intent is Intent.UNKNOWN or result.speech_act != "action"


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["primary", "secondary", "unknown"])
@pytest.mark.parametrize("bad_label", [None, [], {}, ["action"], 1, True, "execute"])
async def test_all_malformed_speech_act_labels_fail_closed(field: str, bad_label: Any) -> None:
    query = "Refund order A; return my shoes"
    data = json.loads(
        verdict(
            "unknown" if field == "unknown" else "refund",
            "" if field == "unknown" else "Refund order A",
            []
            if field == "unknown"
            else [{"intent": "return", "confidence": 0.8, "query": "return my shoes"}],
        )
    )
    if field == "unknown":
        data["confidence"] = 0
    if field == "secondary":
        data["co_detected"][0]["speech_act"] = bad_label
    else:
        data["speech_act"] = bad_label
    result = await LLMIntentDetector(model(json.dumps(data))).detect_multi(query)
    assert result.primary.intent is Intent.UNKNOWN
    assert (result.primary.metadata or {}).get("invalid_response") is True
    assert not result.co_detected


def test_llm_prompt_has_one_model_frame_per_shared_span() -> None:
    prompt = LLMIntentDetector(model(""))._build_multi_intent_system_prompt()
    assert "one entry per order" not in prompt
    assert "one entry for the entire shared span" in prompt
    assert "parser" in prompt and "expands" in prompt


@pytest.mark.asyncio
async def test_model_duplicate_shared_span_is_rejected_not_expanded_twice() -> None:
    query = "Refund orders ORD1001 and ORD1002 because damaged"
    content = verdict("refund", query, [{"intent": "refund", "confidence": 0.8, "query": query}])
    result = await LLMIntentDetector(model(content)).detect_multi(query)
    assert result.primary.intent is Intent.UNKNOWN
    assert not result.co_detected


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit", [False, True])
async def test_optional_speech_act_compatibility_preserves_conditional_history_and_shared_reads(
    explicit: bool,
) -> None:
    spans = [
        ("refund", "If ORD1001 has already shipped, refund it", "conditional"),
        ("track_shipping", "I tracked ORD1002 last week", "historical"),
        ("track_shipping", "Track orders ORD1003 and ORD1004", "action"),
    ]
    data = json.loads(
        verdict(
            spans[0][0],
            spans[0][1],
            [{"intent": intent, "query": span, "confidence": 0.8} for intent, span, _ in spans[1:]],
        )
    )
    if explicit:
        data["speech_act"] = spans[0][2]
        for item, (_, _, act) in zip(data["co_detected"], spans[1:], strict=True):
            item["speech_act"] = act
    query = "; ".join(span for _, span, _ in spans)
    result = await LLMIntentDetector(model(json.dumps(data))).detect_multi(query)
    assert result.speech_act == "conditional"
    assert [(item.speech_act, item.entity_index) for item in result.co_detected] == [
        ("historical", 0),
        ("action", 0),
        ("action", 1),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        "如果符合条件，请退款 ORD1001",
        "If a refund is possible, refund ORD1001",
        "如果订单已经发货，请退款 ORD1001",
    ],
)
async def test_rule_condition_owns_consequent_and_retains_conditional_marker(query: str) -> None:
    result = await RuleBasedIntentDetector().detect_multi(query)
    assert result.primary.intent is Intent.REFUND
    assert result.speech_act == "conditional"
    assert (result.primary_query or query) == query
    assert not result.co_detected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query,expected",
    [
        ("Refund ORD1001 that I bought last week", "action"),
        ("Refund my already returned item ORD1001", "action"),
        ("If ORD1001 was already returned, refund it", "conditional"),
        ("I received a refund for ORD1001 last week", "historical"),
        ("My return for ORD1001 was processed yesterday", "historical"),
        ("I asked for a refund for ORD1001 last week", "historical"),
    ],
)
async def test_task_history_and_current_request_chronology_in_both_detectors(
    query: str, expected: str
) -> None:
    assert classify_speech_act(query) == expected
    rule = await RuleBasedIntentDetector().detect_multi(query)
    assert rule.speech_act == expected
    llm = await LLMIntentDetector(model(verdict("refund", query))).detect_multi(query)
    assert llm.primary.intent is Intent.REFUND
    assert llm.speech_act == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("intent", ["refund", "return", "query_order", "track_shipping"])
async def test_llm_whole_input_shared_expansion_keeps_secondary_source_evidence(
    intent: str,
) -> None:
    query = {
        "refund": "Refund orders ORD1001 and ORD1002",
        "return": "Return orders ORD1001 and ORD1002",
        "query_order": "Check orders ORD1001 and ORD1002",
        "track_shipping": "Track orders ORD1001 and ORD1002",
    }[intent]
    result = await LLMIntentDetector(model(verdict(intent, ""))).detect_multi(query)
    assert result.primary.intent.value == intent
    assert len(result.co_detected) == 1
    assert result.co_detected[0].query == query
    assert result.co_detected[0].entity_index == 1


@pytest.mark.parametrize(
    "query",
    [
        "I want a refund for the item I already returned",
        "I need a refund for ORD1001 that I returned last week",
    ],
)
def test_current_noun_form_request_is_not_history(query: str) -> None:
    assert classify_speech_act(query) == "action"
