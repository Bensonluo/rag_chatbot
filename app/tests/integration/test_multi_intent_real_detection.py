"""Compiled dialogue journeys using the real rule detector, not verdict fixtures."""

import time

import pytest

from app.services.intent.rule_based import RuleBasedIntentDetector
from app.tests.integration.test_multi_intent_queue import invoke, setup_graph


@pytest.mark.parametrize("fallback", [False, True])
async def test_hybrid_annotation_preserves_primary_admission_semantics(fallback: bool) -> None:
    from unittest.mock import AsyncMock, Mock, patch

    from app.core.exceptions import ExternalServiceError
    from app.models.enums.intent import Intent
    from app.services.intent.base import IntentDetection, IntentResult
    from app.services.intent.hybrid import HybridIntentDetector

    rule = RuleBasedIntentDetector()
    detection = IntentDetection(
        primary=IntentResult(Intent.REFUND, 0.8),
        primary_query="source",
        speech_act="conditional",
        entity_index=1,
    )
    llm = Mock()
    llm.detect_multi = AsyncMock(side_effect=ExternalServiceError(service="llm", message="outage"))
    with patch.object(rule, "detect_multi", AsyncMock(return_value=detection)):
        result = await HybridIntentDetector(
            rule, llm, confidence_threshold=0.9 if fallback else 0.7
        ).detect_multi("source")
    assert result.speech_act == "conditional"
    assert result.entity_index == 1


async def test_legacy_llm_subclass_keeps_custom_single_intent_contract() -> None:
    from typing import Any
    from unittest.mock import Mock

    from app.models.enums.intent import Intent
    from app.services.intent.base import IntentResult
    from app.services.intent.llm_based import LLMIntentDetector

    class CustomIntentLeg(LLMIntentDetector):
        async def detect_with_confidence(
            self, query: str, context: dict[str, Any] | None = None
        ) -> IntentResult:
            assert query == "headache" and context == {"custom": True}
            return IntentResult(Intent.GENERAL_KNOWLEDGE, 0.9)

    service = Mock()
    result = await CustomIntentLeg(service).detect_multi("headache", {"custom": True})
    assert result.primary.intent is Intent.GENERAL_KNOWLEDGE
    service.generate.assert_not_called()


async def test_descriptive_lead_in_is_context_not_a_discarded_request() -> None:
    query = "This thing arrived broken and I want to file a complaint: ORD1001 screen cracked"
    result = await RuleBasedIntentDetector().detect_multi(query)
    assert result.primary.intent.value == "complaint"
    assert (result.primary_query or query) == query
    assert not result.co_detected


async def test_real_two_order_reads_complete_in_source_order() -> None:
    graph, calls, _, _ = setup_graph(intent_detector=RuleBasedIntentDetector())
    result = await invoke(graph, "query order ORD1001; track my package ORD1002")
    assert [(intent, args["order_id"]) for intent, args in calls] == [
        ("query_order", "ORD1001"),
        ("track_shipping", "ORD1002"),
    ]
    assert not result["state_stack"]
    assert len(result["executed_tools"]) == 2


async def test_real_same_label_writes_require_two_confirmations() -> None:
    graph, calls, _, _ = setup_graph(intent_detector=RuleBasedIntentDetector())
    staged = await invoke(
        graph,
        "refund order ORD1001 because damaged; refund order ORD1002 because defective",
    )
    assert not calls
    assert staged["pending_confirmation"]["args"]["order_id"] == "ORD1001"
    second = await invoke(graph, "yes")
    assert [(intent, args["order_id"]) for intent, args in calls] == [("refund", "ORD1001")]
    assert second["pending_confirmation"]["args"]["order_id"] == "ORD1002"
    assert (
        staged["pending_confirmation"]["action_id"] != second["pending_confirmation"]["action_id"]
    )
    done = await invoke(graph, "yes")
    assert [args["order_id"] for _, args in calls] == ["ORD1001", "ORD1002"]
    assert not done["pending_confirmation"] and not done["state_stack"]


async def test_real_policy_question_never_stages_return() -> None:
    graph, calls, faq, _ = setup_graph(intent_detector=RuleBasedIntentDetector())
    result = await invoke(graph, "return policy")
    assert not calls and not result.get("pending_confirmation")
    faq.match.assert_awaited_once_with("return policy")


async def test_real_cancel_discards_second_write() -> None:
    graph, calls, _, _ = setup_graph(intent_detector=RuleBasedIntentDetector())
    await invoke(
        graph,
        "refund order ORD1001 because damaged; return order ORD1002 because defective",
    )
    result = await invoke(graph, "cancel")
    assert not result["pending_confirmation"]
    await invoke(graph, "yes")
    assert not calls


@pytest.mark.parametrize(
    "query",
    [
        "Do not refund order ORD1001; refund order ORD1002 because damaged",
        'He said "refund order ORD1001"; refund order ORD1002 because damaged',
        "Refund order ORD1001, actually refund order ORD1002 because damaged",
    ],
)
async def test_filtered_clause_cannot_supply_another_order(query: str) -> None:
    graph, calls, _, _ = setup_graph(intent_detector=RuleBasedIntentDetector())
    staged = await invoke(graph, query)
    assert staged["pending_confirmation"]["args"]["order_id"] == "ORD1002"
    assert staged["message"] == query
    assert not calls


async def test_new_order_does_not_complete_old_refund() -> None:
    graph, calls, _, _ = setup_graph(intent_detector=RuleBasedIntentDetector())
    await graph.aupdate_state(
        {"configurable": {"thread_id": "queue"}},
        {
            "intent": "refund",
            "filled_slots": {"order_id": "ORD1001"},
            "pending_slots": ["reason"],
            "task_executed": False,
        },
    )
    staged = await invoke(
        graph, "refund order ORD1002 because damaged; what is your return policy?"
    )
    assert staged["pending_confirmation"]["args"]["order_id"] == "ORD1002"
    assert not calls
    await invoke(graph, "yes")
    assert [args["order_id"] for _, args in calls] == ["ORD1002"]


@pytest.mark.parametrize(
    "query",
    [
        "I am not asking for a refund for ORD1001 because damaged; what is your policy?",
        "Refund order ORD1001 because damaged unless it has shipped; what is your policy?",
        "What does refund order mean?",
    ],
)
async def test_nonrequest_text_never_stages_a_write(query: str) -> None:
    graph, calls, _, _ = setup_graph(intent_detector=RuleBasedIntentDetector())
    result = await invoke(graph, query)
    assert not result.get("pending_confirmation")
    assert not calls


async def test_direct_knowledge_secondary_is_not_lost() -> None:
    graph, _, faq, llm = setup_graph(
        ("faq", "What is your policy?"), ("general_knowledge", "What is gravity?")
    )
    result = await invoke(graph)
    faq.match.assert_awaited_once_with("What is your policy?")
    llm.generate.assert_awaited_once()
    assert "What is gravity?" in str(llm.generate.await_args)
    assert "FAQ answer:" in result["response"]
    assert not result["state_stack"]


async def test_expiry_terminates_chain_before_unrelated_faq() -> None:
    graph, calls, _, _ = setup_graph(intent_detector=RuleBasedIntentDetector())
    staged = await invoke(
        graph, "refund order ORD1001 because damaged; return order ORD1002 because defective"
    )
    expired = dict(staged["pending_confirmation"])
    expired["staged_at"] = time.time() - 9999999
    await graph.aupdate_state(
        {"configurable": {"thread_id": "queue"}}, {"pending_confirmation": expired}
    )
    result = await invoke(graph, "yes")
    assert not result["pending_confirmation"]
    assert not result["active_request_id"]
    assert not [item for item in result["state_stack"] if item.get("queued_from") == "multi_intent"]
    unrelated = await invoke(graph, "what is your return policy?")
    assert not unrelated["pending_confirmation"]
    assert not calls


async def test_real_shared_verb_two_orders_both_confirmed() -> None:
    graph, calls, _, _ = setup_graph(intent_detector=RuleBasedIntentDetector())
    staged = await invoke(graph, "Refund orders ORD1001 and ORD1002 because damaged")
    assert staged["pending_confirmation"]["args"]["order_id"] == "ORD1001"
    second = await invoke(graph, "yes")
    assert second["pending_confirmation"]["args"]["order_id"] == "ORD1002"
    done = await invoke(graph, "yes")
    assert [args["order_id"] for _, args in calls] == ["ORD1001", "ORD1002"]
    assert not done["pending_confirmation"] and not done["state_stack"]


async def test_real_historical_plus_policy_answers_policy_without_task() -> None:
    graph, calls, faq, _ = setup_graph(intent_detector=RuleBasedIntentDetector())
    result = await invoke(
        graph,
        "I already returned order ORD1001 because damaged; what is your return policy?",
    )
    assert not result.get("pending_confirmation")
    assert not calls
    assert result["intent"] == "policy"


async def test_real_pure_history_asks_clarification_instead_of_staging() -> None:
    graph, calls, _, _ = setup_graph(intent_detector=RuleBasedIntentDetector())
    result = await invoke(graph, "I already returned order ORD1001 because damaged.")
    assert not result.get("pending_confirmation")
    assert not calls
    assert "check its status" in result["response"] or "查询" in result["response"]


async def test_real_alternative_consultation_never_stages_a_write() -> None:
    graph, calls, _, _ = setup_graph(intent_detector=RuleBasedIntentDetector())
    result = await invoke(graph, "Should I return or refund order ORD1001 because damaged?")
    assert not result.get("pending_confirmation")
    assert not calls
    assert result["intent"] not in {"refund", "return"}
    assert result["admission_clarify"]


async def test_real_conditional_write_offers_verification_first() -> None:
    graph, calls, _, _ = setup_graph(intent_detector=RuleBasedIntentDetector())
    result = await invoke(graph, "If the item is damaged, refund order ORD1001 because defective")
    assert not result.get("pending_confirmation")
    assert not calls
    assert "verify" in result["response"] or "核实" in result["response"]


async def test_real_llm_outage_historical_never_stages_a_degraded_write() -> None:
    from unittest.mock import AsyncMock, Mock

    from app.core.exceptions import ExternalServiceError
    from app.services.intent.hybrid import HybridIntentDetector
    from app.services.intent.llm_based import LLMIntentDetector

    service = Mock(spec=["generate"])
    service.generate = AsyncMock(
        side_effect=ExternalServiceError(service="llm", message="simulated outage")
    )
    detector = HybridIntentDetector(
        RuleBasedIntentDetector(), LLMIntentDetector(service), confidence_threshold=0.7
    )
    graph, calls, _, _ = setup_graph(intent_detector=detector)
    result = await invoke(
        graph,
        "I already returned order ORD1001 because damaged; what is your return policy?",
    )
    assert not result.get("pending_confirmation")
    assert not calls


async def test_real_llm_outage_explicit_imperative_refund_still_stages() -> None:
    from unittest.mock import AsyncMock, Mock

    from app.core.exceptions import ExternalServiceError
    from app.services.intent.hybrid import HybridIntentDetector
    from app.services.intent.llm_based import LLMIntentDetector

    service = Mock(spec=["generate"])
    service.generate = AsyncMock(
        side_effect=ExternalServiceError(service="llm", message="simulated outage")
    )
    detector = HybridIntentDetector(
        RuleBasedIntentDetector(), LLMIntentDetector(service), confidence_threshold=0.7
    )
    graph, calls, _, _ = setup_graph(intent_detector=detector)
    staged = await invoke(graph, "Please refund order ORD1001 because damaged")
    assert staged["pending_confirmation"]["args"]["order_id"] == "ORD1001"
    await invoke(graph, "yes")
    assert [args["order_id"] for _, args in calls] == ["ORD1001"]


async def test_real_llm_outage_vague_write_demands_restatement() -> None:
    from unittest.mock import AsyncMock, Mock

    from app.core.exceptions import ExternalServiceError
    from app.services.intent.hybrid import HybridIntentDetector
    from app.services.intent.llm_based import LLMIntentDetector

    service = Mock(spec=["generate"])
    service.generate = AsyncMock(
        side_effect=ExternalServiceError(service="llm", message="simulated outage")
    )
    detector = HybridIntentDetector(
        RuleBasedIntentDetector(), LLMIntentDetector(service), confidence_threshold=0.7
    )
    graph, calls, _, _ = setup_graph(intent_detector=detector)
    result = await invoke(graph, "a refund for ORD1001 would be great")
    assert not result.get("pending_confirmation")
    assert not calls
    assert "restate" in result["response"] or "重述" in result["response"]
