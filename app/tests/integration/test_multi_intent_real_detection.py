"""Compiled dialogue journeys using the real rule detector, not verdict fixtures."""

import time

import pytest

from app.services.intent.rule_based import RuleBasedIntentDetector
from app.tests.integration.test_multi_intent_queue import invoke, setup_graph


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
