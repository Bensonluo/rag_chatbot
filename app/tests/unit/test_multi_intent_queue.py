"""Serial request queue invariants, independent of classifier implementation."""

from typing import Any

from app.models.enums.intent import Intent
from app.services.dialogue.multi_intent import (
    discard_queue,
    enqueue_requests,
    pop_queued,
    pop_resumable_frame,
)
from app.services.intent.base import CoDetectedIntent, IntentDetection, IntentResult


def detection(
    primary: str = "query_order",
    query: str = "query order ORD1001",
    *others: tuple[str, str, float],
) -> IntentDetection:
    return IntentDetection(
        primary=IntentResult(Intent(primary), 0.9),
        primary_query=query,
        co_detected=[
            CoDetectedIntent(intent=Intent(intent), confidence=score, query=text)
            for intent, text, score in others
        ],
    )


def test_same_intent_requests_have_distinct_identity_and_fifo_queries() -> None:
    interrupted = {"intent": "refund", "filled_slots": {"order_id": "OLD"}}
    updates = enqueue_requests(
        {"message": "compound", "state_stack": [interrupted]},
        detection(
            "query_order",
            "query order ORD1001",
            ("query_order", "query order ORD1002", 0.1),
            ("query_order", "query order ORD1003", 0.99),
        ),
    )
    stack = updates["state_stack"]
    assert len({item["request_id"] for item in stack[1:]}) == 2
    first, remaining = pop_queued(stack)
    second, remaining = pop_queued(remaining)
    assert first is not None and second is not None
    assert [first["query"], second["query"]] == ["query order ORD1002", "query order ORD1003"]
    assert first["filled_slots"] == second["filled_slots"] == {}
    first["filled_slots"]["order_id"] = "changed"
    assert not second["filled_slots"]
    assert remaining == [interrupted]


def test_capacity_is_explicit_and_counts_requests_not_intent_kinds() -> None:
    updates = enqueue_requests(
        {"message": "compound"},
        detection(
            "faq",
            "first?",
            ("faq", "second?", 0.9),
            ("faq", "third?", 0.8),
            ("faq", "fourth?", 0.7),
        ),
    )
    assert len(updates["state_stack"]) == 2
    assert updates["multi_intent_overflow"] == ["fourth?"]


def test_queue_discard_preserves_interruption_stack() -> None:
    interrupted = {"intent": "refund"}
    assert discard_queue({"state_stack": [interrupted, {"queued_from": "multi_intent"}]}) == {
        "state_stack": [interrupted]
    }


def test_pop_resumable_frame_skips_queued_entries() -> None:
    interrupted = {"intent": "complaint", "filled_slots": {"order_id": "ORD1001"}}
    stack: list[dict[str, Any]] = [
        {"intent": "query_order", "queued_from": "multi_intent", "query": "q1"},
        interrupted,
        {"intent": "track_shipping", "queued_from": "multi_intent", "query": "q2"},
    ]
    frame, remaining = pop_resumable_frame(stack)
    assert frame is interrupted
    assert all(item.get("queued_from") == "multi_intent" for item in remaining)
    assert len(remaining) == 2


def test_pop_resumable_frame_never_consumes_queue_only_stack() -> None:
    stack = [
        {"intent": "query_order", "queued_from": "multi_intent", "query": "q1"},
        {"intent": "track_shipping", "queued_from": "multi_intent", "query": "q2"},
    ]
    frame, remaining = pop_resumable_frame(stack)
    assert frame is None
    assert remaining == stack
