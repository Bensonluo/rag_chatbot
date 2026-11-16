"""Offline compiled-graph proofs of serial multi-request scheduling."""

import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph

from app.models.enums.intent import Intent
from app.services.dialogue.graph import build_dialogue_graph
from app.services.dialogue.tools import ToolDefinition, ToolRegistry, ToolResult
from app.services.intent.base import (
    CoDetectedIntent,
    IntentDetection,
    IntentDetector,
    IntentResult,
)
from app.services.llm.base import LLMResponse


class Detector(IntentDetector):
    """Contract-shaped detections, with exact subrequests supplied by fixtures."""

    def __init__(self, *requests: tuple[str, str]) -> None:
        self.requests = requests

    async def detect_multi(
        self, query: str, context: dict[str, Any] | None = None
    ) -> IntentDetection:
        pieces = self.requests if query == "compound" else (("confirm", query),)
        if query in {"cancel", "deny", "handoff"}:
            pieces = ((query, query),)
        return IntentDetection(
            primary=IntentResult(Intent(pieces[0][0]), 0.95),
            primary_query=pieces[0][1],
            co_detected=[
                CoDetectedIntent(intent=Intent(i), confidence=0.9, query=q) for i, q in pieces[1:]
            ],
        )

    async def detect_with_confidence(
        self, query: str, context: dict[str, Any] | None = None
    ) -> IntentResult:
        return (await self.detect_multi(query, context)).primary

    async def detect(self, query: str, context: dict[str, Any] | None = None) -> Intent:
        return (await self.detect_with_confidence(query, context)).intent


def setup_graph(
    *requests: tuple[str, str], **kwargs: Any
) -> tuple[CompiledStateGraph[Any], list[tuple[str, dict[str, Any]]], Mock, Mock]:
    calls: list[tuple[str, dict[str, Any]]] = []
    registry = ToolRegistry()
    for intent in ("query_order", "track_shipping", "refund", "return"):

        def handler(args: dict[str, Any], intent: str = intent) -> dict[str, Any]:
            calls.append((intent, dict(args)))
            return {"order_id": args["order_id"], "status": "done"}

        registry.register(
            ToolDefinition(
                name=intent,
                intent=intent,
                description=intent,
                required_slots=["order_id"],
                handler=handler,
                requires_confirmation=intent in {"refund", "return"},
            )
        )
    llm = Mock()
    llm.generate = AsyncMock(
        side_effect=lambda messages, **kwargs: LLMResponse(
            content="Answer: " + messages[-1].content.split("\n")[1], model="offline"
        )
    )
    faq = Mock()
    faq.match = AsyncMock(
        side_effect=lambda q: SimpleNamespace(faq_id=q, question=q, answer="FAQ answer: " + q)
    )
    graph = build_dialogue_graph(
        intent_detector=kwargs.pop("intent_detector", Detector(*requests)),
        slot_filler=None,
        tool_registry=registry,
        retrieval_pipeline=kwargs.pop("retrieval_pipeline", {}),
        llm_service=llm,
        faq_service=kwargs.pop("faq_service", faq),
        checkpointer=MemorySaver(),
        **kwargs,
    )
    return graph, calls, faq, llm


async def invoke(
    graph: CompiledStateGraph[Any],
    message: str = "compound",
    thread: str = "queue",
    **configurable: Any,
) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        await graph.ainvoke(
            {"message": message, "session_id": 1, "user_id": 1},
            {"configurable": {"thread_id": thread, **configurable}, "recursion_limit": 60},
        ),
    )


async def test_three_reads_with_agent_fallback_fit_default_graph_recursion_limit() -> None:
    agent = Mock()
    agent.run = AsyncMock(side_effect=NotImplementedError)
    graph, calls, _, _ = setup_graph(
        ("query_order", "query order ORD1001"),
        ("query_order", "query order ORD1002"),
        ("query_order", "query order ORD1003"),
        agent_service=agent,
    )
    result = await graph.ainvoke(
        {"message": "compound", "session_id": 1, "user_id": 1},
        {"configurable": {"thread_id": "default-limit"}},
    )
    assert [args["order_id"] for _, args in calls] == ["ORD1001", "ORD1002", "ORD1003"]
    assert len(result["executed_tools"]) == 3 and not result["state_stack"]


async def test_two_reads_run_in_same_turn_with_isolated_messages() -> None:
    graph, calls, _, _ = setup_graph(
        ("query_order", "query order ORD1001"), ("track_shipping", "track order ORD1002")
    )
    result = await invoke(graph)
    assert [(i, a["order_id"]) for i, a in calls] == [
        ("query_order", "ORD1001"),
        ("track_shipping", "ORD1002"),
    ]
    assert len(result["executed_tools"]) == 2
    assert "ORD1001" in result["response"] and "ORD1002" in result["response"]
    assert result["message"] == "compound"
    assert result["state_stack"] == []


async def test_write_pauses_faq_then_confirmation_advances_automatically() -> None:
    graph, calls, faq, _ = setup_graph(
        ("refund", "refund order ORD1001 because damaged"), ("faq", "Who pays postage?")
    )
    staged = await invoke(graph)
    assert staged["pending_confirmation"]
    assert not calls and not faq.match.await_count
    result = await invoke(graph, "yes")
    assert len(calls) == 1
    assert faq.match.await_args is not None
    assert faq.match.await_args.args == ("Who pays postage?",)
    assert "FAQ answer:" in result["response"]
    assert result["sources"] == ["faq:Who pays postage?"]
    assert result["message"] == "yes"


async def test_two_writes_require_two_separate_confirmations() -> None:
    graph, calls, _, _ = setup_graph(
        ("refund", "refund order ORD1001 because damaged"),
        ("return", "return order ORD1002 because defective"),
    )
    first = await invoke(graph)
    second = await invoke(graph, "yes")
    assert len(calls) == 1
    assert second["pending_confirmation"]["intent"] == "return"
    assert second["pending_confirmation"]["args"]["order_id"] == "ORD1002"
    assert first["pending_confirmation"]["action_id"] != second["pending_confirmation"]["action_id"]
    third = await invoke(graph, "yes")
    assert len(calls) == 2 and not third["pending_confirmation"]
    assert not third["active_request_id"]


async def test_same_intent_two_orders_do_not_share_slots() -> None:
    graph, calls, _, _ = setup_graph(
        ("query_order", "query order ORD1001"), ("query_order", "query order ORD1002")
    )
    await invoke(graph)
    assert [a["order_id"] for _, a in calls] == ["ORD1001", "ORD1002"]


async def test_two_faqs_keep_both_source_snapshots_and_stream_equals_result() -> None:
    graph, _, _, _ = setup_graph(("faq", "First question?"), ("faq", "Second question?"))
    queue: asyncio.Queue[str] = asyncio.Queue()
    result = await invoke(graph, stream_queue=queue)
    assert result["sources"] == ["faq:First question?", "faq:Second question?"]
    assert [c["faq_id"] for c in result["citations"]] == ["First question?", "Second question?"]
    assert len({c["version_id"] for c in result["citations"]}) == 2
    chunks = []
    while not queue.empty():
        chunks.append(queue.get_nowait())
    assert "".join(chunks) == result["response"]
    assert result["shared_cache_blocked"]


@pytest.mark.parametrize("override", ["cancel", "deny", "handoff"])
async def test_override_never_runs_queued_write(override: str) -> None:
    graph, calls, _, _ = setup_graph(
        ("refund", "refund order ORD1001 because damaged"),
        ("return", "return order ORD1002 because defective"),
    )
    await invoke(graph)
    result = await invoke(graph, override)
    assert not calls
    assert not result["pending_confirmation"]
    assert not [s for s in result["state_stack"] if s.get("queued_from") == "multi_intent"]
    await invoke(graph, "yes")
    assert not calls


async def test_slot_answer_plus_new_request_is_preserved_and_queued() -> None:
    graph, calls, _, _ = setup_graph(("unknown", "damaged"), ("query_order", "query order ORD1002"))
    config: RunnableConfig = {"configurable": {"thread_id": "queue"}}
    await graph.aupdate_state(
        config,
        {
            "intent": "refund",
            "filled_slots": {"order_id": "ORD1001"},
            "pending_slots": ["reason"],
            "task_executed": False,
        },
    )
    staged = await invoke(graph)
    assert staged["pending_confirmation"]["args"] == {"order_id": "ORD1001", "reason": "damaged"}
    assert len(staged["state_stack"]) == 1
    await invoke(graph, "yes")
    assert [(i, a["order_id"]) for i, a in calls] == [
        ("refund", "ORD1001"),
        ("query_order", "ORD1002"),
    ]


async def test_real_detector_keeps_active_slot_answer_before_new_request() -> None:
    from app.services.intent.rule_based import RuleBasedIntentDetector

    graph, calls, _, _ = setup_graph(intent_detector=RuleBasedIntentDetector())
    config: RunnableConfig = {"configurable": {"thread_id": "queue"}}
    await graph.aupdate_state(
        config,
        {
            "intent": "refund",
            "filled_slots": {"order_id": "ORD1001"},
            "pending_slots": ["reason"],
            "task_executed": False,
        },
    )
    staged = await invoke(graph, "damaged; also query order ORD1002")
    assert staged["pending_confirmation"]["args"] == {"order_id": "ORD1001", "reason": "damaged"}
    assert not calls
    result = await invoke(graph, "yes")
    assert [(intent, args["order_id"]) for intent, args in calls] == [
        ("refund", "ORD1001"),
        ("query_order", "ORD1002"),
    ]
    assert not result["state_stack"]


async def test_missing_slots_pause_queue_and_other_sessions_do_not_resume_it() -> None:
    graph, calls, _, _ = setup_graph(
        ("query_order", "query my order"), ("track_shipping", "track order ORD1002")
    )
    paused = await invoke(graph)
    assert paused["pending_slots"] == ["order_id"] and not calls
    assert len(paused["state_stack"]) == 1
    other = await invoke(graph, "yes", thread="other")
    assert not calls and not other.get("state_stack")
    result = await invoke(graph, "ORD1001")
    assert [a["order_id"] for _, a in calls] == ["ORD1001", "ORD1002"]
    assert result["message"] == "ORD1001"


async def test_legacy_single_detector_without_multi_keeps_single_path() -> None:
    detector = SimpleNamespace(
        detect_with_confidence=AsyncMock(return_value=IntentResult(Intent.QUERY_ORDER, 0.9))
    )
    graph, calls, _, _ = setup_graph(intent_detector=detector)
    result = await invoke(graph, "query order ORD1001")
    assert len(calls) == 1 and len(result["executed_tools"]) == 1
    detector.detect_with_confidence.assert_awaited_once()


async def test_confirmation_with_a_new_request_executes_one_write_then_new_read() -> None:
    detector = Detector(("refund", "refund order ORD1001 because damaged"), ("faq", "Existing?"))
    graph, calls, _, _ = setup_graph(intent_detector=detector)
    staged = await invoke(graph)
    assert staged["pending_confirmation"]
    detector.requests = (("confirm", "yes"), ("query_order", "query order ORD1002"))
    result = await invoke(graph)
    assert [(intent, args["order_id"]) for intent, args in calls] == [
        ("refund", "ORD1001"),
        ("query_order", "ORD1002"),
    ]
    assert not result["pending_confirmation"]
    assert result["message"] == "compound"


async def test_overflow_is_visible_and_never_silently_executed() -> None:
    graph, _, faq, _ = setup_graph(
        ("faq", "one?"), ("faq", "two?"), ("faq", "three?"), ("faq", "four?")
    )
    result = await invoke(graph)
    assert faq.match.await_count == 3
    assert "four?" in result["response"] and "not queued" in result["response"]


async def test_rag_then_read_executes_both_without_stale_evidence() -> None:
    graph, calls, _, llm = setup_graph(
        ("policy", "What is your policy?"), ("query_order", "query order ORD1002"), faq_service=None
    )
    result = await invoke(graph)
    assert len(calls) == 1
    assert "ORD1002" in result["response"]
    assert result["retrieval_ran"] is False  # no retrieval service was wired
    assert llm.generate.await_count == 2


async def test_rag_and_read_use_their_own_queries_and_keep_citations() -> None:
    search = Mock()
    search.search = AsyncMock(
        return_value=[
            SimpleNamespace(
                document_id="policy-doc",
                content="Policy evidence",
                score=0.9,
                metadata={"version_id": "v1"},
            )
        ]
    )
    graph, calls, _, llm = setup_graph(
        ("policy", "What is your policy?"),
        ("query_order", "query order ORD1002"),
        faq_service=None,
        retrieval_pipeline={"hybrid_search": search},
    )
    result = await invoke(graph)
    assert search.search.await_args is not None
    assert search.search.await_args.args[0].query == "What is your policy?"
    assert [a["order_id"] for _, a in calls] == ["ORD1002"]
    assert result["sources"] == ["policy-doc"]
    assert result["citations"][0]["version_id"] == "v1"
    assert result["retrieval_ran"]
    assert "Reference documents:" in llm.generate.await_args_list[0].args[0][-1].content
    assert "Reference documents:" not in llm.generate.await_args_list[1].args[0][-1].content
    assert "随时可以继续" not in result["response"]


async def test_primary_multi_request_does_not_reuse_completed_same_intent_slots() -> None:
    graph, calls, _, _ = setup_graph(
        ("query_order", "query order ORD1001"), ("query_order", "query order ORD1002")
    )
    config: RunnableConfig = {"configurable": {"thread_id": "queue"}}
    await graph.aupdate_state(
        config,
        {"intent": "query_order", "filled_slots": {"order_id": "OLD9999"}, "task_executed": True},
    )
    await invoke(graph)
    assert [a["order_id"] for _, a in calls] == ["ORD1001", "ORD1002"]


async def test_new_primary_question_does_not_steal_secondary_active_slot_answer() -> None:
    graph, calls, _, _ = setup_graph(("faq", "Who pays postage?"), ("unknown", "damaged"))
    config: RunnableConfig = {"configurable": {"thread_id": "queue"}}
    await graph.aupdate_state(
        config,
        {
            "intent": "refund",
            "filled_slots": {"order_id": "ORD1001"},
            "pending_slots": ["reason"],
            "task_executed": False,
        },
    )
    result = await invoke(graph)
    assert result["pending_confirmation"]["args"]["reason"] == "damaged"
    assert result["state_stack"][0]["query"] == "Who pays postage?"
    await invoke(graph, "yes")
    assert len(calls) == 1


async def test_compound_turn_bypasses_l0_cache_and_never_writes_shared_answer() -> None:
    from app.services.chat.answer_cache import CachedAnswer
    from app.services.chat.chat_service import ChatService

    cache = Mock()
    cache.get = AsyncMock(
        return_value=CachedAnswer(response="stale partial", sources=["doc"], intent="faq")
    )
    cache.put = AsyncMock()
    graph, _, faq, _ = setup_graph(("faq", "First?"), ("faq", "Second?"), answer_cache=cache)
    result = await graph.ainvoke(
        {"message": "compound", "session_id": 1}, {"configurable": {"thread_id": "anonymous"}}
    )
    assert faq.match.await_count == 2
    cache.get.assert_not_awaited()
    chat = ChatService(graph=graph, answer_cache=cache)
    await chat._maybe_cache_answer("compound", result, user_id=0)
    cache.put.assert_not_awaited()


async def test_agent_terminal_read_advances_but_agent_handoff_stops() -> None:
    agent = Mock()
    agent.run = AsyncMock(
        return_value=SimpleNamespace(
            response="Read completed",
            tool_trace=[{"tool": "query_order", "ok": True}],
            pending_confirmation=None,
            citations=[],
            outcome="done",
            handoff_ticket_id=None,
        )
    )
    graph, _, faq, _ = setup_graph(
        ("query_order", "query order ORD1001"), ("faq", "Next?"), agent_service=agent
    )
    result = await invoke(graph)
    assert faq.match.await_count == 1
    assert agent.run.await_args is not None
    assert agent.run.await_args.kwargs["user_message"] == "query order ORD1001"
    assert "Read completed" in result["response"] and "FAQ answer" in result["response"]

    agent.run.return_value.outcome = "handoff"
    agent.run.return_value.handoff_ticket_id = 12
    stopped = await invoke(graph, thread="handoff")
    assert stopped["handoff_ticket_id"] == 12
    assert not stopped["state_stack"]
    assert faq.match.await_count == 1


async def test_agent_helper_trace_does_not_complete_active_task_or_advance_queue() -> None:
    agent = Mock()
    agent.run = AsyncMock(
        return_value=SimpleNamespace(
            response="Please provide the order number",
            tool_trace=[{"tool": "get_recent_orders", "ok": True}],
            pending_confirmation=None,
            citations=[],
            outcome="done",
            handoff_ticket_id=None,
        )
    )
    graph, _, faq, _ = setup_graph(
        ("query_order", "query order ORD1001"), ("faq", "Next?"), agent_service=agent
    )
    result = await invoke(graph)
    assert not faq.match.await_count
    assert len(result["state_stack"]) == 1
    assert not result["task_executed"]


async def test_emotion_discards_existing_queue_before_any_confirmation() -> None:
    graph, calls, faq, _ = setup_graph(
        ("refund", "refund order ORD1001 because damaged"), ("faq", "Next?")
    )
    await invoke(graph)
    result = await invoke(graph, "terrible service")
    assert not calls and not faq.match.await_count
    assert result["handoff_reason"] == "emotion"
    assert not result["pending_confirmation"]
    assert not [s for s in result["state_stack"] if s.get("queued_from") == "multi_intent"]


async def test_suspended_active_task_keeps_its_queue_paused_during_unrelated_question() -> None:
    detector = Detector(("query_order", "query my order"), ("query_order", "query order ORD1002"))
    graph, calls, faq, _ = setup_graph(intent_detector=detector)
    await invoke(graph)
    detector.requests = (("faq", "Other question?"),)
    result = await invoke(graph)
    assert not calls
    assert faq.match.await_count == 1
    assert len(result["state_stack"]) == 2  # one queued and one interrupted
    assert len([s for s in result["state_stack"] if s.get("queued_from") == "multi_intent"]) == 1
    detector.requests = (("query_order", "query order ORD1001"),)
    await invoke(graph)
    assert [args["order_id"] for _, args in calls] == ["ORD1001", "ORD1002"]


@pytest.mark.parametrize("override", ["cancel", "deny", "handoff"])
async def test_override_among_secondary_requests_vetoes_hidden_actions(override: str) -> None:
    graph, calls, _, _ = setup_graph(
        ("query_order", "query order ORD1001"),
        (override, override),
        ("refund", "refund order ORD1002 because damaged"),
    )
    result = await invoke(graph)
    assert not calls
    assert not result.get("pending_confirmation")
    assert not [s for s in result.get("state_stack", []) if s.get("queued_from") == "multi_intent"]


async def test_durable_unknown_pauses_then_one_completion_stages_only_next_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services.dialogue.tools import ToolResult

    registry = ToolRegistry(action_ledger=Mock())
    for intent in ("refund", "return"):
        registry.register(
            ToolDefinition(
                name=intent,
                intent=intent,
                description=intent,
                required_slots=["order_id", "reason"],
                handler=lambda args: {},
                requires_confirmation=True,
            )
        )
    execute = AsyncMock(return_value=ToolResult(False, {}, execution_status="unknown"))
    monkeypatch.setattr(registry, "execute", execute)
    graph = build_dialogue_graph(
        intent_detector=Detector(
            ("refund", "refund order ORD1001 because damaged"),
            ("return", "return order ORD1002 because defective"),
        ),
        slot_filler=None,
        tool_registry=registry,
        retrieval_pipeline={},
        llm_service=None,
        checkpointer=MemorySaver(),
    )
    staged = await invoke(graph)
    first_id = staged["pending_confirmation"]["action_id"]
    uncertain = await invoke(graph, "yes")
    assert uncertain["pending_confirmation"]["action_id"] == first_id
    assert len(uncertain["state_stack"]) == 1
    assert execute.await_count == 1
    execute.return_value = ToolResult(
        True,
        {"order_id": "ORD1001", "status": "done"},
        execution_status="completed",
        action_id=first_id,
    )
    second = await invoke(graph, "yes")
    assert execute.await_count == 2  # exactly one dispatch per confirm turn
    second_id = second["pending_confirmation"]["action_id"]
    assert second_id != first_id
    assert second["pending_confirmation"]["args"]["order_id"] == "ORD1002"
    assert second["action_execution"]["action_id"] == first_id
    execute.return_value = ToolResult(
        True,
        {"order_id": "ORD1002", "status": "done"},
        execution_status="completed",
        action_id=second_id,
    )
    finished = await invoke(graph, "yes")
    assert not finished["pending_confirmation"]
    assert execute.await_args is not None
    assert execute.await_args.kwargs["action_id"] == second_id
    assert execute.await_args_list[0].kwargs["action_id"] == first_id


@pytest.mark.parametrize("first_intent", ["policy", "chitchat"])
async def test_multi_stream_runs_claim_and_pii_gates_before_publishing(first_intent: str) -> None:
    from app.services.guardrails.base import GuardrailService
    from app.services.guardrails.output_guard import DefaultOutputGuardrail

    search = Mock()
    search.search = AsyncMock(
        return_value=[
            SimpleNamespace(
                document_id="policy", content="退货期限是七天。", score=0.9, metadata={}
            )
        ]
    )
    graph, _, _, llm = setup_graph(
        (first_intent, "退货期限是多少？"),
        ("policy", "退货有什么要求？"),
        faq_service=None,
        retrieval_pipeline={"hybrid_search": search},
        guardrail_service=GuardrailService(output_guard=DefaultOutputGuardrail()),
    )
    llm.generate.side_effect = None
    llm.generate.return_value = LLMResponse(
        content="退货期限是30天。联系 audit@example.invalid。", model="offline"
    )
    queue: asyncio.Queue[str] = asyncio.Queue()
    result = await invoke(graph, stream_queue=queue)
    chunks = []
    while not queue.empty():
        chunks.append(queue.get_nowait())
    assert "".join(chunks) == result["response"]
    assert "audit@example.invalid" not in result["response"]
    assert "30天" not in result["response"]
    assert "[REDACTED]" in result["response"]
    assert not result["citations"]  # replaced claims cannot keep old evidence


class SlotAnswerDetector(IntentDetector):
    """Stable single refund request that counts multi-detection preflights."""

    def __init__(self) -> None:
        self.multi_calls = 0

    async def detect_multi(
        self, query: str, context: dict[str, Any] | None = None
    ) -> IntentDetection:
        self.multi_calls += 1
        return IntentDetection(
            primary=IntentResult(Intent.REFUND, 0.95),
            primary_query="refund because damaged",
        )

    async def detect_with_confidence(
        self, query: str, context: dict[str, Any] | None = None
    ) -> IntentResult:
        return IntentResult(Intent.REFUND, 0.95)

    async def detect(self, query: str, context: dict[str, Any] | None = None) -> Intent:
        return Intent.REFUND


@pytest.mark.asyncio
async def test_bare_order_slot_answer_skips_multi_detection_preflight() -> None:
    detector = SlotAnswerDetector()
    graph, calls, _, _ = setup_graph(intent_detector=detector)
    first = await invoke(graph, "refund because damaged", thread="slot-preflight")
    assert first["pending_slots"]  # waiting for the order id
    assert detector.multi_calls == 1
    second = await invoke(graph, "ORD1001", thread="slot-preflight")
    # A bare order reference is definitionally not a compound request:
    # no detector call on the slot-answer hot path, and the answer still
    # binds and stages the refund exactly as before.
    assert detector.multi_calls == 1
    assert second["pending_confirmation"]["args"]["order_id"] == "ORD1001"
    assert not calls


@pytest.mark.asyncio
async def test_mixed_slot_answer_still_runs_multi_detection() -> None:
    detector = SlotAnswerDetector()
    graph, _, _, _ = setup_graph(intent_detector=detector)
    await invoke(graph, "refund because damaged", thread="slot-preflight-mixed")
    assert detector.multi_calls == 1
    await invoke(graph, "ORD1001 顺便帮我查一下物流", thread="slot-preflight-mixed")
    assert detector.multi_calls == 2


@pytest.mark.asyncio
async def test_failed_action_visibly_cancels_queued_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A durable execution conflict (not a business rejection, which lets
    # the chain continue) terminates the request chain; the discard must
    # be visible instead of silent (invariant 9).
    registry = ToolRegistry(action_ledger=Mock())
    for intent in ("refund", "return"):
        registry.register(
            ToolDefinition(
                name=intent,
                intent=intent,
                description=intent,
                required_slots=["order_id"],
                handler=lambda args: {},
                requires_confirmation=True,
            )
        )
    execute = AsyncMock(return_value=ToolResult(False, {}, execution_status="conflict"))
    monkeypatch.setattr(registry, "execute", execute)
    graph = build_dialogue_graph(
        intent_detector=Detector(
            ("refund", "refund order ORD1001 because damaged"),
            ("return", "return order ORD1002 because defective"),
        ),
        slot_filler=None,
        tool_registry=registry,
        retrieval_pipeline={},
        llm_service=None,
        checkpointer=MemorySaver(),
    )
    staged = await invoke(graph)
    assert staged["pending_confirmation"]
    failed = await invoke(graph, "yes")
    assert execute.await_count == 1  # the queued return was never dispatched
    assert failed["state_stack"] == []
    assert "return order ORD1002 because defective" in failed["response"]
    assert "please send them again" in failed["response"]
