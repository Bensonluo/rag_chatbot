"""Compiled lifecycle regressions for request admission and safe continuations."""

from typing import Any

import pytest

from app.models.enums.intent import Intent
from app.services.intent.base import CoDetectedIntent, IntentDetection, IntentResult
from app.services.intent.rule_based import RuleBasedIntentDetector
from app.tests.integration.test_multi_intent_queue import Detector, invoke, setup_graph


class AdmissionDetector(Detector):
    """Deterministic speech-act fixtures isolate admission from lexical parsing."""

    def __init__(self, primary: tuple[str, str, str], *secondary: tuple[str, str, str]) -> None:
        self.frames = (primary, *secondary)

    async def detect_multi(
        self, query: str, context: dict[str, Any] | None = None
    ) -> IntentDetection:
        if query != "compound":
            return await super().detect_multi(query, context)
        intent, span, act = self.frames[0]
        return IntentDetection(
            primary=IntentResult(Intent(intent), 0.95),
            primary_query=span,
            speech_act=act,
            co_detected=[
                CoDetectedIntent(Intent(i), 0.9, query=q, speech_act=a)
                for i, q, a in self.frames[1:]
            ],
        )


async def test_clarification_never_enters_agent_or_changes_existing_task() -> None:
    from unittest.mock import AsyncMock, Mock

    agent = Mock()
    agent.run = AsyncMock(side_effect=AssertionError("Non-request reached agent"))
    graph, calls, _, _ = setup_graph(
        intent_detector=AdmissionDetector(("refund", "Already refunded ORD1002", "historical")),
        agent_service=agent,
    )
    prior = {"order_id": "ORD1001"}
    await graph.aupdate_state(
        {"configurable": {"thread_id": "queue"}},
        {"intent": "refund", "filled_slots": prior, "pending_slots": ["reason"]},
    )
    result = await invoke(graph)
    assert result["filled_slots"] == prior
    assert result["pending_slots"] == ["reason"]
    assert not result.get("pending_confirmation") and not calls
    agent.run.assert_not_awaited()


async def test_conditional_acknowledgment_executes_only_read() -> None:
    graph, calls, _, _ = setup_graph(intent_detector=RuleBasedIntentDetector())
    result = await invoke(graph, "If the item is damaged, refund order ORD1001 because defective")
    assert not result.get("pending_confirmation") and not calls
    read = await invoke(graph, "yes")
    assert [(i, a["order_id"]) for i, a in calls] == [("query_order", "ORD1001")]
    assert not read.get("pending_confirmation")
    await invoke(graph, "yes")
    assert len(calls) == 1


async def test_clarification_does_not_poison_next_explicit_request() -> None:
    graph, calls, _, _ = setup_graph(intent_detector=RuleBasedIntentDetector())
    await invoke(graph, "I already returned order ORD1001 because damaged")
    result = await invoke(graph, "Please refund order ORD1002 because defective")
    assert not result.get("admission_clarify")
    assert result["pending_confirmation"]["args"]["order_id"] == "ORD1002"
    await invoke(graph, "yes")
    assert [a["order_id"] for _, a in calls] == ["ORD1002"]


async def test_historical_read_does_not_run_new_query() -> None:
    detector = AdmissionDetector(("query_order", "I already checked order ORD1001", "historical"))
    graph, calls, _, _ = setup_graph(intent_detector=detector)
    result = await invoke(graph)
    assert not calls and not result.get("pending_confirmation")
    assert result.get("admission_clarify")


async def test_informational_write_label_routes_to_knowledge() -> None:
    detector = AdmissionDetector(("refund", "What does refund mean?", "informational"))
    graph, calls, faq, _ = setup_graph(intent_detector=detector)
    result = await invoke(graph)
    assert not calls and not result.get("pending_confirmation")
    faq.match.assert_awaited_once_with("What does refund mean?")
    assert result["intent"] == "faq"


async def test_secondary_history_cannot_complete_existing_write_slots() -> None:
    detector = AdmissionDetector(
        ("policy", "What is your policy?", "informational"),
        ("refund", "I already refunded ORD1001 because damaged", "historical"),
    )
    graph, calls, _, _ = setup_graph(intent_detector=detector)
    await graph.aupdate_state(
        {"configurable": {"thread_id": "queue"}},
        {"intent": "refund", "filled_slots": {"order_id": "ORD1001"}, "pending_slots": ["reason"]},
    )
    result = await invoke(graph)
    assert not calls and not result.get("pending_confirmation")
    assert "FAQ answer:" in result["response"]


@pytest.mark.parametrize("act", ["historical", "conditional", "informational", "alternative"])
async def test_nonaction_cancel_secondary_cannot_veto_current_read(act: str) -> None:
    detector = AdmissionDetector(
        ("query_order", "query order ORD1001", "action"),
        ("cancel", "cancel order ORD1002", act),
    )
    graph, calls, _, _ = setup_graph(intent_detector=detector)
    await invoke(graph)
    assert [(i, a["order_id"]) for i, a in calls] == [("query_order", "ORD1001")]


async def test_informational_primary_cannot_supply_active_write_reason() -> None:
    graph, calls, faq, _ = setup_graph(intent_detector=RuleBasedIntentDetector())
    await graph.aupdate_state(
        {"configurable": {"thread_id": "queue"}},
        {"intent": "refund", "filled_slots": {"order_id": "ORD1001"}, "pending_slots": ["reason"]},
    )
    result = await invoke(graph, "How do I refund order ORD1001 because damaged?")
    assert not calls and not result.get("pending_confirmation")
    faq.match.assert_awaited_once()


async def test_conditional_read_cannot_supply_active_write_reason() -> None:
    graph, calls, _, _ = setup_graph(intent_detector=RuleBasedIntentDetector())
    await graph.aupdate_state(
        {"configurable": {"thread_id": "queue"}},
        {"intent": "refund", "filled_slots": {"order_id": "ORD1001"}, "pending_slots": ["reason"]},
    )
    await invoke(graph, "If damaged, refund order ORD1001 because defective")
    result = await invoke(graph, "yes")
    assert not result.get("pending_confirmation")
    assert [(i, a["order_id"]) for i, a in calls] == [("query_order", "ORD1001")]


@pytest.mark.parametrize("override", ["cancel", "handoff", "deny"])
async def test_conditional_ack_respects_same_turn_override(override: str) -> None:
    graph, calls, _, _ = setup_graph(intent_detector=RuleBasedIntentDetector())
    await invoke(graph, "If damaged, refund order ORD1001 because defective")
    message = {
        "handoff": "yes; transfer me to a human",
        "deny": "yes; no",
        "cancel": "yes; cancel",
    }[override]
    result = await invoke(graph, message)
    assert not calls
    assert result["intent"] == override
    assert not result.get("pending_admission_request")


async def test_authorized_conditional_read_bypasses_unrestricted_agent() -> None:
    from unittest.mock import AsyncMock, Mock

    from app.services.agent.service import AgentService
    from app.services.dialogue.tools import ToolDefinition, ToolRegistry
    from app.services.llm.base import LLMResponse

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="refund",
            intent="refund",
            description="refund",
            required_slots=["order_id"],
            handler=lambda args: {"status": "done"},
            requires_confirmation=True,
        )
    )
    llm = Mock()
    llm.generate_with_tools = AsyncMock(
        return_value=LLMResponse(
            content="",
            model="offline",
            tool_calls=[
                {
                    "id": "wrong-write",
                    "name": "refund",
                    "arguments": '{"order_id":"ORD1001","reason":"defective"}',
                }
            ],
        )
    )
    graph, calls, _, _ = setup_graph(
        intent_detector=RuleBasedIntentDetector(),
        agent_service=AgentService(llm, registry),
    )
    await invoke(graph, "If damaged, refund order ORD1001 because defective")
    result = await invoke(graph, "yes")
    assert not result.get("pending_confirmation")
    assert [(i, a["order_id"]) for i, a in calls] == [("query_order", "ORD1001")]
    llm.generate_with_tools.assert_not_awaited()


def _query_policy() -> dict[str, Any]:
    return {
        "version": 1,
        "request_id": "read-request",
        "mode": "read_only",
        "user_id": 1,
        "intents": ["query_order"],
        "target_order_id": "ORD1001",
    }


def _forbidden_agent() -> Any:
    from unittest.mock import AsyncMock, Mock

    agent = Mock()
    agent.run = AsyncMock(side_effect=AssertionError("Query grant reached unrestricted agent"))
    return agent


@pytest.mark.parametrize("interruption", ["what is your return policy?", "transfer me to a human"])
async def test_read_policy_survives_interruption_and_resume(interruption: str) -> None:
    agent = _forbidden_agent()
    graph, calls, _, _ = setup_graph(intent_detector=RuleBasedIntentDetector(), agent_service=agent)
    await invoke(graph, "If damaged, refund because defective")
    ack = await invoke(graph, "yes")
    assert ack["pending_slots"] == ["order_id"]
    assert not ack.get("pending_admission_request")
    original = ack["request_policy"]
    interrupted = await invoke(graph, interruption)
    suspended = interrupted["state_stack"][-1]
    assert suspended["request_policy"] == original
    assert suspended["request_policy_required"] is True
    resumed = await invoke(graph, "yes")
    assert resumed["request_policy"]["request_id"] == original["request_id"]
    assert resumed["request_policy"]["mode"] == "read_only"
    assert not resumed.get("pending_confirmation")
    final = await invoke(graph, "ORD1001")
    assert [(i, a["order_id"]) for i, a in calls] == [("query_order", "ORD1001")]
    assert not final.get("request_policy") and not final.get("request_policy_required")
    assert not final.get("pending_admission_request")
    agent.run.assert_not_awaited()


@pytest.mark.parametrize("secondary", ["what is your return policy?", "track order ORD2002"])
async def test_finished_read_clears_policy_before_queued_successor(secondary: str) -> None:
    graph, calls, _, _ = setup_graph(intent_detector=RuleBasedIntentDetector())
    await invoke(graph, "If damaged, refund order ORD1001 because defective")
    final = await invoke(graph, "yes; " + secondary)
    assert calls[0][0] == "query_order"
    if secondary.startswith("track"):
        assert calls[1][0] == "track_shipping" and calls[1][1]["order_id"] == "ORD2002"
    assert not final.get("pending_admission_request")
    assert not final.get("request_policy") and not final.get("request_policy_required")


async def test_read_checkpoint_roundtrip_preserves_policy_and_binds_answer(tmp_path: Any) -> None:
    import json

    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    agent = _forbidden_agent()
    graph, _, _, _ = setup_graph(intent_detector=RuleBasedIntentDetector(), agent_service=agent)
    await invoke(graph, "If damaged, refund because defective")
    ack = await invoke(graph, "yes")
    await invoke(graph, "what is your return policy?")
    snapshot = await graph.aget_state({"configurable": {"thread_id": "queue"}})
    serializer = JsonPlusSerializer()
    kind, payload = serializer.dumps_typed(snapshot.values)
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(payload)
    restored = serializer.loads_typed((kind, checkpoint.read_bytes()))
    assert restored["state_stack"][-1]["request_policy"] == ack["request_policy"]
    # A separate compiled graph consumes the serialized producer's state.
    second, calls, _, _ = setup_graph(
        intent_detector=RuleBasedIntentDetector(), agent_service=agent
    )
    await second.aupdate_state({"configurable": {"thread_id": "queue"}}, restored)
    resumed = await invoke(second, "yes")
    assert json.dumps(resumed["request_policy"], sort_keys=True) == json.dumps(
        ack["request_policy"], sort_keys=True
    )
    final = await invoke(second, "ORD1001")
    assert [(i, a["order_id"]) for i, a in calls] == [("query_order", "ORD1001")]
    assert not final.get("pending_confirmation")
    agent.run.assert_not_awaited()


@pytest.mark.parametrize("boundary", ["execute", "agent", "confirm", "durable"])
@pytest.mark.parametrize(
    "mutation", ["write", "complaint", "target", "user", "missing", "mode", "intents"]
)
async def test_policy_is_enforced_at_every_execution_boundary(
    boundary: str, mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time
    from unittest.mock import AsyncMock, Mock

    from app.services.dialogue.nodes import NodeFactory
    from app.services.dialogue.state import DialogueState
    from app.services.dialogue.tools import ToolDefinition, ToolRegistry

    registry = ToolRegistry()
    handler = Mock(return_value={"status": "done"})
    for intent in ("query_order", "refund", "complaint"):
        registry.register(
            ToolDefinition(
                name=intent,
                intent=intent,
                description=intent,
                required_slots=["order_id"],
                handler=handler,
                requires_confirmation=intent == "refund",
            )
        )
    policy = _query_policy()
    task = (
        "complaint"
        if mutation == "complaint"
        else "refund"
        if mutation == "write"
        else "query_order"
    )
    args = {"order_id": "ORD2002" if mutation == "target" else "ORD1001"}
    if mutation == "mode":
        policy["mode"] = "unknown"
    if mutation == "intents":
        policy["intents"] = ["query_order", "refund"]
    state: DialogueState = {
        "intent": task,
        "user_id": 2 if mutation == "user" else 1,
        "session_id": 1,
        "message": "yes",
        "filled_slots": args,
        "request_policy": None if mutation == "missing" else policy,
        "request_policy_required": True,
        "pending_confirmation": {
            "intent": task,
            "args": args,
            "action_id": "pending",
            "staged_at": time.time(),
        },
    }
    agent = _forbidden_agent()
    factory = NodeFactory(RuleBasedIntentDetector(), None, registry, agent_service=agent)
    lookup = AsyncMock(side_effect=AssertionError("Denied policy reached ledger"))
    monkeypatch.setattr(registry, "lookup_action", lookup)
    result: dict[str, Any] | None
    if boundary == "execute":
        result = await factory.execute_tool_node(state)
    elif boundary == "agent":
        result = await factory.handle_agent_node(state)
    else:
        state["intent"] = "confirm"
        if boundary == "confirm":
            result = await factory._handle_meta_intent(state)
        else:
            result = await factory._handle_durable_confirmation(state, None)
        assert result is not None
    assert not result.get("pending_confirmation")
    assert result.get("admission_clarify") is True
    assert not result.get("task_executed")
    handler.assert_not_called()
    agent.run.assert_not_awaited()
    lookup.assert_not_awaited()


async def test_legacy_frame_remains_compatible_but_new_missing_policy_cannot_stage() -> None:
    from app.services.dialogue.nodes import NodeFactory
    from app.services.dialogue.state import DialogueState
    from app.services.dialogue.tools import ToolDefinition, ToolRegistry

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="refund",
            intent="refund",
            description="refund",
            required_slots=["order_id"],
            handler=lambda args: {"status": "done"},
            requires_confirmation=True,
        )
    )
    factory = NodeFactory(RuleBasedIntentDetector(), None, registry)
    legacy: DialogueState = {"intent": "refund", "filled_slots": {"order_id": "ORD1001"}}
    assert (await factory.execute_tool_node(legacy))["pending_confirmation"]
    legacy["request_policy_required"] = True
    denied = await factory.execute_tool_node(legacy)
    assert not denied.get("pending_confirmation") and denied.get("admission_clarify")


async def test_independent_queued_write_has_own_policy_and_confirmation() -> None:
    graph, calls, _, _ = setup_graph(intent_detector=RuleBasedIntentDetector())
    await invoke(graph, "If damaged, refund order ORD1001 because defective")
    staged = await invoke(graph, "yes; refund order ORD2002 because damaged")
    assert [(i, a["order_id"]) for i, a in calls] == [("query_order", "ORD1001")]
    assert staged["request_policy"]["mode"] == "normal"
    assert staged["pending_confirmation"]["args"]["order_id"] == "ORD2002"
    assert not staged.get("pending_admission_request")
    await invoke(graph, "yes")
    await invoke(graph, "yes")
    assert [i for i, _ in calls] == ["query_order", "refund"]


async def test_nested_tasks_restore_their_own_policy_without_inheriting_a_read_grant() -> None:
    graph, calls, _, _ = setup_graph(intent_detector=RuleBasedIntentDetector())
    normal = await invoke(graph, "refund order ORD2002")
    normal_id = normal["request_policy"]["request_id"]
    await invoke(graph, "If damaged, refund because defective")
    read = await invoke(graph, "yes")
    read_id = read["request_policy"]["request_id"]
    assert normal_id != read_id
    interrupted = await invoke(graph, "what is your return policy?")
    policies = [frame["request_policy"] for frame in interrupted["state_stack"]]
    assert [p["mode"] for p in policies] == ["normal", "read_only"]
    resumed = await invoke(graph, "yes")
    assert resumed["request_policy"]["request_id"] == read_id
    await invoke(graph, "ORD1001")
    staged = await invoke(graph, "refund order ORD2002 because damaged")
    assert staged["request_policy"]["request_id"] == normal_id
    assert staged["request_policy"]["mode"] == "normal"
    assert staged["pending_confirmation"]["args"]["order_id"] == "ORD2002"
    await invoke(graph, "yes")
    assert [(i, a["order_id"]) for i, a in calls] == [
        ("query_order", "ORD1001"),
        ("refund", "ORD2002"),
    ]


async def test_missing_payload_after_checkpoint_resume_fails_closed() -> None:
    agent = _forbidden_agent()
    graph, calls, _, _ = setup_graph(intent_detector=RuleBasedIntentDetector(), agent_service=agent)
    await invoke(graph, "If damaged, refund because defective")
    await invoke(graph, "yes")
    interrupted = await invoke(graph, "what is your return policy?")
    frame = dict(interrupted["state_stack"][-1])
    frame.pop("request_policy")
    assert frame["request_policy_required"] is True
    await graph.aupdate_state({"configurable": {"thread_id": "queue"}}, {"state_stack": [frame]})
    result = await invoke(graph, "yes")
    assert result.get("admission_clarify") and not result.get("pending_confirmation")
    assert not calls
    agent.run.assert_not_awaited()
