"""Actual mock business effects are not repeated after an uncertain chat turn.

HTTP/JWT, SQLite foreign keys, the coordinator, compiled dialogue graph,
AgentService, durable SQL ledger and default mock handlers are real. Only
model/transport boundaries and deliberate failure timing are controlled.
This suite does not exercise a real payment provider or PostgreSQL locks.
"""

import json
import socket
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import pytest
from httpx import Response
from langgraph.checkpoint.memory import MemorySaver
from sqlalchemy import select, text

from app.config.settings import settings
from app.models.database.action_execution import ActionExecution
from app.models.database.session import ChatSession
from app.services.agent.service import AgentService
from app.services.dialogue.action_ledger import ActionLedger
from app.services.dialogue.graph import build_dialogue_graph
from app.services.dialogue.nodes import NodeFactory
from app.services.intent.rule_based import RuleBasedIntentDetector
from app.services.llm.base import LLMResponse
from app.tests.evaluation.test_human_conversation import _bot, _bot_rows, _session_key
from app.tests.evaluation.test_human_conversation import _Case as HumanCase
from app.tests.evaluation.test_human_conversation import case as case


class _LocalToolProvider:
    calls = 0

    async def generate_with_tools(self, messages: Any, tools: Any) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            content="",
            model="local-action-proposal",
            tool_calls=[
                {
                    "id": "local-provider-call",
                    "name": "process_refund",
                    "arguments": json.dumps({"order_id": "ORD1001", "reason": "damaged"}),
                }
            ],
        )


class _FaultCheckpoint(MemorySaver):
    armed = False
    failures = 0
    effects: list[dict[str, Any]]

    async def aput(self, *args: Any, **kwargs: Any) -> Any:
        if self.armed and self.effects:
            self.armed = False
            self.failures += 1
            raise RuntimeError("isolated checkpoint fault after actual mock action")
        return await super().aput(*args, **kwargs)


@dataclass
class _Business:
    case: HumanCase
    monkeypatch: pytest.MonkeyPatch
    checkpoint: _FaultCheckpoint
    provider: _LocalToolProvider
    effects: list[dict[str, Any]] = field(default_factory=list)
    graph: Any = None
    node_failure_armed: bool = False
    pending: dict[str, Any] = field(default_factory=dict)

    def config(self, thread_id: str | None = None) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id or str(self.case.private_id)}}


@pytest.fixture
async def business(
    case: HumanCase, monkeypatch: pytest.MonkeyPatch
) -> AsyncGenerator[_Business, None]:
    dns_attempts: list[str] = []

    def no_dns(*args: Any, **kwargs: Any) -> Any:
        dns_attempts.append(repr(args))
        raise AssertionError("durable action evaluation forbids DNS")

    for name in ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr"):
        monkeypatch.setattr(socket, name, no_dns)
    case.registry.action_ledger = ActionLedger(case.maker)
    checkpoint = _FaultCheckpoint()
    scenario = _Business(case, monkeypatch, checkpoint, _LocalToolProvider())
    checkpoint.effects = scenario.effects
    for intent in ("refund", "return"):
        tool = case.registry.get_tool_for_intent(intent)
        assert tool is not None
        original = tool.handler

        async def record(
            args: dict[str, Any], original: Any = original, intent: str = intent
        ) -> dict[str, Any]:
            # Observe via a separate SQL session before the real handler.
            # A flush on an uncommitted reservation is insufficient.
            async with case.maker() as db:
                reservation = await db.get(ActionExecution, args.get("idempotency_key"))
                assert reservation is not None and reservation.status == "in_progress"
                assert reservation.result is None
            result = original(args)
            assert isinstance(result, dict)
            scenario.effects.append({"intent": intent, "args": dict(args), "result": result})
            return result

        monkeypatch.setattr(tool, "handler", record)
    original_generate = NodeFactory._generate_with_tool

    async def fail_after_action(self: NodeFactory, *args: Any, **kwargs: Any) -> Any:
        if scenario.node_failure_armed and scenario.effects:
            scenario.node_failure_armed = False
            raise RuntimeError("isolated generation fault after actual mock action")
        return await original_generate(self, *args, **kwargs)

    monkeypatch.setattr(NodeFactory, "_generate_with_tool", fail_after_action)
    try:
        yield scenario
    finally:
        assert dns_attempts == []


async def _stage(scenario: _Business, staging: str = "deterministic") -> dict[str, Any]:
    base = scenario.case
    prior_provider_calls = scenario.provider.calls
    scenario.graph = build_dialogue_graph(
        intent_detector=RuleBasedIntentDetector(),
        slot_filler=None,
        tool_registry=base.registry,
        retrieval_pipeline=None,
        llm_service=None,
        handoff_service=base.handoff,
        checkpointer=scenario.checkpoint,
        agent_service=(
            AgentService(scenario.provider, base.registry) if staging == "agent" else None
        ),
    )
    scenario.monkeypatch.setattr(base.chat, "graph", scenario.graph)
    turns = (
        ["I want a refund"]
        if staging == "agent"
        else ["I want a refund", "ORD1001", "damaged item"]
    )
    for message in turns:
        response = await _bot(base, message)
        assert response.status_code == 200
    pending = (await scenario.graph.aget_state(scenario.config())).values["pending_confirmation"]
    assert str(UUID(pending["action_id"])) == pending["action_id"]
    assert not scenario.effects
    assert scenario.provider.calls - prior_provider_calls == (1 if staging == "agent" else 0)
    scenario.pending = dict(pending)
    return scenario.pending


def _events(response: Response) -> list[tuple[str, str]]:
    assert response.status_code == 200
    result = []
    for block in response.text.replace("\r\n", "\n").split("\n\n"):
        kind = "message"
        data = []
        for line in block.split("\n"):
            if line.startswith("event:"):
                kind = line[6:].lstrip(" ")
            elif line.startswith("data:"):
                data.append(line[5:].removeprefix(" "))
        if data:
            result.append((kind, "\n".join(data)))
    return result


def _content(response: Response, transport: str) -> str:
    assert response.status_code == 200, response.text
    if transport == "normal":
        content = response.json()["content"]
        assert isinstance(content, str)
        return content
    events = _events(response)
    assert events[-1] == ("message", "[DONE]")
    assert ("message", "[STREAM_ERROR]") not in events
    return "".join(value for kind, value in events if kind == "message" and value != "[DONE]")


async def _actions(scenario: _Business) -> list[ActionExecution]:
    async with scenario.case.maker() as db:
        return list((await db.scalars(select(ActionExecution))).all())


async def _trigger(scenario: _Business, phase: str) -> None:
    command = (
        "CREATE TRIGGER refuse_action_write BEFORE INSERT ON action_executions "
        if phase == "reservation"
        else "CREATE TRIGGER refuse_action_write BEFORE UPDATE ON action_executions "
        "WHEN NEW.status = 'completed' "
    )
    async with scenario.case.maker() as db:
        await db.execute(
            text(command + "BEGIN SELECT RAISE(ABORT, 'isolated ledger rejection'); END")
        )
        await db.commit()


async def _drop_trigger(scenario: _Business) -> None:
    async with scenario.case.maker() as db:
        await db.execute(text("DROP TRIGGER refuse_action_write"))
        await db.commit()


def _uncertain_copy(content: str, scenario: _Business) -> None:
    assert "refund_id" not in content
    for effect in scenario.effects:
        assert effect["result"]["refund_id"] not in content
    assert "未执行任何更改" not in content
    assert "退款成功" not in content and "已成功退款" not in content


@pytest.mark.parametrize("staging", ["deterministic", "agent"])
@pytest.mark.parametrize("transport", ["normal", "stream"])
@pytest.mark.parametrize("fault", ["checkpoint", "node"])
async def test_actual_http_retry_replays_durable_result_without_repeating_mock_effect(
    business: _Business, staging: str, transport: str, fault: str
) -> None:
    pending = await _stage(business, staging)
    if fault == "checkpoint":
        business.checkpoint.armed = True
    else:
        business.node_failure_armed = True
    first = await _bot(business.case, "confirm", transport, include_completion=True)
    if transport == "normal":
        assert first.status_code == 500
    else:
        events = _events(first)
        assert ("message", "[STREAM_ERROR]") in events
        assert not any(kind == "completion" for kind, _ in events)
    assert len(business.effects) == 1
    actions = await _actions(business)
    assert len(actions) == 1 and actions[0].status == "completed"
    assert actions[0].action_id == pending["action_id"]
    refund_id = business.effects[0]["result"]["refund_id"]
    assert refund_id in json.dumps(actions[0].result)
    retry = await _bot(business.case, "confirm", transport, include_completion=True)
    content = _content(retry, transport)
    assert len(business.effects) == 1
    assert refund_id in content
    latest = (await _bot_rows(business.case))[-1]
    assert latest.content == content
    metadata = latest.message_metadata
    assert isinstance(metadata, dict)
    action = metadata["action_execution"]
    assert action == {
        "action_id": pending["action_id"],
        "status": "completed",
        "replayed": True,
        "tool": "process_refund",
    }
    assert not metadata.get("executed_tools")
    if transport == "normal":
        assert retry.json()["metadata"]["action_execution"] == action
        assert not retry.json()["metadata"].get("executed_tools")
    history = await business.case.client.get(
        "/api/v1/chat/history",
        params={
            "session_id": business.case.private_id,
            "expected_session_key": await _session_key(business.case),
        },
        headers=business.case.headers[1],
    )
    assert history.status_code == 200
    assert history.json()["messages"][-1]["content"] == content
    # A subsequent turn with no pending action must not inherit its receipt.
    # Use a fixed meta response: this fixture deliberately has no text LLM.
    ordinary = await _bot(business.case, "cancel", transport)
    ordinary_content = _content(ordinary, transport)
    fresh = (await _bot_rows(business.case))[-1]
    assert fresh.id > latest.id and fresh.content == ordinary_content
    fresh_metadata = fresh.message_metadata
    assert isinstance(fresh_metadata, dict)
    assert not fresh_metadata.get("action_execution")
    assert not fresh_metadata.get("executed_tools")
    if transport == "normal":
        assert not ordinary.json()["metadata"].get("action_execution")


@pytest.mark.parametrize("transport", ["normal", "stream"])
async def test_actual_sql_reservation_rejection_dispatches_no_handler(
    business: _Business, transport: str
) -> None:
    await _stage(business)
    await _trigger(business, "reservation")
    response = await _bot(business.case, "confirm", transport)
    content = _content(response, transport)
    assert business.effects == []
    assert await _actions(business) == []
    _uncertain_copy(content, business)
    await _drop_trigger(business)


@pytest.mark.parametrize("transport", ["normal", "stream"])
async def test_actual_sql_result_write_rejection_blocks_automatic_retry(
    business: _Business, transport: str
) -> None:
    await _stage(business)
    await _trigger(business, "result")
    first = await _bot(business.case, "confirm", transport)
    _uncertain_copy(_content(first, transport), business)
    assert len(business.effects) == 1
    rows = await _actions(business)
    assert len(rows) == 1 and rows[0].status in {"in_progress", "unknown"}
    assert rows[0].result is None
    await _drop_trigger(business)
    retry = await _bot(business.case, "confirm", transport)
    _uncertain_copy(_content(retry, transport), business)
    assert len(business.effects) == 1


@pytest.mark.parametrize("transport", ["normal", "stream"])
async def test_handler_exception_after_actual_effect_is_unknown_and_never_retried(
    business: _Business, transport: str
) -> None:
    await _stage(business)
    tool = business.case.registry.get_tool_for_intent("refund")
    assert tool is not None
    original = tool.handler

    async def lost_response(args: dict[str, Any]) -> Any:
        await original(args)
        raise RuntimeError("isolated provider lost acknowledgement after actual mock effect")

    business.monkeypatch.setattr(tool, "handler", lost_response)
    for _ in range(2):
        response = await _bot(business.case, "confirm", transport)
        _uncertain_copy(_content(response, transport), business)
        assert len(business.effects) == 1
    rows = await _actions(business)
    assert len(rows) == 1 and rows[0].status in {"in_progress", "unknown"}
    assert rows[0].result is None


@pytest.mark.parametrize("transport", ["normal", "stream"])
async def test_completed_receipt_remains_observable_after_proposal_expiry(
    business: _Business, transport: str
) -> None:
    pending = await _stage(business)
    business.checkpoint.armed = True
    await _bot(business.case, "confirm")
    assert len(business.effects) == 1
    pending["staged_at"] = time.time() - 120
    business.monkeypatch.setattr(settings, "CONFIRMATION_TTL_SECONDS", 1)
    await business.graph.aupdate_state(
        business.config(), {"pending_confirmation": pending, "intent": "refund"}
    )
    response = await _bot(business.case, "confirm", transport)
    assert business.effects[0]["result"]["refund_id"] in _content(response, transport)
    assert len(business.effects) == 1


@pytest.mark.parametrize("staging", ["deterministic", "agent"])
@pytest.mark.parametrize("kind", ["legacy", "expired"])
async def test_expired_unstarted_action_and_legacy_action_never_dispatch(
    business: _Business, staging: str, kind: str
) -> None:
    pending = await _stage(business, staging)
    if kind == "legacy":
        pending.pop("action_id")
    else:
        pending["staged_at"] = time.time() - 120
        business.monkeypatch.setattr(settings, "CONFIRMATION_TTL_SECONDS", 1)
    await business.graph.aupdate_state(
        business.config(), {"pending_confirmation": pending, "intent": "refund"}
    )
    response = await _bot(business.case, "confirm")
    _uncertain_copy(_content(response, "normal"), business)
    assert business.effects == [] and await _actions(business) == []


@pytest.mark.parametrize("command", ["cancel", "no"])
@pytest.mark.parametrize("outcome", ["completed", "unknown"])
async def test_cancel_and_deny_do_not_claim_prior_effect_was_reversed(
    business: _Business, command: str, outcome: str
) -> None:
    await _stage(business)
    if outcome == "completed":
        business.checkpoint.armed = True
        await _bot(business.case, "confirm")
    else:
        await _trigger(business, "result")
        await _bot(business.case, "confirm")
        await _drop_trigger(business)
    assert len(business.effects) == 1
    response = await _bot(business.case, command)
    content = _content(response, "normal")
    assert "未执行任何更改" not in content
    assert "未执行任何操作" not in content
    assert "no changes" not in content.lower()
    assert len(business.effects) == 1
    rows = await _actions(business)
    assert len(rows) == 1 and rows[0].status == outcome


@pytest.mark.parametrize("conflict", ["arguments", "tool", "actor", "session", "scope"])
async def test_transplanted_or_modified_completed_action_never_replays_or_dispatches(
    business: _Business, conflict: str
) -> None:
    pending = await _stage(business)
    business.checkpoint.armed = True
    await _bot(business.case, "confirm")
    assert len(business.effects) == 1
    refund_id = business.effects[0]["result"]["refund_id"]
    target = business.case.private_id
    actor = 1
    thread = str(target)
    if conflict == "arguments":
        pending["args"] = {**pending["args"], "reason": "changed after confirmation"}
    elif conflict == "tool":
        pending["intent"] = "return"
    elif conflict == "actor":
        async with business.case.maker() as db:
            row = await db.get(ChatSession, target)
            assert row is not None
            row.user_id = 2
            await db.commit()
        actor = 2
    elif conflict == "session":
        created = await business.case.client.post(
            "/api/v1/sessions", headers=business.case.headers[1], json={"title": "Other session"}
        )
        assert created.status_code == 200
        target = created.json()["id"]
        thread = str(target)
    else:
        async with business.case.maker() as db:
            row = await db.get(ChatSession, target)
            assert row is not None
            row.automation_epoch += 1
            thread = f"{target}:automation:{row.automation_epoch}"
            await db.commit()
    await business.graph.aupdate_state(
        business.config(thread),
        {"pending_confirmation": pending, "intent": "refund", "session_id": target},
    )
    response = await _bot(business.case, "confirm", session_id=target, actor=actor)
    content = _content(response, "normal")
    assert len(business.effects) == 1
    assert refund_id not in content
    _uncertain_copy(content, business)
    assert len(await _actions(business)) == 1


@pytest.mark.parametrize("actor", [0, 2, 3])
async def test_http_authentication_and_owner_gate_precede_action_execution(
    business: _Business, actor: int
) -> None:
    await _stage(business)
    response = await _bot(business.case, "confirm", actor=actor)
    assert response.status_code == (401 if actor == 0 else 404)
    assert business.effects == []
    assert await _actions(business) == []
