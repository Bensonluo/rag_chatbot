"""Execution identity survives retries independently of graph checkpoints."""

import asyncio
from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy import delete, event, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.database.base import Base
from app.models.database.session import ChatSession
from app.models.database.user import User
from app.services.dialogue.action_ledger import ActionBinding


@pytest.fixture
async def ledger_stack(tmp_path):
    from app.services.dialogue.action_ledger import ActionLedger

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/actions.sqlite3")

    @event.listens_for(engine.sync_engine, "connect")
    def foreign_keys(connection, _record):
        connection.execute("PRAGMA foreign_keys=ON")

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as db:
        db.add(User(id=1, email="ledger@unit.invalid", hashed_password="!"))
        await db.flush()
        db.add(ChatSession(id=10, user_id=1))
        await db.commit()
    try:
        yield maker, ActionLedger(maker)
    finally:
        await engine.dispose()


def _binding(**changes: Any) -> ActionBinding:
    values: dict[str, Any] = {
        "action_id": str(uuid4()),
        "session_id": 10,
        "user_id": 1,
        "scope": "10:automation:1",
        "tool_name": "process_refund",
        "arguments": {"order_id": "ORD1001", "reason": "arrived damaged"},
    }
    return ActionBinding(**(values | changes))


async def test_completed_result_is_replayed_without_a_second_effect(ledger_stack):
    _, ledger = ledger_stack
    binding = _binding()
    calls = []

    async def refund():
        calls.append("effect")
        return {"success": True, "data": {"refund_id": "RF-original"}, "message": "OK"}

    first = await ledger.execute_once(binding, refund)
    second = await ledger.execute_once(binding, refund)
    assert first.status == second.status == "completed"
    assert (
        first.result
        == second.result
        == {"success": True, "data": {"refund_id": "RF-original"}, "message": "OK"}
    )
    assert not first.replayed and second.replayed
    assert calls == ["effect"]


async def test_reservation_insert_refusal_never_calls_handler(ledger_stack):
    maker, ledger = ledger_stack
    async with maker() as db:
        await db.execute(
            text(
                "CREATE TRIGGER refuse_action BEFORE INSERT ON action_executions BEGIN SELECT RAISE(ABORT, 'synthetic reservation refusal'); END"
            )
        )
        await db.commit()
    calls = []

    async def refund():
        calls.append("effect")
        return {"success": True}

    outcome = await ledger.execute_once(_binding(), refund)
    assert outcome.status == "unavailable"
    assert calls == []


async def test_lookup_and_expired_first_attempt_do_not_reserve_or_invoke(ledger_stack):
    from app.models.database.action_execution import ActionExecution

    maker, ledger = ledger_stack
    binding = _binding()

    async def forbidden():
        pytest.fail("Read-only lookup must not invoke a tool")

    assert (await ledger.lookup(binding)).status == "not_started"
    assert (await ledger.execute_once(binding, forbidden, allow_new=False)).status == "not_started"
    async with maker() as db:
        assert await db.scalar(select(func.count()).select_from(ActionExecution)) == 0


async def test_completed_result_replays_even_when_new_dispatch_is_disallowed(ledger_stack):
    _, ledger = ledger_stack
    binding = _binding()

    async def refuse():
        return {"success": False, "data": {}, "message": "Business request refused"}

    first = await ledger.execute_once(binding, refuse)

    async def forbidden():
        pytest.fail("Completed request must be replayed")

    replay = await ledger.execute_once(binding, forbidden, allow_new=False)
    assert first.status == replay.status == "completed"
    assert first.result == replay.result
    assert replay.result["success"] is False and replay.replayed


@pytest.mark.parametrize(
    "changes",
    [
        {"session_id": 11},
        {"user_id": 2},
        {"scope": "10:automation:2"},
        {"tool_name": "process_return"},
        {"arguments": {"order_id": "ORD1002", "reason": "different"}},
    ],
)
async def test_same_id_with_changed_scope_is_denied_without_receipt(ledger_stack, changes):
    _, ledger = ledger_stack
    binding = _binding()

    async def effect():
        return {"success": True, "data": {"refund_id": "private-receipt"}, "message": "OK"}

    await ledger.execute_once(binding, effect)
    changed = _binding(
        **{
            "action_id": binding.action_id,
            "session_id": binding.session_id,
            "user_id": binding.user_id,
            "scope": binding.scope,
            "tool_name": binding.tool_name,
            "arguments": binding.arguments,
        }
        | changes
    )

    async def forbidden():
        pytest.fail("Conflicting action identity must not invoke")

    assert (await ledger.lookup(changed)).status == "conflict"
    result = await ledger.execute_once(changed, forbidden)
    assert result.status == "conflict" and result.result is None


async def test_canonical_argument_order_and_caller_mutation_do_not_change_identity(ledger_stack):
    _, ledger = ledger_stack
    args = {"reason": "damaged", "details": {"b": [1, True], "a": None}}
    binding = _binding(arguments=args)
    args["reason"] = "caller changed its own object"
    calls = []

    async def effect():
        calls.append("effect")
        return {"success": True, "data": {}, "message": "OK"}

    await ledger.execute_once(binding, effect)
    reordered = _binding(
        action_id=binding.action_id,
        arguments={"details": {"a": None, "b": [1, True]}, "reason": "damaged"},
    )
    assert (await ledger.execute_once(reordered, effect)).replayed
    assert calls == ["effect"]


async def test_reservation_is_committed_before_handler_and_concurrent_retry_is_blocked(
    ledger_stack,
):
    from app.models.database.action_execution import ActionExecution

    maker, ledger = ledger_stack
    binding = _binding()
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def effect():
        async with maker() as db:
            row = await db.get(ActionExecution, binding.action_id)
            assert row is not None and row.status == "in_progress"
        calls.append("effect")
        entered.set()
        await release.wait()
        return {"success": True, "data": {}, "message": "OK"}

    task = asyncio.create_task(ledger.execute_once(binding, effect))
    await asyncio.wait_for(entered.wait(), 3)
    try:
        second = await ledger.execute_once(binding, effect)
        assert second.status == "unknown" and second.result is None
        assert calls == ["effect"]
    finally:
        release.set()
    assert (await task).status == "completed"
    assert (await ledger.execute_once(binding, effect)).replayed
    assert calls == ["effect"]


@pytest.mark.parametrize("error", [RuntimeError("provider disconnected"), TimeoutError()])
async def test_handler_exception_remains_unknown_after_service_restart(ledger_stack, error):
    from app.services.dialogue.action_ledger import ActionLedger

    maker, ledger = ledger_stack
    binding = _binding()
    calls = []

    async def uncertain():
        calls.append("effect may have happened")
        raise error

    assert (await ledger.execute_once(binding, uncertain)).status == "unknown"
    assert (await ActionLedger(maker).execute_once(binding, uncertain)).status == "unknown"
    assert calls == ["effect may have happened"]


async def test_completion_update_refusal_does_not_retry_the_effect(ledger_stack):
    from app.models.database.action_execution import ActionExecution

    maker, ledger = ledger_stack
    async with maker() as db:
        await db.execute(
            text(
                "CREATE TRIGGER refuse_completion BEFORE UPDATE ON action_executions WHEN NEW.status='completed' BEGIN SELECT RAISE(ABORT, 'synthetic result refusal'); END"
            )
        )
        await db.commit()
    binding = _binding()
    calls = []

    async def effect():
        calls.append("effect")
        return {"success": True, "data": {"refund_id": "already-issued"}, "message": "OK"}

    first = await ledger.execute_once(binding, effect)
    assert first.status == "unknown" and first.result is None
    assert (await ledger.execute_once(binding, effect)).status == "unknown"
    assert calls == ["effect"]
    async with maker() as db:
        row = await db.get(ActionExecution, binding.action_id)
        assert row.status in {"in_progress", "unknown"} and row.result is None


@pytest.mark.parametrize("after_commit", [False, True])
async def test_reservation_commit_failure_never_dispatches(ledger_stack, after_commit):
    from app.models.database.action_execution import ActionExecution
    from app.services.dialogue.action_ledger import ActionLedger

    maker, _ = ledger_stack

    class RefusedCommit(AsyncSession):
        async def commit(self):
            if after_commit:
                await super().commit()
            raise RuntimeError("synthetic uncertain commit")

    failing = ActionLedger(async_sessionmaker(maker.kw["bind"], class_=RefusedCommit))
    binding = _binding()

    async def forbidden():
        pytest.fail("Failed reservation commit must not dispatch")

    assert (await failing.execute_once(binding, forbidden)).status == "unavailable"
    async with maker() as db:
        row = await db.get(ActionExecution, binding.action_id)
        assert (row is not None) == after_commit
    expected = "unknown" if after_commit else "not_started"
    assert (await ActionLedger(maker).lookup(binding)).status == expected


async def test_completion_commit_ack_failure_does_not_overwrite_committed_receipt(ledger_stack):
    from app.models.database.action_execution import ActionExecution
    from app.services.dialogue.action_ledger import ActionLedger

    maker, _ = ledger_stack

    class UnknownCompletionCommit(AsyncSession):
        async def commit(self):
            reservation = any(isinstance(row, ActionExecution) for row in self.new)
            await super().commit()
            if not reservation:
                raise RuntimeError("commit completed but acknowledgement lost")

    ledger = ActionLedger(async_sessionmaker(maker.kw["bind"], class_=UnknownCompletionCommit))
    binding = _binding()
    calls = []

    async def effect():
        calls.append("effect")
        return {"success": True, "data": {"refund_id": "original"}, "message": "OK"}

    assert (await ledger.execute_once(binding, effect)).status == "unknown"
    replay = await ActionLedger(maker).execute_once(binding, effect)
    assert replay.status == "completed" and replay.replayed
    assert replay.result is not None
    assert replay.result["data"]["refund_id"] == "original"
    assert calls == ["effect"]


async def test_cancelled_handler_keeps_a_durable_non_retryable_reservation(ledger_stack):
    from app.services.dialogue.action_ledger import ActionLedger

    maker, ledger = ledger_stack
    binding = _binding()
    entered = asyncio.Event()
    calls = []

    async def uncertain():
        calls.append("effect may have happened")
        entered.set()
        await asyncio.Event().wait()
        return {}

    task = asyncio.create_task(ledger.execute_once(binding, uncertain))
    await asyncio.wait_for(entered.wait(), 3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert (await ActionLedger(maker).execute_once(binding, uncertain)).status == "unknown"
    assert calls == ["effect may have happened"]


@pytest.mark.parametrize("result", [None, [], {1: "key"}, {"nan": float("nan")}, {"bad": {1, 2}}])
async def test_invalid_result_after_call_is_unknown_and_not_retried(ledger_stack, result):
    _, ledger = ledger_stack
    binding = _binding()
    calls = []

    async def effect():
        calls.append("effect")
        return result

    assert (await ledger.execute_once(binding, effect)).status == "unknown"
    assert (await ledger.execute_once(binding, effect)).status == "unknown"
    assert calls == ["effect"]


async def test_result_limit_counts_utf8_bytes_and_never_reexecutes(ledger_stack):
    from app.services.dialogue.action_ledger import ActionLedger

    maker, _ = ledger_stack
    ledger = ActionLedger(maker, max_result_bytes=64)
    binding = _binding()

    async def oversized():
        return {"message": "中" * 30}

    assert (await ledger.execute_once(binding, oversized)).status == "unknown"

    async def forbidden():
        pytest.fail("Oversized results cannot authorize another execution")

    assert (await ledger.execute_once(binding, forbidden)).status == "unknown"


async def test_deleted_session_retains_fact_and_reused_numeric_id_cannot_replay(ledger_stack):
    from app.models.database.action_execution import ActionExecution

    maker, ledger = ledger_stack
    binding = _binding(user_id=0)

    async def effect():
        return {"success": True, "data": {"refund_id": "retained"}, "message": "OK"}

    await ledger.execute_once(binding, effect)
    async with maker() as db:
        await db.execute(delete(ChatSession).where(ChatSession.id == 10))
        await db.commit()
        row = await db.get(ActionExecution, binding.action_id)
        assert row is not None and row.live_session_id is None
        assert row.scope_session_id == 10 and row.result["data"]["refund_id"] == "retained"
        db.add(ChatSession(id=10, user_id=1))
        await db.commit()

    async def forbidden():
        pytest.fail("Deleted scope cannot replay or reexecute an old action")

    result = await ledger.execute_once(binding, forbidden)
    assert result.status == "conflict" and result.result is None


@pytest.mark.parametrize(
    "changes",
    [
        {"action_id": "not-a-uuid"},
        {"action_id": "00000000000000000000000000000000"},
        {"session_id": True},
        {"session_id": 0},
        {"session_id": 2**31},
        {"user_id": False},
        {"user_id": -1},
        {"scope": " "},
        {"scope": "x" * 97},
        {"tool_name": ""},
        {"arguments": {1: "value"}},
        {"arguments": {"bad": float("inf")}},
        {"arguments": {"bad": (1, 2)}},
    ],
)
def test_invalid_binding_cannot_become_an_execution_identity(changes):
    with pytest.raises(ValueError):
        _binding(**changes)


async def test_binding_mutation_after_construction_is_rejected_before_dispatch(ledger_stack):
    _, ledger = ledger_stack
    binding = _binding()
    binding.arguments["order_id"] = "changed-after-construction"

    async def forbidden():
        pytest.fail("Mutated binding must not invoke a tool")

    with pytest.raises(ValueError, match="changed after construction"):
        await ledger.execute_once(binding, forbidden)


async def test_database_wait_is_bounded_and_never_authorizes_a_handler(ledger_stack):
    from app.services.dialogue.action_ledger import ActionLedger

    maker, _ = ledger_stack

    class DelayedLookup(AsyncSession):
        async def get(self, *args, **kwargs):
            await asyncio.Event().wait()

    ledger = ActionLedger(
        async_sessionmaker(maker.kw["bind"], class_=DelayedLookup), timeout_seconds=0.01
    )
    assert (await asyncio.wait_for(ledger.lookup(_binding()), 1)).status == "unavailable"
