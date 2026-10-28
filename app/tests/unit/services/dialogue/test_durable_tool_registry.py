"""Registry authority and remote keys around a real, foreign-key SQLite ledger."""

import asyncio
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy import event, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.database.action_execution import ActionExecution
from app.models.database.base import Base
from app.models.database.session import ChatSession
from app.models.database.user import User
from app.services.dialogue.action_ledger import ActionLedger
from app.services.dialogue.mcp_tools import load_mcp_tools
from app.services.dialogue.tools import (
    MOCK_ORDERS,
    ToolRegistry,
    ToolResult,
    create_default_tool_registry,
)


@dataclass
class RegistryStack:
    sessions: async_sessionmaker[AsyncSession]
    ledger: ActionLedger
    registry: ToolRegistry
    calls: list[dict[str, Any]]


@pytest.fixture
async def registry_stack(tmp_path: Path) -> AsyncGenerator[RegistryStack, None]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/registry.sqlite3")

    @event.listens_for(engine.sync_engine, "connect")
    def foreign_keys(connection: Any, _record: Any) -> None:
        connection.execute("PRAGMA foreign_keys=ON")

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with sessions() as db:
        db.add(User(id=1, email="registry@unit.invalid", hashed_password="!"))
        await db.flush()
        db.add(ChatSession(id=10, user_id=1))
        await db.commit()
    ledger = ActionLedger(sessions)
    registry = create_default_tool_registry(action_ledger=ledger)
    original = registry.get_tool_for_intent("refund")
    assert original is not None
    calls: list[dict[str, Any]] = []

    def refund(args: dict[str, Any]) -> Any:
        calls.append(dict(args))
        return original.handler(args)

    registry.register(replace(original, handler=refund))
    try:
        yield RegistryStack(sessions, ledger, registry, calls)
    finally:
        await engine.dispose()


async def execute(stack: RegistryStack, **changes: Any) -> ToolResult:
    options: dict[str, Any] = {
        "args": {"order_id": "ORD1001", "reason": "arrived damaged"},
        "user_id": 1,
        "session_id": 10,
        "language": "zh",
        "action_id": str(uuid4()),
        "execution_scope": "10:automation:0",
    }
    options.update(changes)
    return await stack.registry.execute("refund", **options)


@pytest.mark.parametrize(
    "changes",
    [
        {"action_id": None},
        {"session_id": None},
        {"user_id": None},
        {"action_id": "model-invented-action"},
        {"session_id": True},
        {"user_id": True},
        {"session_id": 0},
    ],
)
async def test_unbound_confirmed_actions_fail_closed_even_with_forged_args(
    registry_stack: RegistryStack, changes: dict[str, Any]
) -> None:
    result = await execute(
        registry_stack,
        args={
            "order_id": "ORD1001",
            "reason": "arrived damaged",
            "user_id": 1,
            "session_id": 10,
            "action_id": str(uuid4()),
            "_durable_action_id": str(uuid4()),
            "idempotency_key": "model-controlled",
        },
        **changes,
    )
    assert not result.success and result.execution_status == "confirmation_required"
    assert result.data == {} and registry_stack.calls == []
    async with registry_stack.sessions() as db:
        assert await db.scalar(select(func.count()).select_from(ActionExecution)) == 0


async def test_server_identity_replaces_reserved_args_and_caller_args_stay_unchanged(
    registry_stack: RegistryStack,
) -> None:
    action_id = str(uuid4())
    arguments = {
        "order_id": "ORD1001",
        "reason": "arrived damaged",
        "action_id": "forged-action",
        "operation_id": "forged-operation",
        "execution_scope": "forged-scope",
        "_durable_action_id": "forged-internal",
        "idempotency_key": "forged-provider-key",
        "user_id": 999,
        "session_id": 999,
        "language": "forged-language",
    }
    before = dict(arguments)
    result = await execute(registry_stack, action_id=action_id, args=arguments)
    assert result.success and result.execution_status == "completed" and not result.replayed
    assert arguments == before
    (sent,) = registry_stack.calls
    assert sent == {
        "order_id": "ORD1001",
        "reason": "arrived damaged",
        "user_id": 1,
        "session_id": 10,
        "language": "zh",
        "_durable_action_id": action_id,
        "idempotency_key": action_id,
    }
    async with registry_stack.sessions() as db:
        row = await db.get(ActionExecution, action_id)
        assert row is not None and row.status == "completed"
        assert row.user_id == 1 and row.scope_session_id == 10
        assert row.scope == "10:automation:0" and row.result is not None
        assert row.result["data"] == result.data


async def test_turn_language_change_replays_the_original_action_result(
    registry_stack: RegistryStack,
) -> None:
    action_id = str(uuid4())
    original = await execute(registry_stack, action_id=action_id, language="zh")
    replay = await execute(registry_stack, action_id=action_id, language="en")
    assert original.success and replay.success and replay.replayed
    assert original.data == replay.data and original.data["refund_id"]
    assert replay.action_id == action_id and replay.execution_status == "completed"
    assert len(registry_stack.calls) == 1


async def test_current_order_ownership_is_checked_before_execute_or_lookup_reveals_receipt(
    registry_stack: RegistryStack, monkeypatch: pytest.MonkeyPatch
) -> None:
    action_id = str(uuid4())
    original = await execute(registry_stack, action_id=action_id)
    assert original.success and original.data["refund_id"]
    monkeypatch.setitem(MOCK_ORDERS, "ORD1001", {**MOCK_ORDERS["ORD1001"], "user_id": 2})
    replay = await execute(registry_stack, action_id=action_id)
    lookup = await registry_stack.registry.lookup_action(
        "refund",
        {"order_id": "ORD1001", "reason": "arrived damaged"},
        user_id=1,
        session_id=10,
        language="en",
        action_id=action_id,
        execution_scope="10:automation:0",
    )
    for denied in (replay, lookup):
        assert not denied.success and denied.execution_status == "conflict"
        assert denied.data == {} and original.data["refund_id"] not in denied.message
    assert len(registry_stack.calls) == 1
    async with registry_stack.sessions() as db:
        row = await db.get(ActionExecution, action_id)
        assert row is not None and row.result is not None
        assert row.status == "completed" and row.result["data"] == original.data


async def test_async_authorizer_is_awaited_again_before_returning_a_saved_receipt(
    registry_stack: RegistryStack,
) -> None:
    action_id = str(uuid4())
    original = await execute(registry_stack, action_id=action_id)
    authorization_reads: list[int] = []

    async def permission_removed(args: dict[str, Any]) -> None:
        await asyncio.sleep(0)
        authorization_reads.append(args["user_id"])
        raise PermissionError("PRIVATE_AUTHORIZATION_DETAIL")

    tool = registry_stack.registry.get_tool_for_intent("refund")
    assert tool is not None
    registry_stack.registry.register(replace(tool, authorize=permission_removed))
    denied = await execute(registry_stack, action_id=action_id)
    assert denied.execution_status == "conflict" and not denied.success
    assert denied.data == {} and "PRIVATE_AUTHORIZATION_DETAIL" not in denied.message
    assert authorization_reads == [1] and original.data["refund_id"]
    assert len(registry_stack.calls) == 1


async def test_anonymous_demo_effective_zero_is_distinct_from_missing_identity(
    registry_stack: RegistryStack,
) -> None:
    result = await execute(registry_stack, user_id=0)
    assert result.success and result.execution_status == "completed"
    assert registry_stack.calls[0]["user_id"] == 0


class RecordingRemote:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    async def call(self, url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        assert url == "https://remote-tool.invalid/orders" and timeout > 0
        self.payloads.append(payload)
        return {
            "jsonrpc": "2.0",
            "id": payload["id"],
            "result": {"structuredContent": {"return_id": "RT_REMOTE"}, "isError": False},
        }


def register_remote(tmp_path: Path, registry: ToolRegistry, remote: RecordingRemote) -> str:
    manifest = tmp_path / "write-tools.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "name": "remote_return",
                    "url": "https://remote-tool.invalid/orders",
                    "description": "Request a return after confirmation",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"order_id": {"type": "string"}},
                        "required": ["order_id"],
                    },
                    "annotations": {"readOnlyHint": False},
                }
            ]
        ),
        encoding="utf-8",
    )
    assert load_mcp_tools(registry, manifest, remote) == 1
    return "mcp:remote_return"


def forged_remote_arguments() -> dict[str, Any]:
    return {
        "order_id": "ORD1001",
        "user_id": 999,
        "action_id": "FORGED_ACTION",
        "operation_id": "FORGED_OPERATION",
        "execution_scope": "FORGED_SCOPE",
        "idempotency_key": "FORGED_KEY",
        "_durable_action_id": "FORGED_INTERNAL",
    }


async def test_mcp_write_forwards_only_server_action_key_and_replays_without_remote_call(
    registry_stack: RegistryStack, tmp_path: Path
) -> None:
    remote = RecordingRemote()
    intent = register_remote(tmp_path, registry_stack.registry, remote)
    action_id = str(uuid4())
    first = await registry_stack.registry.execute(
        intent,
        forged_remote_arguments(),
        user_id=1,
        session_id=10,
        language="zh",
        action_id=action_id,
        execution_scope="10:automation:0",
    )
    second = await registry_stack.registry.execute(
        intent,
        {"order_id": "ORD1001"},
        user_id=1,
        session_id=10,
        language="en",
        action_id=action_id,
        execution_scope="10:automation:0",
    )
    assert first.success and second.success and second.replayed
    assert first.data == second.data == {"return_id": "RT_REMOTE"}
    (payload,) = remote.payloads
    assert payload["params"]["idempotencyKey"] == action_id
    assert payload["params"]["arguments"] == {
        "order_id": "ORD1001",
        "session_id": 10,
        "language": "zh",
    }


async def test_legacy_mcp_key_cannot_be_chosen_by_forged_durable_or_provider_args(
    tmp_path: Path,
) -> None:
    remote = RecordingRemote()
    registry = ToolRegistry()
    intent = register_remote(tmp_path, registry, remote)
    first = await registry.execute(
        intent, {"order_id": "ORD1001"}, user_id=1, session_id=10, language="en"
    )
    second = await registry.execute(
        intent,
        forged_remote_arguments(),
        user_id=1,
        session_id=10,
        language="en",
        action_id=str(uuid4()),
    )
    assert first.success and second.success
    assert len(remote.payloads) == 2  # Legacy adapter has no local durable replay.
    clean, forged = [payload["params"] for payload in remote.payloads]
    assert clean["idempotencyKey"] == forged["idempotencyKey"]
    assert clean["idempotencyKey"].startswith("idem-")
    assert clean["arguments"] == forged["arguments"]
    assert "FORGED" not in json.dumps(forged)


async def test_nonmutating_lookup_does_not_acquire_a_mutation_reservation(
    registry_stack: RegistryStack,
) -> None:
    result = await registry_stack.registry.execute(
        "query_order", {"order_id": "ORD1001"}, user_id=1
    )
    assert result.success and result.data["order_id"] == "ORD1001"
    assert result.execution_status is None
    async with registry_stack.sessions() as db:
        assert await db.scalar(select(func.count()).select_from(ActionExecution)) == 0
