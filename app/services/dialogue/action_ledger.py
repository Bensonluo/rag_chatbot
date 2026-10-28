"""Reserve an action before calling it; never retry an unsettled execution.

This ledger is independent of graph checkpoints. It does not provide an
atomic transaction with an external provider: a durable in-progress row
deliberately blocks replay after a crash whose outcome cannot be known.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import UUID

from sqlalchemy import update
from sqlalchemy.exc import IntegrityError

from app.models.database.action_execution import ActionExecution

ActionStatus = Literal["completed", "unknown", "unavailable", "conflict", "not_started"]


def _json_bytes(value: Any) -> bytes:
    """Reject non-JSON values rather than silently changing an operation."""

    def validate(item: Any) -> None:
        if item is None or type(item) in (str, bool, int, float):
            return
        if type(item) is list:
            for child in item:
                validate(child)
            return
        if type(item) is dict and all(type(key) is str for key in item):
            for child in item.values():
                validate(child)
            return
        raise ValueError("Action values must contain only JSON types and string keys")

    try:
        validate(value)
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError, UnicodeError) as exc:
        raise ValueError("Action values must be valid JSON") from exc


@dataclass(frozen=True)
class ActionBinding:
    action_id: str
    session_id: int
    user_id: int
    scope: str
    tool_name: str
    arguments: dict[str, Any]
    _arguments_hash: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            valid_id = type(self.action_id) is str and str(UUID(self.action_id)) == self.action_id
        except (ValueError, AttributeError):
            valid_id = False
        if not valid_id:
            raise ValueError("Action ID must be a canonical UUID string")
        if type(self.session_id) is not int or not 1 <= self.session_id <= 2_147_483_647:
            raise ValueError("Action session must be a positive 31-bit integer")
        if type(self.user_id) is not int or self.user_id < 0:
            raise ValueError("Action user must be a nonnegative integer")
        for value, maximum in ((self.scope, 96), (self.tool_name, 255)):
            if type(value) is not str or not value.strip() or len(value) > maximum:
                raise ValueError("Action scope and tool name must be bounded nonblank strings")
        if type(self.arguments) is not dict:
            raise ValueError("Action arguments must be an object")
        canonical = _json_bytes(self.arguments)
        # Isolate the binding from changes to the caller's original args.
        object.__setattr__(self, "arguments", json.loads(canonical))
        object.__setattr__(self, "_arguments_hash", hashlib.sha256(canonical).hexdigest())

    def arguments_hash(self) -> str:
        if hashlib.sha256(_json_bytes(self.arguments)).hexdigest() != self._arguments_hash:
            raise ValueError("Action binding arguments changed after construction")
        return self._arguments_hash


@dataclass(frozen=True)
class ActionOutcome:
    status: ActionStatus
    result: dict[str, Any] | None = None
    replayed: bool = False


class ActionLedger:
    def __init__(
        self,
        session_maker: Callable[[], Any],
        *,
        timeout_seconds: float = 3.0,
        max_result_bytes: int = 65_536,
    ) -> None:
        if (
            isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("Ledger timeout must be finite and positive")
        if type(max_result_bytes) is not int or max_result_bytes < 1:
            raise ValueError("Ledger result limit must be a positive integer")
        self._session_maker = session_maker
        self._timeout = timeout_seconds
        self._max_result_bytes = max_result_bytes

    def _result_snapshot(self, value: Any) -> dict[str, Any]:
        if type(value) is not dict:
            raise ValueError("Action result must be an object")
        raw = _json_bytes(value)
        if len(raw) > self._max_result_bytes:
            raise ValueError("Action result exceeds its storage limit")
        result: dict[str, Any] = json.loads(raw)
        return result

    @staticmethod
    def _matches(row: ActionExecution, binding: ActionBinding) -> bool:
        return (
            row.scope_session_id == binding.session_id
            and row.live_session_id == binding.session_id
            and row.user_id == binding.user_id
            and row.scope == binding.scope
            and row.tool_name == binding.tool_name
            and row.arguments_hash == binding.arguments_hash()
        )

    async def lookup(self, binding: ActionBinding) -> ActionOutcome:
        binding.arguments_hash()
        try:
            async with asyncio.timeout(self._timeout), self._session_maker() as db:
                row = await db.get(ActionExecution, binding.action_id)
                if row is None:
                    return ActionOutcome("not_started")
                if not self._matches(row, binding):
                    return ActionOutcome("conflict")
                if row.status == "completed":
                    try:
                        result = self._result_snapshot(row.result)
                    except ValueError:
                        return ActionOutcome("unknown")
                    return ActionOutcome("completed", result, replayed=True)
                return ActionOutcome("unknown")
        except Exception:  # noqa: BLE001 - durable state is unavailable, never permission to execute
            return ActionOutcome("unavailable")

    async def execute_once(
        self,
        binding: ActionBinding,
        invoke: Callable[[], Awaitable[dict[str, Any]]],
        *,
        allow_new: bool = True,
    ) -> ActionOutcome:
        arguments_hash = binding.arguments_hash()
        if type(allow_new) is not bool:
            raise ValueError("allow_new must be a boolean")
        if not allow_new:
            return await self.lookup(binding)
        try:
            async with asyncio.timeout(self._timeout), self._session_maker() as db:
                db.add(
                    ActionExecution(
                        action_id=binding.action_id,
                        scope_session_id=binding.session_id,
                        live_session_id=binding.session_id,
                        user_id=binding.user_id,
                        scope=binding.scope,
                        tool_name=binding.tool_name,
                        arguments_hash=arguments_hash,
                        status="in_progress",
                        result=None,
                    )
                )
                await db.commit()
        except IntegrityError:
            existing = await self.lookup(binding)
            # A refused INSERT (FK, trigger, etc.) is not a reservation.
            return ActionOutcome("unavailable") if existing.status == "not_started" else existing
        except Exception:  # noqa: BLE001 - uncertain commit must never dispatch the tool
            return ActionOutcome("unavailable")

        try:
            result = self._result_snapshot(await invoke())
            async with asyncio.timeout(self._timeout), self._session_maker() as db:
                changed = await db.execute(
                    update(ActionExecution)
                    .where(*self._conditions(binding))
                    .values(status="completed", result=result)
                    .returning(ActionExecution.action_id)
                )
                if changed.scalar_one_or_none() != binding.action_id:
                    raise RuntimeError("Action reservation changed before completion")
                await db.commit()
            return ActionOutcome("completed", result)
        except asyncio.CancelledError:
            # The committed reservation is already safe if cancellation
            # interrupts this best-effort status update as well.
            await self._mark_unknown(binding)
            raise
        except Exception:  # noqa: BLE001 - call may have taken effect; do not repeat it
            await self._mark_unknown(binding)
            return ActionOutcome("unknown")

    @staticmethod
    def _conditions(binding: ActionBinding) -> tuple[Any, ...]:
        return (
            ActionExecution.action_id == binding.action_id,
            ActionExecution.scope_session_id == binding.session_id,
            ActionExecution.user_id == binding.user_id,
            ActionExecution.scope == binding.scope,
            ActionExecution.tool_name == binding.tool_name,
            ActionExecution.arguments_hash == binding.arguments_hash(),
            ActionExecution.status == "in_progress",
        )

    async def _mark_unknown(self, binding: ActionBinding) -> None:
        try:
            async with asyncio.timeout(self._timeout), self._session_maker() as db:
                await db.execute(
                    update(ActionExecution)
                    .where(*self._conditions(binding))
                    .values(status="unknown")
                )
                await db.commit()
        except Exception:  # noqa: BLE001 - a retained in_progress row also blocks all retries
            pass
