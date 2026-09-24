"""
LLM function-calling agent service.

Runs a bounded ReAct-style loop over the LLM's native function-calling
API: the model sees the tool schemas (JSON Schema, exported by
ToolRegistry), decides which tool to call with which arguments, the
service executes it through the registry (which enforces per-user
authorization), feeds the result back, and repeats until the model
produces a final answer — or the step budget is exhausted.

Safety invariants inherited from the deterministic pipeline and NOT
relaxed here:

- Irreversible tools (refund / return) are never executed inside the
  loop. The requested action is staged as ``pending_confirmation``
  with a fixed-template confirmation question; it only executes when
  the user's next turn resolves to the ``confirm`` meta intent (the
  A3 gate, unchanged).
- Tool execution goes through ``ToolRegistry.execute`` so ownership
  checks (``_owned_order``) apply with the caller's real user_id.
- The loop is bounded (``max_steps``) — a model that keeps requesting
  tools cannot spin or stack latency on the chat path.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from app.services.llm.base import LLMMessage
from app.services.llm.budget import LLMBudgetExceeded

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "你是电商平台的智能客服助手。你可以调用工具查询订单状态、查询物流、"
    "提交投诉，以及发起退款和退货。\n"
    "规则：\n"
    "1. 只处理用户本人的订单；用户没有提供订单号时，先调用 get_recent_orders "
    "查询其本人最近的订单，从结果中选择订单，仅当查不到订单或无法确定时才向用户"
    "询问，不要编造订单号。\n"
    "2. 退款、退货是不可逆操作：工具会代你向用户确认，不要试图绕过。\n"
    "3. 工具返回错误时，如实告知用户并给出下一步建议；无法解决的，建议转人工客服。\n"
    "4. 用简体中文回答，友好、专业、简洁。"
)

_FALLBACK_RESPONSE = (
    "抱歉，您的请求处理超时了。您可以重新描述问题，或回复「转人工」由人工客服为您处理。"
)

# Audit-trace summaries are bounded so a chatty tool result cannot
# bloat the persisted metadata.
_TRACE_SUMMARY_MAX = 200


@dataclass
class AgentResult:
    """Outcome of one agent run.

    Attributes:
        response: Final user-facing text (model answer, confirmation
            question, or fallback apology).
        pending_confirmation: Staged irreversible action (intent +
            args) awaiting the user's confirm/deny — mirrors the
            slot-pipeline's staging shape.
        executed_tools: Names of tools actually executed (for tests
            and tracing).
        tool_trace: Structured audit trail of every tool call —
            ``{"tool", "ok", "args", "summary"}`` per entry. Persisted
            with the assistant message so irreversible actions
            (refunds) are traceable after the fact.
        truncated: True when the step budget ran out before a final
            answer.
    """

    response: str
    pending_confirmation: dict[str, Any] | None = None
    executed_tools: list[str] = field(default_factory=list)
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    truncated: bool = False


class AgentService:
    """Bounded function-calling loop over an LLM and a ToolRegistry."""

    def __init__(
        self,
        llm_service: Any,
        tool_registry: Any,
        max_steps: int = 3,
    ) -> None:
        """
        Args:
            llm_service: LLM provider exposing ``generate_with_tools``
                (GLM / OpenAI clients, or the ResilientLLMService wrap
                of either).
            tool_registry: Registry exporting function schemas and
                enforcing per-user authorization on execute.
            max_steps: Maximum model rounds per run.
        """
        self._llm = llm_service
        self._tools = tool_registry
        self._max_steps = max(1, max_steps)

    async def run(
        self,
        user_message: str,
        user_id: int | None = None,
        context_note: str = "",
        history: list[LLMMessage] | None = None,
    ) -> AgentResult:
        """Answer one user message, orchestrating tools as needed.

        Args:
            user_message: The user's current message.
            user_id: Authenticated caller id, threaded into every tool
                execution for ownership checks.
            context_note: Compact prior-turn context (e.g. an already
                known order id) prepended as a system note so the
                agent does not re-ask for information the dialogue
                already collected.

        Returns:
            AgentResult: Final response plus any staged action.
        """
        tools = self._tools.to_function_schemas()
        messages: list[LLMMessage] = [LLMMessage(role="system", content=SYSTEM_PROMPT)]
        if context_note:
            messages.append(LLMMessage(role="system", content=context_note))
        # Prior turns sit between the policy blocks and the current ask,
        # so tool decisions can resolve "那运费谁出？" against what was
        # already agreed earlier in the session.
        if history:
            messages.extend(history)
        messages.append(LLMMessage(role="user", content=user_message))

        executed: list[str] = []
        trace: list[dict[str, Any]] = []

        for step in range(self._max_steps):
            try:
                response = await self._llm.generate_with_tools(messages, tools)
            except LLMBudgetExceeded:
                # The request's LLM call budget is spent: stop the loop
                # with what has already run — degrading beats failing
                # the whole turn.
                logger.warning("Agent stopped at step %d: request LLM budget exhausted", step)
                return AgentResult(
                    response=_FALLBACK_RESPONSE,
                    executed_tools=executed,
                    tool_trace=trace,
                    truncated=True,
                )
            calls = response.tool_calls or []

            if not calls:
                return AgentResult(
                    response=response.content or _FALLBACK_RESPONSE,
                    executed_tools=executed,
                    tool_trace=trace,
                )

            # No budget left for another model round after executing —
            # stop instead of starting work we cannot finish.
            if step == self._max_steps - 1:
                logger.warning(
                    "Agent step budget exhausted after %d rounds; tools requested again",
                    self._max_steps,
                )
                return AgentResult(
                    response=_FALLBACK_RESPONSE,
                    executed_tools=executed,
                    tool_trace=trace,
                    truncated=True,
                )

            # Echo the model's tool request back into history (the API
            # requires the assistant message before its tool results).
            messages.append(
                LLMMessage(
                    role="assistant",
                    content=response.content or "",
                    tool_calls=[_to_wire_format(c) for c in calls],
                )
            )

            for call in calls:
                staged = self._maybe_stage_confirmation(call)
                if staged is not None:
                    # Keep the audit trail of tools already run before
                    # the staged (unexecuted) irreversible action.
                    staged.executed_tools = executed
                    staged.tool_trace = trace
                    return staged
                await self._execute_call(call, messages, executed, trace, user_id)

        return AgentResult(
            response=_FALLBACK_RESPONSE,
            executed_tools=executed,
            tool_trace=trace,
            truncated=True,
        )

    # ── Internals ────────────────────────────────────────────────────────

    def _maybe_stage_confirmation(self, call: dict[str, Any]) -> AgentResult | None:
        """Stage an irreversible tool instead of executing it.

        Returns a terminal AgentResult when the requested tool is
        confirmation-gated; None when the call may proceed.
        """
        from app.services.dialogue.nodes import _build_confirmation_summary

        tool = self._tools.get_tool_by_name(call.get("name", ""))
        if tool is None or not tool.requires_confirmation:
            return None
        args = _parse_arguments(call)
        return AgentResult(
            response=_build_confirmation_summary(tool, args),
            pending_confirmation={"intent": tool.intent, "args": args},
        )

    async def _execute_call(
        self,
        call: dict[str, Any],
        messages: list[LLMMessage],
        executed: list[str],
        trace: list[dict[str, Any]],
        user_id: int | None,
    ) -> None:
        """Execute one tool call and append its result to the history."""
        tool = self._tools.get_tool_by_name(call.get("name", ""))
        if tool is None:
            unknown = f"未知工具: {call.get('name', '')}"
            messages.append(_tool_result_message(call.get("id", ""), {"error": unknown}))
            trace.append(
                {
                    "tool": str(call.get("name", "")),
                    "ok": False,
                    "args": _parse_arguments(call),
                    "summary": unknown,
                }
            )
            return

        args = _parse_arguments(call)
        result = await self._tools.execute(tool.intent, args, user_id=user_id)
        executed.append(tool.name)
        payload = result.data if result.success else {"error": result.message}
        messages.append(_tool_result_message(call.get("id", ""), payload))
        trace.append(
            {
                "tool": tool.name,
                "ok": result.success,
                "args": args,
                "summary": json.dumps(payload, ensure_ascii=False, default=str)[
                    :_TRACE_SUMMARY_MAX
                ],
            }
        )


def _parse_arguments(call: dict[str, Any]) -> dict[str, Any]:
    """Parse a tool call's arguments JSON, tolerating model garbage."""
    raw = call.get("arguments") or "{}"
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError):
        return {}


def _to_wire_format(call: dict[str, Any]) -> dict[str, Any]:
    """Rebuild the wire-format tool_call the API expects in history."""
    return {
        "id": call.get("id", ""),
        "type": "function",
        "function": {
            "name": call.get("name", ""),
            "arguments": call.get("arguments", "{}"),
        },
    }


def _tool_result_message(call_id: str, payload: dict[str, Any]) -> LLMMessage:
    """Serialize a tool result as a role=tool history message."""
    return LLMMessage(
        role="tool",
        content=json.dumps(payload, ensure_ascii=False, default=str),
        tool_call_id=call_id,
    )
