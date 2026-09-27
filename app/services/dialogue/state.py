"""
Dialogue state definition for LangGraph StateGraph.

Defines the TypedDict state that flows through all graph nodes,
representing the complete dialogue context including intent, slots,
routing decisions, tool results, and intent switch stack.
"""

from typing import Any, TypedDict
from uuid import uuid4


class ToolExecutionContext(TypedDict):
    """Last tool-bearing turn, retained only for human handoff context."""

    turn_id: str
    tool_result: dict[str, Any]
    executed_tools: list[dict[str, Any]]


class TurnState(TypedDict, total=False):
    """Transient fields replaced at the start of every new graph invocation.

    Tool results, retrieval and output in this state all belong to turn_id.
    They must never be carried into a later turn by the checkpointer.
    """

    turn_id: str
    prev_intent: str
    confidence: float
    slot_prompt: str
    route: str
    route_after_agent: str
    route_after_faq: str
    route_after_cache: str
    tool_name: str
    tool_result: dict[str, Any]
    executed_tools: list[dict[str, Any]]
    retrieved_docs: list[dict[str, Any]]
    sources: list[str]
    # True when a retrieval leg actually ran this turn (empty results
    # included) — begin_turn pre-clears retrieved_docs to [], so the
    # generation-side no-evidence branch cannot key on emptiness alone.
    retrieval_ran: bool
    # True when a retrieval leg raised instead of answering: the
    # evidence gap is a service outage, not missing knowledge, and the
    # user-facing copy must say so (retry/handoff vs rephrase).
    retrieval_degraded: bool
    response: str
    handoff_reason: str
    handoff_ticket_id: int
    blocked: bool
    blocked_reason: str
    # True when this turn's generation folded in session history or
    # cross-session user facts: the answer is personal to this dialogue
    # and must never enter a shared (cross-session) cache — see the
    # L0/L1 write gates in chat_service / nodes.
    personalized: bool


class DialogueState(TurnState, total=False):
    """
    State flowing through the dialogue graph.

    LangGraph merges returned dicts into the state automatically.
    Each node returns only the fields it wants to update. Fields declared
    here survive new turns; inherited TurnState fields are reset first.
    """

    # Input
    message: str
    session_id: int
    user_id: int

    # Intent
    intent: str

    # Slots
    filled_slots: dict[str, Any]
    pending_slots: list[str]

    # Tool execution
    # Historical context is deliberately separate from current results:
    # only the handoff node consumes it, never response generation.
    last_tool_execution: ToolExecutionContext
    # Set while an irreversible tool is staged awaiting explicit user
    # confirmation: {"intent": ..., "args": {...}}
    pending_confirmation: dict[str, Any] | None
    # True from the turn a task's tool actually ran (direct execution,
    # confirm-resolved staging, or an agent tool call) until the next
    # intent switch consumes it. An executed task is terminal: it is
    # never pushed onto the switch stack, never auto-resumed, and no
    # longer captures follow-ups — re-entering it risks re-executing an
    # irreversible action (double refund).
    task_executed: bool

    # Intent switch stack (manually managed, not a reducer)
    state_stack: list[dict[str, Any]]


def begin_turn(state: DialogueState) -> DialogueState:
    """Reset per-turn outputs before guardrails, caches or routing run.

    Returning only transient updates preserves slots, confirmations,
    task terminality and suspended tasks. Each collection is fresh.
    A resumed checkpoint inside a turn does not traverse this entry node.
    """
    updates: DialogueState = {
        "turn_id": uuid4().hex,
        "prev_intent": "",
        "confidence": 0.0,
        "slot_prompt": "",
        "route": "",
        "route_after_agent": "",
        "route_after_faq": "",
        "route_after_cache": "",
        "tool_name": "",
        "tool_result": {},
        "executed_tools": [],
        "retrieved_docs": [],
        "sources": [],
        "retrieval_ran": False,
        "retrieval_degraded": False,
        "response": "",
        "handoff_reason": "",
        "handoff_ticket_id": 0,
        "blocked": False,
        "blocked_reason": "",
        "personalized": False,
    }
    if state.get("tool_result") or state.get("executed_tools"):
        updates["last_tool_execution"] = {
            "turn_id": state.get("turn_id", ""),
            "tool_result": state.get("tool_result") or {},
            "executed_tools": state.get("executed_tools") or [],
        }
    return updates
