"""
Dialogue state definition for LangGraph StateGraph.

Defines the TypedDict state that flows through all graph nodes,
representing the complete dialogue context including intent, slots,
routing decisions, tool results, and intent switch stack.
"""

from typing import Any, TypedDict


class DialogueState(TypedDict, total=False):
    """
    State flowing through the dialogue graph.

    LangGraph merges returned dicts into the state automatically.
    Each node returns only the fields it wants to update.
    """

    # Input
    message: str
    session_id: int
    user_id: int

    # Intent
    intent: str
    prev_intent: str
    confidence: float

    # Slots
    filled_slots: dict[str, Any]
    pending_slots: list[str]
    slot_prompt: str

    # Routing
    route: str
    # Agent-path routing decision: "agent_done" ends the turn,
    # "agent_fallback" rejoins the slot pipeline (see handle_agent_node).
    route_after_agent: str
    # FAQ fast-path routing decision: "hit" ends the turn with the
    # curated answer, "miss" continues into RAG (see faq_lookup_node).
    route_after_faq: str

    # Tool execution
    tool_name: str
    tool_result: dict[str, Any]
    # Structured agent audit trail ({"tool", "ok", "args", "summary"}
    # per executed call) — persisted with the assistant message and
    # included in handoff context. Declared explicitly per the
    # route_after_agent lesson: undeclared keys are silently dropped.
    executed_tools: list[dict[str, Any]]
    # Set while an irreversible tool is staged awaiting explicit user
    # confirmation: {"intent": ..., "args": {...}}
    pending_confirmation: dict[str, Any] | None

    # RAG
    retrieved_docs: list[dict[str, Any]]
    sources: list[str]

    # Output
    response: str

    # Human handoff: why the handoff fired (explicit / emotion /
    # refund_threshold) and the created ticket id, if any.
    handoff_reason: str
    handoff_ticket_id: int

    # Intent switch stack (manually managed, not a reducer)
    state_stack: list[dict[str, Any]]

    # Guardrail
    blocked: bool
    blocked_reason: str
