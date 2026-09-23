"""
Dialogue state definition for LangGraph StateGraph.

Defines the TypedDict state that flows through all graph nodes,
representing the complete dialogue context including intent, slots,
routing decisions, tool results, and intent switch stack.
"""

from typing import TypedDict


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
    filled_slots: dict
    pending_slots: list[str]
    slot_prompt: str

    # Routing
    route: str

    # Tool execution
    tool_name: str
    tool_result: dict
    # Set while an irreversible tool is staged awaiting explicit user
    # confirmation: {"intent": ..., "args": {...}}
    pending_confirmation: dict | None

    # RAG
    retrieved_docs: list[dict]
    sources: list[str]

    # Output
    response: str

    # Human handoff: why the handoff fired (explicit / emotion /
    # refund_threshold) and the created ticket id, if any.
    handoff_reason: str
    handoff_ticket_id: int

    # Intent switch stack (manually managed, not a reducer)
    state_stack: list[dict]

    # Guardrail
    blocked: bool
    blocked_reason: str
