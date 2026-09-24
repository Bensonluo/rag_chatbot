"""
LangGraph StateGraph builder for dialogue management.

Assembles the node graph, conditional edges, and memory checkpointer
into a compiled graph ready for invocation.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.services.dialogue.state import DialogueState

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph

    from app.services.agent.service import AgentService  # noqa: F401 (type refs below)
    from app.services.dialogue.tools import ToolRegistry
    from app.services.faq.store import FAQService
    from app.services.graph.retrieval.graph_retrieval_service import (
        GraphRetrievalService,
    )
    from app.services.guardrails.base import GuardrailService
    from app.services.handoff.service import HandoffService
    from app.services.intent.base import IntentDetector
    from app.services.llm.base import LLMMessage, LLMServiceBase
    from app.services.slot_filling.base import SlotFiller

logger = logging.getLogger(__name__)


def build_dialogue_graph(
    intent_detector: IntentDetector,
    # Nullable in practice: the API wiring passes None whenever slot
    # filling is disabled, and the slot node treats None as "skip filling".
    slot_filler: SlotFiller | None,
    tool_registry: ToolRegistry,
    retrieval_pipeline: dict[str, Any] | None,
    llm_service: LLMServiceBase | None,
    guardrail_service: GuardrailService | None = None,
    graph_retrieval_service: GraphRetrievalService | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    handoff_service: HandoffService | None = None,
    agent_service: AgentService | None = None,
    faq_service: FAQService | None = None,
    history_provider: Callable[[int], Awaitable[list[LLMMessage]]] | None = None,
) -> CompiledStateGraph[Any]:
    """Build and compile the dialogue StateGraph.

    Args:
        intent_detector: Service implementing detect_with_confidence().
        slot_filler: Slot filling service (reserved for future LLM-based filling).
        tool_registry: ToolRegistry instance for Function Calling.
        retrieval_pipeline: Dict with optional "hybrid_search" key for RAG.
        llm_service: LLMServiceBase implementation for response generation.
        guardrail_service: Optional GuardrailService for input safety checks.
        graph_retrieval_service: Optional service for graph-based retrieval.
        checkpointer: Optional shared checkpointer (e.g. Postgres-backed)
            for horizontal scaling; defaults to a process-local MemorySaver.
        handoff_service: Optional HandoffService for human-agent
            escalation tickets; when None the handoff node still
            responds but creates no ticket.
        agent_service: Optional AgentService (LLM function-calling
            loop). When provided, task intents route through the agent
            node, which falls back to the slot pipeline whenever the
            provider lacks tool support or the loop fails.
        faq_service: Optional FAQService (curated semantic match).
            When provided, RAG-bound intents first try the FAQ fast
            path; a hit ends the turn with a pre-approved answer and
            a miss continues into retrieval unchanged.

    Returns:
        Compiled StateGraph with the requested checkpointer.
    """
    from app.services.dialogue.nodes import NodeFactory

    factory = NodeFactory(
        intent_detector=intent_detector,
        history_provider=history_provider,
        slot_filler=slot_filler,
        tool_registry=tool_registry,
        retrieval_pipeline=retrieval_pipeline,
        llm_service=llm_service,
        guardrail_service=guardrail_service,
        graph_retrieval_service=graph_retrieval_service,
        handoff_service=handoff_service,
        agent_service=agent_service,
        faq_service=faq_service,
    )

    graph = StateGraph(DialogueState)

    # ── Nodes ────────────────────────────────────────────────────────────────
    graph.add_node("guardrail", factory.guardrail_node)
    graph.add_node("detect_intent", factory.detect_intent_node)
    graph.add_node("handle_switch", factory.handle_switch_node)
    graph.add_node("route_intent", factory.route_intent_node)
    graph.add_node("collect_slots", factory.collect_slots_node)
    graph.add_node("execute_tool", factory.execute_tool_node)
    graph.add_node("rag_lookup", factory.rag_lookup_node)
    graph.add_node("generate_response", factory.generate_response_node)
    graph.add_node("direct_response", factory.direct_response_node)
    graph.add_node("handle_handoff", factory.handle_handoff_node)
    graph.add_node("handle_agent", factory.handle_agent_node)
    graph.add_node("faq_lookup", factory.faq_lookup_node)

    # ── Fixed edges ──────────────────────────────────────────────────────────
    graph.add_edge(START, "guardrail")

    # After guardrail: skip intent detection when user is answering a slot prompt.
    graph.add_conditional_edges(
        "guardrail",
        factory.should_skip_intent,
        {
            "skip": "collect_slots",
            "full": "detect_intent",
        },
    )

    graph.add_edge("detect_intent", "handle_switch")
    graph.add_edge("handle_switch", "route_intent")

    # ── Conditional edges ────────────────────────────────────────────────────

    # After routing: fan out to the correct sub-pipeline.
    graph.add_conditional_edges(
        "route_intent",
        factory.route_by_intent,
        {
            "task": "collect_slots",
            "rag": "faq_lookup",
            "direct": "direct_response",
            "meta": "generate_response",
            "handoff": "handle_handoff",
            "agent": "handle_agent",
        },
    )

    # FAQ fast path: a curated hit ends the turn; a miss (or unwired
    # service) falls through to the full RAG pipeline unchanged.
    graph.add_conditional_edges(
        "faq_lookup",
        factory.route_after_faq,
        {
            "hit": END,
            "miss": "rag_lookup",
        },
    )

    # Agent mode: a successful run is terminal; a failed one (provider
    # without function calling, LLM outage) rejoins the deterministic
    # slot pipeline mid-flow so the user is still served.
    graph.add_conditional_edges(
        "handle_agent",
        factory.route_after_agent,
        {
            "agent_done": END,
            "agent_fallback": "collect_slots",
        },
    )

    # After slot collection: proceed to tool execution or ask for more info.
    graph.add_conditional_edges(
        "collect_slots",
        factory.check_slots,
        {
            "complete": "execute_tool",
            "missing": "generate_response",
        },
    )

    # After tool execution: irreversible tools stage the action and emit a
    # fixed confirmation question (straight to END, no LLM rewording);
    # everything else flows into response generation.
    graph.add_conditional_edges(
        "execute_tool",
        factory.after_execute_tool,
        {
            "confirm": END,
            "done": "generate_response",
        },
    )

    # ── Terminal edges ───────────────────────────────────────────────────────
    graph.add_edge("rag_lookup", "generate_response")
    graph.add_edge("direct_response", END)
    graph.add_edge("generate_response", END)
    graph.add_edge("handle_handoff", END)

    # ── Compile with the requested checkpointing backend ─────────────────────
    compiled = graph.compile(
        checkpointer=checkpointer if checkpointer is not None else MemorySaver()
    )

    logger.info("Dialogue graph compiled successfully with %d nodes", len(graph.nodes))

    return compiled
