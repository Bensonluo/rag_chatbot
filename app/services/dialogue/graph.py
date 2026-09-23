"""
LangGraph StateGraph builder for dialogue management.

Assembles the node graph, conditional edges, and memory checkpointer
into a compiled graph ready for invocation.
"""

import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.services.dialogue.state import DialogueState

logger = logging.getLogger(__name__)


def build_dialogue_graph(
    intent_detector,
    slot_filler,
    tool_registry,
    retrieval_pipeline,
    llm_service,
    guardrail_service=None,
    graph_retrieval_service=None,
    checkpointer=None,
    handoff_service=None,
    agent_service=None,
):
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

    Returns:
        Compiled StateGraph with the requested checkpointer.
    """
    from app.services.dialogue.nodes import NodeFactory

    factory = NodeFactory(
        intent_detector=intent_detector,
        slot_filler=slot_filler,
        tool_registry=tool_registry,
        retrieval_pipeline=retrieval_pipeline,
        llm_service=llm_service,
        guardrail_service=guardrail_service,
        graph_retrieval_service=graph_retrieval_service,
        handoff_service=handoff_service,
        agent_service=agent_service,
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
            "rag": "rag_lookup",
            "direct": "direct_response",
            "meta": "generate_response",
            "handoff": "handle_handoff",
            "agent": "handle_agent",
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
