"""
LangGraph StateGraph builder for dialogue management.

Assembles the node graph, conditional edges, and memory checkpointer
into a compiled graph ready for invocation.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.services.dialogue.state import DialogueState, begin_turn

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph

    from app.services.agent.service import AgentService  # noqa: F401 (type refs below)
    from app.services.chat.answer_cache import AnswerCacheService
    from app.services.chat.semantic_cache import SemanticCacheService
    from app.services.dialogue.tools import ToolRegistry
    from app.services.faq.store import FAQService
    from app.services.graph.retrieval.graph_retrieval_service import (
        GraphRetrievalService,
    )
    from app.services.guardrails.base import GuardrailService
    from app.services.handoff.service import HandoffService
    from app.services.intent.base import IntentDetector
    from app.services.llm.base import LLMMessage, LLMServiceBase
    from app.services.retrieval.retrieval_cache import RetrievalCacheService
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
    system_prompt: str | None = None,
    user_facts_provider: Callable[[int], Awaitable[list[str]]] | None = None,
    answer_cache: AnswerCacheService | None = None,
    semantic_cache: SemanticCacheService | None = None,
    retrieval_cache: RetrievalCacheService | None = None,
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
        answer_cache: Optional AnswerCacheService (L0 exact-match
            cache). When provided, the post-guardrail lookup node
            replays a previously grounded answer in ~5ms on an exact
            key hit; a miss (or None) leaves the pipeline unchanged.
        semantic_cache: Optional SemanticCacheService (L1 near-duplicate
            cache). When provided, the direct tier replays cached
            smalltalk answers on semantic hits; a miss (or None)
            leaves the direct path unchanged.

    Returns:
        Compiled StateGraph with the requested checkpointer.
    """
    from app.services.dialogue.nodes import NodeFactory

    factory = NodeFactory(
        intent_detector=intent_detector,
        history_provider=history_provider,
        system_prompt=system_prompt,
        user_facts_provider=user_facts_provider,
        slot_filler=slot_filler,
        tool_registry=tool_registry,
        retrieval_pipeline=retrieval_pipeline,
        llm_service=llm_service,
        guardrail_service=guardrail_service,
        graph_retrieval_service=graph_retrieval_service,
        handoff_service=handoff_service,
        agent_service=agent_service,
        faq_service=faq_service,
        answer_cache=answer_cache,
        semantic_cache=semantic_cache,
        retrieval_cache=retrieval_cache,
    )

    graph = StateGraph(DialogueState)

    # Buffer multi-request turns until each sub-answer has passed its
    # normal claim/PII gates. Single-request token streaming is unchanged.
    def serving_node(node: Callable[..., Awaitable[dict[str, Any]]]) -> Callable[..., Any]:
        async def serve(state: DialogueState, config: RunnableConfig) -> dict[str, Any]:
            if state.get("multi_intent_turn"):
                config = {
                    **config,
                    "configurable": {
                        **(config.get("configurable") or {}),
                        "stream_queue": None,
                        "streamed_response": False,
                        "generation_evidence_replaced": False,
                    },
                }
            return await node(state, config)

        return serve

    # ── Nodes ────────────────────────────────────────────────────────────────
    graph.add_node("begin_turn", begin_turn)
    graph.add_node("guardrail", factory.guardrail_node)
    graph.add_node("prepare_requests", factory.prepare_requests_node)
    graph.add_node("answer_cache", factory.answer_cache_lookup_node)
    graph.add_node("detect_intent", factory.detect_intent_node)
    graph.add_node("handle_switch", factory.handle_switch_node)
    graph.add_node("route_intent", factory.route_intent_node)
    graph.add_node("collect_slots", factory.collect_slots_node)
    graph.add_node("execute_tool", serving_node(factory.execute_tool_node))
    graph.add_node("rag_lookup", factory.rag_lookup_node)
    graph.add_node("generate_response", serving_node(factory.generate_response_node))
    graph.add_node("direct_response", serving_node(factory.direct_response_node))
    graph.add_node("handle_handoff", serving_node(factory.handle_handoff_node))
    graph.add_node("handle_agent", serving_node(factory.handle_agent_node))
    graph.add_node("faq_lookup", serving_node(factory.faq_lookup_node))
    graph.add_node("finish_subrequest", factory.finish_subrequest_node)

    # ── Fixed edges ──────────────────────────────────────────────────────────
    graph.add_edge(START, "begin_turn")
    graph.add_edge("begin_turn", "guardrail")

    # A blocked input ends the turn at the gate: nothing downstream may
    # run — not the cache, not intent detection, not tools, not LLM
    # generation. The refusal is emitted by the guardrail node itself.
    graph.add_conditional_edges(
        "guardrail",
        factory.route_after_guardrail,
        {
            "blocked": END,
            "continue": "prepare_requests",
        },
    )

    # Multi-request discovery precedes L0 replay and slot-prompt shortcuts:
    # a cached partial answer must never hide another independent request.
    # Single-request cache hits still end here after normal output checks.
    graph.add_conditional_edges(
        "answer_cache",
        factory.route_after_cache,
        {
            "hit": END,
            "skip": "collect_slots",
            "full": "detect_intent",
        },
    )

    graph.add_edge("prepare_requests", "answer_cache")
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

    # FAQ hits and successful agent runs join the same completion boundary
    # as generated answers. Misses/fallbacks retain their existing pipeline.
    graph.add_conditional_edges(
        "faq_lookup",
        factory.route_after_faq,
        {
            "hit": "finish_subrequest",
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
            "agent_done": "finish_subrequest",
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

    # Irreversible tools stage a fixed confirmation question without LLM
    # rewording. The completion boundary pauses instead of dispatching more.
    graph.add_conditional_edges(
        "execute_tool",
        factory.after_execute_tool,
        {
            "confirm": "finish_subrequest",
            "done": "generate_response",
        },
    )

    # ── Terminal edges ───────────────────────────────────────────────────────
    graph.add_edge("rag_lookup", "generate_response")
    graph.add_edge("direct_response", "finish_subrequest")
    graph.add_edge("generate_response", "finish_subrequest")
    graph.add_edge("handle_handoff", "finish_subrequest")
    graph.add_conditional_edges(
        "finish_subrequest",
        factory.route_after_subrequest,
        {"next": "route_intent", "done": END},
    )

    # ── Compile with the requested checkpointing backend ─────────────────────
    compiled = graph.compile(
        checkpointer=checkpointer if checkpointer is not None else MemorySaver()
    )

    logger.info("Dialogue graph compiled successfully with %d nodes", len(graph.nodes))

    return compiled
