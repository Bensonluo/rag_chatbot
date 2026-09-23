"""
Factory for creating chat service instances.

Builds the LangGraph dialogue graph and wires it into ChatService,
while keeping backward-compatible service construction.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from app.config.settings import settings
from app.core.exceptions import ValidationError
from app.repositories.message_repository import MessageRepository
from app.repositories.session_repository import SessionRepository
from app.services.chat.chat_service import ChatService
from app.services.intent import IntentFactory
from app.services.intent.base import IntentDetector
from app.services.llm.base import LLMServiceBase
from app.services.memory import MemoryFactory
from app.services.memory.base import MemoryStrategy

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver

    from app.services.chat.knowledge_gap_recorder import KnowledgeGapRecorder
    from app.services.chat.persistence import ChatMessagePersister
    from app.services.graph.community import GlobalSearchService
    from app.services.graph.retrieval import (
        GraphRetrievalService,
        MultiPathRetrievalFusion,
    )
    from app.services.guardrails.base import GuardrailService
    from app.services.slot_filling.base import SlotFiller

logger = logging.getLogger(__name__)


class ChatServiceFactory:
    """
    Factory for creating chat service instances.

    Provides methods for creating chat service with proper dependency injection.
    """

    @staticmethod
    def create(
        llm_service: LLMServiceBase,
        memory_strategy: MemoryStrategy,
        intent_detector: IntentDetector,
        retrieval_pipeline: dict[str, Any] | None = None,  # noqa: ARG004 - legacy
        graph_retrieval_service: GraphRetrievalService | None = None,  # noqa: ARG004
        global_search_service: GlobalSearchService | None = None,  # noqa: ARG004
        multi_path_fusion: MultiPathRetrievalFusion | None = None,  # noqa: ARG004
        slot_filler: SlotFiller | None = None,  # noqa: ARG004 - legacy signature
    ) -> ChatService:
        """
        Legacy factory method — builds ChatService without LangGraph.

        .. deprecated::
            Use :meth:`create_with_defaults` which constructs a LangGraph
            dialogue graph internally.

        Args:
            llm_service: LLM service
            memory_strategy: Memory strategy
            intent_detector: Intent detector
            retrieval_pipeline: Optional retrieval pipeline
            graph_retrieval_service: Optional graph retrieval service
            global_search_service: Optional global search service
            multi_path_fusion: Optional multi-path fusion
            slot_filler: Optional slot filler

        Returns:
            ChatService with no graph (backward compat only)
        """
        if llm_service is None:
            raise ValidationError("llm_service is required")
        if memory_strategy is None:
            raise ValidationError("memory_strategy is required")
        if intent_detector is None:
            raise ValidationError("intent_detector is required")

        return ChatService(
            graph=None,
            llm_service=llm_service,
            memory_strategy=memory_strategy,
        )

    @staticmethod
    def create_with_defaults(
        llm_service: LLMServiceBase,
        message_repo: MessageRepository,
        session_repo: SessionRepository,  # noqa: ARG004 - interface symmetry
        memory_type: str = "optimized",
        intent_type: str = "hybrid",
        retrieval_pipeline: dict[str, Any] | None = None,
        graph_retrieval_service: GraphRetrievalService | None = None,
        global_search_service: GlobalSearchService | None = None,  # noqa: ARG004
        multi_path_fusion: MultiPathRetrievalFusion | None = None,  # noqa: ARG004
        slot_filler: SlotFiller | None = None,
        guardrail_service: GuardrailService | None = None,
        persister: ChatMessagePersister | None = None,
        gap_recorder: KnowledgeGapRecorder | None = None,
        checkpointer: BaseCheckpointSaver[Any] | None = None,
        **memory_kwargs: Any,
    ) -> ChatService:
        """
        Create chat service with default memory, intent, and LangGraph graph.

        Args:
            llm_service: LLM service for generation and intent detection
            message_repo: Message repository for memory
            session_repo: Session repository
            memory_type: Type of memory strategy
                (``sliding_window``, ``summarization``, ``hybrid``, ``optimized``)
            intent_type: Type of intent detector
                (``rule_based``, ``llm_based``, ``hybrid``)
            retrieval_pipeline: Optional retrieval pipeline dict
            graph_retrieval_service: Optional GraphRAG retrieval service
            global_search_service: Optional global search service
            multi_path_fusion: Optional multi-path retrieval fusion
            slot_filler: Optional slot filler
            guardrail_service: Optional guardrail service
            persister: Optional ChatMessagePersister for request-scoped
                turn persistence and history reads
            gap_recorder: Optional KnowledgeGapRecorder for knowledge-gap
                telemetry on unanswered knowledge turns
            checkpointer: Optional shared LangGraph checkpointer
                (Postgres-backed) for horizontally scaled deployments
            **memory_kwargs: Additional parameters for memory strategy

        Returns:
            ChatService: Configured chat service with compiled LangGraph
        """
        # Create embedding service if using optimized memory
        from app.services.embeddings import EmbeddingFactory

        embedding_service = None
        if memory_type == "optimized":
            embedding_service = EmbeddingFactory.create_from_settings()

        # Create memory strategy
        memory_strategy = MemoryFactory.create(
            memory_type=memory_type,
            message_repo=message_repo,
            llm_service=llm_service if memory_type in ["summarization", "hybrid"] else None,
            embedding_service=embedding_service,
            **memory_kwargs,
        )

        # Create intent detector
        intent_detector = IntentFactory.create(
            detector_type=intent_type,
            llm_service=llm_service if intent_type in ["llm_based", "hybrid"] else None,
        )

        # Create tool registry and build LangGraph dialogue graph
        graph = None
        try:
            from app.services.agent import AgentService
            from app.services.dialogue.graph import build_dialogue_graph
            from app.services.dialogue.tools import create_default_tool_registry
            from app.services.faq import create_faq_service
            from app.services.handoff import create_handoff_service

            tool_registry = create_default_tool_registry()
            agent_service = None
            if settings.AGENT_TOOLS_ENABLED and llm_service is not None:
                agent_service = AgentService(
                    llm_service=llm_service,
                    tool_registry=tool_registry,
                    max_steps=settings.AGENT_MAX_STEPS,
                )
            # FAQ fast path needs an embedding service for semantic
            # matching; without one (non-optimized memory setups) it
            # stays unwired and RAG serves those questions as before.
            faq_service = None
            if settings.FAQ_ENABLED and embedding_service is not None:
                faq_service = create_faq_service(
                    embedding_service=embedding_service,
                    data_file=settings.FAQ_DATA_FILE or None,
                    threshold=settings.FAQ_SIMILARITY_THRESHOLD,
                )
            graph = build_dialogue_graph(
                intent_detector=intent_detector,
                slot_filler=slot_filler,
                tool_registry=tool_registry,
                retrieval_pipeline=retrieval_pipeline,
                llm_service=llm_service,
                guardrail_service=guardrail_service,
                graph_retrieval_service=graph_retrieval_service,
                checkpointer=checkpointer,
                handoff_service=create_handoff_service(),
                agent_service=agent_service,
                faq_service=faq_service,
            )
        except Exception as exc:
            logger.warning("Failed to build LangGraph dialogue graph: %s", exc)
            logger.info("Falling back to ChatService without graph")

        return ChatService(
            graph=graph,
            llm_service=llm_service,
            memory_strategy=memory_strategy,
            guardrail_service=guardrail_service,
            persister=persister,
            gap_recorder=gap_recorder,
        )
