"""
Optimized context builder with relevance-based filtering.

Implements smart token budgeting and relevance filtering to reduce
token usage and improve response quality.
"""

from typing import Any

from app.repositories.message_repository import MessageRepository
from app.services.embeddings import EmbeddingFactory
from app.services.embeddings.base import EmbeddingServiceBase
from app.services.memory.base import MemoryStrategy, MessageContent


class OptimizedContextBuilder(MemoryStrategy):
    """
    Optimized context builder that filters irrelevant chat history.

    Features:
    - Relevance-based filtering using embeddings
    - Smart token budgeting
    - Prepends the latest session summary (when one exists) and keeps
      it overlap-free with the message window
    - Dynamic context window adjustment
    """

    def __init__(
        self,
        message_repo: MessageRepository,
        embedding_service: EmbeddingServiceBase | None = None,
        max_recent_messages: int = 3,
        max_relevant_messages: int = 5,
        relevance_threshold: float = 0.5,
        token_budget: int = 4096,
    ) -> None:
        """
        Initialize optimized context builder.

        Args:
            message_repo: Message repository
            embedding_service: Embedding service for relevance scoring
            max_recent_messages: Always include N most recent messages
            max_relevant_messages: Maximum relevant historical messages
            relevance_threshold: Minimum similarity score (0.0-1.0)
            token_budget: Maximum tokens for chat history
        """
        self.message_repo = message_repo
        self.embedding_service = embedding_service or EmbeddingFactory.create(provider="local")
        self.max_recent_messages = max_recent_messages
        self.max_relevant_messages = max_relevant_messages
        self.relevance_threshold = relevance_threshold
        self.token_budget = token_budget

    async def get_context(
        self,
        session_id: int,
        max_tokens: int | None = None,
        current_query: str | None = None,
    ) -> list[MessageContent]:
        """
        Get optimized context with relevance filtering.

        Args:
            session_id: Chat session ID
            current_query: Current user query (for relevance filtering)
            max_tokens: Override token budget

        Returns:
            List[MessageContent]: Optimized list of messages
        """
        # 1. Get all messages and convert ORM rows to MessageContent dicts —
        # the base-class token helpers subscript by key (m["content"]).
        rows = await self.message_repo.get_recent_messages(
            session_id=session_id,
            limit=100,  # Get more, will filter
        )
        # Long sessions: a compression pass (SummarizationMemory) stores a
        # system-role summary. Consume it instead of replaying the covered
        # turns — drop rows older than the summary, using the same
        # boundary the archive does (created_at < summary.created_at).
        summary = await self.message_repo.get_latest_summary(session_id)
        # System-role rows are compression artifacts (summaries), not
        # conversation turns. Dropping them also keeps the latest summary
        # from appearing twice — as a raw row AND the wrapped prepend.
        from app.models.enums.message import MessageRole

        rows = [row for row in rows if row.role != MessageRole.SYSTEM]
        if summary is not None:
            rows = [row for row in rows if row.created_at >= summary.created_at]
        all_messages = [
            MessageContent(role=row.role, content=row.content, timestamp=row.created_at)
            for row in rows
        ]
        # The repo returns newest-first; flip to chronological so window
        # slicing and the truncate_by_tokens "newest last" contract both
        # behave as documented.
        all_messages.reverse()

        if not all_messages:
            if summary is not None:
                return [
                    MessageContent(
                        role="system",
                        content=f"Previous conversation: {summary.content}",
                        timestamp=summary.created_at,
                    )
                ]
            return []

        # 2. Always include most recent messages (for continuity)
        recent_count = min(self.max_recent_messages, len(all_messages))
        recent_messages = all_messages[-recent_count:]

        # 3. Filter older messages by relevance (if query provided)
        older_messages = all_messages[:-recent_count]
        relevant_messages: list[MessageContent] = []

        if current_query and older_messages:
            relevant_messages = await self._filter_by_relevance(
                messages=older_messages, query=current_query, max_count=self.max_relevant_messages
            )

        # 4. Combine chronologically: (optional summary), relevant older
        # block, then the recent window — reads like a conversation log,
        # newest last.
        combined = relevant_messages + recent_messages
        if summary is not None:
            combined.insert(
                0,
                MessageContent(
                    role="system",
                    content=f"Previous conversation: {summary.content}",
                    timestamp=summary.created_at,
                ),
            )

        # 5. Truncate by tokens if needed
        budget = max_tokens or self.token_budget
        if budget:
            combined = await self.truncate_by_tokens(combined, budget)

        return combined

    async def _filter_by_relevance(
        self, messages: list[MessageContent], query: str, max_count: int = 5
    ) -> list[MessageContent]:
        """
        Filter messages by semantic similarity to query.

        Args:
            messages: List of historical messages
            query: Current query to compare against
            max_count: Maximum number of relevant messages to return

        Returns:
            List[MessageContent]: Most relevant messages
        """
        if not messages:
            return []

        try:
            # 1. Embed query once
            query_embedding = await self.embedding_service.embed_single(query)

            # 2. Score each message (position kept so picking by score
            #    can still present the picks chronologically)
            scored_messages = []
            for position, msg in enumerate(messages):
                # Embed message
                msg_embedding = await self.embedding_service.embed_single(msg["content"])

                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, msg_embedding)

                # Only include if above threshold
                if similarity >= self.relevance_threshold:
                    scored_messages.append((position, msg, similarity))

            # 3. Pick top N by relevance, then present chronologically
            scored_messages.sort(key=lambda x: x[2], reverse=True)
            picked = sorted(scored_messages[:max_count], key=lambda x: x[0])
            return [msg for _, msg, _ in picked]

        except Exception:
            # Fallback: return the newest of these older messages if
            # embedding fails (messages are chronological → last entries
            # are the newest).
            return messages[-max_count:]

    @staticmethod
    def _cosine_similarity(embedding1: list[float], embedding2: list[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            float: Similarity score (-1.0 to 1.0)
        """
        try:
            import numpy as np

            v1 = np.array(embedding1)
            v2 = np.array(embedding2)

            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return float(dot_product / (norm1 * norm2))

        except Exception:
            return 0.0

    async def estimate_token_savings(self, session_id: int, current_query: str) -> dict[str, Any]:
        """
        Estimate token savings from using relevance filtering.

        Args:
            session_id: Chat session ID
            current_query: Current query

        Returns:
            dict: Savings statistics
        """
        # Get all messages
        all_messages = await self.message_repo.get_recent_messages(session_id=session_id, limit=100)

        if not all_messages:
            return {"savings_percent": 0, "tokens_saved": 0}

        # Count tokens in all messages
        all_tokens = sum(len(msg.content) // 4 for msg in all_messages)

        # Get optimized context
        optimized = await self.get_context(session_id=session_id, current_query=current_query)

        # Count tokens in optimized messages (get_context returns dicts)
        optimized_tokens = sum(len(msg["content"]) // 4 for msg in optimized)

        # Calculate savings
        tokens_saved = all_tokens - optimized_tokens
        savings_percent = (tokens_saved / all_tokens * 100) if all_tokens > 0 else 0

        return {
            "total_messages": len(all_messages),
            "selected_messages": len(optimized),
            "original_tokens": all_tokens,
            "optimized_tokens": optimized_tokens,
            "tokens_saved": tokens_saved,
            "savings_percent": round(savings_percent, 1),
        }

    async def add_message(self, session_id: int, message: MessageContent) -> None:
        """Add a message to the repository."""
        # Create message in database
        from app.models.database.message import Message
        from app.models.enums.message import MessageRole, MessageStatus

        db_message = Message(
            session_id=session_id,
            role=MessageRole(message["role"]),
            content=message["content"],
            status=MessageStatus.COMPLETED,
        )

        await self.message_repo.create(db_message)

    async def clear_session(self, session_id: int) -> None:
        """Clear all messages for a session."""
        await self.message_repo.delete_by_session(session_id)
