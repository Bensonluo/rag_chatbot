"""
Community summarization service.

Generates LLM summaries for each detected community and stores them
in the graph for later retrieval by global search queries.
"""

import logging
from typing import Any

from app.services.embeddings.base import EmbeddingServiceBase
from app.services.graph.base import (
    CommunitySummary,
    GraphClient,
)
from app.services.llm.base import LLMMessage, LLMServiceBase

logger = logging.getLogger(__name__)

_SUMMARY_PROMPT = """Summarize the following group of related entities from a customer service knowledge graph.

Entities ({entity_count} total):
{entity_list}

Provide:
1. A concise title (5-10 words)
2. A summary paragraph describing what this community represents, key products/issues/solutions involved, and notable patterns.

Return ONLY a JSON object:
{{"title": "...", "summary": "..."}}"""


class CommunitySummarizationService:
    """Generate LLM summaries for each community."""

    def __init__(
        self,
        llm_service: LLMServiceBase,
        graph_client: GraphClient,
        embedding_service: EmbeddingServiceBase,
    ) -> None:
        self._llm = llm_service
        self._graph = graph_client
        self._embeddings = embedding_service

    async def generate_summaries(
        self, communities: dict[int, list[CommunitySummary]]
    ) -> list[CommunitySummary]:
        """Generate summaries for all communities at all levels."""
        all_summaries: list[CommunitySummary] = []

        for level, level_communities in communities.items():
            for community in level_communities:
                if community.entity_count < 2:
                    continue

                summary = await self._summarize_community(community, level)
                if summary:
                    all_summaries.append(summary)

        return all_summaries

    async def _summarize_community(
        self, community: CommunitySummary, level: int
    ) -> CommunitySummary | None:
        entity_list = "\n".join(f"  - {name}" for name in community.entities[:50])

        prompt = _SUMMARY_PROMPT.format(
            entity_count=community.entity_count,
            entity_list=entity_list,
        )

        messages = [
            LLMMessage(
                role="system",
                content="You summarize knowledge graph communities. Output only valid JSON.",
            ),
            LLMMessage(role="user", content=prompt),
        ]

        try:
            response = await self._llm.generate(messages=messages, max_tokens=512)
            result = self._parse_summary(response.content)
        except Exception as e:
            logger.warning("Community summary failed: %s", e)
            result = None

        title = result["title"] if result else community.title
        summary_text = result["summary"] if result else ""

        embedding = await self._embeddings.embed_single(f"{title}: {summary_text}")

        return CommunitySummary(
            community_id=community.community_id,
            level=level,
            title=title,
            summary=summary_text,
            entity_count=community.entity_count,
            entities=community.entities,
            embedding=embedding,
        )

    @staticmethod
    def _parse_summary(content: str) -> dict[str, Any] | None:
        import json

        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:])
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        try:
            parsed: object = json.loads(text)
        except json.JSONDecodeError:
            return None
        # The prompt demands a {"title", "summary"} object; a bare string
        # or list means the model broke contract, not that parsing failed.
        return parsed if isinstance(parsed, dict) else None
