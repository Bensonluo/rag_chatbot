"""
Graph service base interface and data models.

Provides abstract interface for graph database operations, mirroring the
VectorClient ABC pattern from app/services/retrieval/vector_base.py.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GraphEntity:
    """An entity node in the knowledge graph."""

    id: str
    name: str
    type: str
    properties: dict[str, Any] = field(default_factory=dict)
    description: str | None = None
    embedding: list[float] | None = None


@dataclass
class GraphRelation:
    """A relationship (edge) in the knowledge graph."""

    id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: str
    properties: dict[str, Any] = field(default_factory=dict)
    description: str | None = None


@dataclass
class GraphSearchResult:
    """Result from a graph query, analogous to SearchResult."""

    content: str
    entities: list[GraphEntity]
    relations: list[GraphRelation]
    score: float
    source_type: str  # "text_to_cypher" | "graph_embedding" | "community_summary"
    metadata: dict[str, Any] | None = None


@dataclass
class CommunitySummary:
    """A community (cluster) summary for global context."""

    community_id: str
    level: int
    title: str
    summary: str
    entity_count: int
    entities: list[str]
    embedding: list[float] | None = None


class GraphClient(ABC):
    """Abstract base class for graph database clients."""

    @abstractmethod
    async def connect(self) -> None:
        pass

    @abstractmethod
    async def close(self) -> None:
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        pass

    @abstractmethod
    async def add_entities(self, entities: list[GraphEntity]) -> list[str]:
        pass

    @abstractmethod
    async def add_relations(self, relations: list[GraphRelation]) -> list[str]:
        pass

    @abstractmethod
    async def execute_cypher(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    async def search_entities_by_embedding(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        entity_types: list[str] | None = None,
    ) -> list[GraphSearchResult]:
        pass

    @abstractmethod
    async def get_entity_neighborhood(
        self, entity_name: str, max_hops: int = 2, limit: int = 50
    ) -> list[GraphSearchResult]:
        pass

    @abstractmethod
    async def get_schema(self) -> str:
        pass

    @abstractmethod
    async def get_stats(self) -> dict[str, int]:
        pass


class GraphClientError(Exception):
    """Base exception for graph client errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
