"""
Graph API endpoints for GraphRAG operations.

Provides REST API for graph queries, schema inspection, community management,
structured data import, and health checks.

Authorization: writes and expensive graph-wide computations (structured
import, community detection) require an admin; read/introspection
endpoints require an authenticated user. Only ``/health`` stays
anonymous for liveness probes.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.api.deps import get_current_active_user, require_admin
from app.models.database.user import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/graph", tags=["graph"])

_graph_client = None


def set_graph_client(client: Any) -> None:
    """Set the graph client instance (called during startup)."""
    global _graph_client
    _graph_client = client


def get_graph_client() -> Any | None:
    """Return the connected graph client, or ``None`` when GraphRAG is disabled."""
    return _graph_client


def _get_client() -> Any:
    client = get_graph_client()
    if client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph service not initialized. Set GRAPH_RAG_ENABLED=true.",
        )
    return client


# --- Schemas ---


class GraphQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language query")
    max_results: int = Field(20, gt=0, le=100)


class GraphQueryResponse(BaseModel):
    results: list[dict[str, Any]]
    count: int


class StructuredImportRequest(BaseModel):
    csv_content: str = Field(..., min_length=1)
    entity_mappings: list[dict[str, Any]]
    relation_mappings: list[dict[str, Any]] | None = None
    delimiter: str = Field(",", max_length=1)


# --- Endpoints ---


@router.get("/health")
async def graph_health() -> dict[str, str]:
    """Check graph database connectivity (anonymous: liveness probe)."""
    try:
        client = _get_client()
        healthy = await client.health_check()
        return {"status": "healthy" if healthy else "unhealthy"}
    except HTTPException:
        return {"status": "disabled"}
    except Exception as e:
        logger.error("Graph health check failed: %s", e)
        return {"status": "error"}


@router.get("/stats")
async def graph_stats(
    current_user: User = Depends(get_current_active_user),  # noqa: ARG001
) -> dict[str, Any]:
    """Return entity and relation counts."""
    client = _get_client()
    try:
        stats: dict[str, Any] = await client.get_stats()
        return stats
    except Exception as e:
        logger.error("Graph stats failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to load graph stats") from e


@router.get("/schema")
async def graph_schema(
    current_user: User = Depends(get_current_active_user),  # noqa: ARG001
) -> dict[str, Any]:
    """Return the current graph schema."""
    client = _get_client()
    try:
        schema = await client.get_schema()
        return {"schema": schema}
    except Exception as e:
        logger.error("Graph schema failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to load graph schema") from e


@router.post("/query", response_model=GraphQueryResponse)
async def graph_query(
    request: GraphQueryRequest,
    current_user: User = Depends(get_current_active_user),  # noqa: ARG001
) -> GraphQueryResponse:
    """Execute a natural language query via Text-to-Cypher."""
    from app.services.graph.retrieval import TextToCypherService
    from app.services.llm import LLMFactory

    client = _get_client()

    try:
        llm_service = LLMFactory.create_from_settings()
        t2c = TextToCypherService(
            llm_service=llm_service,
            graph_client=client,
            max_results=request.max_results,
        )
        results = await t2c.query(request.query, max_results=request.max_results)
        return GraphQueryResponse(
            results=[
                {
                    "content": r.content,
                    "score": r.score,
                    "entities": [{"name": e.name, "type": e.type} for e in r.entities],
                    "source_type": r.source_type,
                }
                for r in results
            ],
            count=len(results),
        )
    except Exception as e:
        logger.error("Graph query failed: %s", e)
        raise HTTPException(status_code=500, detail="Graph query failed") from e


@router.post("/import/structured")
async def import_structured(
    request: StructuredImportRequest,
    admin: User = Depends(require_admin),  # noqa: ARG001
) -> dict[str, int]:
    """Import structured CSV data into the knowledge graph (admin only)."""
    from app.services.graph.extraction.structured_importer import (
        ColumnMapping,
        RelationMapping,
        StructuredDataImporter,
    )

    client = _get_client()

    try:
        importer = StructuredDataImporter()
        entity_mappings = [
            ColumnMapping(
                entity_type=m["entity_type"],
                name_column=m["name_column"],
                property_columns=m.get("property_columns"),
                description_column=m.get("description_column"),
            )
            for m in request.entity_mappings
        ]

        relation_mappings = None
        if request.relation_mappings:
            relation_mappings = [
                RelationMapping(
                    source_entity_type=r["source_entity_type"],
                    source_name_column=r["source_name_column"],
                    target_entity_type=r["target_entity_type"],
                    target_name_column=r["target_name_column"],
                    relation_type=r["relation_type"],
                    relation_properties=r.get("relation_properties"),
                )
                for r in request.relation_mappings
            ]

        entities, relations = importer.import_csv(
            request.csv_content,
            entity_mappings,
            relation_mappings,
            delimiter=request.delimiter,
        )

        entity_ids = await client.add_entities(entities) if entities else []
        relation_ids = await client.add_relations(relations) if relations else []

        return {
            "entities_imported": len(entity_ids),
            "relations_imported": len(relation_ids),
        }
    except Exception as e:
        logger.error("Structured import failed: %s", e)
        raise HTTPException(status_code=500, detail="Structured import failed") from e


@router.post("/communities/detect")
async def detect_communities(
    min_size: int = Query(3, gt=0),
    max_levels: int = Query(5, gt=0),
    admin: User = Depends(require_admin),  # noqa: ARG001
) -> dict[str, Any]:
    """Run community detection on the knowledge graph (admin only)."""
    from app.services.graph.community import CommunityDetectionService

    client = _get_client()

    try:
        service = CommunityDetectionService(client)
        communities = await service.detect_communities(
            min_community_size=min_size,
            max_levels=max_levels,
        )

        total = sum(len(comms) for comms in communities.values())
        return {
            "levels": len(communities),
            "total_communities": total,
            "details": {str(level): len(comms) for level, comms in communities.items()},
        }
    except Exception as e:
        logger.error("Community detection failed: %s", e)
        raise HTTPException(status_code=500, detail="Community detection failed") from e
