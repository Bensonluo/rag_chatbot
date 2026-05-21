"""
Graph API endpoints for GraphRAG operations.

Provides REST API for graph queries, schema inspection, community management,
structured data import, and health checks.
"""
import logging
from typing import Optional, List

from fastapi import APIRouter, HTTPException, status, UploadFile, File
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/graph", tags=["graph"])

_graph_client = None


def set_graph_client(client):
    """Set the graph client instance (called during startup)."""
    global _graph_client
    _graph_client = client


def _get_client():
    if _graph_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph service not initialized. Set GRAPH_RAG_ENABLED=true.",
        )
    return _graph_client


# --- Schemas ---


class GraphQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language query")
    max_results: int = Field(20, gt=0, le=100)


class GraphQueryResponse(BaseModel):
    results: List[dict]
    count: int


class StructuredImportRequest(BaseModel):
    csv_content: str = Field(..., min_length=1)
    entity_mappings: List[dict]
    relation_mappings: Optional[List[dict]] = None
    delimiter: str = Field(",", max_length=1)


# --- Endpoints ---


@router.get("/health")
async def graph_health():
    """Check graph database connectivity."""
    try:
        client = _get_client()
        healthy = await client.health_check()
        return {"status": "healthy" if healthy else "unhealthy"}
    except HTTPException:
        return {"status": "disabled"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@router.get("/stats")
async def graph_stats():
    """Return entity and relation counts."""
    client = _get_client()
    try:
        stats = await client.get_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/schema")
async def graph_schema():
    """Return the current graph schema."""
    client = _get_client()
    try:
        schema = await client.get_schema()
        return {"schema": schema}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=GraphQueryResponse)
async def graph_query(request: GraphQueryRequest):
    """Execute a natural language query via Text-to-Cypher."""
    from app.services.graph.retrieval import TextToCypherService
    from app.services.llm import LLMFactory
    from app.config.settings import get_settings

    client = _get_client()
    settings = get_settings()

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
        raise HTTPException(status_code=500, detail="Graph query failed")


@router.post("/import/structured")
async def import_structured(request: StructuredImportRequest):
    """Import structured CSV data into the knowledge graph."""
    from app.services.graph.extraction.structured_importer import (
        StructuredDataImporter,
        ColumnMapping,
        RelationMapping,
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
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/communities/detect")
async def detect_communities(
    min_size: int = Field(3, gt=0),
    max_levels: int = Field(5, gt=0),
):
    """Run community detection on the knowledge graph."""
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
            "details": {
                str(level): len(comms) for level, comms in communities.items()
            },
        }
    except Exception as e:
        logger.error("Community detection failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
