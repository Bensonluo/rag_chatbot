"""
Neo4j graph database client implementation.

Uses the official neo4j Python driver with async support.
"""
import logging
from typing import Optional, List, Dict, Any

from app.services.graph.base import (
    GraphClient,
    GraphEntity,
    GraphRelation,
    GraphSearchResult,
    GraphClientError,
)

logger = logging.getLogger(__name__)

_UNSAFE_KEYWORDS = frozenset({
    "CREATE", "DELETE", "DETACH", "SET ", "REMOVE", "MERGE",
    "DROP", "FOREACH",
})


class Neo4jClient(GraphClient):
    """Neo4j implementation of GraphClient using the official async driver."""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
        max_connection_pool_size: int = 50,
        connection_timeout: float = 30.0,
    ) -> None:
        self._uri = uri
        self._user = user
        self._password = password
        self._database = database
        self._pool_size = max_connection_pool_size
        self._timeout = connection_timeout
        self._driver: Any = None

    async def connect(self) -> None:
        try:
            from neo4j import AsyncGraphDatabase

            self._driver = AsyncGraphDatabase.driver(
                self._uri,
                auth=(self._user, self._password),
                max_connection_pool_size=self._pool_size,
                connection_timeout=self._timeout,
            )
            await self._driver.verify_connectivity()
            await self._ensure_constraints()
            logger.info("Connected to Neo4j at %s", self._uri)
        except Exception as e:
            raise GraphClientError(
                f"Failed to connect to Neo4j: {e}",
                details={"uri": self._uri},
            ) from e

    async def close(self) -> None:
        if self._driver:
            await self._driver.close()
            self._driver = None

    async def health_check(self) -> bool:
        try:
            if not self._driver:
                return False
            await self._driver.verify_connectivity()
            return True
        except Exception:
            return False

    async def add_entities(self, entities: List[GraphEntity]) -> List[str]:
        if not entities:
            return []

        ids: List[str] = []
        for ent in entities:
            safe_type = _sanitize_label(ent.type)
            query = f"""
            MERGE (node:Entity:`{safe_type}` {{name: $name}})
            ON CREATE SET node.id = $id
            SET node += $props
            RETURN node.id AS id
            """
            props: Dict[str, Any] = {
                **ent.properties,
                "type": ent.type,
            }
            if ent.description:
                props["description"] = ent.description
            if ent.embedding:
                props["embedding"] = ent.embedding

            results = await self._run_query(
                query,
                {"name": ent.name, "id": ent.id, "props": props},
            )
            if results and results[0].get("id"):
                ids.append(results[0]["id"])
        return ids

    async def add_relations(self, relations: List[GraphRelation]) -> List[str]:
        if not relations:
            return []

        ids: List[str] = []
        for rel in relations:
            safe_type = _sanitize_label(rel.relation_type)
            query = f"""
            MATCH (a:Entity {{id: $source_id}})
            MATCH (b:Entity {{id: $target_id}})
            MERGE (a)-[r:`{safe_type}`]->(b)
            SET r += $props
            RETURN type(r) AS rel_type
            """
            props: Dict[str, Any] = {"id": rel.id, **rel.properties}
            if rel.description:
                props["description"] = rel.description

            await self._run_query(
                query,
                {
                    "source_id": rel.source_entity_id,
                    "target_id": rel.target_entity_id,
                    "props": props,
                },
            )
            ids.append(rel.id)
        return ids

    async def execute_cypher(
        self, query: str, params: Optional[dict] = None
    ) -> List[Dict[str, Any]]:
        _validate_read_only(query)
        return await self._run_query(query, params or {})

    async def search_entities_by_embedding(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        entity_types: Optional[List[str]] = None,
    ) -> List[GraphSearchResult]:
        index_query = """
        CALL db.index.vector.queryNodes('entity_embedding', $k, $embedding)
        YIELD node, score
        RETURN node.name AS name, node.type AS type, node.id AS id,
               node.description AS description, score
        ORDER BY score DESC
        """
        params: Dict[str, Any] = {"k": top_k, "embedding": query_embedding}

        try:
            results = await self._run_query(index_query, params)
        except Exception:
            results = []

        if not results:
            results = await self._fallback_embedding_search(
                query_embedding, top_k, entity_types
            )

        search_results: List[GraphSearchResult] = []
        for r in results:
            entity = GraphEntity(
                id=r.get("id", ""),
                name=r.get("name", ""),
                type=r.get("type", ""),
                description=r.get("description"),
            )
            search_results.append(
                GraphSearchResult(
                    content=entity.description or entity.name,
                    entities=[entity],
                    relations=[],
                    score=r.get("score", 0.0),
                    source_type="graph_embedding",
                    metadata={"entity_type": entity.type},
                )
            )
        return search_results[:top_k]

    async def _fallback_embedding_search(
        self,
        query_embedding: List[float],
        top_k: int,
        entity_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        type_filter = ""
        if entity_types:
            labels = " OR ".join(f"e:{t}" for t in entity_types)
            type_filter = f"WHERE ({labels}) AND e.embedding IS NOT NULL"
        else:
            type_filter = "WHERE e.embedding IS NOT NULL"

        query = f"""
        MATCH (e:Entity)
        {type_filter}
        WITH e, gds.similarity.cosine(e.embedding, $embedding) AS score
        ORDER BY score DESC
        LIMIT $k
        RETURN e.name AS name, e.type AS type, e.id AS id,
               e.description AS description, score
        """
        try:
            return await self._run_query(
                query, {"embedding": query_embedding, "k": top_k}
            )
        except Exception:
            return []

    async def get_entity_neighborhood(
        self, entity_name: str, max_hops: int = 2, limit: int = 50
    ) -> List[GraphSearchResult]:
        query = f"""
        MATCH path = (e:Entity {{name: $name}})-[*1..{max_hops}]-(neighbor:Entity)
        WITH e, neighbor, relationships(path) AS rels
        WITH e, neighbor, last(rels) AS rel
        RETURN DISTINCT
            e.name AS source_name, e.type AS source_type,
            type(rel) AS rel_type,
            neighbor.name AS target_name, neighbor.type AS target_type,
            neighbor.description AS description
        LIMIT $limit
        """
        results = await self._run_query(
            query, {"name": entity_name, "limit": limit}
        )

        search_results: List[GraphSearchResult] = []
        for r in results:
            source = GraphEntity(id="", name=r["source_name"], type=r["source_type"])
            target = GraphEntity(id="", name=r["target_name"], type=r["target_type"])
            relation = GraphRelation(
                id="",
                source_entity_id="",
                target_entity_id="",
                relation_type=r["rel_type"],
            )
            content = (
                f"{source.name} ({source.type}) --[{r['rel_type']}]--> "
                f"{target.name} ({target.type})"
            )
            if r.get("description"):
                content += f": {r['description']}"

            search_results.append(
                GraphSearchResult(
                    content=content,
                    entities=[source, target],
                    relations=[relation],
                    score=1.0,
                    source_type="graph_embedding",
                )
            )
        return search_results

    async def get_schema(self) -> str:
        labels_query = "CALL db.labels() YIELD label RETURN label ORDER BY label"
        rels_query = (
            "CALL db.relationshipTypes() YIELD relationshipType "
            "RETURN relationshipType ORDER BY relationshipType"
        )

        try:
            labels = await self._run_query(labels_query, {})
            rels = await self._run_query(rels_query, {})
        except Exception as e:
            return f"Unable to read schema: {e}"

        parts = ["Node Labels:"]
        for l in labels:
            parts.append(f"  - {l['label']}")

        parts.append("\nRelationship Types:")
        for r in rels:
            parts.append(f"  - {r['relationshipType']}")

        return "\n".join(parts)

    async def get_stats(self) -> Dict[str, int]:
        entity_count = await self._run_query(
            "MATCH (e:Entity) RETURN count(e) AS count", {}
        )
        rel_count = await self._run_query(
            "MATCH ()-[r]->() RETURN count(r) AS count", {}
        )
        return {
            "entities": entity_count[0]["count"] if entity_count else 0,
            "relations": rel_count[0]["count"] if rel_count else 0,
        }

    async def _run_query(self, query: str, params: dict) -> List[Dict[str, Any]]:
        if not self._driver:
            raise GraphClientError("Not connected to Neo4j")

        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, params)
            records = await result.data()
            return records

    async def _ensure_constraints(self) -> None:
        constraints = [
            "CREATE CONSTRAINT entity_name IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            "CREATE CONSTRAINT entity_id IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE e.id IS UNIQUE",
        ]
        for c in constraints:
            try:
                await self._run_query(c, {})
            except Exception:
                logger.debug("Constraint already exists or not supported: %s", c)

        vector_index = """
        CREATE VECTOR INDEX entity_embedding IF NOT EXISTS
        FOR (e:Entity) ON (e.embedding)
        OPTIONS {
            indexConfig: {
                `vector.dimensions`: 1024,
                `vector.similarity_function`: 'cosine'
            }
        }
        """
        try:
            await self._run_query(vector_index, {})
        except Exception:
            logger.debug("Vector index creation skipped (may already exist)")


def _validate_read_only(query: str) -> None:
    upper = query.upper()
    for keyword in _UNSAFE_KEYWORDS:
        if keyword in upper:
            raise GraphClientError(
                f"Only read-only queries allowed. Found: {keyword}",
                details={"query": query[:200]},
            )


def _sanitize_label(label: str) -> str:
    sanitized = "".join(c for c in label if c.isalnum() or c == "_")
    if not sanitized:
        raise GraphClientError(f"Invalid label: {label}")
    return sanitized
