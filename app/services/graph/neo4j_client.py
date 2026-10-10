"""
Neo4j graph database client implementation.

Uses the official neo4j Python driver with async support.
"""

import json
import logging
from typing import Any

from app.services.graph.base import (
    GraphClient,
    GraphClientError,
    GraphEntity,
    GraphRelation,
    GraphSearchResult,
)

logger = logging.getLogger(__name__)

_UNSAFE_KEYWORDS = frozenset(
    {
        "CREATE",
        "DELETE",
        "DETACH",
        "SET ",
        "REMOVE",
        "MERGE",
        "DROP",
        "FOREACH",
    }
)


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

    async def add_entities(self, entities: list[GraphEntity]) -> list[str]:
        if not entities:
            return []

        ids: list[str] = []
        for ent in entities:
            safe_type = _sanitize_label(ent.type)
            query = f"""
            MERGE (node:Entity {{name: $name, type: $type}})
            ON CREATE SET node.id = $id
            SET node:`{safe_type}`, node += $props
            RETURN node.id AS id
            """
            props: dict[str, Any] = {
                **ent.properties,
                "type": ent.type,
            }
            if ent.description:
                props["description"] = ent.description
            if ent.embedding:
                props["embedding"] = ent.embedding

            results = await self._run_query(
                query,
                {"name": ent.name, "type": ent.type, "id": ent.id, "props": props},
            )
            if results and results[0].get("id"):
                ids.append(results[0]["id"])
        return ids

    async def add_relations(self, relations: list[GraphRelation]) -> list[str]:
        if not relations:
            return []

        ids: list[str] = []
        for rel in relations:
            safe_type = _sanitize_label(rel.relation_type)
            query = f"""
            MATCH (a:Entity {{id: $source_id}})
            MATCH (b:Entity {{id: $target_id}})
            MERGE (a)-[r:`{safe_type}`]->(b)
            SET r += $props
            RETURN type(r) AS rel_type
            """
            props: dict[str, Any] = {"id": rel.id, **rel.properties}
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

    async def replace_document_graph(
        self,
        document_id: str,
        version_id: str,
        entities: list[GraphEntity],
        relations: list[GraphRelation],
    ) -> dict[str, int]:
        """Commit a complete document graph in one retryable transaction.

        Shared Entity nodes contain identity and a projection of live source
        records. Descriptions, embeddings and custom properties belong to
        DocumentEntity records, so deleting one source cannot leave its claims
        on an entity still supported by another document. Relations are scoped
        to their source document, rather than overwriting another source's edge.
        """
        if not document_id or not version_id:
            raise GraphClientError("Document and version IDs are required")
        entity_ids = {entity.id for entity in entities}
        if len(entity_ids) != len(entities) or any(not entity.id for entity in entities):
            raise GraphClientError("Document entity IDs must be nonempty and unique")
        for entity in entities:
            if not entity.name.strip():
                raise GraphClientError("Entity names must be nonempty")
            _sanitize_label(entity.type)
            _source_chunk_ids(entity.properties)
        for relation in relations:
            if not relation.id:
                raise GraphClientError("Relation IDs must be nonempty")
            _sanitize_label(relation.relation_type)
            _source_chunk_ids(relation.properties)
            if (
                relation.source_entity_id not in entity_ids
                or relation.target_entity_id not in entity_ids
            ):
                raise GraphClientError("Relation endpoints must belong to the document graph")

        async def replace(tx: Any) -> dict[str, int]:
            await self._lock_document(tx, document_id)
            await self._clear_document_graph(tx, document_id)
            stored_ids: dict[str, str] = {}
            for entity in entities:
                safe_type = _sanitize_label(entity.type)
                records = await _transaction_query(
                    tx,
                    f"""
                    MERGE (node:Entity {{name: $name, type: $type}})
                    ON CREATE SET node.id = $id, node.document_managed = true
                    SET node:`{safe_type}`,
                        node.provenance_lock = coalesce(node.provenance_lock, 0) + 1
                    WITH node
                    FOREACH (_ IN CASE WHEN node.source_document_ids IS NULL THEN [1] ELSE [] END |
                        SET node.original_description = node.description,
                            node.original_embedding = node.embedding)
                    MERGE (source:DocumentEntity {{source_document_id: $document_id,
                                                  entity_id: node.id}})
                    SET source.source_version_id = $version_id,
                        source.description = coalesce($description, source.description),
                        source.embedding = coalesce($embedding, source.embedding),
                        source.properties_json = $properties_json,
                        source.source_chunk_ids = reduce(
                            ids = coalesce(source.source_chunk_ids, []), id IN $chunk_ids |
                            CASE WHEN id IN ids THEN ids ELSE ids + id END)
                    MERGE (source)-[:SUPPORTS_ENTITY]->(node)
                    RETURN node.id AS id
                    """,
                    {
                        "id": entity.id,
                        "name": entity.name,
                        "type": entity.type,
                        "document_id": document_id,
                        "version_id": version_id,
                        "description": entity.description,
                        "embedding": entity.embedding,
                        "properties_json": _source_properties_json(entity.properties),
                        "chunk_ids": _source_chunk_ids(entity.properties),
                    },
                )
                if len(records) != 1 or not records[0].get("id"):
                    raise GraphClientError("Entity write did not return a stored ID")
                stored_ids[entity.id] = records[0]["id"]

            stored_relations: set[str] = set()
            for relation in relations:
                safe_type = _sanitize_label(relation.relation_type)
                records = await _transaction_query(
                    tx,
                    f"""
                    MATCH (a:Entity {{id: $source_id}}), (b:Entity {{id: $target_id}})
                    MERGE (a)-[r:`{safe_type}` {{source_document_id: $document_id}}]->(b)
                    ON CREATE SET r.id = $id
                    SET r.source_version_id = $version_id,
                        r.description = $description,
                        r.properties_json = $properties_json,
                        r.source_chunk_ids = reduce(
                            ids = coalesce(r.source_chunk_ids, []), id IN $chunk_ids |
                            CASE WHEN id IN ids THEN ids ELSE ids + id END)
                    RETURN r.id AS id
                    """,
                    {
                        "id": relation.id,
                        "source_id": stored_ids[relation.source_entity_id],
                        "target_id": stored_ids[relation.target_entity_id],
                        "document_id": document_id,
                        "version_id": version_id,
                        "description": relation.description,
                        "properties_json": _source_properties_json(relation.properties),
                        "chunk_ids": _source_chunk_ids(relation.properties),
                    },
                )
                if len(records) != 1 or not records[0].get("id"):
                    raise GraphClientError("Relation write did not return a stored ID")
                stored_relations.add(records[0]["id"])

            await self._refresh_entity_sources(tx, list(set(stored_ids.values())))
            await _transaction_query(
                tx,
                "MATCH (document:IngestedDocument {id: $document_id}) "
                "SET document.version_id = $version_id",
                {"document_id": document_id, "version_id": version_id},
            )
            return {"entities": len(set(stored_ids.values())), "relations": len(stored_relations)}

        return await self._document_transaction(replace)

    async def delete_document_graph(self, document_id: str) -> dict[str, int]:
        if not document_id:
            raise GraphClientError("Document ID is required")

        async def delete(tx: Any) -> dict[str, int]:
            await self._lock_document(tx, document_id)
            counts = await self._clear_document_graph(tx, document_id)
            await _transaction_query(
                tx,
                "MATCH (document:IngestedDocument {id: $document_id}) DELETE document",
                {"document_id": document_id},
            )
            return counts

        return await self._document_transaction(delete)

    async def _document_transaction(self, operation: Any) -> dict[str, int]:
        if not self._driver:
            raise GraphClientError("Not connected to Neo4j")
        try:
            async with self._driver.session(database=self._database) as session:
                result: dict[str, int] = await session.execute_write(operation)
                return result
        except GraphClientError:
            raise
        except Exception as exc:
            raise GraphClientError(f"Document graph transaction failed: {exc}") from exc

    @staticmethod
    async def _lock_document(tx: Any, document_id: str) -> None:
        await _transaction_query(
            tx,
            "MERGE (document:IngestedDocument {id: $document_id}) "
            "SET document.provenance_lock = coalesce(document.provenance_lock, 0) + 1",
            {"document_id": document_id},
        )

    async def _clear_document_graph(self, tx: Any, document_id: str) -> dict[str, int]:
        records = await _transaction_query(
            tx,
            """
            MATCH (:Entity)-[r]->(:Entity)
            WHERE r.source_document_id = $document_id
            WITH collect(r) AS relations
            FOREACH (r IN relations | DELETE r)
            RETURN size(relations) AS count
            """,
            {"document_id": document_id},
        )
        relation_count = records[0]["count"] if records else 0
        records = await _transaction_query(
            tx,
            """
            MATCH (source:DocumentEntity {source_document_id: $document_id})
                  -[:SUPPORTS_ENTITY]->(node:Entity)
            SET node.provenance_lock = coalesce(node.provenance_lock, 0) + 1
            WITH collect(DISTINCT node.id) AS ids, collect(DISTINCT source) AS sources
            FOREACH (source IN sources | DETACH DELETE source)
            RETURN ids
            """,
            {"document_id": document_id},
        )
        ids = records[0]["ids"] if records else []
        await self._refresh_entity_sources(tx, ids)
        records = await _transaction_query(
            tx,
            """
            MATCH (node:Entity)
            WHERE node.id IN $ids AND node.document_managed = true
                  AND NOT (node)--()
            WITH collect(node) AS nodes
            FOREACH (node IN nodes | DELETE node)
            RETURN size(nodes) AS count
            """,
            {"ids": ids},
        )
        return {
            "entities": records[0]["count"] if records else 0,
            "relations": relation_count,
        }

    @staticmethod
    async def _refresh_entity_sources(tx: Any, ids: list[str]) -> None:
        if not ids:
            return
        await _transaction_query(
            tx,
            """
            MATCH (node:Entity) WHERE node.id IN $ids
            OPTIONAL MATCH (source:DocumentEntity)-[:SUPPORTS_ENTITY]->(node)
            WITH node, source ORDER BY source.source_document_id
            WITH node, collect(source) AS sources
            SET node.source_document_ids = [source IN sources | source.source_document_id],
                node.source_version_ids = [source IN sources | source.source_version_id],
                node.source_chunk_ids = reduce(ids = [], source IN sources |
                    ids + coalesce(source.source_chunk_ids, [])),
                node.description = coalesce(head([source IN sources
                    WHERE source.description IS NOT NULL | source.description]), node.original_description),
                node.embedding = coalesce(head([source IN sources
                    WHERE source.embedding IS NOT NULL | source.embedding]), node.original_embedding)
            """,
            {"ids": ids},
        )

    async def execute_cypher(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        _validate_read_only(query)
        return await self._run_query(query, params or {})

    async def search_entities_by_embedding(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        entity_types: list[str] | None = None,
    ) -> list[GraphSearchResult]:
        index_query = """
        CALL db.index.vector.queryNodes('entity_embedding', $k, $embedding)
        YIELD node, score
        WHERE $entity_types IS NULL OR node.type IN $entity_types
        RETURN node.name AS name, node.type AS type, node.id AS id,
               node.description AS description, score,
               node.source_document_ids AS source_document_ids,
               node.source_version_ids AS source_version_ids,
               node.source_chunk_ids AS source_chunk_ids
        ORDER BY score DESC
        """
        params: dict[str, Any] = {
            "k": top_k,
            "embedding": query_embedding,
            "entity_types": entity_types or None,
        }

        try:
            results = await self._run_query(index_query, params)
        except Exception:
            results = []

        if not results:
            results = await self._fallback_embedding_search(query_embedding, top_k, entity_types)

        search_results: list[GraphSearchResult] = []
        for r in results:
            provenance = {
                key: r[key]
                for key in ("source_document_ids", "source_version_ids", "source_chunk_ids")
                if r.get(key) is not None
            }
            entity = GraphEntity(
                id=r.get("id", ""),
                name=r.get("name", ""),
                type=r.get("type", ""),
                description=r.get("description"),
                properties=provenance,
            )
            search_results.append(
                GraphSearchResult(
                    content=entity.description or entity.name,
                    entities=[entity],
                    relations=[],
                    score=r.get("score", 0.0),
                    source_type="graph_embedding",
                    metadata={"entity_type": entity.type, **provenance},
                )
            )
        return search_results[:top_k]

    async def _fallback_embedding_search(
        self,
        query_embedding: list[float],
        top_k: int,
        entity_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        # Community Neo4j has no GDS plugin by default. Compute cosine with
        # built-in Cypher so provider dimensions other than the legacy 1024
        # vector index remain searchable, without silently returning no hits.
        query = """
        MATCH (e:Entity)
        WHERE e.embedding IS NOT NULL
              AND size(e.embedding) = size($embedding)
              AND ($entity_types IS NULL OR e.type IN $entity_types)
        WITH e,
             reduce(dot = 0.0, i IN range(0, size($embedding) - 1) |
                    dot + e.embedding[i] * $embedding[i]) AS dot,
             sqrt(reduce(norm = 0.0, value IN e.embedding | norm + value * value)) AS en,
             sqrt(reduce(norm = 0.0, value IN $embedding | norm + value * value)) AS qn
        WHERE en > 0 AND qn > 0
        WITH e, (1.0 + dot / (en * qn)) / 2.0 AS score
        ORDER BY score DESC
        LIMIT $k
        RETURN e.name AS name, e.type AS type, e.id AS id,
               e.description AS description, score,
               e.source_document_ids AS source_document_ids,
               e.source_version_ids AS source_version_ids,
               e.source_chunk_ids AS source_chunk_ids
        """
        try:
            return await self._run_query(
                query,
                {"embedding": query_embedding, "k": top_k, "entity_types": entity_types or None},
            )
        except Exception:
            return []

    async def get_entity_neighborhood(
        self, entity_name: str, max_hops: int = 2, limit: int = 50
    ) -> list[GraphSearchResult]:
        query = f"""
        MATCH path = (e:Entity {{name: $name}})-[*1..{max_hops}]-(neighbor:Entity)
        WITH e, neighbor, relationships(path) AS rels
        WITH e, neighbor, last(rels) AS rel
        RETURN DISTINCT
            e.id AS source_id, e.name AS source_name, e.type AS source_type,
            rel.id AS relation_id, type(rel) AS rel_type,
            neighbor.id AS target_id, neighbor.name AS target_name,
            neighbor.type AS target_type, neighbor.description AS description,
            rel.source_document_id AS source_document_id,
            rel.source_version_id AS source_version_id,
            rel.source_chunk_ids AS source_chunk_ids
        LIMIT $limit
        """
        results = await self._run_query(query, {"name": entity_name, "limit": limit})

        search_results: list[GraphSearchResult] = []
        for r in results:
            source = GraphEntity(
                id=r.get("source_id", ""), name=r["source_name"], type=r["source_type"]
            )
            target = GraphEntity(
                id=r.get("target_id", ""), name=r["target_name"], type=r["target_type"]
            )
            provenance = {
                key: r[key]
                for key in ("source_document_id", "source_version_id", "source_chunk_ids")
                if r.get(key) is not None
            }
            relation = GraphRelation(
                id=r.get("relation_id", ""),
                source_entity_id=source.id,
                target_entity_id=target.id,
                relation_type=r["rel_type"],
                properties=provenance,
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
                    metadata=provenance or None,
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
        for label in labels:
            parts.append(f"  - {label['label']}")

        parts.append("\nRelationship Types:")
        for r in rels:
            parts.append(f"  - {r['relationshipType']}")

        return "\n".join(parts)

    async def get_stats(self) -> dict[str, int]:
        entity_count = await self._run_query("MATCH (e:Entity) RETURN count(e) AS count", {})
        rel_count = await self._run_query(
            "MATCH (:Entity)-[r]->(:Entity) RETURN count(r) AS count", {}
        )
        return {
            "entities": entity_count[0]["count"] if entity_count else 0,
            "relations": rel_count[0]["count"] if rel_count else 0,
        }

    async def _run_query(self, query: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        if not self._driver:
            raise GraphClientError("Not connected to Neo4j")

        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, params)
            records: list[dict[str, Any]] = await result.data()
            return records

    async def _ensure_constraints(self) -> None:
        constraints = [
            # Older installs made name globally unique, preventing the same
            # term from being both e.g. a product and a policy. Keep identity
            # unique on (name, type), matching the MERGE used by both writers.
            "DROP CONSTRAINT entity_name IF EXISTS",
            "CREATE CONSTRAINT entity_name_type IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE (e.name, e.type) IS UNIQUE",
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT ingested_document_id IF NOT EXISTS "
            "FOR (d:IngestedDocument) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT document_entity_source IF NOT EXISTS "
            "FOR (s:DocumentEntity) REQUIRE (s.source_document_id, s.entity_id) IS UNIQUE",
        ]
        for c in constraints:
            try:
                await self._run_query(c, {})
            except Exception as exc:
                raise GraphClientError(f"Failed to ensure graph identity constraints: {c}") from exc

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


async def _transaction_query(
    transaction: Any, query: str, params: dict[str, Any]
) -> list[dict[str, Any]]:
    result = await transaction.run(query, params)
    records: list[dict[str, Any]] = await result.data()
    return records


def _source_chunk_ids(properties: dict[str, Any]) -> list[str]:
    chunk_ids = properties.get("source_chunk_ids", [])
    if not isinstance(chunk_ids, list) or any(
        not isinstance(chunk_id, str) or not chunk_id for chunk_id in chunk_ids
    ):
        raise GraphClientError("source_chunk_ids must be a list of nonempty strings")
    return list(dict.fromkeys(chunk_ids))


def _source_properties_json(properties: dict[str, Any]) -> str:
    # LLM extraction can contain nested maps, which Neo4j cannot persist as
    # properties. Keep that source-specific data intact as JSON, and reserve
    # identity/provenance fields for arguments verified by the caller.
    return json.dumps(
        {
            key: value
            for key, value in properties.items()
            if key not in {"source_document_id", "source_version_id", "source_chunk_ids"}
        },
        ensure_ascii=False,
        allow_nan=False,
    )


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
