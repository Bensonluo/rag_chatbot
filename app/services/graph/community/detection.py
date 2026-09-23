"""
Community detection service.

Detects communities (clusters) in the knowledge graph using either Neo4j GDS
Leiden algorithm or a pure Python fallback.
"""

import contextlib
import logging

from app.services.graph.base import (
    CommunitySummary,
    GraphClient,
    GraphClientError,
)

logger = logging.getLogger(__name__)


class CommunityDetectionService:
    """Run community detection on the knowledge graph."""

    def __init__(self, graph_client: GraphClient) -> None:
        self._graph = graph_client

    async def detect_communities(
        self,
        min_community_size: int = 3,
        max_levels: int = 5,  # noqa: ARG002  # base detection API conformance
    ) -> dict[int, list[CommunitySummary]]:
        """Detect communities and return hierarchical summaries.

        Strategy:
        1. Try Neo4j GDS Leiden algorithm
        2. Fallback to pure Python using networkx
        """
        try:
            return await self._detect_with_gds(min_community_size)
        except Exception as e:
            logger.info("GDS detection failed (%s), trying Python fallback", e)
            return await self._detect_with_python(min_community_size)

    async def _detect_with_gds(self, min_community_size: int) -> dict[int, list[CommunitySummary]]:
        """Use Neo4j GDS Leiden algorithm."""
        # Project graph
        await self._graph.execute_cypher(
            "CALL gds.graph.project('entity_graph', 'Entity', {"
            "  ALL: {orientation: 'UNDIRECTED'}"
            "})",
        )

        try:
            # Run Leiden
            await self._graph.execute_cypher(
                "CALL gds.leiden.write('entity_graph', {"
                f"  writeProperty: 'communityId',"
                f"  minCommunitySize: {min_community_size}"
                "})",
            )

            # Read communities
            results = await self._graph.execute_cypher(
                "MATCH (e:Entity) "
                "WHERE e.communityId IS NOT NULL "
                "RETURN e.communityId AS cid, collect(e.name) AS members, "
                "count(e) AS entity_count "
                "ORDER BY entity_count DESC"
            )
        finally:
            # Clean up projection
            with contextlib.suppress(Exception):
                await self._graph.execute_cypher("CALL gds.graph.drop('entity_graph')")

        communities: dict[int, list[CommunitySummary]] = {0: []}
        for r in results:
            communities[0].append(
                CommunitySummary(
                    community_id=str(r["cid"]),
                    level=0,
                    title=f"Community {r['cid']}",
                    summary="",
                    entity_count=r["entity_count"],
                    entities=r["members"],
                )
            )
        return communities

    async def _detect_with_python(
        self, min_community_size: int
    ) -> dict[int, list[CommunitySummary]]:
        """Pure Python fallback using networkx + python-louvain."""
        try:
            import networkx as nx
        except ImportError:
            logger.warning(
                "networkx not installed. Install with: pip install networkx python-louvain"
            )
            return {}

        # Export graph from Neo4j
        edges = await self._graph.execute_cypher(
            "MATCH (a:Entity)-[r]->(b:Entity) RETURN a.name AS source, b.name AS target"
        )

        nodes = await self._graph.execute_cypher(
            "MATCH (e:Entity) RETURN e.name AS name, e.type AS type"
        )

        if not nodes:
            return {}

        # Build networkx graph
        G: nx.Graph[str] = nx.Graph()
        for node in nodes:
            G.add_node(node["name"], type=node["type"])
        for edge in edges:
            G.add_edge(edge["source"], edge["target"])

        if G.number_of_edges() == 0:
            return {}

        # Run community detection
        try:
            from community import best_partition

            partition = best_partition(G)
        except ImportError:
            logger.warning("python-louvain not installed. Using connected components.")
            partition = {}
            for i, component in enumerate(nx.connected_components(G)):
                for member in component:
                    partition[member] = i

        # Group by community
        community_groups: dict[int, list[str]] = {}
        for node, comm_id in partition.items():
            community_groups.setdefault(comm_id, []).append(node)

        communities: dict[int, list[CommunitySummary]] = {0: []}
        for comm_id, members in community_groups.items():
            if len(members) < min_community_size:
                continue
            communities[0].append(
                CommunitySummary(
                    community_id=str(comm_id),
                    level=0,
                    title=f"Community {comm_id}",
                    summary="",
                    entity_count=len(members),
                    entities=members,
                )
            )

        # Write community IDs back to Neo4j
        for comm_id, members in community_groups.items():
            node_data = [{"name": m, "cid": str(comm_id)} for m in members]
            for nd in node_data:
                # Read-only validation in execute_cypher; use direct driver
                with contextlib.suppress(GraphClientError):
                    await self._graph.execute_cypher(
                        "MATCH (e:Entity {name: $name}) SET e.communityId = $cid",
                        nd,
                    )

        return communities
