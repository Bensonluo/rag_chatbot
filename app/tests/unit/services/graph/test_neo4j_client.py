"""Unit tests for Neo4j client with mocked driver."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.graph.base import GraphClientError, GraphEntity, GraphRelation
from app.services.graph.neo4j_client import Neo4jClient, _sanitize_label, _validate_read_only


class TestNeo4jClient:
    def setup_method(self):
        self.client = Neo4jClient(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="test",
            database="neo4j",
        )
        self.client._driver = MagicMock()

    @pytest.mark.asyncio
    async def test_health_check_healthy(self):
        self.client._driver.verify_connectivity = AsyncMock(return_value=None)
        assert await self.client.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_no_driver(self):
        self.client._driver = None
        assert await self.client.health_check() is False

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self):
        self.client._driver.verify_connectivity = AsyncMock(side_effect=Exception("no connection"))
        assert await self.client.health_check() is False

    @pytest.mark.asyncio
    async def test_add_entities(self):
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.data = AsyncMock(return_value=[{"id": "e1"}])
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        self.client._driver.session = MagicMock(return_value=mock_session)

        entities = [
            GraphEntity(id="e1", name="登录失败", type="Issue", description="用户无法登录"),
        ]
        ids = await self.client.add_entities(entities)
        assert "e1" in ids

    @pytest.mark.asyncio
    async def test_add_entities_empty(self):
        ids = await self.client.add_entities([])
        assert ids == []

    @pytest.mark.asyncio
    async def test_add_relations(self):
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.data = AsyncMock(return_value=[{"rel_type": "RESOLVES"}])
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        self.client._driver.session = MagicMock(return_value=mock_session)

        from app.services.graph.base import GraphRelation

        relations = [
            GraphRelation(
                id="r1", source_entity_id="e1", target_entity_id="e2", relation_type="RESOLVES"
            ),
        ]
        ids = await self.client.add_relations(relations)
        assert "r1" in ids

    @pytest.mark.asyncio
    async def test_execute_cypher_validates_read_only(self):
        with pytest.raises(GraphClientError, match="Only read-only"):
            await self.client.execute_cypher("CREATE (n:Test)")

    @pytest.mark.asyncio
    async def test_get_stats(self):
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.data = AsyncMock(return_value=[{"count": 42}])
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        self.client._driver.session = MagicMock(return_value=mock_session)

        stats = await self.client.get_stats()
        assert stats["entities"] == 42
        assert stats["relations"] == 42

    @pytest.mark.asyncio
    async def test_replace_commerce_projection_runs_atomically(self):
        transaction = AsyncMock()

        async def execute_write(operation, *args, **kwargs):
            return await operation(transaction)

        class SessionContext:
            async def __aenter__(self) -> "SessionContext":
                return self

            async def __aexit__(self, *args: object) -> bool:
                return False

        session = SessionContext()
        session.execute_write = execute_write  # type: ignore[attr-defined]
        self.client._driver.session = MagicMock(return_value=session)

        category = GraphEntity(
            id="commerce-category:speaker",
            name="商品类目：音箱",
            type="CommerceCategory",
            description="音频播放",
            properties={"commerce_projection": True},
            embedding=[1.0],
        )
        attribute = GraphEntity(
            id="commerce-attribute:speaker:connection",
            name="音箱属性：连接",
            type="CommerceAttribute",
            description="蓝牙或有线",
            properties={"commerce_projection": True},
        )
        relation = GraphRelation(
            id="HAS_COMMERCE_ATTRIBUTE:speaker:connection",
            source_entity_id=category.id,
            target_entity_id=attribute.id,
            relation_type="HAS_COMMERCE_ATTRIBUTE",
            description="音箱拥有连接属性",
        )

        counts = await self.client.replace_commerce_projection([category, attribute], [relation])

        assert counts == {"entities": 2, "relations": 1}
        queries = [call.args[0] for call in transaction.run.await_args_list]
        assert any("DETACH DELETE node" in query for query in queries)
        assert any("CommerceCategory" in query for query in queries)
        assert any("HAS_COMMERCE_ATTRIBUTE" in query for query in queries)


class TestValidateReadOnly:
    def test_match_query_passes(self):
        _validate_read_only("MATCH (n) RETURN n")

    def test_create_query_fails(self):
        with pytest.raises(GraphClientError, match="CREATE"):
            _validate_read_only("CREATE (n:Test)")

    def test_delete_query_fails(self):
        with pytest.raises(GraphClientError, match="DELETE"):
            _validate_read_only("MATCH (n) DELETE n")

    def test_merge_query_fails(self):
        with pytest.raises(GraphClientError, match="MERGE"):
            _validate_read_only("MERGE (n:Test {name: 'x'})")


class TestSanitizeLabel:
    def test_valid_label(self):
        assert _sanitize_label("RESOLVES") == "RESOLVES"

    def test_strips_special_chars(self):
        assert _sanitize_label("HAS-FEATURE") == "HASFEATURE"

    def test_empty_label_raises(self):
        with pytest.raises(GraphClientError, match="Invalid"):
            _sanitize_label("---")
