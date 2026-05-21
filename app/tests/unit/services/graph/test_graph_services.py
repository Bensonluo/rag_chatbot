"""Unit tests for graph base models and interfaces."""
import pytest

from app.services.graph.base import (
    GraphEntity,
    GraphRelation,
    GraphSearchResult,
    CommunitySummary,
    GraphClientError,
)


class TestGraphEntity:
    def test_create_entity(self):
        entity = GraphEntity(
            id="123", name="二甲双胍", type="Drug",
            properties={"brand": "格华止"}, description="降糖药物",
        )
        assert entity.id == "123"
        assert entity.name == "二甲双胍"
        assert entity.type == "Drug"
        assert entity.properties["brand"] == "格华止"
        assert entity.embedding is None

    def test_entity_defaults(self):
        entity = GraphEntity(id="1", name="Test", type="Test")
        assert entity.properties == {}
        assert entity.description is None
        assert entity.embedding is None


class TestGraphRelation:
    def test_create_relation(self):
        rel = GraphRelation(
            id="r1",
            source_entity_id="e1",
            target_entity_id="e2",
            relation_type="MANUFACTURES",
            properties={"since": "2020"},
        )
        assert rel.relation_type == "MANUFACTURES"
        assert rel.properties["since"] == "2020"


class TestGraphSearchResult:
    def test_create_result(self):
        entity = GraphEntity(id="1", name="Test", type="Drug")
        result = GraphSearchResult(
            content="test content",
            entities=[entity],
            relations=[],
            score=0.95,
            source_type="graph_embedding",
        )
        assert result.score == 0.95
        assert result.source_type == "graph_embedding"
        assert len(result.entities) == 1

    def test_result_metadata(self):
        result = GraphSearchResult(
            content="test", entities=[], relations=[],
            score=0.5, source_type="text_to_cypher",
            metadata={"query": "test query"},
        )
        assert result.metadata["query"] == "test query"


class TestCommunitySummary:
    def test_create_summary(self):
        summary = CommunitySummary(
            community_id="c1", level=0, title="Diabetes Drugs",
            summary="Community of diabetes-related drugs",
            entity_count=5, entities=["二甲双胍", "阿卡波糖"],
        )
        assert summary.level == 0
        assert summary.entity_count == 5
        assert summary.embedding is None


class TestGraphClientError:
    def test_error_with_details(self):
        err = GraphClientError("connection failed", details={"uri": "bolt://localhost"})
        assert err.message == "connection failed"
        assert err.details["uri"] == "bolt://localhost"

    def test_error_without_details(self):
        err = GraphClientError("generic error")
        assert err.details == {}
