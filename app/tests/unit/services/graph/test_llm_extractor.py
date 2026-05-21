"""Unit tests for LLM entity extractor."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.services.graph.extraction.llm_extractor import LLMEntityExtractor


class MockLLMResponse:
    def __init__(self, content: str):
        self.content = content


class TestLLMEntityExtractor:
    def setup_method(self):
        self.mock_llm = AsyncMock()
        self.extractor = LLMEntityExtractor(llm_service=self.mock_llm)

    @pytest.mark.asyncio
    async def test_extract_valid_json(self):
        self.mock_llm.generate = AsyncMock(
            return_value=MagicMock(
                content='{"entities": [{"name": "二甲双胍", "type": "Drug", "description": "降糖药", "properties": {"brand": "格华止"}}], "relations": [{"source": "二甲双胍", "target": "糖尿病", "type": "TREATS", "properties": {}}]}'
            )
        )

        result = await self.extractor.extract("二甲双胍是一种降糖药物，用于治疗2型糖尿病")
        assert len(result.entities) == 1
        assert result.entities[0]["name"] == "二甲双胍"
        assert result.entities[0]["type"] == "Drug"
        assert len(result.relations) == 1
        assert result.relations[0]["type"] == "TREATS"

    @pytest.mark.asyncio
    async def test_extract_invalid_json(self):
        self.mock_llm.generate = AsyncMock(
            return_value=MagicMock(content="not json at all")
        )

        result = await self.extractor.extract("some text")
        assert result.entities == []
        assert result.relations == []

    @pytest.mark.asyncio
    async def test_extract_with_markdown_fences(self):
        self.mock_llm.generate = AsyncMock(
            return_value=MagicMock(
                content='```json\n{"entities": [{"name": "测试", "type": "Drug", "description": "", "properties": {}}], "relations": []}\n```'
            )
        )

        result = await self.extractor.extract("test text")
        assert len(result.entities) == 1
        assert result.entities[0]["name"] == "测试"

    @pytest.mark.asyncio
    async def test_extract_filters_unknown_entity_type(self):
        self.mock_llm.generate = AsyncMock(
            return_value=MagicMock(
                content='{"entities": [{"name": "test", "type": "UnknownType", "properties": {}}], "relations": []}'
            )
        )

        result = await self.extractor.extract("test")
        assert len(result.entities) == 0

    @pytest.mark.asyncio
    async def test_extract_handles_llm_error(self):
        self.mock_llm.generate = AsyncMock(side_effect=Exception("LLM error"))

        result = await self.extractor.extract("test text")
        assert result.entities == []
        assert result.raw_text == "test text"

    @pytest.mark.asyncio
    async def test_extract_batch(self):
        self.mock_llm.generate = AsyncMock(
            return_value=MagicMock(
                content='{"entities": [{"name": "a", "type": "Drug", "properties": {}}], "relations": []}'
            )
        )

        results = await self.extractor.extract_batch(["text1", "text2"])
        assert len(results) == 2
