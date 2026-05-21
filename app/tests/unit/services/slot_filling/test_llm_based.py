"""Unit tests for LLMSlotFiller with mocked LLM."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.services.slot_filling.llm_based import LLMSlotFiller
from app.models.enums.intent import Intent


class TestLLMSlotFiller:
    def setup_method(self):
        self.mock_llm = AsyncMock()
        self.filler = LLMSlotFiller(llm_service=self.mock_llm)

    @pytest.mark.asyncio
    async def test_extract_valid_slots(self):
        self.mock_llm.generate = AsyncMock(
            return_value=MagicMock(
                content='{"slots": [{"slot_type": "drug", "value": "二甲双胍"}, {"slot_type": "channel", "value": "零售"}]}'
            )
        )
        result = await self.filler.fill_slots("二甲双胍在零售渠道的销售情况")
        assert result.has_slots()
        assert len(result.slots) == 2
        assert result.slots[0].source == "llm"

    @pytest.mark.asyncio
    async def test_extract_empty_slots(self):
        self.mock_llm.generate = AsyncMock(
            return_value=MagicMock(content='{"slots": []}')
        )
        result = await self.filler.fill_slots("今天天气怎么样")
        assert not result.has_slots()

    @pytest.mark.asyncio
    async def test_handles_invalid_json(self):
        self.mock_llm.generate = AsyncMock(
            return_value=MagicMock(content="not json")
        )
        result = await self.filler.fill_slots("test")
        assert not result.has_slots()

    @pytest.mark.asyncio
    async def test_handles_markdown_fences(self):
        self.mock_llm.generate = AsyncMock(
            return_value=MagicMock(
                content='```json\n{"slots": [{"slot_type": "drug", "value": "测试药"}]}\n```'
            )
        )
        result = await self.filler.fill_slots("test")
        assert result.has_slots()
        assert result.slots[0].value == "测试药"

    @pytest.mark.asyncio
    async def test_filters_unknown_slot_type(self):
        self.mock_llm.generate = AsyncMock(
            return_value=MagicMock(
                content='{"slots": [{"slot_type": "unknown_type", "value": "test"}]}'
            )
        )
        result = await self.filler.fill_slots("test")
        assert not result.has_slots()

    @pytest.mark.asyncio
    async def test_handles_llm_error(self):
        self.mock_llm.generate = AsyncMock(side_effect=Exception("LLM error"))
        result = await self.filler.fill_slots("test")
        assert not result.has_slots()

    @pytest.mark.asyncio
    async def test_skip_chitchat_intent(self):
        result = await self.filler.fill_slots("你好", intent=Intent.CHITCHAT)
        assert not result.has_slots()
        assert result.metadata == {"method": "skipped"}
