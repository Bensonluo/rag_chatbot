"""Unit tests for HybridSlotFiller."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.services.slot_filling.hybrid import HybridSlotFiller
from app.services.slot_filling.rule_based import RuleBasedSlotFiller
from app.services.slot_filling.llm_based import LLMSlotFiller


class TestHybridSlotFiller:
    def setup_method(self):
        self.rule_based = RuleBasedSlotFiller()
        self.mock_llm = AsyncMock()
        self.llm_based = LLMSlotFiller(llm_service=self.mock_llm)

    @pytest.mark.asyncio
    async def test_rule_based_finds_slots_no_llm_call(self):
        hybrid = HybridSlotFiller(
            rule_based=self.rule_based, llm_based=self.llm_based,
        )
        result = await hybrid.fill_slots("iPhone的屏幕碎了")
        assert result.has_slots()
        assert result.get_slot_value("product") == "iPhone"
        assert result.metadata["method"] == "rule_based"
        self.mock_llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_llm_fallback_when_no_rule_slots(self):
        self.mock_llm.generate = AsyncMock(
            return_value=MagicMock(
                content='{"slots": [{"slot_type": "product", "value": "测试设备"}]}'
            )
        )
        hybrid = HybridSlotFiller(
            rule_based=self.rule_based, llm_based=self.llm_based,
        )
        result = await hybrid.fill_slots("我的测试设备出问题了")
        assert result.has_slots()
        assert result.metadata["method"] == "llm_fallback"
        self.mock_llm.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_fallback_when_disabled(self):
        hybrid = HybridSlotFiller(
            rule_based=self.rule_based, llm_based=self.llm_based,
            llm_fallback=False,
        )
        result = await hybrid.fill_slots("随便聊聊")
        assert not result.has_slots()
        self.mock_llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_fallback_when_no_llm_service(self):
        hybrid = HybridSlotFiller(
            rule_based=self.rule_based, llm_based=None,
        )
        result = await hybrid.fill_slots("随便聊聊")
        assert not result.has_slots()
