"""Unit tests for RuleBasedSlotFiller."""
import pytest

from app.services.slot_filling.rule_based import RuleBasedSlotFiller
from app.models.enums.intent import Intent


class TestRuleBasedSlotFiller:
    def setup_method(self):
        self.filler = RuleBasedSlotFiller()

    @pytest.mark.asyncio
    async def test_drug_keyword(self):
        result = await self.filler.fill_slots("二甲双胍的副作用有哪些")
        assert result.has_slots()
        assert result.get_slot_value("drug") == "二甲双胍"

    @pytest.mark.asyncio
    async def test_channel_keyword(self):
        result = await self.filler.fill_slots("零售渠道的销售数据")
        assert result.has_slots()
        assert result.get_slot_value("channel") == "零售"

    @pytest.mark.asyncio
    async def test_time_period_year(self):
        result = await self.filler.fill_slots("2024年的销售情况")
        assert result.has_slots()
        assert result.get_slot_value("time_period") == "2024"

    @pytest.mark.asyncio
    async def test_time_period_quarter(self):
        result = await self.filler.fill_slots("第一季度销售额")
        assert result.has_slots()
        assert result.get_slot_value("time_period") == "Q1"

    @pytest.mark.asyncio
    async def test_multiple_slots(self):
        result = await self.filler.fill_slots("二甲双胍在零售渠道2024年的销售情况")
        assert result.has_slots()
        slots = result.to_filters()
        assert "drug" in slots
        assert "channel" in slots
        assert "time_period" in slots

    @pytest.mark.asyncio
    async def test_company_keyword(self):
        result = await self.filler.fill_slots("诺华的最新药物管线")
        assert result.has_slots()
        assert result.get_slot_value("company") == "诺华"

    @pytest.mark.asyncio
    async def test_indication_keyword(self):
        result = await self.filler.fill_slots("糖尿病药物市场规模")
        assert result.has_slots()
        assert result.get_slot_value("indication") == "糖尿病"

    @pytest.mark.asyncio
    async def test_region_keyword(self):
        result = await self.filler.fill_slots("华东地区的药品销售")
        assert result.has_slots()
        assert result.get_slot_value("region") == "华东"

    @pytest.mark.asyncio
    async def test_no_slots_for_unknown(self):
        result = await self.filler.fill_slots("今天天气怎么样")
        assert not result.has_slots()

    @pytest.mark.asyncio
    async def test_skip_chitchat_intent(self):
        result = await self.filler.fill_slots("你好", intent=Intent.CHITCHAT)
        assert not result.has_slots()
        assert result.metadata == {"method": "skipped"}

    @pytest.mark.asyncio
    async def test_processes_question_intent(self):
        result = await self.filler.fill_slots("二甲双胍怎么用", intent=Intent.QUESTION)
        assert result.has_slots()
        assert result.get_slot_value("drug") == "二甲双胍"

    @pytest.mark.asyncio
    async def test_english_keyword_normalization(self):
        result = await self.filler.fill_slots("metformin的市场表现")
        assert result.has_slots()
        assert result.get_slot_value("drug") == "二甲双胍"

    @pytest.mark.asyncio
    async def test_channel_电商_maps_to_线上(self):
        result = await self.filler.fill_slots("电商渠道的OTC药品")
        assert result.has_slots()
        assert result.get_slot_value("channel") == "线上"

    @pytest.mark.asyncio
    async def test_metadata(self):
        result = await self.filler.fill_slots("二甲双胍")
        assert result.metadata["method"] == "rule_based"
        assert result.metadata["slot_count"] >= 1
