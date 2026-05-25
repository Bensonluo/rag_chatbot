"""Unit tests for RuleBasedSlotFiller."""
import pytest

from app.services.slot_filling.rule_based import RuleBasedSlotFiller
from app.models.enums.intent import Intent


class TestRuleBasedSlotFiller:
    def setup_method(self):
        self.filler = RuleBasedSlotFiller()

    @pytest.mark.asyncio
    async def test_product_keyword(self):
        result = await self.filler.fill_slots("iPhone的屏幕碎了")
        assert result.has_slots()
        assert result.get_slot_value("product") == "iPhone"

    @pytest.mark.asyncio
    async def test_product_keyword_airpods(self):
        result = await self.filler.fill_slots("AirPods连不上")
        assert result.has_slots()
        assert result.get_slot_value("product") == "AirPods"

    @pytest.mark.asyncio
    async def test_issue_keyword(self):
        result = await self.filler.fill_slots("手机蓝屏了怎么办")
        assert result.has_slots()
        assert result.get_slot_value("issue") == "蓝屏"

    @pytest.mark.asyncio
    async def test_platform_keyword(self):
        result = await self.filler.fill_slots("iOS上的微信闪退")
        assert result.has_slots()
        assert result.get_slot_value("platform") == "iOS"

    @pytest.mark.asyncio
    async def test_category_keyword(self):
        result = await self.filler.fill_slots("退款政策是什么")
        assert result.has_slots()
        assert result.get_slot_value("category") == "退款"

    @pytest.mark.asyncio
    async def test_feature_keyword(self):
        result = await self.filler.fill_slots("WiFi连不上")
        assert result.has_slots()
        assert result.get_slot_value("feature") == "WiFi"

    @pytest.mark.asyncio
    async def test_time_period_year(self):
        result = await self.filler.fill_slots("2024年的订单情况")
        assert result.has_slots()
        assert result.get_slot_value("time_period") == "2024"

    @pytest.mark.asyncio
    async def test_time_period_relative(self):
        result = await self.filler.fill_slots("最近的物流状态")
        assert result.has_slots()
        assert result.get_slot_value("time_period") == "最近"

    @pytest.mark.asyncio
    async def test_multiple_slots(self):
        result = await self.filler.fill_slots("iOS上的iPhone蓝屏了")
        assert result.has_slots()
        slots = result.to_filters()
        assert "product" in slots
        assert "platform" in slots

    @pytest.mark.asyncio
    async def test_no_slots_for_unknown(self):
        result = await self.filler.fill_slots("随便聊聊")
        assert not result.has_slots()

    @pytest.mark.asyncio
    async def test_skip_chitchat_intent(self):
        result = await self.filler.fill_slots("你好", intent=Intent.CHITCHAT)
        assert not result.has_slots()
        assert result.metadata == {"method": "skipped"}

    @pytest.mark.asyncio
    async def test_processes_faq_intent(self):
        result = await self.filler.fill_slots("iPhone怎么用", intent=Intent.FAQ)
        assert result.has_slots()
        assert result.get_slot_value("product") == "iPhone"

    @pytest.mark.asyncio
    async def test_metadata(self):
        result = await self.filler.fill_slots("iPhone")
        assert result.metadata["method"] == "rule_based"
        assert result.metadata["slot_count"] >= 1
