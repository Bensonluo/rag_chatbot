"""Unit tests for SlotFillingResult and ExtractedSlot data models."""
from app.services.slot_filling.base import ExtractedSlot, SlotFillingResult


class TestExtractedSlot:
    def test_to_filter(self):
        slot = ExtractedSlot(
            slot_type="drug", entity_type="Drug",
            value="二甲双胍", normalized_value="二甲双胍",
            confidence=0.9, source="rule",
        )
        assert slot.to_filter() == {"drug": "二甲双胍"}

    def test_to_entity_hint(self):
        slot = ExtractedSlot(
            slot_type="drug", entity_type="Drug",
            value="二甲双胍", normalized_value="二甲双胍",
            confidence=0.9, source="rule",
        )
        hint = slot.to_entity_hint()
        assert hint == {"type": "Drug", "name": "二甲双胍"}


class TestSlotFillingResult:
    def test_empty_result(self):
        result = SlotFillingResult()
        assert not result.has_slots()
        assert result.to_filters() == {}
        assert result.to_entity_hints() == []
        assert result.get_slot("drug") is None
        assert result.get_slot_value("drug") is None
        assert result.slot_types() == set()

    def test_single_slot(self):
        slot = ExtractedSlot(
            slot_type="drug", entity_type="Drug",
            value="二甲双胍", normalized_value="二甲双胍",
        )
        result = SlotFillingResult(slots=[slot], raw_query="test")
        assert result.has_slots()
        assert result.to_filters() == {"drug": "二甲双胍"}
        assert len(result.to_entity_hints()) == 1

    def test_multiple_slots(self):
        slots = [
            ExtractedSlot(slot_type="drug", entity_type="Drug", value="二甲双胍", normalized_value="二甲双胍"),
            ExtractedSlot(slot_type="channel", entity_type="Channel", value="零售", normalized_value="零售"),
        ]
        result = SlotFillingResult(slots=slots, raw_query="二甲双胍在零售渠道")
        assert result.to_filters() == {"drug": "二甲双胍", "channel": "零售"}
        assert result.slot_types() == {"drug", "channel"}
        assert result.get_slot_value("drug") == "二甲双胍"
        assert result.get_slot_value("channel") == "零售"

    def test_last_slot_wins_in_filters(self):
        slots = [
            ExtractedSlot(slot_type="drug", entity_type="Drug", value="二甲双胍", normalized_value="二甲双胍"),
            ExtractedSlot(slot_type="drug", entity_type="Drug", value="metformin", normalized_value="二甲双胍"),
        ]
        result = SlotFillingResult(slots=slots, raw_query="test")
        assert result.to_filters()["drug"] == "二甲双胍"
