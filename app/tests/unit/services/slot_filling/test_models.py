"""Unit tests for SlotFillingResult and ExtractedSlot data models."""
from app.services.slot_filling.base import ExtractedSlot, SlotFillingResult


class TestExtractedSlot:
    def test_to_filter(self):
        slot = ExtractedSlot(
            slot_type="product", entity_type="Product",
            value="产品A", normalized_value="产品A",
            confidence=0.9, source="rule",
        )
        assert slot.to_filter() == {"product": "产品A"}

    def test_to_entity_hint(self):
        slot = ExtractedSlot(
            slot_type="product", entity_type="Product",
            value="产品A", normalized_value="产品A",
            confidence=0.9, source="rule",
        )
        hint = slot.to_entity_hint()
        assert hint == {"type": "Product", "name": "产品A"}


class TestSlotFillingResult:
    def test_empty_result(self):
        result = SlotFillingResult()
        assert not result.has_slots()
        assert result.to_filters() == {}
        assert result.to_entity_hints() == []
        assert result.get_slot("product") is None
        assert result.get_slot_value("product") is None
        assert result.slot_types() == set()

    def test_single_slot(self):
        slot = ExtractedSlot(
            slot_type="product", entity_type="Product",
            value="产品A", normalized_value="产品A",
        )
        result = SlotFillingResult(slots=[slot], raw_query="test")
        assert result.has_slots()
        assert result.to_filters() == {"product": "产品A"}
        assert len(result.to_entity_hints()) == 1

    def test_multiple_slots(self):
        slots = [
            ExtractedSlot(slot_type="product", entity_type="Product", value="产品A", normalized_value="产品A"),
            ExtractedSlot(slot_type="platform", entity_type="Platform", value="iOS", normalized_value="iOS"),
        ]
        result = SlotFillingResult(slots=slots, raw_query="产品A在iOS上的问题")
        assert result.to_filters() == {"product": "产品A", "platform": "iOS"}
        assert result.slot_types() == {"product", "platform"}
        assert result.get_slot_value("product") == "产品A"
        assert result.get_slot_value("platform") == "iOS"

    def test_last_slot_wins_in_filters(self):
        slots = [
            ExtractedSlot(slot_type="product", entity_type="Product", value="产品A", normalized_value="产品A"),
            ExtractedSlot(slot_type="product", entity_type="Product", value="产品A Pro", normalized_value="产品A Pro"),
        ]
        result = SlotFillingResult(slots=slots, raw_query="test")
        assert result.to_filters()["product"] == "产品A Pro"
