"""Factory for creating slot filler instances."""
from typing import Optional

from app.services.slot_filling.base import SlotFiller
from app.services.slot_filling.rule_based import RuleBasedSlotFiller
from app.services.llm.base import LLMServiceBase


class SlotFillerFactory:
    """Factory for creating slot filler instances."""

    @staticmethod
    def create(
        filler_type: str = "rule_based",
        llm_service: Optional[LLMServiceBase] = None,
    ) -> SlotFiller:
        if filler_type == "rule_based":
            return RuleBasedSlotFiller()
        elif filler_type == "hybrid":
            if llm_service is None:
                from app.services.slot_filling.rule_based import RuleBasedSlotFiller as RB
                return RB()
            from app.services.slot_filling.llm_based import LLMSlotFiller
            from app.services.slot_filling.hybrid import HybridSlotFiller
            return HybridSlotFiller(
                rule_based=RuleBasedSlotFiller(),
                llm_based=LLMSlotFiller(llm_service=llm_service),
            )
        else:
            raise ValueError(f"Unknown filler_type: {filler_type}. Must be rule_based or hybrid")

    @staticmethod
    def create_from_settings(
        llm_service: Optional[LLMServiceBase] = None,
    ) -> Optional[SlotFiller]:
        from app.config.settings import get_settings

        settings = get_settings()

        if not settings.SLOT_FILLING_ENABLED:
            return None

        return SlotFillerFactory.create(
            filler_type=settings.SLOT_FILLING_TYPE,
            llm_service=llm_service,
        )
