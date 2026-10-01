"""SlotFillerFactory contract (coverage-honesty tick, review 2026-10-01 #10).

The factory is the env-switchable seam behind ``SLOT_FILLING_TYPE`` and
the chat endpoint's lazy-initialization fallback (app/api/v1/chat.py).
It sat in the coverage omit list since inception, so the suite never
pinned its contract — un-hiding it showed 0%. These tests pin each
selection branch so the seam cannot silently break when the strategy
set or the settings flag changes.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.services.llm.base import LLMServiceBase
from app.services.slot_filling.factory import SlotFillerFactory
from app.services.slot_filling.hybrid import HybridSlotFiller
from app.services.slot_filling.rule_based import RuleBasedSlotFiller


def test_rule_based_selection() -> None:
    filler = SlotFillerFactory.create(filler_type="rule_based")
    assert isinstance(filler, RuleBasedSlotFiller)


def test_hybrid_selection_builds_both_legs() -> None:
    llm = MagicMock(spec=LLMServiceBase)
    filler = SlotFillerFactory.create(filler_type="hybrid", llm_service=llm)
    assert isinstance(filler, HybridSlotFiller)


def test_hybrid_without_llm_degrades_to_rule_based() -> None:
    """No LLM wired (keyless environments): the factory degrades to the
    rule leg instead of raising — slot filling must not be the reason a
    degraded startup refuses to serve."""
    filler = SlotFillerFactory.create(filler_type="hybrid", llm_service=None)
    assert isinstance(filler, RuleBasedSlotFiller)


def test_unknown_type_raises() -> None:
    with pytest.raises(ValueError, match="Unknown filler_type"):
        SlotFillerFactory.create(filler_type="nonsense")


@patch("app.config.settings.get_settings")
def test_create_from_settings_disabled_returns_none(mock_settings: MagicMock) -> None:
    mock = MagicMock()
    mock.SLOT_FILLING_ENABLED = False
    mock_settings.return_value = mock
    assert SlotFillerFactory.create_from_settings() is None


@patch("app.config.settings.get_settings")
def test_create_from_settings_follows_configured_type(mock_settings: MagicMock) -> None:
    mock = MagicMock()
    mock.SLOT_FILLING_ENABLED = True
    mock.SLOT_FILLING_TYPE = "rule_based"
    mock_settings.return_value = mock
    assert isinstance(SlotFillerFactory.create_from_settings(), RuleBasedSlotFiller)
