"""Unit tests for output guardrail."""
import pytest

from app.services.guardrails.output_guard import DefaultOutputGuardrail


class TestOutputGuardrail:
    def setup_method(self):
        self.guard = DefaultOutputGuardrail()

    def test_clean_output_passes(self):
        result = self.guard.check("二甲双胍是一种常用的降糖药物。")
        assert result.passed is True
        assert result.action == "allow"
        assert not result.violations

    def test_empty_output_passes(self):
        result = self.guard.check("")
        assert result.passed is True

    def test_pii_phone_redacted(self):
        result = self.guard.check("患者电话13812345678，请跟进")
        assert result.was_redacted
        assert "13812345678" not in result.sanitized_content
        assert "[REDACTED]" in result.sanitized_content

    def test_pii_email_redacted(self):
        result = self.guard.check("联系方式: patient@hospital.com")
        assert result.was_redacted
        assert "patient@hospital.com" not in result.sanitized_content

    def test_pii_bank_card_redacted(self):
        result = self.guard.check("银行卡号6222021234567890123")
        assert result.was_redacted

    def test_no_pii_passes_through(self):
        content = "建议患者每日服用二甲双胍500mg"
        result = self.guard.check(content)
        assert result.action == "allow"
        assert result.sanitized_content == content

    def test_pii_disabled(self):
        guard = DefaultOutputGuardrail(enable_pii_redaction=False)
        result = guard.check("电话13812345678")
        assert not result.was_redacted
        assert "13812345678" in result.sanitized_content

    def test_multiple_pii_types(self):
        result = self.guard.check("电话13812345678，邮箱test@example.com")
        assert result.was_redacted
        assert "pii_phone_cn" in result.violations
        assert "pii_email" in result.violations
