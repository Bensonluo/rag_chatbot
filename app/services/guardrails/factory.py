"""Factory for creating guardrail service instances."""
from typing import Optional

from app.services.guardrails.base import GuardrailService
from app.services.guardrails.input_guard import DefaultInputGuardrail
from app.services.guardrails.output_guard import DefaultOutputGuardrail


class GuardrailFactory:
    """Factory for creating guardrail service instances."""

    @staticmethod
    def create(
        enable_input: bool = True,
        enable_output: bool = True,
        enable_pii_redaction: bool = True,
    ) -> Optional[GuardrailService]:
        input_guard = DefaultInputGuardrail(enable_pii_redaction=enable_pii_redaction) if enable_input else None
        output_guard = DefaultOutputGuardrail(enable_pii_redaction=enable_pii_redaction) if enable_output else None

        if input_guard is None and output_guard is None:
            return None

        return GuardrailService(input_guard=input_guard, output_guard=output_guard)

    @staticmethod
    def create_from_settings() -> Optional[GuardrailService]:
        from app.config.settings import get_settings

        settings = get_settings()

        if not settings.GUARDRAILS_ENABLED:
            return None

        return GuardrailFactory.create(
            enable_input=settings.GUARDRAILS_INPUT_ENABLED,
            enable_output=settings.GUARDRAILS_OUTPUT_ENABLED,
            enable_pii_redaction=settings.GUARDRAILS_PII_REDACTION_ENABLED,
        )
