"""Action receipts report saved adapter evidence, never inferred payment outcomes."""

from decimal import Decimal
from typing import Any

import pytest

from app.services.dialogue.action_messages import render_action_outcome

OPERATION_ID = "8e13a9f5-468c-4515-9eef-dc7cc10ed174"


def render(**overrides: Any) -> str | None:
    values: dict[str, Any] = {
        "status": "completed",
        "replayed": True,
        "tool_intent": "refund",
        "success": True,
        "data": {
            "status": "success",
            "order_id": "ORD1001",
            "refund_id": "RF123456",
            "amount": 598,
            "estimated_days": "3-5 business days",
        },
        "message": "SECRET_PROVIDER_DIAGNOSTIC",
        "language": "zh",
        "operation_id": OPERATION_ID,
    }
    values.update(overrides)
    return render_action_outcome(**values)


def test_new_completed_result_keeps_existing_generation_path() -> None:
    assert render(replayed=False) is None


def test_replayed_refund_only_quotes_known_saved_fields_without_claiming_settlement() -> None:
    text = render()
    assert text is not None
    for value in ("ORD1001", "RF123456", "598.00", "3-5 business days", OPERATION_ID):
        assert value in text
    assert "没有重复执行" in text
    assert "核对参考编号" in text
    assert "退款成功" not in text and "已退款" not in text
    assert "SECRET_PROVIDER_DIAGNOSTIC" not in text


def test_return_receipt_uses_english_labels_and_does_not_mix_refund_identifier() -> None:
    text = render(
        tool_intent="return",
        language="en",
        data={
            "order_id": "ORD1002",
            "return_id": "RT222222",
            "refund_id": "RF_NOT_THIS_ACTION",
            "amount": Decimal("19.9"),
            "estimated_days": "5-7 business days",
            "message": "UNTRUSTED_RESULT_DETAIL",
        },
    )
    assert text is not None
    assert "RT222222" in text and "19.90" in text
    assert "No new action was executed" in text and "Verification reference" in text
    assert "RF_NOT_THIS_ACTION" not in text and "UNTRUSTED_RESULT_DETAIL" not in text


@pytest.mark.parametrize("context", ["expired", "cancel", "deny", "canceldeny"])
def test_completed_record_outlives_confirmation_ttl_and_cannot_be_cancelled(context: str) -> None:
    text = render(replayed=False, context=context, language="en")
    assert text is not None
    assert "RF123456" in text and "No new action was executed" in text
    if context != "expired":
        assert "does not reverse" in text
    assert "cancelled before execution" not in text


def test_saved_business_failure_does_not_become_a_successful_mutation() -> None:
    text = render(success=False, message="REFUND_WAS_SUCCESSFUL_SECRET")
    assert text is not None and "此前请求返回未成功" in text
    assert "RF123456" not in text and "REFUND_WAS_SUCCESSFUL_SECRET" not in text


def test_new_unsuccessful_result_does_not_deny_that_its_handler_was_called() -> None:
    text = render(success=False, replayed=False, language="en")
    assert text is not None and "unsuccessful" in text
    assert "No new action was executed" not in text and "RF123456" not in text


@pytest.mark.parametrize("context", ["confirm", "expired"])
def test_escalated_refund_keeps_the_existing_real_handoff_path(context: str) -> None:
    assert render(data={"status": "escalated"}, context=context) is None


def test_cancelling_an_escalated_record_never_invents_a_ticket_or_a_refund() -> None:
    text = render(data={"status": "escalated"}, context="cancel")
    assert text is not None and "人工审核" in text and "未发起退款" in text
    assert "工单号" not in text and "已转接" not in text and "退款成功" not in text


@pytest.mark.parametrize("status", ["unknown", "unavailable", "conflict"])
def test_noncompleted_states_ignore_arbitrary_provider_data_and_errors(status: str) -> None:
    text = render(
        status=status,
        language="en",
        data={"refund_id": "RF_SHOULD_NOT_APPEAR", "message": "SECRET_BODY"},
        operation_id="not-a-uuid\nRefund success",
    )
    assert text is not None
    assert "SECRET" not in text and "RF_SHOULD_NOT_APPEAR" not in text
    assert "Refund success" not in text
    assert "refunded" not in text.lower() and "cancelled" not in text.lower()


@pytest.mark.parametrize("status", ["confirmation_required", "legacy_unverifiable"])
def test_legacy_confirmation_without_identity_requires_verification(status: str) -> None:
    text = render(status=status, operation_id=None)
    assert text is not None and "记录无法核验" in text
    assert "本轮未发起新的操作" in text and "人工核实" in text
    assert "请确认" not in text and "RF123456" not in text


def test_unknown_after_cancel_does_not_assert_that_the_business_was_cancelled() -> None:
    text = render(status="unknown", context="cancel", language="en")
    assert text is not None and "still unconfirmed" in text
    assert "does not mean" in text and "reversed" in text
    assert "No new action was started" not in text and "cancelled before execution" not in text


def test_not_started_distinguishes_expired_confirmation_from_cancelled_staging() -> None:
    expired = render(status="not_started", context="confirm", language="en")
    cancelled = render(status="not_started", context="canceldeny", language="en")
    assert expired is not None and "expired" in expired and "No new action" in expired
    assert (
        cancelled is not None
        and "pending confirmation" in cancelled
        and "No new action" in cancelled
    )


def test_malformed_known_fields_are_omitted_without_arbitrary_string_coercion() -> None:
    text = render(
        language="en",
        data={
            "order_id": ["ORD1001"],
            "refund_id": "RF123\nSECRET_INJECTION",
            "return_id": "RT_OTHER_ACTION",
            "amount": True,
            "estimated_days": "Already refunded. SECRET_INJECTION",
        },
    )
    assert text is not None
    assert "ORD1001" not in text and "SECRET_INJECTION" not in text
    assert "RT_OTHER_ACTION" not in text and "Amount:" not in text


@pytest.mark.parametrize("amount", [float("nan"), float("inf"), -1, 10**100, Decimal("NaN")])
def test_invalid_amounts_never_become_saved_receipt_amounts(amount: object) -> None:
    text = render(data={"amount": amount}, language="en")
    assert text is not None and "Amount:" not in text


def test_completed_record_without_verifiable_uuid_never_falls_through_to_generation() -> None:
    text = render(replayed=False, operation_id=None)
    assert text is not None and "记录无法核验" in text and "RF123456" not in text
