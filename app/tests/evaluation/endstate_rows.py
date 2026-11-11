"""End-state assertion catalog: scripted journeys -> authoritative final state.

Wave-1 features are pinned by asserting the FINAL durable state (DB rows,
receipts, tickets, HTTP status), not intermediate prose. Each
:class:`EndStateRow` names the feature, the scripted journey, and the
final-state contract; ``test_endstate_catalog.py`` maps ``row_id`` to the
journey implementation. Rows live here as data so the coverage index
(``COVERAGE.md``) and the eval stay in sync.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EndStateRow:
    """One table row: journey -> authoritative end state."""

    row_id: str
    feature: str
    journey: str
    final_state: str


ENDSTATE_ROWS: tuple[EndStateRow, ...] = (
    EndStateRow(
        row_id="happy_path_paid",
        feature="purchase state machine + idempotency receipts",
        journey="search -> cart -> quote -> confirm order -> confirm mock pay, then replay both confirmations",
        final_state=(
            "stage=paid with order_id/payment_id set and the active slot released; "
            "exactly one order object, one payment object, one order/mock_pay operation "
            "each; replays return the same state without a second receipt"
        ),
    ),
    EndStateRow(
        row_id="decline_preserves_cart",
        feature="explicit confirmation gate",
        journey="reach a quoted purchase, then the visitor explicitly declines (no tool call)",
        final_state=(
            "stage stays quoted with the cart lines and the unconsumed confirmation "
            "intact (approval card still armed); zero order/payment receipts"
        ),
    ),
    EndStateRow(
        row_id="spend_limit_refusal_ticket",
        feature="spend limits + ticket escalation",
        journey="quote a total above a monkeypatched single-order limit, then confirm the order",
        final_state=(
            "no order or payment object exists; the view carries a structured "
            "spend_limit block with ticket_id; exactly one high-priority "
            "reason=spend_limit ticket bound to the visitor's chat session"
        ),
    ),
    EndStateRow(
        row_id="quote_ttl_requote",
        feature="quote TTL (300s) + re-quote gate",
        journey="quote, advance the clock past the TTL, attempt to order, re-quote, then order and pay",
        final_state=(
            "the expired confirmation is unusable (no order from it, approval card "
            "disarmed); a fresh quote mints a new confirmation with a later expiry; "
            "the journey then completes to paid with exactly one order/payment"
        ),
    ),
    EndStateRow(
        row_id="routing_fallback_support_http",
        feature="semantic routing fallback (selection-only)",
        journey="router LLM raises; an ambiguous no-keyword message must land on support over real HTTP",
        final_state=(
            "the support lane answers (no commerce UI) while the entire business "
            "snapshot (purchases, operations, provider objects, receipts) is unchanged"
        ),
    ),
    EndStateRow(
        row_id="routing_rule_table",
        feature="degraded keyword routing table",
        journey="rule_route_decision over ambiguous/unroutable/shopping/support strings",
        final_state=(
            "every input lands on a bounded lane; ambiguous and keywordless input "
            "defaults to support; clear shopping terms stay shopping"
        ),
    ),
    EndStateRow(
        row_id="mcp_decline_before_mutation",
        feature="MCP write-tool elicitation confirmation",
        journey="decline the update_cart elicitation, then accept the identical mutation",
        final_state=(
            "the declined call surfaces user_declined and mutates nothing: the "
            "accepted retry still starts at cart_version 1"
        ),
    ),
    EndStateRow(
        row_id="mcp_accept_idempotent_replay",
        feature="MCP structural idempotency keys",
        journey="accept update_cart twice with identical arguments, run quote -> order -> sandbox_pay, replay sandbox_pay",
        final_state=(
            "each identical retry replays the original committed receipt byte-for-byte; "
            "the flow terminates in status=paid with one logical payment"
        ),
    ),
    EndStateRow(
        row_id="prefs_redline_keys_rejected",
        feature="visitor preference redline boundary",
        journey="attempt to store quota./limit./permission./policy. keys (case variants) as preferences",
        final_state=(
            "every redline key is rejected at the boundary with ValueError and "
            "nothing is stored — no preference rows, no audit rows"
        ),
    ),
    EndStateRow(
        row_id="prefs_confirmed_not_downgraded",
        feature="confirmed preferences never downgraded by inferred",
        journey="save a confirmed preference, then save the same key as inferred with a different value",
        final_state=(
            "the confirmed value and source survive verbatim (the inferred write is a "
            "no-op); a later confirmed write still upgrades the stored value"
        ),
    ),
    EndStateRow(
        row_id="legacy_history_503",
        feature="legacy history fails 503 (no silent empty 200)",
        journey="wire a persister-less/memory-less ChatService as the guest chat service and read history",
        final_state=(
            "GET /api/v1/guest/history answers 503 instead of an empty 200; the "
            "normal chat service keeps answering"
        ),
    ),
)
