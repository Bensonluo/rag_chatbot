# Shopping-agent evaluation coverage index

Wave-1 shopping-agent features mapped to the evaluations that pin them, plus
the explicit gap list for wave-2 features landing in parallel. Test names are
`file::test` so a failing feature points straight at its evals. The two data
modules (`redteam_corpus.py`, `endstate_rows.py`) keep the corpora as data so
this index and the evals stay in sync.

## Wave-1 feature -> eval map

| Wave-1 feature | Eval(s) |
| --- | --- |
| Provider-LLM agent completes purchase with tools only (search -> cart -> quote -> order -> pay) | `test_guest_commerce_flow.py::test_provider_agent_completes_purchase_with_tools_only`, `test_guest_commerce_flow.py::test_stream_exposes_router_agent_and_tool_trace_from_entry` |
| Final paid end state + idempotent confirmation replays (single receipt per kind) | `test_endstate_catalog.py::test_endstate_row[happy_path_paid]` |
| Explicit confirmation gate (decline preserves cart and the unconsumed confirmation) | `test_endstate_catalog.py::test_endstate_row[decline_preserves_cart]`, `test_guest_commerce_flow.py::test_model_cannot_order_before_a_quote` |
| Spend limits: refusal view + high-priority handoff ticket, no order | `test_endstate_catalog.py::test_endstate_row[spend_limit_refusal_ticket]` (unit backing: `app/tests/unit/services/commerce/test_spend_limit.py`) |
| Quote TTL (300s) expiry forces re-quote; expired card disarms | `test_endstate_catalog.py::test_endstate_row[quote_ttl_requote]`, `test_guest_commerce_flow.py::test_expired_quote_requires_new_quote_and_agent_confirmation` |
| Semantic routing fallback is selection-only (router failure -> support over HTTP, business state frozen) | `test_endstate_catalog.py::test_endstate_row[routing_fallback_support_http]` |
| Degraded keyword routing table (ambiguous/tie defaults to support, bounded lanes) | `test_endstate_catalog.py::test_endstate_row[routing_rule_table]` (unit backing: `app/tests/unit/services/commerce/test_conversation_routing_fallback.py`) |
| Provider data spotlighting/delimiting (`<catalog_data>`/`<graph_data>`/`<session_memory>`) | `test_catalog_injection_hardening.py::test_poisoned_catalog_reaches_model_only_as_delimited_data`, `::test_poisoned_reply_history_stays_delimited_on_the_next_turn`, `::test_poisoned_catalog_turn_creates_no_transaction_side_effects`, `::test_tool_payloads_carry_source_notice_and_delimiters` |
| Injection red-team corpus (payload x placement x channel, 28 cases, 7 channels) | `test_injection_redteam_corpus.py::test_injection_redteam_corpus_case`, `::test_corpus_covers_every_corpus_channel` (data: `redteam_corpus.py`) |
| Guest cookie identity/CSRF/private sessions for the shopping lane | `test_guest_conversations.py` (all 12 tests, e.g. `::test_independent_cookie_jars_get_distinct_visitors_and_cookie_possession_is_identity`) |
| Cross-visitor isolation of purchases | `test_guest_commerce_flow.py::test_cross_visitor_agent_never_touches_first_purchase` |
| Shopping-agent task fencing (one operation generation per turn) | `test_commerce_task_fencing.py` (all 7 tests) |
| Durable business actions / receipt replays under transport+SQL faults | `test_durable_business_actions.py` (all 9 tests), `test_guest_commerce_flow.py::test_order_timeout_query_never_pays_or_duplicates`, `::test_payment_timeout_query_confirms_once` |
| Dynamic category minting (provider-owned facts only) | `test_injection_redteam_corpus.py::test_injection_redteam_corpus_case[minted_name_payload]`, `[minted_description_payload]` (unit backing: mint tests in `app/tests/unit/services/commerce/`) |
| Visitor preference memory: redline keys rejected, confirmed never downgraded | `test_endstate_catalog.py::test_endstate_row[prefs_redline_keys_rejected]`, `::test_endstate_row[prefs_confirmed_not_downgraded]` (unit backing: `app/tests/unit/services/commerce/test_preferences.py`) |
| MCP provider surface: elicitation decline mutates nothing; structural idempotency keys replay committed receipts | `test_endstate_catalog.py::test_endstate_row[mcp_decline_before_mutation]`, `::test_endstate_row[mcp_accept_idempotent_replay]` (unit backing: `app/tests/unit/commerce_provider/test_mcp_server.py`) |
| Legacy history wiring fails 503 (no silent empty 200) | `test_endstate_catalog.py::test_endstate_row[legacy_history_503]` |
| Refund manual-review escalation in the shopping lane | `test_refund_manual_review.py` (all 9 tests) |
| Support/shopping lane switching without side effects | `test_guest_commerce_flow.py::test_abrupt_public_guest_intent_switch_matrix`, `::test_policy_detour_preserves_cart_and_agent_continues` |

## Known suspect: framed-block breakout (kept loud, not fixed here)

Four corpus cases are `xfail(strict=True)`: `stock_name_forge_close_cn`,
`stock_name_forge_close_en`, `stock_name_forge_close_reopen`,
`minted_description_forge_close`. `delimited()` in
`app/services/commerce/tools.py` does not neutralize the literal delimiter
tokens inside untrusted provider data, so a payload embedding
`</catalog_data>` terminates the block early and its instruction tail reaches
the model outside any delimited block. These flip to XPASS (and fail loudly)
once the framing is hardened; see the wave-2 eval report.

## Gaps (wave-2 features landing in parallel; add rows/cases when they merge)

| Wave-2 feature (in flight) | Owner files seen in-repo | Eval debt |
| --- | --- | --- |
| Order timeline (event chain per purchase) | `app/services/commerce/timeline.py`, `app/tests/unit/services/commerce/test_order_timeline.py` | No end-state catalog row; add a `timeline_*` row to `endstate_rows.py` once the contract settles |
| Proactive price watch (watch ORM, WatchCard) | price-watch models + `app/tests/unit/services/commerce/test_price_watch.py` (in progress) | No corpus channel for watch-notification payloads and no end-state row for watch lifecycle; add both after merge |
| `delimited()` token neutralization (framing hardening) | fix pending against `app/services/commerce/tools.py` | The 4 xfail corpus cases above are the acceptance eval; keep them strict |
| Preference -> conversation wiring | future wiring of `PreferenceMemoryService` into the commerce context | `preference_value` corpus channel currently pins "confined or absent"; extend the driver once preferences become model-visible |
| Daily spend-limit window | `COMMERCE_SPEND_LIMIT_DAILY_CENTS` path | End-state covers the single-order limit only; add a daily-window row (multi-order accumulation) |
