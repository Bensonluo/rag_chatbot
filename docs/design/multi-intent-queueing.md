# Serial multi-request handling

## Scope and invariants

Handle multiple independently actionable requests in one customer message
without introducing parallel writes or a new agent framework. Reuse the
suspended-task stack, with a distinct `queued_from: multi_intent` entry kind.
A request contains an intent **and its source subquery**; an intent label alone
cannot distinguish a refund for order A from a refund for order B.

The detection and execution boundaries are deliberately separate:

1. Identify independent source-grounded requests, not every lexical match.
2. Dispatch requests sequentially, in source order.
3. Finish read-only and knowledge requests without an extra acknowledgment.
4. Pause for missing slots or an irreversible-action confirmation.
5. A confirmation authorizes one staged action only. The next write must stage
   its own confirmation, even when both writes were requested together.
6. Cancellation, denial and human escalation must not silently advance a write.
7. Never merge another request's order ID, slots, pending confirmation or result
   into the current request.
8. Keep suspended tasks distinct from unstarted queued requests.
9. Bound queued work; an overflow must be visible rather than silently lost.
10. Preserve evidence attribution, output safety checks and cache isolation.

## Detection safety

`return policy` is one policy question, not a policy question plus a return
command. Similarly, negated, quoted, conditional and deferred actions must not
become actionable tasks merely because their keywords are present. The rule
leg must be conservative; a more expressive LLM leg still has to validate the
schema, label, finite confidence and source span before its result is accepted.

Same-label requests are distinct when their subqueries differ. No label-only
`set()` is used to count them as one request. An unknown or malformed model
verdict must not fall through substring matching to a money-moving action.

## Offline evaluation

```bash
.venv/bin/python scripts/run_multi_intent_eval.py --json report.json
```

The default experiment is fully offline and keyless. It compares the frozen
single-intent method with `detect_multi` on the **same synthetic corpus**.
No environment variable can silently enable paid model calls.

The corpus separates task-plus-knowledge, task-plus-task, same-label requests,
knowledge-plus-knowledge, English/mixed-language, three-request and single-
request controls. The single controls detect the regression where broad
multi-label keyword output appears to improve recall but adds spurious writes.

Report both label-level precision/recall (with multiplicity) and exact
source-query request matching. The latter checks whether individual task
arguments can actually be isolated. The primary-label metric requires the
first expected request, not merely membership anywhere in the answer set.
An additional counter records extra write-intent predictions.

These scores measure classification on a synthetic corpus. Graph-level tests
separately verify confirmations, slot isolation, persistence, cancellation,
streamed output and actual mocked tool invocations. Neither metric is a
measurement of production traffic prevalence.

## Design references

- Rasa explains why detecting multiple labels is not enough: requests may need
  ordering, conflict resolution or clarification:
  https://rasa.com/blog/its-about-time-we-get-rid-of-intents
- Rasa's flow-start guards distinguish recognition from permission to start:
  https://rasa.com/docs/reference/primitives/starting-flows
- LangGraph graph state and reducer semantics are documented at:
  https://docs.langchain.com/oss/python/langgraph/graph-api

The choice here is serial execution with isolated request frames, not a claim
that all real multi-intent problems are inherently sequential.
