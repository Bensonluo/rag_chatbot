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

## Request admission

Detection can describe zero, one or several current requests. Business
labels are candidates, not execution authority. Each primary and secondary
frame carries `speech_act` (`action`, `informational`, `historical`,
`conditional`, or `alternative`) and `entity_index` for shared order targets.
The rule leg classifies these deterministically; the LLM protocol validates
labels and source evidence, and hybrid annotation preserves both fields.

Before override routing, active-slot merging or queueing, admission:

- keeps historical task mentions as context, asking a fixed clarification
  only when no current request remains;
- routes informational business frames into the guarded FAQ lane;
- never stages a write from an option comparison or an unverified condition;
- offers an order-status read for conditional writes. A subsequent explicit
  confirmation authorizes that read only, once, within five minutes. The
  original write is not automatically authorized by an order-status result;
- requires an explicit imperative before admitting a fallback write when the
  model leg failed or returned an invalid/weaker verdict.

Knowledge requests retain the existing FIFO completion/aggregation path.
Clarifications bypass the agent and slot extraction, preserving existing task
slots. Historical secondary frames cannot complete another task's missing
slots. A new invocation resets the transient clarification flag.

An action over several orders shares its source span, but each frame owns one
`entity_index`; each write still requires a separate confirmation. Expansion
is limited to supported order-scoped tasks. `cancel` remains dialogue control
(cancel the current operation), not a new order-cancellation tool.

### Task-owned execution policy

A conditional query proposal is not a reusable execution permission. Its
acknowledgment consumes the expiring proposal and creates an explicitly
read-only query task. Execution policy belongs to that task: request identity,
current user and order target travel with its progress through slot collection,
checkpoint serialization, interruption, suspension and resumption.

The prerequisite's read-only policy permits `query_order` only. It does not
mean every tool with `requires_confirmation=False` is safe: complaint creation
can also mutate business state. Restoring progress must never expand that
permission, and a different queued task must not inherit its predecessor's
policy or target.

The deterministic query lane enforces the narrow permission without offering
write tools to an unrestricted agent. Checks at staging, execution and
confirmation resolution reject mismatched intent, user or target before an
unintended action can be prepared or performed. Task consent does not replace
current account/resource authorization in the underlying business tools.

Policy and lifecycle transitions are recorded together: suspension snapshots
policy, resumption restores it, and completion/control/queue advancement
clears or replaces it before returning. Legacy frames without policy retain
legacy compatibility; a new-style frame with a missing, invalid or unknown
policy must not silently receive normal permissions. No credentials are stored
in the policy record and the model cannot grant itself permissions.

### Admission challenge report

The same CLI also evaluates a separate 32-case synthetic challenge set with
explicit source/entity annotations. Candidate admission and actual compiled
lifecycle observations are reported separately. The lifecycle probe uses
in-memory checkpoints and fake tools; only real `pending_confirmation`
snapshots count as staged writes. It measures single-request contamination,
false staged/executed writes, multi-request complete recovery and unnecessary
clarifications with explicit populations and denominators. It does not invent
missing slots or acknowledge an unintended write to improve completion.

The CLI retains its JSON report and exits nonzero on a safety failure or graph
error; CI uploads the artifact even on failure. The original 40-case detector
comparison remains separate. These finite synthetic sets are regression
instruments, not production error-rate estimates or calibrated probabilities.

## Design references

- Rasa explains why detecting multiple labels is not enough: requests may need
  ordering, conflict resolution or clarification:
  https://rasa.com/blog/its-about-time-we-get-rid-of-intents
- Rasa's flow-start guards distinguish recognition from permission to start:
  https://rasa.com/docs/reference/primitives/starting-flows
- LangGraph graph state and reducer semantics are documented at:
  https://docs.langchain.com/oss/python/langgraph/graph-api
- OpenAI's per-call approval and persisted run-state semantics:
  https://openai.github.io/openai-agents-python/human_in_the_loop/
- OWASP's minimum permissions and complete-mediation guidance:
  https://genai.owasp.org/llmrisk/llm062025-excessive-agency/

These sources support scoped authorization and durable state ownership. They
do not prescribe this project's exact policy fields or make checkpointing an
automatic authorization mechanism.

The choice here is serial execution with isolated request frames, not a claim
that all real multi-intent problems are inherently sequential.
