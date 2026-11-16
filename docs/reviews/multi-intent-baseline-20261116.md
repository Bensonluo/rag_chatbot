# Offline multi-intent baseline

## Experiment scope

- Base commit: `ffe54216b5067e024ac7cbe9d9392eeca093ace3`
- Branch: `feat/multi-intent-queueing`
- Layer: deterministic rule-based detector; no hosted API calls.
- Corpus: 40 synthetic cases, 74 expected requests. Thirty compound inputs
  plus ten single-request controls; same-label requests retain multiplicity.
- Command: `.venv/bin/python scripts/run_multi_intent_eval.py --json report.json`

The experiment compares the frozen single-intent API with the multi-intent
API on identical inputs. The source corpus is versioned alongside the scorer.

## Frozen single-intent API result

- Matched labels: 39 of 74 expected requests (52.70% recall).
- Label precision: 39 of 40 predictions (97.50%).
- Exact source-query coverage: 10 of 74 (13.51%). Compound inputs are not
  decomposed by this API, so a correct label is not proof of isolated arguments.
- Exact request sets: 10 of 40 cases (25.00%).
- Extra write-intent predictions: 0.

Multi-intent results are emitted by the same command. Its per-case JSON
records predictions, expected requests and the matching counts, so apparent
recall improvements can be checked for false positives and slot contamination.

## Scoring correction

An earlier draft counted unique intent labels and called membership in any
expected label a primary hit. That collapses two refunds for different orders
into one and cannot test source-query isolation. Those exploratory percentages
are superseded by this request-level corpus and scorer. They are not production
traffic statistics and are not comparable to this report.

## Multi-request API result

On the same 40 cases / 74 requests:

- Matched labels: 73 of 74 (98.65% recall).
- Label precision: 73 of 73 (100.00%).
- Exact source-query coverage: 71 of 74 (95.95%).
- Exact request sets: 38 of 40 (95.00%).
- Extra write-intent predictions: 0 on this corpus.

The two non-exact cases remain in the dataset: a terminal question mark
changes the exact span comparison, and `also file a complaint` is not split by
the conservative rule boundary heuristic. They are not removed to improve scores.
Broader negation, quotation, supersession and confirmation-expiry checks live
in the regression tests rather than being inferred from this small corpus.

## Separate gates

Detection scores do not prove that requests executed. The compiled-graph tests
must independently cover automatic sequential advancement, multiple write
confirmations, missing slots, cancellation, handoff, same-label order isolation,
checkpoint persistence, citation aggregation and stream output.

## Acceptance results

All commands ran locally with Python 3.12 against the final implementation.

- Clean base: 4,170 passed, 3 skipped, 4 xfailed; coverage 93.71%.
- Final full suite: 4,332 passed, 3 skipped, 4 xfailed, 5 warnings;
  coverage 93.83%; pytest exit code 0. No pipeline was used to mask the status.
- Added feature tests: 162, including real rule-detector-to-graph journeys.
- `ruff check app/ scripts/run_multi_intent_eval.py`: passed.
- `ruff format --check app/ scripts/run_multi_intent_eval.py`: passed.
- `mypy app/`: no issues in 574 source files.
- `pip check`: no broken requirements.
- Independent lifecycle review: PASS after resolving all six initial findings.
  Final boundary-delta review: 202 tests passed, including the seven independent
  negative regressions and the existing routing evaluation.

The independently reviewed 23-file snapshot, before this acceptance note,
was `a15e97912144e90b192048ad228771b7f920c15ea23bb2f3d1ec8ce927365921`.
Its manifest hashes compact JSON containing sorted `(path, SHA-256(file bytes))`
pairs for every changed tracked or untracked file. Production code and tests
were unchanged when this note was added.

These are offline detector and compiled-graph results. Hosted-model behavior,
production traffic rates and deployed business outcomes are separate experiments.
