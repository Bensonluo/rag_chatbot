## What changed

<!-- One or two sentences. Reference the issue with "Closes #N" when applicable. -->

## Why

<!-- The problem or motivation. Link the issue/discussion if there is one. -->

## How it was verified

<!-- Concrete evidence, not "tests pass": test names/counts, commands run and their outcome.
     Example: `pytest app/tests/unit/services/commerce/ -q` → 128 passed, 0 failed. -->

- [ ] `pytest` full suite passes (run the targeted module first if scoping)
- [ ] `ruff check app/` and `ruff format app/` are clean
- [ ] `mypy app/` passes with no new ignores

## Checklist

- [ ] Tests written before or with the change (red → green); product-code-only PRs will be asked for tests
- [ ] No gates were weakened (coverage floor, mypy strictness, lint rules) to make this pass
- [ ] Docs updated in the same PR if this changes documented behavior or claims
- [ ] Commerce invariants preserved: idempotency keys, CAS versioning, staged confirmations, terminal completed actions
- [ ] No secrets, credentials, or private deployment details in the diff
