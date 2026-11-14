# Contributing

Thanks for your interest in Agent Store. This project is an engineering-first showcase: every claim in the README should be verifiable from the code and tests, so contributions are held to the same standard.

## Development Setup

```bash
# 1. Create an isolated environment (never install globally)
python3 -m venv .venv
source .venv/bin/activate

# 2. Install with dev dependencies
pip install -e ".[dev]"

# 3. Run the test suite (coverage minimum is enforced by pyproject.toml)
pytest

# 4. Start the API locally
uvicorn app.main:create_app --factory --reload --host 0.0.0.0 --port 8000
```

Optional services (PostgreSQL, Redis, Qdrant, Neo4j, commerce provider) start with `docker-compose up -d`. Most of the test suite runs on SQLite/in-memory backends and does not require them.

### Frontend (`frontend/`)

The standalone Agent Store page is a static-export Next.js app. Node 22:

```bash
cd frontend
npm install
cp .env.example .env.local   # point NEXT_PUBLIC_API_URL at your API (dev only)
npm run dev                  # http://localhost:3000
```

Gates (CI enforces both): `npm run type-check` and `npm run build`.

## Quality Gates — All Must Pass

Run these before opening a PR:

```bash
pytest                 # full suite; coverage floor enforced in pyproject.toml
ruff check app/        # lint
ruff format app/       # formatting (run `ruff format app/` to fix)
mypy app/              # strict type checking — no ignores, narrow types properly
```

Notes:

- The suite is large (4,000+ tests). If your change is scoped, run the targeted module first, then the full suite before the PR.
- New behavior needs tests written **before or with** the fix (red → green). A PR that only changes product code without tests will be asked for tests.
- Do not lower gates (coverage thresholds, mypy strictness, lint rules) to make a change pass. If a rule genuinely blocks a legitimate fix, discuss it in the issue first.
- Manual smoke scripts under `scripts/smoke/` run against live providers and are not part of `pytest`.

## Commit & PR Conventions

- Commits: `<type>: <description>` with types `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `ci`.
- One logical change per commit; avoid mixing product code with unrelated formatting churn.
- PRs: describe **what changed, why, and how it was verified** (test names/counts, commands run). "Tests pass" without specifics is not verification.
- Never force-push to shared branches; no `git amend` on pushed commits — add new commits instead.

## Project Conventions

- **Language**: code, comments, commit messages, and docs are English-first (the product serves a global audience).
- **Architecture**: factory/strategy patterns throughout (`app/services/*/factory.py`); providers are switchable via env vars. New capabilities should follow the same pattern rather than hard-wiring a provider.
- **Commerce invariants**: idempotency keys, CAS versioning, staged confirmations, and terminal completed actions are the core of the project's value proposition. Changes that weaken these invariants will not be accepted.
- **Docs honesty**: capability claims (counts, metrics, status) must match what the code and tests actually do. If a contribution changes behavior, update the affected docs in the same PR.

## Reporting Issues

Use the issue templates (bug report / feature request). For security reports, see [SECURITY.md](SECURITY.md) — do not open public issues for vulnerabilities.
