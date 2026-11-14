# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.1] - 2026-11-14

### Added

- README opens with a hero screenshot of the bundled frontend's landing
  page (captured from the shipped Docker image at 1440×900).
- Bundled frontend ships a favicon (`frontend/app/icon.svg`, Next.js App Router
  file convention): an indigo shopping-bag mark with an agent spark, emitted as
  its own static route with the `<link rel="icon">` tag injected in both root
  and `/store` basePath builds.
- Custom bilingual not-found page for the bundled frontend
  (`app/not-found.tsx`): brand design tokens, dark-mode aware, and a
  link home in both languages — served with a real 404 status through the
  fixed nginx `error_page`.
- Frontend unit in the full-stack compose stack now has a healthcheck
  (busybox `wget` against the served root page), so `docker compose ps`
  reports it alongside the API, Postgres and Redis units.

### Fixed

- Bundled frontend nginx configs no longer serve a 200 homepage for unknown
  paths (soft-404): `try_files` now falls through to a hard 404 and
  `error_page` serves Next.js's generated not-found page. The BASE_PATH
  Dockerfile stage copies that page to the mount root so `/store`-prefixed
  builds return the branded 404 too. Verified against a live nginx container
  in both root and `/store` prefix mounts.
- qdrant-client 1.20 (part of the 2026-11-14 dependency refresh) removed the
  deprecated `client.search()` API: vector search now uses the universal
  `query_points()` API and reads hits from the `QueryResponse` envelope;
  affected unit-test mocks follow the new contract.

### Changed

- Release hygiene: `pyproject.toml` version re-aligned to the shipped v0.4.0
  release (it still read 0.3.0 while the tag, CHANGELOG and bundled frontend
  `package.json` were already at 0.4.0 — `/health` and installed-distribution
  metadata reported the stale number). New release-hygiene pin tests fail the
  suite when `pyproject.toml`, the latest CHANGELOG release heading and the
  bundled frontend version ever drift apart again.
- README evaluation-baseline table now lists the reproduce command for all
  three gates, including the LLM-tier answer-quality eval (needs a model key).
- Dependency refresh, part 2 (landed directly on main; the PR channel is
  reserved for external contributors): pypdf 6.21.4 → 7.0.1 — the 7.0 text
  extraction majors were regression-run against the full document-parsing
  suite (173 green, layout mode included) — and CI actions checkout v4 → v7,
  setup-python v5 → v7 (node24 runtime, no input changes). The Dependabot
  version-update config is removed; CVE alerts are unaffected (repo-level
  setting) and dependency bumps now ship as reviewed direct commits.

## [0.4.0] - 2026-11-14

### Added

- Documentation index (`docs/README.md`) mapping every tracked design contract,
  operations runbook, and review/evidence document, linked from the README.
- Runnable MCP client example (`examples/mcp_shopping_client.py`): connects to the
  commerce provider over Streamable HTTP with the official `mcp` SDK and walks
  the read-tool surface; verified against a live provider.
- Dependabot configuration: weekly grouped updates for pip and GitHub Actions,
  gated by the existing CI checks.

### Fixed

- Security: patched dnspython CVE-2026-128089 (2.9.0 → 2.9.2) and
  opentelemetry-api CVE-2026-126360 (1.45.1 → 1.45.2, with the whole
  instrumentation family moving to 0.66b2 as one release train) in the
  runtime lock; pip-audit is clean again.

### Changed

- Live-demo links point at the product's own page (benluo.art/store) — the
  standalone frontend shipped in this repo.
- Public-repo hygiene: machine-local absolute paths in the 2026-09-26 review are
  now repo-relative GitHub line anchors; `docker-compose.fullstack.yml` no longer
  references the author's portfolio frontend or ships a server-IP CORS default.
- The standalone Agent Store branding page moved into this repository
  (`frontend/`): a static-export Next.js app with the guest chat, served by
  nginx with a same-origin `/api` proxy, wired into the full-stack compose
  profile. The repository is now the canonical home of the product frontend.
- README/Makefile quality-gate numbers aligned with the enforced
  `fail_under = 70` coverage floor and the 2026-11-14 measured baseline
  (4146 passed / 93.66%).

## [0.3.0] - 2026-11-13

### Added

**Retrieval resilience (self-healing vector store)**

- Empty-knowledge-base honesty: a fresh stack with zero indexed documents answers
  "no relevant documents found" instead of reporting a retrieval outage — only a
  genuine dual-leg failure (vector and keyword both erroring) surfaces as degraded.
- Embedding-dimension drift self-heal: switching `EMBEDDING_PROVIDER` between
  models of different dimensionality (e.g. 512-dim BGE era → 1024-dim GLM
  embedding-2) is detected at ensure time; an empty collection is recreated at
  the provider's dims, a non-empty one fails fast with both dimensions and
  concrete re-index instructions instead of opaque per-search 400s.
- Vanished-collection reconciliation: a collection deleted mid-process (volume
  wipe, operator error) no longer 404s every call until restart — search and
  upsert reset the readiness latch, re-ensure, and retry once; a second failure
  propagates untouched.
- The BM25 warm-up scroll path now ensures the collection like every other
  access path, removing a misleading boot-time 404 skip on fresh stacks.

**Observability**

- `retrieval_collection_self_heals_total` Prometheus counter (label `reason`:
  `dim_drift` / `vanished`) makes every silent recovery alertable.
- Two shipped alert rules: `RetrievalSelfHealsRising` (sustained self-heals =
  upstream churns the collection) and `RetrievalFilterFallbackShareHigh`
  (fallback share pinned >90% = the chunk-metadata filter contract is dead).
- Behavioral promtool unit tests for the new alerts run in CI with the deployed
  Prometheus engine alongside the containment suite.

**Agent guardrails**

- `get_catalog_stats` tool: catalog-scale claims (category counts, SKU totals)
  now come from the authoritative provider instead of LLM imagination.
- Malformed order IDs never reach a confirmation gate — a staged irreversible
  action with a garbage order number triggers a re-ask in the user's language.
- Prompt rules: collect one missing field per turn (question form, no form
  dumps), no internal query narration, no numeric claims without tool backing.

### Fixed

- GLM embeddings now default to the standard PAAS endpoint
  (`GLM_EMBEDDING_BASE_URL`) regardless of `GLM_BASE_URL`: the Coding Plan chat
  endpoint does not serve embedding models (HTTP 400 error 1211), and the
  factory no longer forwards local BGE checkpoint names to API providers.
- Docker build layering: dependency install is keyed on metadata + lock only,
  code-only rebuilds take seconds; `pip check` promotes the lock from
  constraints to a build-time closure invariant.

## [0.2.0] - 2026-11-12

### Added

**Dialogue engine and agents**

- LangGraph dialogue engine with intent switching, skip-intent optimization, and
  mid-collection escape hatches for handoff requests.
- LLM function-calling agent loop with confirmation gates for irreversible
  actions: TTL-bound staged actions, an idempotency ledger, CAS state
  transitions, and post-run fences that isolate stale tasks after crashes.
- Slot filling (rule-based / LLM / hybrid) wired into vector-search filters and
  graph entity hints; agent turns project deterministic slot state.
- Memory strategies (sliding window, rolling summaries, hybrid, and
  relevance-filtered optimized) plus cross-session user-facts extraction and
  recall with a per-user TTL cache.
- Curated FAQ fast path, follow-up query rewriting for retrieval, and
  deterministic no-evidence / service-degraded replies.
- Epoch-scoped L0 answer cache on the chat path.

**Agentic commerce (shopping agent)**

- Anonymous cookie-identity guest visitors — the public demo journey requires
  no registration.
- End-to-end shopping agent: product search → cart → quote → explicit confirm →
  pay → order status → refund, on durable purchase sessions with instance-key
  isolation.
- Standalone internal commerce provider: deterministic 154,624-SKU catalog
  (151 categories × 32 brands × 8 models × 4 variants) over a SQLite WAL core,
  Bearer-auth HTTP contract, and hidden fault-injection routes for tests.
- Order timeline assembled from authoritative rows; order lookup by provider
  order id; recent-orders agent context tool; executable refunds gated to paid
  orders.
- Price watch (watch, notify-on-change) — `app/services/commerce/watch.py`.
- Commerce MCP server (`app.commerce_provider.mcp_server`, `mcp` extra)
  exposing store capabilities over Model Context Protocol.

**Human handoff and QA**

- Staff workbench (`/support`): ticket queue with priority routing, claim and
  two-way messaging, resolution lifecycle, and LLM conversation summaries
  attached to tickets.
- Session resolution tracking (resolved / reopened), containment metrics, and a
  QA review queue sampling containment wins without positive evidence.
- Knowledge-gap telemetry feeding a resolve-lifecycle worklist.

**Retrieval and knowledge**

- GraphRAG on Neo4j: entity extraction, Text-to-Cypher, graph embedding search,
  Leiden community detection with global search — behind `GRAPH_RAG_ENABLED`.
- True Okapi BM25 with CJK bigram tokenization; retrieval candidate pool
  widened (pool 10 → rerank → top-3) so the reranker can rescue deep hits.
- Deterministic offline evaluations: bilingual retrieval golden set (three-arm
  comparison) and routing golden set, plus a live probe suite replaying known
  production bug families.
- Server-side ACL injection seam for retrieval filters (private-document
  groundwork).

**Platform and operations**

- Alembic migrations replacing `create_all` bootstrapping; shared Postgres
  dialogue checkpointer for horizontal scaling.
- SSE streaming with heartbeats, disconnect handling, TTFT/duration/outcome
  metrics, and pipeline trace events multiplexed onto the stream.
- OpenTelemetry per-stage tracing; Prometheus alert rules validated with
  promtool; Grafana dashboards.
- Redis-backed distributed rate limiting (global sliding window + per-IP chat
  budget) that degrades to in-memory buckets when Redis is unreachable.
- Backup/restore drill in CI: pg_dump round-trip with CHECK-constraint text
  normalization.
- LLM resilience chain (GLM → MiniMax → Kimi → OpenAI → Anthropic) with a
  light/primary tier split and per-request call budgets with graceful agent
  stop.
- Multi-provider embeddings (local BGE-M3 default, GLM, OpenAI) with caching.
- Guardrails: input injection detection, sentence-buffered streaming PII
  redaction, prompt hardening, and a deterministic policy claim gate.

**Community and repository**

- CONTRIBUTING.md, SECURITY.md, Code of Conduct (Contributor Covenant 3.0),
  issue templates, PR template, and CI status badge.
- Dependency lockfile (`requirements-lock.txt`) with pip-audit in CI; smoke
  scripts consolidated under `scripts/smoke/`.

### Changed

- Demo deployments share a single public session by default;
  `DEMO_MODE=false` switches to strict per-user authorization.
- Replies follow the user's language with an English fallback; personas are
  pinned to a platform-assistant identity (no model/vendor
  self-identification).
- Containment-rate naming unified across repo, API, Prometheus, and Grafana
  (contained ≠ resolved).
- Hybrid retrieval reranks a widened candidate pool instead of reranking the
  final top-k.

### Fixed

Highlights only — see git history for the complete list.

- Cross-session leaks: L0/L1 answer-cache eligibility gates, feedback message
  ownership (cross-tenant rating and cache eviction), and session incarnation
  keys after delete/recreate.
- Staged irreversible actions: confirmation TTL expiry, retry idempotency,
  mid-collection handoff escape, and task fences that stop crash-orphaned
  actions from executing.
- Demo session parent rows are created before first-turn handoff tickets, so
  tickets and history no longer fail on foreign keys.
- Streaming PII redaction and streamed policy claims gated at sentence
  boundaries.
- Intent routing: ASCII keyword word boundaries, tie arbitration, English
  global coverage, elapsed-time FAQ phrasings, and unknown knowledge questions
  falling back to retrieval.
- Alembic `fileConfig` no longer disables application loggers; taxonomy
  variant columns migrated to JSONB on PostgreSQL.
- CI parity: `mcp` extra installed in all jobs; lockfile CVEs cleared
  (pip-audit reports no known vulnerabilities).

### Security

- python-jose → PyJWT migration, removing the unpatched `ecdsa` dependency
  (PYSEC-2026-1325); existing tokens remain valid.
- Knowledge-base management admin-gated in both demo and strict modes;
  `no-store` on sensitive responses.
- Auth error responses sanitized (no library or database internals); explicit
  `SECRET_KEY` required in production.

## [0.1.0] - 2026-03-08

Initial release.

- FastAPI RAG chatbot service with Qdrant vector search.
- BGE-M3 local embeddings (no API key required).
- Hybrid search combining vector and BM25 legs.
- Document ingestion pipeline and Docker Compose deployment.
