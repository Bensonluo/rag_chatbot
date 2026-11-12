# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
