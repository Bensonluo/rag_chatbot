# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- The commerce MCP server's agent-facing surface is now English-first
  bilingual: all 8 tool descriptions, the server instructions, the 4
  elicitation confirmations, and the module-authored user-facing errors
  lead with English and keep the Chinese mirror (the internal GLM agent
  reads both). Shared parameter aliases (`owner_user_id`, id, query) now
  carry schema descriptions too. A handshake-level test pins every tool
  description as EN-lead + zh-mirror so the surface cannot drift back to
  zh-only.
- The demo's most-read response models (`ChatResponse`,
  `FeedbackResponse`) now ship hand-written OpenAPI examples too, so
  `/docs` shows a realistic commerce-agent reply — answer copy, intent,
  a FAQ citation excerpt, message id — instead of a generated stub. A
  sibling contract test validates response examples against their
  models, mirroring the request-example pin.
- CI now builds both Docker images the quick-start ships (`docker
  compose up` builds exactly these): a new `docker-images` job gates the
  API and frontend images on every push and PR via build-push-action
  (`push: false`) with GitHub Actions layer caching — a broken
  Dockerfile layer or an uninstallable lockfile fails the PR, not the
  user's first `up`.
- The storefront's four public request models (`ChatRequest`,
  `GuestBootstrapRequest`, `GuestChatRequest`, `FeedbackCreate`) now ship
  hand-written OpenAPI examples, so `/docs` pre-fills realistic request
  bodies instead of auto-generated ones that fail the models' own
  validation (e.g. `session_id: 0`). A schema contract test validates
  every example against its model — examples can never silently rot.
- The embedded live-chat panel is now wrapped in an error boundary: a
  render-time crash inside the chat island degrades to a bilingual
  fallback ("the rest of the page is unaffected") with a retry button,
  instead of React unmounting the entire storefront — which is its
  documented behavior for uncaught errors. Pinned by three jest tests
  (containment, retry remount, locale-following fallback).
- The docs-hygiene drift pins now cover the ZH quickstart too: the 环境要求
  line's backticked port map and Python floor must match docker-compose.yml
  and pyproject exactly, like the English Requirements block. The new pin
  immediately caught the ZH mirror lagging behind — it omitted the
  monitoring profile's Prometheus `9090` and Grafana `3001`.
- Every one of the 66 REST endpoints now carries a human OpenAPI `summary`
  (Swagger UI shows one line per operation instead of bare route ids);
  coverage is complete across all ten routers and stays consistent with the
  summaries the auth/sessions routers already had.
- The eight public visitor endpoints (`/api/v1/guest/*`) now document the
  error codes their handlers actually raise (`responses=` metadata with a
  one-line cause per code — expired cookie, same-origin check, rate budget,
  human-handoff lock, provider unavailability). The documented sets come
  from the real code paths (the `commerce_errors` mapper, identity/CSRF
  guards, and the chat rate limiter), and a schema contract test pins the
  chat and bootstrap sets so the storefront cannot silently drop them.
- Error-response documentation extended to the authenticated surface:
  sessions (9 endpoints), chat (4), and feedback (2) now declare their
  handler-raised codes through a shared `make_error_responses` factory in
  the OpenAPI module (one code→cause table per router, per-endpoint
  subsets). The sweep also exposed and fixed a real defect: updating a
  missing session surfaced as 500 (service `NotFoundError` uncaught) — now
  404, pinned by an integration regression test, and the schema contract
  test now pins error sets on the chat/feedback/sessions operations too.
- Error-response documentation completed across the whole API: documents
  (10), handoff (9), commerce admin (11), graph (6), analytics (2), and
  auth (4) declare their real raise paths the same way — upload limits
  (413), file types (400), KB-writer/admin gates (403), taxonomy conflicts
  (409), graph store outages (503). A completeness contract test now fails
  the suite if any `/api/v1` operation (health probes aside) ships without
  at least one documented error code.
- The bundled frontend now speaks the API's full error taxonomy: chat and
  feedback failures map 409 (conversation changed), 423 (human agent took
  over), and 503 (service unavailable) to their own bilingual copy instead
  of a generic "could not finish" message, with a guest-chat test pinning
  the handoff-lock and outage wording.
- A rate-limited chat turn now names the server-stated wait: the transport
  reads the `Retry-After` header the chat budget limiter already sends and
  the failure copy shows "try again in Ns" (bilingual) instead of vague
  advice — pinned by a guest-chat test with a controlled 429 response.
- README Quick Start opens with a Requirements block derived from the actual
  stack: Compose v2 + loopback-only port map (API `8000–8003`, Postgres 5432,
  Redis 6379, Qdrant 6333/6334, Neo4j 7474/7687, monitoring adds 9090/3001),
  one-LLM-key minimum (GLM default), the two embedding paths (GLM-key API
  embeddings vs. keyless local BGE-M3 needing the 6 GB+ torch extra), and the
  `cp .env.example` footgun warning — mirrored condensed in the Chinese
  quickstart.
- The Requirements block is now pinned by docs-hygiene contract tests: the
  documented port set must equal the host ports `docker-compose.yml` actually
  publishes (range-expanded on both sides), and the stated Python floor must
  match `pyproject.toml`'s `requires-python` — a new service port fails CI
  until the README names it, and a removed mapping fails CI until the README
  drops it.
- A changed-conversation rejection (409) now offers a "Refresh conversation"
  button instead of text-only advice: clicking it re-runs the visitor
  allocation (bootstrap reuse returns the current session without rotating
  the identity) and re-reads history, so the client heals the stale session
  key in one tap — pinned by a guest-chat test driving the 409, the refresh,
  and the re-synced server state.
- Every documented error response in the OpenAPI storefront now carries its
  body schema: the shared `ErrorDetail` (`{"detail": string}`) model that
  handler-raised `HTTPException`s actually return, wired through the same
  `make_error_responses` factory — 422 stays description-only because
  FastAPI's body-validation failures return the structured `ValidationError`
  array, not handler prose. A schema contract test pins that every
  documented non-422 error code declares a JSON body referencing
  `ErrorDetail`.
- Landing-page accessibility round driven by a Lighthouse audit of the live
  store (a11y 0.94): the page now has a `<main>` landmark (pinned by the
  store-page contract test), the muted-text token `--ink-faint` darkens one
  notch to slate-600 (7:1 on the light panels; it drove all 13 flagged
  contrast failures), and the `--primary` token now carries actual Indigo 600
  instead of the mislabeled Indigo 400 value (white-on-primary ≈ 6.3:1,
  fixing the CTA button).
- The README hero screenshot is recaptured at a 960px-deep viewport so both
  confirmation-gate pipeline rows (buy and support) render fully instead of
  the support row being cut at the bottom edge, and a docs-hygiene pin now
  fails the suite when any README local image reference points at a file
  that does not exist — screenshots are binary assets, so a rename or a
  missed `git add` would otherwise ship a broken image silently.
- The README's MCP example client (`examples/mcp_shopping_client.py`) is now
  rot-pinned: a test launches the real MCP server as a subprocess on an
  ephemeral port and drives the example's own `run()` through the complete
  purchase journey over the wire (search → cart → quote → order →
  idempotent replay → sandbox payment). An SDK rename, schema change, or
  moved flag now fails the suite instead of shipping a broken demo.

### Fixed

- The storefront's chat status badge was hardcoded `● online` — it claimed
  the API was up even while it was down, in English, on a bilingual page.
  The chat island already probed real health every 30s (`apiStatus`
  checking/online/offline); that verdict is now lifted to the landing
  header via an `onApiStatusChange` callback and rendered bilingually
  through the locale trees. Static HTML honestly ships `● 连接中…` /
  `● checking…`; pinned by a static-render honesty test and two runtime
  lift tests (healthy→online, failing→offline). The hard-bounds baseline
  row was refreshed to the measured suite (4,170 tests · 93.71%) in the
  same honesty pass.
- The frontend Docker image no longer crashes `next build` when built
  without build args: `ARG NEXT_PUBLIC_SITE_URL=""` set the variable to
  an empty string, which Next.js treats as a value (so the code's `??`
  fallback never fired and `new URL("")` failed page-data collection).
  Every arg-less build since the ARG was introduced was broken — server
  deploys only worked because they pass the real URL. The ARG now
  defaults to the code's own fallback (`http://localhost:3000`), and all
  four SITE_URL reads use `||` so a set-but-empty env means "unset".
  Caught by the new CI docker-images job on its first run.
- The standalone store unit's nginx cache headers never fired: the
  locations were plain prefixes (`/_next/static/`), but the server unit
  mounts the export under `BASE_PATH=/store`, where assets arrive as
  `/store/_next/static/...` and fell through to the generic location
  with no cache header at all — found by post-deploy verification of the
  live store (hashed assets answered 200 with no `Cache-Control`). Both
  nginx variants now use BASE_PATH-agnostic regex locations (optional
  single-segment prefix), validated with `nginx -t` in the runtime image,
  and the caching pins follow the new shape.

## [0.4.2] - 2026-11-15

### Added

- The OpenAPI schema behind the DEBUG-gated `/docs` storefront is now
  enriched and wired in: metadata for all ten router tags, real servers
  (local development + the public storefront), MIT license, repository
  contact, and the JWT bearer scheme. Title/description/version are read
  from the app constructor — the module can no longer drift from the
  package version the way its unwired predecessor (placeholder
  `example.com` hosts, hardcoded `1.0.0`) silently had. New schema
  contract tests pin out placeholders and require every route tag to have
  metadata.
- The runnable MCP shopping example now walks the complete purchase journey:
  cart → quote → order → sandbox pay, answering every live elicitation
  confirmation by hand (or `--yes` for scripted runs), deliberately calling
  `create_order` twice to show the structural request key replaying the same
  order — and a declined gate aborting with zero mutations. Verified
  end-to-end over the wire against a running provider, both accept and
  decline paths. The README shows the run as a collapsible terminal
  capture, and the similar-projects section now maps the 2026
  agentic-commerce landscape (platform MCP servers, ACP/UCP protocols).
- The bundled frontend's landing page is now fully bilingual: all copy
  (hero, flows, trust/service cards, hard bounds, footer) follows the
  visitor's locale with a manual EN / 中文 toggle in the nav, matching the
  chat's existing language behavior. Copy lives in a typed
  zh/en dictionary (`frontend/lib/i18n/store-copy.ts`) and a jest contract
  test asserts both trees stay structurally identical.

- `CITATION.cff` (validated against schema 1.2.0): GitHub renders a "cite
  this repository" entry from it, with the version pinned to releases by
  the release-hygiene tests (now a four-way sync: pyproject, CHANGELOG,
  frontend package.json, CITATION.cff).
- MCP tool safety metadata: every tool on the commerce MCP server now
  declares spec `annotations` — reads are `readOnlyHint`; additive writes
  (quote, order, sandbox pay) declare `idempotentHint` (from their
  structural request keys) and an explicit `destructiveHint: false`
  (omitted hints default to destructive in client risk models), while
  `update_cart` honestly declares its whole-replace `destructiveHint`.
  Every tool also carries a human `title`. A handshake contract test
  pins the full annotation map, so agent clients can gate calls without
  reading docstrings.
- Search/AI-agent readiness for the bundled frontend: an explicit
  allow-all `robots.txt` with a sitemap pointer (`app/robots.ts`), a
  generated `sitemap.xml` (`app/sitemap.ts`, both static-export safe via
  `dynamic = "force-static"`), and schema.org `SoftwareApplication`
  JSON-LD on the landing page with the version auto-synced from
  `package.json`. The product welcomes agents — it is one.
- Supply-chain hardening of the CI surface: every GitHub Actions use is
  now pinned to a full commit SHA (tags are mutable), CI jobs run with
  an explicit least-privilege `permissions: contents: read` token,
  CodeQL static analysis runs on pushes/PRs plus a weekly sweep, and
  `CODEOWNERS` routes external PRs to the maintainer automatically.

### Fixed

- The storefront no longer reflows for non-Chinese first visits: the
  static export used to render the zh tree and swap to the visitor's
  locale during hydration, visibly shifting the hero (Lighthouse CLS
  0.207, mobile). Both locale trees now ship in the HTML behind a CSS
  gate on `<html data-locale>`, pinned before first paint by a
  parse-blocking bootstrap (stored preference, else system locale) —
  Lighthouse CLS on a cold en-US visit drops to 0 while the visitor
  still gets their language at first paint.
- The landing page's hard-bounds strip showed a stale test baseline
  (4,146 / 93.66%); it now carries the current measured numbers
  (4,148 / 93.71%), pinned by the page contract test.
- The runnable MCP shopping example (`examples/mcp_shopping_client.py`)
  now speaks the `httpx2` fork the `mcp` 2.x SDK is actually built on —
  a plain `httpx` client fails the SDK's type surface on the injected
  transport. Re-verified live end-to-end against a running provider:
  connect, annotated tool listing, search, product detail, clean
  session teardown.

### Changed

- The frontend's two nginx serving variants now set cache policy:
  content-hashed Next.js build assets (`/_next/static/`) are served
  `public, max-age=31536000, immutable` — the header Next.js itself
  sets when it serves those files — and un-hashed `/images/` assets get
  a bounded day of cache. Returning visitors skip re-downloading the
  JS/CSS bundles; a docs-hygiene test pins the policy in both configs.
- `pyproject.toml` Homepage now points at the product's own page
  (benluo.art/store/) instead of the author's portfolio write-up.

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
