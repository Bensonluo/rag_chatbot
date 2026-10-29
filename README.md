<div align="center">

# Agentic RAG Smart Customer Service

**An enterprise-grade Agentic RAG customer service system connecting multi-turn business tasks, versioned knowledge, evidence-backed answers, and human handoff. Built with FastAPI and LangGraph.**

[![Live Demo](https://img.shields.io/badge/LIVE-DEMO-brightgreen?style=for-the-badge&logo=vercel)](https://benluo.art/projects/rag-chatbot/)
[![GitHub stars](https://img.shields.io/github/stars/Bensonluo/rag_chatbot?style=for-the-badge)](https://github.com/Bensonluo/rag_chatbot/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-FF6B6B)](https://github.com/langchain-ai/langgraph)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

`intent routing → slot filling / tool call / FAQ fast path / agent loop / hybrid retrieval → claim-gated streaming → handoff`

</div>

---

## 📌 Table of Contents

- [Why This Project](#-why-this-project)
- [Key Highlights](#-key-highlights)
- [How It Works](#-how-it-works)
- [Quick Start](#-quick-start)
- [Live Demo](#-live-demo)
- [Architecture](#-architecture)
- [Configuration](#-configuration)
- [Testing & Quality](#-testing--quality)
- [Roadmap](#-roadmap)
- [中文说明](#-中文说明)

---

## 💡 Why This Project

Customer service needs to resolve business requests while keeping policy answers, task state, and staff handoffs consistent. Users interrupt a refund flow to ask a question; policies change after an answer was sent; human agents need the original context rather than a fresh conversation.

The system connects a 12-node LangGraph dialogue engine with hybrid retrieval, versioned document ingestion, claim-gated streaming, and assigned-agent handoff. Business actions use a ToolRegistry; completed actions remain terminal so task resumption cannot silently execute them twice.

### Enterprise architecture and delivery

- **Knowledge lifecycle:** retain original documents, process immutable revisions asynchronously, recover interrupted ingestion, and preserve the answer-time citation snapshot.
- **Access and continuity:** private sessions enforce authenticated ownership in both Demo and strict modes; assigned staff can read context and send separate human replies. PostgreSQL checkpoints persist dialogue state.
- **Operations:** epoch-scoped caches, Redis rate limiting, metrics and alert rules, tracing hooks, replica deployment templates, and a PostgreSQL backup/recovery runbook.
- **Integration:** five bundled mock business-tool handlers define the refund/return/order/logistics/complaint interface; connect enterprise business APIs through the registry. Multi-tenant knowledge isolation remains on the roadmap.

Deployment commands below start a single API instance. Replica templates, optional GraphRAG and monitoring are configured separately. Recorded verification results and their scope are listed under [Testing & Quality](#-testing--quality).

---

## ✨ Key Highlights

| Capability | Implementation | Business purpose |
|---|---|---|
| Stateful task execution | 12-node StateGraph, five-way routing, slot collection, task stack, ToolRegistry | Complete requests across interruptions without repeating completed actions |
| Versioned support knowledge | Retained sources, immutable revisions, retryable ingestion, stable passage IDs | Trace an answer back to the policy version used at the time |
| Retrieval and answer control | Vector + BM25 + RRF + reranking, optional GraphRAG, sentence-level claim gates | Retrieve relevant evidence and check curated policy facts before streaming |
| Human handoff | Private sign-in workspace, tickets/queue, assigned context, two-way human messages, atomic final reply/closure | Carry context into staff handling and deliver a saved explanation |
| Operational control | Epoch-scoped L0/L1/L2 cache, metrics/alerts, tracing, rate limits | Inspect latency, outcomes, cache consistency and load |

**Recorded quality baseline (2026-10-28):** 3720 passing core tests, 3 opt-in scenarios skipped by default, 93.57% core coverage. Containment rate measures sessions without handoff or downvote; confirmed resolution, reopen counts and human QA are tracked separately. Failed, completed and latest-failed reply cohorts are also reported independently.

[Answer citation contract](docs/design/answer-citations.md) · [Staff handoff context](docs/operations/handoff-context.md) · [Backup and recovery](docs/operations/backup-and-recovery.md)

---

## 🔄 How It Works

A user says *"I want a refund"*. Here's what happens:

```
1. guardrail        → scan for prompt injection, redact PII
2. detect_intent    → classify: refund (task) | policy (rag) | chitchat (direct) ...
3. handle_switch    → if intent changed, push current state to stack
4. route_intent     → dispatch to the right branch
   ├── task  → collect_slots → execute_tool → generate_response
   ├── faq   → faq_lookup (curated answer, no retrieval hop)
   ├── rag   → rag_lookup   → generate_response
   ├── agent → handle_agent (LLM tool loop)
   └── direct → direct_response
5. generate_response → claim-gated streaming (policy numbers verified
                        sentence-by-sentence before they leave the stream)
6. handle_handoff   → emotion/complexity trigger → ticket + queue/AHT info
7. checkpoint       → save state by session_id (Postgres checkpointer,
                      resume on next turn)
```

### Intent Switch & Resume — the killer feature

```
User: "我要退款"
→ refund, filled={}, pending=[order_id, reason]
→ "请提供您的订单号"

User: "订单号12345"
→ refund, filled={order_id:"12345"}, pending=[reason]
→ "请问退款原因是什么？"

User: "退货政策是什么"        ← 🔄 intent switched!
→ push refund state to stack
→ policy → RAG 检索
→ "退货政策是7天无理由..."
   + "您之前的退款申请需要继续吗？"

User: "继续，原因是质量问题"
→ pop stack → resume refund + filled={order_id:"12345"}
→ filled={order_id:"12345", reason:"质量问题"}, complete
→ execute_tool: refund → {status:success, refund_id:"RF123456"}
→ "退款已受理，退款单号RF123456"
```

Two safety rules bound the resume feature:

- **Executed tasks are terminal.** Once a task's tool has actually run (a refund
  confirmed, an order queried), it is never pushed onto the stack and never
  auto-resumed — resurrecting it could only re-run an irreversible action
  (a second refund the user never asked for).
- **Unrecognized input is answered, not hijacked.** An "unknown" classification
  never silently resumes a suspended task: the question gets a direct answer
  plus a visible hint that the suspended task is still there; resume happens
  only on an explicit "继续" or by restating the task.

---

## 🚀 Quick Start

### Option 1: Docker Compose (recommended)

```bash
git clone https://github.com/Bensonluo/rag_chatbot.git
cd rag_chatbot

# Configure environment
cp .env.example .env
# Edit .env: set GLM_API_KEY (and optionally MINIMAX/KIMI for speed tiers)

# Start the service stack (API + PostgreSQL + Redis + Qdrant + Neo4j)
docker-compose up -d

# Or with monitoring stack (Prometheus + Grafana)
docker-compose --profile monitoring up -d

# Verify
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

### Option 2: Try the Live Demo first

Don't want to deploy? **[Try it online →](https://benluo.art/projects/rag-chatbot/)** — no signup, runs in your browser.

> 🔑 **Note on API keys**: configure at least one supported LLM key. Embeddings default to the local [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) model, so the first start downloads model weights and can take several minutes.

### Option 3: Local development

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Start PostgreSQL + Redis + Qdrant via docker-compose
docker-compose up -d postgres redis qdrant

# Run migrations & dev server
alembic upgrade head
uvicorn app.main:create_app --factory --reload --host 0.0.0.0 --port 8000
```

📖 **API docs** (when running): http://localhost:8000/docs

---

## 🌐 Live Demo

The chatbot is deployed and interactive at:

> 🎯 **[benluo.art/projects/rag-chatbot/](https://benluo.art/projects/rag-chatbot/)**

Try these scenarios:
- 🔄 **Intent switch**: Ask "我要退款", partially answer, then ask "退货政策是什么"
- 📚 **RAG retrieval**: Ask about return policy, shipping, warranty
- 🛠️ **Function Calling**: Complete a refund/return/order query workflow

---

## 🏗️ Architecture

### Tech Stack

| Layer | Choice | Why |
|-------|--------|-----|
| **Dialogue orchestration** | LangGraph StateGraph | Conditional routing + checkpointing |
| **API framework** | FastAPI + async | High concurrency, OpenAPI built-in |
| **Vector DB** | Qdrant | Fast hybrid search, open-source |
| **Graph DB** (optional) | Neo4j | Knowledge graph for GraphRAG |
| **Relational DB** | PostgreSQL | Sessions, users, feedback |
| **Cache** | Redis | Rate limiting + L0/L1/L2 answer/semantic/retrieval caches (epoch-scoped) |
| **LLM** | GLM / OpenAI / Anthropic | Auto-fallback by API key availability |
| **Embeddings** | BGE-M3 (local) | Free, multilingual, no API key |
| **Observability** | OpenTelemetry + Prometheus + Grafana | Inspect demo traces and metrics |

### LangGraph Dialogue Graph

```
START
  │
  ▼
guardrail (输入安全检查)
  │
  ▼
detect_intent (意图识别)
  │   cancel 优先 → 槽位值保持任务意图 → 元意图保持
  ▼
handle_switch (意图切换)
  │   推栈保存 / 弹栈恢复
  ▼
route_intent (路由决策)
  │
  ├── task  → collect_slots (槽位收集)
  │             │
  │             ├── complete → execute_tool → generate_response → END
  │             └── missing  → generate_response (追问) → END
  │
  ├── faq   → faq_lookup (策展命中直出) → generate_response → END
  │
  ├── rag   → rag_lookup → generate_response → END
  │
  ├── agent → handle_agent (工具循环) → generate_response → END
  │
  ├── direct → direct_response → END
  │
  └── meta  → generate_response → END
              │
              ▼ (情绪/复杂度触发)
          handle_handoff (转人工工单) → END
```

### Project Structure

```
rag_chatbot/
├── app/
│   ├── api/v1/              # REST endpoints (chat, auth, feedback, docs)
│   ├── services/
│   │   ├── dialogue/        # 🎯 LangGraph engine
│   │   │   ├── graph.py     #   StateGraph construction + compile
│   │   │   ├── nodes.py     #   12 dialogue nodes + conditional edges
│   │   │   ├── state.py     #   DialogueState TypedDict
│   │   │   └── tools.py     #   ToolRegistry + 5 mock handlers
│   │   ├── agent/           # Agent tool loop (bounded iterations)
│   │   ├── faq/             # Curated FAQ fast path
│   │   ├── facts/           # Claim gate (stream) + curated facts + user_facts memory
│   │   ├── handoff/         # Human handoff tickets, queue/AHT, knowledge gaps
│   │   ├── retrieval/       # Hybrid search + rerank + Qdrant
│   │   ├── graph/           # GraphRAG (Neo4j, extraction, community)
│   │   ├── intent/          # Rule + LLM + hybrid intent detection
│   │   ├── slot_filling/    # Per-intent slot schemas
│   │   ├── guardrails/      # Input/output safety
│   │   ├── memory/          # 4 memory strategies
│   │   └── llm/             # Multi-provider LLM clients + <think> stream filter
│   ├── models/              # ORM + Pydantic schemas
│   ├── repositories/        # Async data access layer
│   └── tests/               # Unit + integration + e2e
├── deploy/deploy-server.sh  # One-command service deploy (port-pinned, pre-flight gated)
├── deploy/k8s/              # Kubernetes manifests
├── deploy/prometheus/       # Alert rules (SLA / TTFT / claim gate / handoff queue)
└── docker-compose*.yml
```

---

## ⚙️ Configuration

### Core Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `DATABASE_URL` | PostgreSQL connection string | — | ✅ |
| `REDIS_URL` | Redis connection string | — | ✅ |
| `QDRANT_URL` | Qdrant vector DB URL | — | ✅ |
| `SECRET_KEY` | JWT secret key | — | ✅ |
| `GLM_API_KEY` | Zhipu AI GLM API key | — | (any primary LLM) |
| `MINIMAX_API_KEY` | MiniMax backup + light-tier key | — | ❌ |
| `KIMI_API_KEY` | Kimi Coding generation fallback provider; optional intent override via `INTENT_LLM_PROVIDER=kimi` | — | ❌ |
| `OPENAI_API_KEY` | OpenAI API key | — | (any primary LLM) |
| `ANTHROPIC_API_KEY` | Anthropic API key | — | (any primary LLM) |
| `EMBEDDING_PROVIDER` | Embedding source | `local` | ❌ |
| `GRAPH_RAG_ENABLED` | Enable GraphRAG (needs Neo4j) | `false` | ❌ |

### Pluggable Strategies

The system is designed for swappable components:

```bash
# Intent detection: hybrid | rule_based | llm_based
INTENT_TYPE=hybrid
CONFIDENCE_THRESHOLD=0.7

# Memory: optimized | sliding_window | summarization | hybrid
MEMORY_TYPE=optimized
MEMORY_MAX_RECENT=3
MEMORY_TOKEN_BUDGET=4096

# Retrieval weights
VECTOR_WEIGHT=0.7
USE_RERANKING=true
GRAPH_RAG_FUSION_WEIGHT=0.3   # only if GraphRAG enabled

# Guardrails
GUARDRAILS_ENABLED=true
GUARDRAILS_HARDENING_ENABLED=true

# Authorization posture (review 2026-09-26, #1): DEMO_MODE=true keeps the
# public demo open for unmarked shared sessions. Private sessions always
# require their authenticated owner. Knowledge management and QA require
# active admins in both modes; false enforces token-only chat identity
# and ownership on every session.
DEMO_MODE=true
```

See [.env.example](.env.example) for the full list.

---

## Document ingestion for support knowledge

Chat answers and history include `citations`: reference snapshots with the
document revision, passage, available page/section location, and an excerpt.
Streaming clients opt in with `include_citations: true` to receive the final
`event: citations` frame before `[DONE]`; existing clients retain the text-only
protocol. Historical references keep their original versions after a document
update. These are references supplied for the answer, not proof of each claim.
See the [citation contract](docs/design/answer-citations.md) for compatibility,
location semantics, cache behavior, and source-deletion boundaries.

Uploads retain the original source in PostgreSQL and return **HTTP 202** with
`document_id`, `version_id`, `revision`, and `status`. Existing clients that
expected a completed upload at HTTP 201 must poll the revision endpoint.
The application worker processes queued revisions outside the upload request;
Postgres document locks serialize replacements across replicas. Interrupted
`processing` jobs resume after restart using the same chunk IDs.

| Operation | Endpoint |
|---|---|
| Upload text/Markdown | `POST /api/v1/documents/upload` |
| Upload PDF/TXT/MD | `POST /api/v1/documents/upload/file` |
| List documents | `GET /api/v1/documents` |
| List revisions | `GET /api/v1/documents/{document_id}/versions` |
| Processing status and warnings | `GET /api/v1/documents/versions/{version_id}` |
| Retry a failed/partial latest revision | `POST /api/v1/documents/versions/{version_id}/retry` |
| Download retained original | `GET /api/v1/documents/versions/{version_id}/source` |
| Read normalized text for source offsets | `GET /api/v1/documents/versions/{version_id}/text` |
| Delete indexes and retained sources | `DELETE /api/v1/documents/{document_id}` |
| Check registry, vector store, and worker | `GET /api/v1/documents/health` |

Knowledge lifecycle management requires an active administrator in both Demo
and strict modes, including catalog/version inspection, retained originals and
normalized text. Open registration grants customer access only; customers may
search published evidence but cannot manage shared knowledge. Sensitive document
responses and handled auth/validation errors use private,no-store headers. See
the [knowledge access contract](docs/design/knowledge-access.md).

Text upload example (with an active administrator bearer token):

```json
{
  "document_id": "refund-policy-us",
  "title": "Refund policy",
  "file_type": "md",
  "content": "# Refunds\n\n## Exceptions\nOpened products require review.",
  "metadata": {"lang": "en", "topic": "refunds", "product": "physical-goods", "region": "US"}
}
```

Use the same `document_id` for a replacement. Repeated identical submissions
return the same revision; changes to content, policy scope, or processing
configuration create another revision. Status transitions are
`queued → processing → indexed`; failures are explicit `failed` or `partial`
(the latter means the vector index succeeded but optional graph indexing failed).
Informational OCR/blank-page warnings do not turn a complete index into a partial job.
A failed embedding step preserves the previous index. Qdrant replacement writes
the new points before removing superseded points; it is not a transaction across
Postgres, Qdrant, and Neo4j. New retrieval evidence and cache hits must match the
committed active-version registry; failed/unactivated candidates are withheld.
Publication can temporarily leave no eligible evidence if old points were removed
before a registry failure; retry converges on the retained revision. Existing
active versions remain eligible during a retry attempt. Unregistered raw corpora
must be re-ingested through the durable jobs. Do not run another uncoordinated
writer against the same document IDs. See the
[publication contract and migration boundary](docs/design/knowledge-publication.md).

Each Qdrant point stores `chunk_id`, parent `document_id`, `content`, and metadata:

| Metadata | Contract |
|---|---|
| `document_id`, `version_id`, `chunk_id` | Stable policy identity, immutable revision, deterministic passage ID |
| `index`, `start_pos`, `end_pos` | Chunk order and exact character interval in the retained normalized text |
| `chunk_size`, `chunking_strategy` | Actual character count and splitter used |
| `title`, `original_filename`, `file_type`, `source` | Original source identity, never temporary paths |
| `page_start`, `page_end` | PDF page range; absent for plain text |
| `section_path` | Markdown heading hierarchy, excluding code blocks |
| `lang`, `topic`, `product`, `region`, `category` | Validated author-supplied support scope, usable as retrieval filters |
| `content_hash` | SHA-256 of the uploaded bytes, shared by all chunks in that source revision |
| `pipeline_version`, `embedding_provider`, `embedding_model` | Processing provenance |

The design has three identity levels. `document_id` means a logical policy and
survives edits; supply it explicitly for predictable replacements. `version_id`
identifies one immutable revision of its source, scope, and processing configuration.
`chunk_id` identifies one passage within that revision. It is generated deterministically
from the document/revision, splitter, chunk index, start offset, and content. A retry
therefore overwrites the same points, while a new revision cannot accidentally reuse
the previous revision's passage IDs. Identical wording in different policies still
has different IDs because its provenance differs.

Source locations use zero-based Python string positions, with an exclusive `end_pos`:
`chunk.content == normalized_text[start_pos:end_pos]`. These are character offsets,
not byte offsets, PDF coordinates, or token counts. Normalized text is retained
alongside the original bytes so this relationship remains inspectable after newline
normalization or OCR. PDF pages are one-based; text files have no invented page
numbers. Markdown `section_path` describes the heading ancestry at the start of the
passage; an overlapping chunk can also contain material from the adjacent section.

Support scope is copied onto every chunk so the vector and keyword stores can filter
without a database join. Authors supply `lang`, `topic`, `product`, `region`, and
`category`; the pipeline does not guess an authoritative policy scope from prose.
For example, a caller can filter a refund query to the relevant product and region.
Choose consistent values when uploading; these are exact filter values, and merely
storing them does not automatically select the customer's region. Scope metadata
is not an access-control boundary. Tenant isolation, permissions, approval state,
and effective-date rules would need explicit validation and enforcement in every
retrieval path before becoming supported policy controls.

Processing provenance answers a different question: how was this evidence built?
`content_hash` detects identical uploaded bytes; it is not the document's identity.
`pipeline_version`, the saved chunking configuration, and embedding provider/model
make rebuilds and debugging traceable. Job state, retries, errors, originals, and
uploader identity live in the PostgreSQL revision registry instead of being copied
into every searchable passage. Chunk metadata is intentionally a small retrieval
and citation record, while the registry owns the ingestion lifecycle.

Reserved system fields cannot be overwritten by upload metadata. Uploader identity
stays in the registry, outside retrievable chunk payloads. Vector search, keyword
search, reranking, and graph fusion keep passage identity separate from parent
document identity. Page references remain in retrieval metadata; answer citations
retain the parent document IDs, including evidence retrieved through the graph.
Curated FAQ and fact-check tables are maintained separately; uploading a policy
does not automatically rewrite those reviewed answers or numeric facts.

PDF extraction preserves page boundaries and layout whitespace. Scanned pages use
local Poppler + Tesseract OCR with English and Simplified Chinese language packs
included in the Docker image. A page requiring unavailable/failed OCR fails the
job with an actionable error, rather than silently omitting its evidence. Blank
pages are reported; encrypted PDFs require an unlocked source. Complex tables and
OCR output still need author review against the retained original.

Run `alembic upgrade head` before starting a source checkout; the standard container
entrypoint runs migrations. Rebuild the image to install the OCR binaries. Local
hosts also need `pdftoppm`, `tesseract`, and the configured language packs to process
scans. `DOCUMENT_MAX_BYTES` defaults to 50 MiB, chunk size/overlap to 512/50
characters; long documents are no longer silently truncated. Configure these via
`.env.example`. Original bytes are stored in Postgres, so include them in database
backup/retention sizing. Changing embedding providers/models requires a new vector
collection and re-ingestion; queued jobs with an outdated model configuration fail
explicitly. Pre-existing chunks have no recoverable original source: re-upload
those documents to obtain the new provenance and revision lifecycle.

PostgreSQL backup and new-database recovery are covered by a disposable
PostgreSQL 16 drill, including historical citations, document bytes, handoff
state, and same-database dialogue checkpoints. Follow the
[backup and recovery runbook](docs/operations/backup-and-recovery.md) to select
the correct Compose stack and verify a restore. Qdrant/Neo4j recovery and
separately configured checkpoint databases require their own recovery plans.

---

## 🧪 Testing & Quality

```bash
# Maintained core suite (70% minimum; currently 3720 passed + 3 opt-in skipped / 93.57%)
pytest

# Targeted runs
pytest app/tests/unit/services/dialogue/ -v    # LangGraph tests
pytest app/tests/unit/services/intent/ -v      # Intent detection
# Legacy audit only (contains superseded contracts; not expected to be all green)
pytest app/tests --no-cov

# Use promtool from the deployed Prometheus 2.47.2 release
promtool check rules deploy/prometheus/alerts.yml
promtool test rules deploy/prometheus/containment-alerts.test.yml
```

Answer evaluation separates given-context point coverage from evidence-grounded
pipeline grading. The last live run (2026-10-22, operational-evidence rubric) passed
23 judge controls and 14 bilingual turns using GLM generation and MiniMax judging.
This remains a synthetic isolated measurement, not overall product acceptance.
See the [evaluation runbook](docs/operations/answer-quality-evaluation.md) and
[recorded results](docs/reviews/answer-quality-results-2026-10-22-round9.json).
A separate zero-LLM strict-mode route test now covers owner chat, handoff, claimed
staff context with citations, and denial after resolution.

The live probe checks response/session identity, queued-ticket IDs and policy citation
anchors. Optional authenticated mode allocates owned sessions and reads their history;
a separate staff client can observe ticket/session/owner pairs without claiming tickets.
Default anonymous PASS without a staff observer leaves ticket persistence unverified;
random IDs provide best-effort session isolation.
See the [probe runbook](docs/operations/live-probe.md) for credential configuration,
side effects, bounded queue scans and evidence limits.

### Code Quality Gates

```bash
# Full checks used by CI
ruff check app/
ruff format --check app/
mypy app/
```

Local verification on 2026-10-28: 3720 tests pass, with three opt-in scenarios
skipped by default (PostgreSQL concurrency, assigned-transcript locking, and
the Docker recovery drill);
selected core coverage is 93.57%, not a coverage claim for chat/handoff/API code.
Maintained Ruff lint passes for `app/`, `migrations/`, and the three CLI paths
`scripts/init_db.py`, `scripts/document_import.py`, and `scripts/probe_live.py`.
Format checking passes for `app/` plus `scripts/probe_live.py` (496 files);
strict Mypy passes for 495 `app/` Python files and those three CLI files.
Authenticated HTTP/SSE regressions cover chat ownership, guarded text and
citations, plus ASGI 2.3/2.4 disconnect persistence; the earlier socket-blocked
run is recorded in the [boundary evidence](docs/reviews/http-stream-results-2026-10-23.json).
The refund-review regression covers 24 socket-blocked cases (22 real service/graph,
2 strict JWT HTTP): high-value review tickets, write failures and retry, cancellation,
order access, single stream emission and low-value controls. See the [refund review evidence](docs/reviews/refund-review-results-2026-10-28.json).
Context-aware L0/L1 replay also passes 12 real service/graph cases plus a separate
14-contrast actual semantic-cache audit using local I/O seams. Anonymous stateless
hits remain available; applicable or unavailable history blocks shared replay and
write. See the [cache-context evidence](docs/reviews/context-cache-results-2026-10-28.json).
Persisted reply failures now remain visible beside containment: 15 real SQLite/JWT HTTP
cases verify cohort counts, latest-reply ordering, Prometheus output and outage handling.
Grafana separates counts, ratios and refresh age; see the [metric contract](docs/operations/session-quality-metrics.md).
The probe revision passes 82 focused cases, including six real route scenarios with
SQLite/FAQ/JWT: owned histories and separate staff ticket observation, database refusal,
uncited answers and denied observer access. No live model or production service was used.
See the [probe evidence](docs/reviews/live-probe-results-2026-10-28.json).
Containment alerts now separate fresh business snapshots, unavailable statistics and
clock skew. Native Prometheus timing tests are required by a dedicated CI job configured with
the same engine version as deployment; see the [alert runbook](docs/operations/containment-alerts.md).
Rule evaluation tests do not prove external notification delivery.

The following database checks were verified in earlier iterations and were not
rerun on October 28. All 8 PostgreSQL concurrency tests
and the SQLite/PostgreSQL migration round trips pass in isolated databases.
The new PostgreSQL backup/recovery drill also passes with explicit Docker opt-in.
CI migration/recovery commands were verified locally; remote CI was not triggered.
See the [iteration record](docs/reviews/product-quality-2026-10-16.md) for evidence
and migration precautions. On October 28, the companion `portfolio-fe` passed
177 tests (148 Jest + 29 citation/stream protocol tests), lint, strict TypeScript
checks and an 11-page static build after a clean install. Failed or interrupted
chat replies retain their partial text and cannot become canned business replies;
completion requires the stream's explicit terminator. Next and its ESLint config
were upgraded from 16.3.8 to 16.4.3; runtime npm audit reports zero advisories
(development dependencies excluded). Browser visual/interaction verification was
unavailable. This is the round-17 baseline; see the
[frontend evidence](docs/reviews/frontend-response-results-2026-10-28.json).
The round-18 iteration passes 227 frontend tests (172 Jest + 55 protocol), lint,
strict TypeScript and the 11-page static build. It carries committed message IDs/status through REST,
opt-in SSE and history, and restricts feedback to completed assistant replies.
The shared Demo page still requires an existing token to rate replies and has
no new sign-in flow in that round. See the
[message/feedback contract](docs/design/message-feedback.md) and
[verification results](docs/reviews/message-feedback-results-2026-10-28.json).

The round-20 companion frontend adds `/support/` with real sign-in, private customer sessions,
assigned staff messaging and a saved final explanation plus closure. Its complete suite passes
263 tests (208 Jest + 55 protocol), lint, strict TypeScript and a 12-page static build.
That round passed 3232 backend tests, including 83 new default cases. Existing 93.35% coverage
measures the selected core modules; it does not measure chat/coordinator/handoff/API code.
Private sessions also enforce owner checks in Demo mode, private feedback requires ownership,
and QA access requires admin in both modes. Shared legacy tickets have no private human channel.

Round20 introduced migration `a31e8d7f6290` and required draining old app instances **before**
allowing private consultations; current builds require the newer head described below.
Private sessions/human rows/nonzero automation generations block downgrade;
restarting old Demo binaries can bypass privacy. Local SQLite migration round trips and
schema alignment pass. PostgreSQL advisory-lock coordination has a separate bounded pool and
CI-selected independent-connection regression, but was not executed on real PG this round;
both local Docker socket contexts were unavailable. No browser, production or real payment
acceptance is claimed. See the [human conversation contract](docs/design/human-conversation.md)
and [round-20 evidence](docs/reviews/human-conversation-results-2026-10-28.json).

Round21 makes customer resolution evidence consistent across AI and human messages.
New human customer messages use the existing conservative confirmation/reopen policy in
the same transaction as their text; UUID replay cannot overwrite a later result, and staff
text/closure is not customer confirmation. Human rows do not increase bot-served cohorts.
The workspace also prevents delayed GET/ACK responses from regressing ticket state and
rejects invalid JSON enum types. No new outcome API, schema or confirmation buttons were added.
Full regression: 3288 backend tests (3 existing opt-in skipped), and 272 frontend tests
(217 Jest + 55 protocol), lint/types and 12 static pages. See the [followup evidence](docs/reviews/human-followup-results-2026-10-28.json).

Round22 requires an active administrator for knowledge lifecycle management in
both Demo and strict modes, including retained originals and unpublished version
metadata. Open registration no longer grants knowledge operator access. Sensitive
document responses and handled auth/validation errors are private,no-store; ordinary
customer search and shared Demo chat are preserved. New actual JWT/SQLite/jobs/local
Qdrant coverage adds 32 cases; the role matrix grows from 3 to 10 (net 7), bringing the
full backend suite to 3327. Frontend code is unchanged from the prior 272-test baseline.
See the [access contract](docs/design/knowledge-access.md) and
[round22 evidence](docs/reviews/knowledge-access-results-2026-10-28.json).

Round23 adds a durable SQL execution ledger for confirmed refund/return and MCP write proposals.
Checkpoint failures and fresh-confirm retries replay the saved result for the same UUID;
unknown outcomes block another dispatch, and cancellation cannot claim an earlier request
was reversed. A saved adapter result is distinct from business success. Production startup
requires migration `b42f9c8e7310`; drain old instances before enabling this guarantee.
The new124 default cases cover ledger/migration, both graph/AgentService proposal paths,
actual HTTP/SSE/SQL faults, authority, metadata and MCP parameter binding. Local SQLite
full migration round trips pass; real PG, external payment/MCP and browser acceptance remain
unverified. See the [durable action contract](docs/design/durable-actions.md) and
[round23 evidence](docs/reviews/durable-actions-results-2026-10-28.json).

Round24 (historical baseline) preserves support drafts and immutable human-send attempts across reload,
consultation switches and same-user sign-in recovery. Drafts keep their original
AI/human recipient until an explicit move. New customer sends bind expected_ticket_id;
old-ticket attempts cannot enter a newer ticket. Closed staff attempts have an explicit
saved-work recovery entry. New 13 actual backend cases bring the full suite to 3464;
frontend passes 288 Jest + 55 Node = 343, lint, strict types and 12 static pages.
Local recovery is bounded and cannot replace server authorization or business idempotency.
See the [continuity contract](docs/design/human-conversation.md) and
[round24 evidence](docs/reviews/support-continuity-results-2026-10-28.json).

Round25 (historical baseline) binds workspace conversation operations and local recovery to a server-generated
`session_key`, preventing an old draft or retry from entering a replacement consultation
when numeric session/ticket IDs are reused. Private content requests carry the expected key;
clear-history rotates the key and automation epoch before cleanup. Journal v2 includes the
key; legacy v1 recovery requires explicit local discard and is never silently adopted.
The authenticated operator probe also carries and checks the original server key.
Upgrade all clients and backends to migration head `c53a7d9e8421` (after `b42f9c8e7310`),
with old instances drained and SQLite writers stopped during migration.
New 66 backend cases (17 migration, 42 actual lifecycle, 7 operator) bring the full suite
to 3530 passed and 3 opt-in skipped. New 11 frontend cases (7 journal, 4 actual React)
bring its complete suite to 354 (299 Jest + 55 Node), with lint, strict types and 12 static
pages passing. The frozen manifests cover 501 backend Python and 86 frontend source/config
files. These are local SQLite/JWT and controlled-fetch checks; real PG, browser/hydration,
production migration and external commerce remain unverified. The activity-ordering
candidate is deferred, and the purchase integration design remains proposed.
See the [session incarnation contract](docs/design/session-incarnation.md) and
[round25 evidence](docs/reviews/session-incarnation-results-2026-10-28.json).

Round26 binds feedback and QA mutations to the object and evidence actually selected.
Private feedback requires the displayed session key; the server reauthorizes inside the
conversation gate and atomically binds the SQL update to owner, parent, key and a completed
assistant answer. A committed immutable receipt fixes the original message/session/key;
metrics and best-effort cache cleanup run outside the gate so slow cleanup cannot hold
up a later conversation action. QA queue/context now provides a session key and opaque
review version; verdicts require both and reject a proposal made stale by a new question,
feedback, human message or review correction. Explicit GET review permits a fresh correction.
The projection is streamed in batches of 256, but still reads O(N) bot rows; this is not
an audit log, proof of transcript comprehension or a PostgreSQL/capacity acceptance result.
The final complete backend suite passes 3581 cases with 3 opt-in skips and 93.57% selected
core coverage. The earlier 3580-case run is a historical pre-fix result; the final suite
includes the post-commit cleanup regression. New default cases total 51: 15 feedback unit
and 36 actual HTTP/SQLite scenarios. Frontend at that checkpoint passes 367 (312 Jest + 55 Node);
its 13-case increase comprises 7 feedback guards and 6 preserved locale tests from concurrent
workspace changes. Frozen source covers 502 backend Python and 86 frontend source/config
files, with 14 backend and 6 frontend changes versus round25. No new migration is needed;
head remains `c53a7d9e8421` and clients must
adopt the new private feedback/QA contract. See the [quality mutation contract](docs/design/quality-mutations.md)
and [round26 evidence](docs/reviews/quality-mutations-results-2026-10-28.json).

### Public visitor chat and Agent text commerce (current)

The public website uses server-issued HttpOnly session cookies and a private
conversation per browser, without registration. The existing `/support/` workspace
is retained; product work focuses on the public experience.

Mock shopping is now an Agent orchestration demo with no purchase buttons. The
bounded Agent discovers the provider's authoritative category taxonomy, maps
open-ended natural-language needs onto one or more standard categories, performs
structured multi-category search/pagination, understands references and
constraints, updates a multi-line cart, prepares a quote, creates an order and
settles mock payment entirely through chat text. `commit_checkout` and
`mock_pay` are Agent tools only when the current turn carries a fresh, explicit
authorization for the latest displayed purchase version; handlers recheck owner,
stage, confirmation/payment and generation. The two irreversible steps require
separate confirmations.

Commerce facts live in the separate internal `commerce-provider` FastAPI service:
a deterministic 100,000,000-SKU addressable catalog with a 25-category taxonomy,
SQLite-backed multi-line carts, quotes, stock deltas, orders, payments and
idempotency receipts. The API reaches it only through `ExternalCommercePort`;
provider failure makes shopping explicitly unavailable and never falls back to a
local or regex planner. `PurchaseService` keeps only durable safety orchestration —
owner binding, immutable quote confirmations, version CAS, operation idempotency
and recovery. Unknown or processing results only query the original operation;
clearing chat does not delete or repeat a purchase. Catalog turns render the
LLM's labeled bundle advice plus authoritative typed provider facts, while cart,
quote, order and payment stages remain server-rendered and the model cannot
invent success. All commerce is visibly simulated.

Current text-commerce backend gate: Ruff, formatting and strict Mypy pass; the
complete suite passes **3803 tests with 3 configured skips, 5 warnings** (core
coverage 93.53%). A real local HTTP smoke drove the external adapter through
search → multi-line cart → quote → order → payment → status, including stable
request-key replay and refusal of a conflicting replay. The production canary
then completed the same text-only journey against the separate internal
provider with three cart lines, one order, one payment and zero purchase-action
requests. A later production browser canary sent the open-ended
“budget ¥1,500 remote-office setup” request and received labeled bundle advice
plus candidates from monitor, keyboard, mouse, headset, webcam and dock
categories; persisted metadata recorded `planner_mode=provider`, one
`commerce_discover_catalog` call and seven category searches, with no purchase
mutation. See the
[external provider deployment record](docs/operations/external-commerce-provider-deployment-2026-10-29.md)
and the earlier [Agent text-commerce record](docs/operations/agent-text-commerce-deployment-2026-10-29.md).

New migrations are `d64b8c0f9532` (anonymous visitors) and `e75c9d1064a3`
(commerce framework), following `c53a7d9e8421`. The 2026-10-29 deployment kept
the database at `e75c9d1064a3` and restored no old image or backup.
`GUEST_COOKIE_SECURE=true` requires HTTPS; disable it only for local HTTP.
Cookie identity represents its browser, with no real-person or cross-device
recovery promise. See the [public commerce contract](docs/design/public-commerce.md).
The round27 numbers below are the historical button-card baseline:
3709 backend tests, 398 frontend checks, strict types/lint/build and a native
Chrome mock journey passed then. See
[round27 evidence](docs/reviews/public-commerce-results-2026-10-28.json).
Neither the historical run nor the current mock deployment verifies real
payments, supplier integration, production concurrency/capacity or a real
commerce rollout.

### Historical button reliability milestone (2026-10-28)

The preceding button-card implementation committed its conversation task
boundary before invoking Agent or business operations. A failed fence blocked
the operation; later model errors, cancellation or chat-save failures could not
reactivate an earlier refund proposal. Commerce requests waited at most 30
seconds, then required explicit status recovery without resubmission. Its local
gate passed **3720 backend** and **407 frontend** tests plus lint, strict types
and a 12-page build. The timeout path used real React and controlled fetch; the
prior milestone used real Chrome. These reliability semantics carry forward,
while its action-ref UI contract is obsolete. See the
[milestone record](docs/reviews/public-purchase-milestone-2026-10-28.md) and
[reliability evidence](docs/reviews/public-commerce-reliability-2026-10-28.json).

| Metric | Value |
|--------|-------|
| Maintained test results | **3720 passed, 3 opt-in skipped** |
| Maintained core coverage | **93.57%** (70% minimum enforced locally; main dialogue path graph/nodes/tools measured since 2026-10-01) |
| Python files (app/) | 495 |
| Python files in app/tests/ | 261 (255 test_ modules) |

---

## 🗺️ Roadmap

- [x] LangGraph StateGraph with 12 nodes + conditional routing
- [x] Intent switch & resume via state stack
- [x] Task terminality guard (executed tasks never suspended or auto-resumed)
- [x] Hybrid retrieval (vector + BM25 + rerank)
- [x] GraphRAG with Neo4j (optional)
- [x] OpenTelemetry tracing + Prometheus metrics
- [x] 3720 passing core tests, 93.57% coverage; routing golden-set eval (19 cases / 31 turns over the real compiled graph) + retrieval baseline; runtime dependency lock (requirements-lock.txt) + pip-audit in CI
- [x] FAQ fast path (curated answers skip retrieval)
- [x] Agent tool loop with bounded iterations
- [x] Claim-gated streaming (policy numbers verified before release)
- [x] Cross-session user memory (user_facts + TTL recall cache)
- [x] Human handoff: private login workspace, assigned-agent context, bidirectional human messages, atomic final reply/closure and claimed-session automation pause
- [x] Postgres checkpointer (AsyncPostgresSaver, MemorySaver fallback)
- [x] Prometheus alert rules (SLA / TTFT / claim gate / handoff queue)
- [x] Epoch-scoped cache pyramid (L0 answer / L1 semantic / L2 retrieval)
- [x] Containment proxy and reply-status cohorts (separate from confirmed resolution, DB-bridged gauges)
- [x] 2→N scaling readiness (nginx upstream template + dns_sd multi-replica metrics)
- [x] Graceful shutdown choreography (background refreshers stop before pools close)
- [ ] Fine-tuned intent classifier (replace LLM-based with small specialized model)
- [ ] A/B testing framework for prompt variants
- [ ] Multi-tenant knowledge bases

---

## 🤝 Contributing

PRs welcome! Especially:
- 🐛 Bug fixes — please include a failing test
- ✨ New dialogue node types or slot strategies
- 📚 More GraphRAG use cases (currently: customer service KB)
- 🌍 i18n improvements

For development setup, follow the local-development commands above and run `pytest` before opening a PR.

---

## 📜 License

[MIT](LICENSE) — free for personal and commercial use.

If this project helped you, please ⭐ star the repo — it helps others discover it.

---

## 📬 Contact

- 💼 **Portfolio**: [benluo.art](https://benluo.art)
- 🐙 **GitHub**: [@Bensonluo](https://github.com/Bensonluo)
- 💬 **Issues**: [GitHub Issues](https://github.com/Bensonluo/rag_chatbot/issues)

---

## 🇨🇳 中文说明

**Agentic RAG 企业级智能客服系统** — 面向企业知识服务与业务办理，连接多轮任务、版本化知识、可追溯回答与人工工单，覆盖对话编排、检索治理、安全门控和运维观测。

### 核心亮点

- **LangGraph StateGraph 对话图**:12 节点编排(安全检查 → 意图识别 → 意图切换 → 路由 → 槽位/工具/FAQ/RAG/Agent → 生成 → 转人工)
- **业务意图分类**:5 种任务型(退款/退货/订单查询/物流追踪/投诉)+ 知识型 + 对话型 + 元意图
- **五路路由**:task → 槽位收集 → 工具执行;faq → 策展命中直出;rag → 检索生成;agent → 工具循环;direct → LLM 直答
- **多轮槽位收集**:正则提取 + 短消息回退,缺槽追问
- **Function Calling**:ToolRegistry + 5 个 Mock 工具 + Agent 工具循环
- **意图切换与恢复**:State Stack 推栈保存/弹栈恢复;已执行任务为终态,不再挂起或自动恢复(防止二次执行)
- **声明门控流式输出**:句子缓冲 + 策展事实校验,越界声明拦截/改写后才放出
- **跨会话用户记忆**:user_facts 长期画像 + TTL 召回缓存
- **人工接续**:情绪/复杂度触发工单，登录后私人咨询、客服领取与双向消息、最终说明和关闭同事务；接单后暂停机器人，旧共享 Demo 仅原历史和行政关闭
- **GraphRAG**:Neo4j 知识图谱 + Text-to-Cypher + 社区发现
- **混合检索**:向量 + BM25 → RRF 融合 → Cross-Encoder 重排序
- **全链路可观测**:OpenTelemetry + Prometheus 指标(TTFT/时长/结果)+ SLA/策略门控告警规则
- **缓存金字塔**:L0 答案缓存 / L1 语义缓存 / L2 检索结果缓存,KB epoch 轮换整体失效,无逐键驱逐
- **自动处理率 KPI（containment rate）**:北极星指标,无转人工工单且无差评的会话占比;口径是「未升级」,不等同「已解决」;Prometheus DB 桥接 gauge
- **横向扩展就绪**:单机即节点模板(compose 端口段 + nginx upstream + dns_sd 多副本抓取),2→N 不重构
- **Checkpoint 持久化**:LangGraph Postgres Checkpointer(本地降级 MemorySaver)

### 快速开始

```bash
git clone https://github.com/Bensonluo/rag_chatbot.git
cd rag_chatbot
cp .env.example .env  # 填入 GLM_API_KEY
docker-compose up -d
```

访问 http://localhost:8000/docs 查看 API 文档。

> 部署与接入：默认启动单 API 实例；企业业务接口通过 ToolRegistry 接入，鉴权、数据留存、监控与副本按部署环境配置。多租户知识隔离列在路线图，当前验证结果见 Testing & Quality。
