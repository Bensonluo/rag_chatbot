<div align="center">

# Agentic RAG Smart Customer Service

**A personal technical demo powered by LangGraph — 12-node dialogue graph, Function Calling, agent tool loop, hybrid RAG, claim-gated streaming, and human handoff.**

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

Most "RAG chatbot" tutorials stop at a single vector search call. **Real customer service is much harder**:

- ❌ Users interrupt midway ("wait, what's the refund policy?") then expect to resume
- ❌ Tasks need multi-turn slot collection (order ID, reason, amount...)
- ❌ You need function calling to actually execute refunds, not just chat about them
- ❌ Knowledge questions and task questions need different handling paths
- ❌ PII leakage and prompt injection are real attack vectors
- ❌ A streaming LLM can state numbers the policy never approved
- ❌ Some turns need a human — the bot has to know when and how to hand off

This project explores those problems with a **LangGraph StateGraph**. It is deliberately an architecture-focused personal demo: easy to read, run, modify, and discuss in an interview or technical review.

> 💬 **What you'll learn**: how to structure a multi-node dialogue graph, design intent routing with task resume, integrate hybrid retrieval (vector + GraphRAG), and expose useful runtime metrics.

### Project scope

- The maintained path is a single-instance technical demo, not a production SLA target.
- PostgreSQL, Redis, Qdrant, and Neo4j data created by Compose can be treated as disposable demo data.
- Authentication, Kubernetes, tracing, and community GraphRAG remain extension examples; they are not part of the default quality gate.
- The core quality gate covers dialogue state, intent/slot processing, guardrails, document ingestion, retrieval, and the connected GraphRAG path.

---

## ✨ Key Highlights

<div align="center">

| 🎯 Dialogue Engine | 🔧 Function Calling | 📚 RAG |
|:---:|:---:|:---:|
| **12-node** StateGraph | **5** mock tools | Hybrid + GraphRAG |
| Intent switch & resume | ToolRegistry pattern | Vector + BM25 + RRF |
| FAQ fast path + agent loop | Task → tool → response | Cross-Encoder reranking |

| 🛡️ Safety | 📊 Observability | 🤝 Human in the loop |
|:---:|:---:|:---:|
| Input/Output guardrails | OpenTelemetry tracing | Handoff tickets + queue/AHT |
| Claim-gated streaming | Prometheus + Grafana | Knowledge-gap closeout |
| PII redaction + `<think>` filtering | TTFT/duration/outcome + alert rules | Cross-session user memory |

| ⚡ Caching | 📈 Funnel KPI | 📦 Scaling |
|:---:|:---:|:---:|
| L0/L1/L2 epoch-scoped pyramid | Containment resolution rate | 2→N node template |
| Answer / semantic / retrieval tiers | Downvote-aware aggregation | nginx upstream + dns_sd |

| 📈 Stats | | |
|:---:|:---:|:---:|
| **2664** passing tests | **3** LLM providers | **93.01%** core coverage |
| **12** dialogue nodes | **5** data stores | **4** memory strategies |

</div>

### 🧠 What makes it different

1. **LangGraph StateGraph, not a chain** — 12 nodes with conditional edges, checkpointed for resume
2. **Intent switch with state stack** — push/pop pattern to save & restore in-flight tasks; executed tasks are terminal (never suspended, never auto-resumed)
3. **Five-way dispatch** — task → tool execution / faq → curated direct answer / rag → retrieval / agent → tool loop / direct → LLM
4. **Claim-gated streaming** — sentence-buffered policy checks gate what leaves the stream; stream and persistence stay identical
5. **GraphRAG (optional)** — Neo4j knowledge graph + Text-to-Cypher + community detection
6. **Hybrid retrieval** — vector (Qdrant) + BM25 keyword → RRF fusion → Cross-Encoder rerank
7. **Human handoff loop** — emotion/complexity-triggered tickets with queue position and AHT, plus knowledge-gap recording
8. **Observable demo runtime** — `/metrics`, TTFT/duration/outcome metrics, SLA + policy-gate alert rules, tracing hooks
9. **Epoch-scoped cache pyramid** — L0 answer / L1 semantic / L2 retrieval caches; every KB mutation rotates an epoch that invalidates all three wholesale — no stale entries, no per-key eviction
10. **Containment KPI** — the share of served sessions with no handoff or downvote, served as a DB-bridged Prometheus gauge. Containment is not confirmed resolution; user confirmations, reopen counts, and human QA verdicts are tracked separately.
11. **Scale-ready node template** — compose port range + nginx upstream template + dns_sd multi-replica scraping: one node today, N replicas without re-architecture

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
# Edit .env: set GLM_API_KEY (or OPENAI_API_KEY / ANTHROPIC_API_KEY)

# Start the demo stack (API + PostgreSQL + Redis + Qdrant + Neo4j)
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
├── deploy/deploy-server.sh  # One-command demo-box deploy (port-pinned, pre-flight gated)
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
| `GLM_API_KEY` | Zhipu AI GLM API key | — | (any LLM) |
| `OPENAI_API_KEY` | OpenAI API key | — | (any LLM) |
| `ANTHROPIC_API_KEY` | Anthropic API key | — | (any LLM) |
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
# public demo open (anonymous chat under a body user_id, history by bare
# session id); false enforces token-only identity, session ownership on
# history read/clear, and admin-only knowledge-base writes.
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

Text upload example (with the normal bearer token):

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
Postgres, Qdrant, and Neo4j. A failure during publication is recorded and invalidates
caches; retry converges on the retained revision. Do not run another uncoordinated
writer against the same document IDs.

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
# Maintained core suite (70% minimum; currently 2664 passed + 2 opt-in skipped / 93.01%)
pytest

# Targeted runs
pytest app/tests/unit/services/dialogue/ -v    # LangGraph tests
pytest app/tests/unit/services/intent/ -v      # Intent detection
# Legacy audit only (contains superseded contracts; not expected to be all green)
pytest app/tests --no-cov
```

Answer evaluation now separates given-context point coverage from evidence-grounded
pipeline grading. The isolated real graph covers 14 bilingual turns with actual
BM25 retrieval, citations, SQLite messages, and handoff outcomes. GLM generation
and independent MiniMax judging passed these synthetic checks on 2026-10-17;
manual review still found a Chinese handoff hint in English replies. This is not
an overall product acceptance result. See the [evaluation runbook](docs/operations/answer-quality-evaluation.md)
and [recorded results](docs/reviews/answer-quality-results-2026-10-17.json).

### Code Quality Gates

```bash
# Full checks used by CI
ruff check app/
ruff format --check app/
mypy app/
```

Local verification on 2026-10-22: 2664 tests pass, with two opt-in scenarios
skipped by default (PostgreSQL concurrency and the Docker recovery drill);
core coverage is 93.01%. Ruff and
full strict Mypy pass (424 Python files).

The following dependency and companion-frontend checks were verified in earlier
iterations and were not rerun on October 22. All 8 PostgreSQL concurrency tests
and the SQLite/PostgreSQL migration round trips pass in isolated databases.
The new PostgreSQL backup/recovery drill also passes with explicit Docker opt-in.
CI migration/recovery commands were verified locally; remote CI was not triggered.
See the [iteration record](docs/reviews/product-quality-2026-10-16.md) for evidence
and migration precautions. The companion `portfolio-fe` now passes all 86
tests (79 Jest + 7 citation protocol tests), lint, strict TypeScript checks, and
a static build of 11 pages. Its legacy test configuration and 125 type errors
are resolved. Runtime npm audit reports zero vulnerabilities; development-tool
dependency advisories remain documented in the iteration record.

| Metric | Value |
|--------|-------|
| Maintained test results | **2664 passed, 2 opt-in skipped** |
| Maintained core coverage | **93.01%** (70% minimum enforced locally; main dialogue path graph/nodes/tools measured since 2026-10-01) |
| Python files (app/) | 424 |
| Python files in app/tests/ | 214 (208 test_ modules) |

---

## 🗺️ Roadmap

- [x] LangGraph StateGraph with 12 nodes + conditional routing
- [x] Intent switch & resume via state stack
- [x] Task terminality guard (executed tasks never suspended or auto-resumed)
- [x] Hybrid retrieval (vector + BM25 + rerank)
- [x] GraphRAG with Neo4j (optional)
- [x] OpenTelemetry tracing + Prometheus metrics
- [x] 2664 passing core tests, 93.01% coverage; routing golden-set eval (19 cases / 31 turns over the real compiled graph) + retrieval baseline; runtime dependency lock (requirements-lock.txt) + pip-audit in CI
- [x] FAQ fast path (curated answers skip retrieval)
- [x] Agent tool loop with bounded iterations
- [x] Claim-gated streaming (policy numbers verified before release)
- [x] Cross-session user memory (user_facts + TTL recall cache)
- [x] Human handoff: tickets, queue position, AHT, knowledge-gap closeout
- [x] Postgres checkpointer (AsyncPostgresSaver, MemorySaver fallback)
- [x] Prometheus alert rules (SLA / TTFT / claim gate / handoff queue)
- [x] Epoch-scoped cache pyramid (L0 answer / L1 semantic / L2 retrieval)
- [x] Containment resolution KPI (downvote-aware, DB-bridged gauge)
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

**Agentic RAG 智能客服** — 基于 LangGraph 构建的个人技术 Demo，重点展示对话状态、混合检索、可靠性与可观测链路。

### 核心亮点

- **LangGraph StateGraph 对话图**:12 节点编排(安全检查 → 意图识别 → 意图切换 → 路由 → 槽位/工具/FAQ/RAG/Agent → 生成 → 转人工)
- **业务意图分类**:5 种任务型(退款/退货/订单查询/物流追踪/投诉)+ 知识型 + 对话型 + 元意图
- **五路路由**:task → 槽位收集 → 工具执行;faq → 策展命中直出;rag → 检索生成;agent → 工具循环;direct → LLM 直答
- **多轮槽位收集**:正则提取 + 短消息回退,缺槽追问
- **Function Calling**:ToolRegistry + 5 个 Mock 工具 + Agent 工具循环
- **意图切换与恢复**:State Stack 推栈保存/弹栈恢复;已执行任务为终态,不再挂起或自动恢复(防止二次执行)
- **声明门控流式输出**:句子缓冲 + 策展事实校验,越界声明拦截/改写后才放出
- **跨会话用户记忆**:user_facts 长期画像 + TTL 召回缓存
- **转人工兜底**:情绪/复杂度触发工单(队列位置、AHT),知识缺口记录与闭环
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

> 项目口径：这是可运行、可扩展、适合技术交流的单实例 Demo；不把多租户、高可用、灾备等生产能力作为完成标准。
