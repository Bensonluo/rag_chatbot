<div align="center">

# GraphRAG Smart Customer Service

**A production-grade enterprise chatbot powered by LangGraph — 9-node dialogue graph, Function Calling, hybrid RAG, and GraphRAG knowledge retrieval.**

[![Live Demo](https://img.shields.io/badge/LIVE-DEMO-brightgreen?style=for-the-badge&logo=vercel)](https://benluo.art/projects/rag-chatbot/)
[![GitHub stars](https://img.shields.io/github/stars/Bensonluo/rag_chatbot?style=for-the-badge)](https://github.com/Bensonluo/rag_chatbot/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-FF6B6B)](https://github.com/langchain-ai/langgraph)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

<!-- 🎬 录制说明:用 lic_ecap/kap 录 30 秒对话演示,放到 docs/assets/demo.gif -->
<!--     录制内容:输入"我要退款" → 多轮槽位收集 → 意图切换"退货政策" → 恢复原任务 -->
<img src="docs/assets/demo.gif" alt="RAG Chatbot Demo" width="80%">

*🎬 Replace this with a 30s GIF of the dialogue flow — see [Recording Guide](#-demo-recording-guide) below*

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

This project solves all of them with a **LangGraph StateGraph** — the same architecture used by enterprises running mission-critical dialogue systems. It's not a demo; it's a reference implementation you can learn from and extend.

> 💬 **What you'll learn**: how to structure a multi-node dialogue graph, design intent routing with task resume, integrate hybrid retrieval (vector + GraphRAG), and ship it with full observability.

---

## ✨ Key Highlights

<div align="center">

| 🎯 Dialogue Engine | 🔧 Function Calling | 📚 RAG |
|:---:|:---:|:---:|
| **9-node** StateGraph | **5** mock tools | Hybrid + GraphRAG |
| Intent switch & resume | ToolRegistry pattern | Vector + BM25 + RRF |
| Slot filling | Task → tool → response | Cross-Encoder reranking |

| 🛡️ Safety | 📊 Observability | 🚀 Production |
|:---:|:---:|:---:|
| Input/Output guardrails | OpenTelemetry tracing | Docker Compose |
| Prompt injection detection | Prometheus + Grafana | K8s manifests |
| PII redaction | Token & latency metrics | Health checks |

| 📈 Stats | | |
|:---:|:---:|:---:|
| **104+** test cases | **3** LLM providers | **80%+** coverage |
| **9** dialogue nodes | **4** data stores | **4** memory strategies |

</div>

### 🧠 What makes it different

1. **LangGraph StateGraph, not a chain** — 9 nodes with conditional edges, checkpointed for resume
2. **Intent switch with state stack** — push/pop pattern to save & restore in-flight tasks
3. **Tri-route dispatch** — task → tool execution / RAG → retrieval / direct → LLM
4. **GraphRAG (optional)** — Neo4j knowledge graph + Text-to-Cypher + community detection
5. **Hybrid retrieval** — vector (Qdrant) + BM25 keyword → RRF fusion → Cross-Encoder rerank
6. **Full-stack observability** — OpenTelemetry + Prometheus + Grafana out of the box

---

## 🔄 How It Works

A user says *"I want a refund"*. Here's what happens:

```
1. guardrail        → scan for prompt injection, redact PII
2. detect_intent    → classify: refund (task) | policy (rag) | chitchat (direct)
3. handle_switch    → if intent changed, push current state to stack
4. route_intent     → dispatch to the right branch
   ├── task → collect_slots → execute_tool → generate_response
   ├── rag   → rag_lookup   → generate_response
   └── direct → direct_response
5. checkpoint       → save state by session_id (resume on next turn)
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

---

## 🚀 Quick Start

### Option 1: Docker Compose (recommended, ~2 min)

```bash
git clone https://github.com/Bensonluo/rag_chatbot.git
cd rag_chatbot

# Configure environment
cp .env.example .env
# Edit .env: set GLM_API_KEY (or OPENAI_API_KEY / ANTHROPIC_API_KEY)

# Start core services (API + PostgreSQL + Redis + Qdrant)
docker-compose up -d

# Or with monitoring stack (Prometheus + Grafana)
docker-compose --profile monitoring up -d

# Verify
curl http://localhost:8000/health
# → {"status":"healthy","database":"connected","redis":"connected",...}
```

### Option 2: Try the Live Demo first

Don't want to deploy? **[Try it online →](https://benluo.art/projects/rag-chatbot/)** — no signup, runs in your browser.

> 🔑 **Note on API keys**: GLM is recommended for Chinese workloads (best cost/quality ratio). The system auto-falls back: GLM → OpenAI → Anthropic. Embeddings default to local BGE-M3 (free, no API key needed).

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
| **Cache** | Redis | Rate limiting, embeddings cache |
| **LLM** | GLM / OpenAI / Anthropic | Auto-fallback by API key availability |
| **Embeddings** | BGE-M3 (local) | Free, multilingual, no API key |
| **Observability** | OpenTelemetry + Prometheus + Grafana | Production-grade tracing & metrics |

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
  ├── rag   → rag_lookup → generate_response → END
  │
  ├── direct → direct_response → END
  │
  └── meta  → generate_response → END
```

### Project Structure

```
rag_chatbot/
├── app/
│   ├── api/v1/              # REST endpoints (chat, auth, feedback, docs)
│   ├── services/
│   │   ├── dialogue/        # 🎯 LangGraph engine
│   │   │   ├── graph.py     #   StateGraph construction + compile
│   │   │   ├── nodes.py     #   9 dialogue nodes + conditional edges
│   │   │   ├── state.py     #   DialogueState TypedDict
│   │   │   └── tools.py     #   ToolRegistry + 5 mock handlers
│   │   ├── retrieval/       # Hybrid search + rerank + Qdrant
│   │   ├── graph/           # GraphRAG (Neo4j, extraction, community)
│   │   ├── intent/          # Rule + LLM + hybrid intent detection
│   │   ├── slot_filling/    # Per-intent slot schemas
│   │   ├── guardrails/      # Input/output safety
│   │   ├── memory/          # 4 memory strategies
│   │   └── llm/             # Multi-provider LLM clients
│   ├── models/              # ORM + Pydantic schemas
│   ├── repositories/        # Async data access layer
│   └── tests/               # Unit + integration + e2e
├── deploy/k8s/              # Kubernetes manifests
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
```

See [.env.example](.env.example) for the full list.

---

## 🧪 Testing & Quality

```bash
# Full test suite with coverage (80% minimum enforced)
pytest

# Targeted runs
pytest app/tests/unit/services/dialogue/ -v    # LangGraph tests
pytest app/tests/unit/services/intent/ -v      # Intent detection
pytest app/tests/integration/ -v               # API integration
```

### Code Quality Gates

```bash
ruff check app/      # Lint
ruff format app/     # Format
mypy app/            # Type check
```

| Metric | Value |
|--------|-------|
| Test cases | **104+** |
| Test coverage | **80%+** (CI-enforced) |
| Python files | ~200 |
| Test files | ~58 |

---

## 🗺️ Roadmap

- [x] LangGraph StateGraph with 9 nodes + conditional routing
- [x] Intent switch & resume via state stack
- [x] Hybrid retrieval (vector + BM25 + rerank)
- [x] GraphRAG with Neo4j (optional)
- [x] OpenTelemetry tracing + Prometheus metrics
- [x] 104+ test cases, 80%+ coverage
- [ ] PostgresSaver checkpointing (production-grade persistence)
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

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup.

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

**GraphRAG 智能客服** — 基于 LangGraph 构建的企业级客服对话系统。

### 核心亮点

- **LangGraph StateGraph 对话图**:9 节点编排(安全检查 → 意图识别 → 意图切换 → 路由 → 槽位/工具/RAG → 生成回复)
- **业务意图分类**:5 种任务型(退款/退货/订单查询/物流追踪/投诉)+ 知识型 + 对话型 + 元意图
- **多轮槽位收集**:正则提取 + 短消息回退,缺槽追问
- **Function Calling**:ToolRegistry + 5 个 Mock 工具
- **意图切换与恢复**:State Stack 推栈保存/弹栈恢复
- **GraphRAG**:Neo4j 知识图谱 + Text-to-Cypher + 社区发现
- **混合检索**:向量 + BM25 → RRF 融合 → Cross-Encoder 重排序
- **全链路可观测**:OpenTelemetry + Prometheus + Grafana

### 快速开始

```bash
git clone https://github.com/Bensonluo/rag_chatbot.git
cd rag_chatbot
cp .env.example .env  # 填入 GLM_API_KEY
docker-compose up -d
```

访问 http://localhost:8000/docs 查看 API 文档。

📖 **部署文档**:参考 [Portfolio Deployment Guide](https://github.com/Bensonluo/portfolio-fe)。

---

<details>
<summary>🎬 Demo Recording Guide (for maintainers)</summary>

### How to record the hero GIF

1. **Tool**: [licecap](https://www.cockos.com/licecap/) (Mac/Win, free) or [kap](https://getkap.co/) (Mac, OSS)
2. **Content** (~30s):
   - 0-5s: Type "我要退款", show bot asking for order_id
   - 5-15s: Provide order_id, watch slot filling
   - 15-20s: Ask "退货政策是什么" mid-flow (show intent switch)
   - 20-30s: Resume original refund task, watch it complete
3. **Save to**: `docs/assets/demo.gif` (keep under 5MB)
4. **Update**: Replace the placeholder `<img>` in the hero section

### Architecture diagram

Use [excalidraw](https://excalidraw.com/) (free) or [mermaid](https://mermaid.live/) to export a clean PNG of the dialogue graph, save to `docs/assets/architecture.png`.

</details>

<!--
RECORDING_TODO:
1. Record demo.gif → docs/assets/demo.gif
2. Draw architecture.png → docs/assets/architecture.png
3. Replace placeholder img tags in hero section
4. Update Live Demo URL (benluo.art → real domain)
-->
