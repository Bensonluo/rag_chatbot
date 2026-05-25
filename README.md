# RAG 智能客服

Production-grade intelligent customer service with LangGraph dialogue management, business intent routing, Function Calling, and RAG knowledge retrieval.

## Features

### LangGraph 对话引擎
- **StateGraph 编排**: 9 节点对话图（安全检查 → 意图识别 → 意图切换 → 路由 → 槽位收集/工具执行/RAG检索 → 生成回复）
- **业务意图分类**: 5 种任务型（退款/退货/订单查询/物流追踪/投诉）+ 知识型 + 对话型 + 元意图
- **多轮槽位收集**: 正则提取 + 短消息回退赋值，缺槽追问直到完整
- **Function Calling**: ToolRegistry + 5 个工具（退款/退货/订单查询/物流追踪/投诉）
- **意图切换恢复**: State Stack 推栈保存/弹栈恢复，中途切换后无缝继续
- **Checkpoint 持久化**: MemorySaver 按 session_id 自动保存/恢复对话状态
- **三路路由**: task → 槽位收集 → 工具执行；rag → 向量/图谱检索；direct → LLM 直出

### Core RAG Capabilities
- **Vector Search**: Qdrant vector database with semantic search
- **Hybrid Search**: Combines semantic vector search + BM25 keyword search with Reciprocal Rank Fusion
- **Smart Embeddings**: Local BGE-M3 (free, default) with GLM/OpenAI API options
- **Document Management**: Upload, search, and manage documents (PDF, TXT, MD)
- **Semantic Chunking**: Multiple strategies (fixed, semantic, recursive)
- **LLM Reranking**: Advanced reranking for better retrieval quality

### Advanced RAG
- **GraphRAG** (optional): Neo4j-powered knowledge graph with entity extraction, community detection, and global search
- **Multi-Path Fusion**: Merges vector + graph retrieval results with configurable weighting
- **Text-to-Cypher**: Natural language to Cypher query translation
- **Community Summarization**: Leiden algorithm-based community detection with LLM summaries

### Chat Intelligence
- **Hybrid Intent Detection**: Rule-based + LLM-powered business intent classification with confidence scores
- **Slot Filling**: Per-intent slot schemas with regex extraction + fallback assignment
- **Memory Management**: Four strategies — optimized (default), sliding window, summarization, hybrid
- **Streaming Responses**: Real-time server-sent events for low latency
- **Multilingual**: Excellent Chinese + English support

### Safety & Observability
- **Input Guardrails**: Prompt injection detection, PII redaction, prompt hardening
- **Output Guardrails**: PII redaction in responses
- **Observability**: OpenTelemetry-based LLM tracing with latency, token usage, and error metrics
- **Feedback Loop**: User thumbs up/down ratings for continuous improvement

### System Features
- **JWT Authentication**: Secure user authentication
- **Rate Limiting**: Token bucket algorithm (in-memory; Redis recommended for multi-instance)
- **Monitoring**: Prometheus metrics, health/readiness checks, optional Grafana dashboards
- **Docker Support**: Full containerized deployment with Docker Compose
- **Multi-Provider LLM**: GLM (default), OpenAI, Anthropic — auto-fallback by available API key

## Architecture

### LangGraph Dialogue Graph

```
START
  │
  ▼
guardrail (输入安全)
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

### Intent Switch & Resume Flow

```
User: "我要退款"
→ refund, filled={}, pending=[order_id, reason]
→ "请提供您的订单号"

User: "订单号12345"
→ refund, filled={order_id:"12345"}, pending=[reason]
→ "请问退款原因是什么？"

User: "退货政策是什么"        ← 意图切换！
→ push refund state to stack
→ policy → RAG 检索
→ "退货政策是7天无理由..."
   + "您之前的退款申请需要继续吗？"

User: "继续，原因是质量问题"
→ pop stack → 恢复 refund + filled={order_id:"12345"}
→ filled={order_id:"12345", reason:"质量问题"}, complete
→ execute_tool: refund → {status:success, refund_id:"RF123456"}
→ "退款已受理，退款单号RF123456"
```

### Data Stores

| Service    | Purpose                              | Required |
|------------|--------------------------------------|----------|
| PostgreSQL | Users, sessions, messages, feedback  | Yes      |
| Redis      | Caching, rate limiting               | Yes      |
| Qdrant     | Vector embeddings, semantic search   | Yes      |
| Neo4j      | Knowledge graph (GraphRAG)           | No       |

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 16+
- Redis 7+
- Qdrant 1.7+
- Docker & Docker Compose (recommended)
- GLM API key (recommended) or OpenAI/Anthropic API key
- Neo4j (optional, for GraphRAG)

### Docker Compose (Recommended)

```bash
# Clone repository
git clone https://github.com/Bensonluo/rag_chatbot.git
cd rag_chatbot

# Create environment file
cat > .env << EOF
# LLM Configuration (GLM recommended for Chinese)
GLM_API_KEY=your-glm-api-key-here
GLM_MODEL=glm-4.5-air

# Embedding Configuration (local = free)
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=bge-m3-v2-zh

# Vector Database
QDRANT_URL=http://localhost:6333

# Database & Cache
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/rag_chatbot
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=your-secret-key-here

# Optional: Enable GraphRAG
GRAPH_RAG_ENABLED=false
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
EOF

# Start core services
docker-compose up -d

# Or start with monitoring (Prometheus + Grafana)
docker-compose --profile monitoring up -d

# Check health
curl http://localhost:8000/health

# View logs
docker-compose logs -f api
```

### Manual Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Run database migrations
alembic upgrade head

# Start development server
uvicorn app.main:create_app --factory --reload --host 0.0.0.0 --port 8000
```

## API Documentation

Once running, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

### Quick API Examples

```bash
# 1. Register user
curl -X POST http://localhost:8000/api/v1/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "securepass123",
    "full_name": "John Doe"
  }'

# 2. Login
TOKEN=$(curl -X POST http://localhost:8000/api/v1/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "securepass123"
  }' | jq -r '.access_token')

# 3. Send chat message (LangGraph dialogue graph)
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "我要退款",
    "session_id": 1
  }'

# 4. Stream response
curl -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "退货政策是什么",
    "session_id": 1
  }'

# 5. Submit feedback
curl -X POST http://localhost:8000/api/v1/feedback \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "message_id": 123,
    "rating": "up"
  }'
```

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `DATABASE_URL` | PostgreSQL connection string | — | Yes |
| `REDIS_URL` | Redis connection string | — | Yes |
| `QDRANT_URL` | Qdrant vector DB URL | — | Yes |
| `SECRET_KEY` | JWT secret key | — | Yes |
| `GLM_API_KEY` | Zhipu AI GLM API key | — | No |
| `GLM_MODEL` | GLM model name | `glm-4.5-air` | No |
| `OPENAI_API_KEY` | OpenAI API key | — | No |
| `ANTHROPIC_API_KEY` | Anthropic API key | — | No |
| `EMBEDDING_PROVIDER` | Embedding source | `local` | No |
| `ENVIRONMENT` | Environment mode | `development` | No |
| `LOG_LEVEL` | Logging level | `INFO` | No |

### Dialogue Configuration

The LangGraph dialogue graph is built automatically by `ChatServiceFactory.create_with_defaults()`. Key components:

- **Intent Detection**: `hybrid` (rule + LLM), `rule_based`, or `llm_based`
- **Slot Schemas**: Defined in `app/services/slot_filling/slot_types.py` per intent
- **Tools**: Registered in `app/services/dialogue/tools.py` (mock handlers for demo)
- **Checkpointer**: `MemorySaver` (dev), swap to `PostgresSaver` for production

```bash
# Intent detection strategy
INTENT_TYPE=hybrid          # hybrid | rule_based | llm_based
CONFIDENCE_THRESHOLD=0.7

# Memory strategy
MEMORY_TYPE=optimized       # optimized | sliding_window | summarization | hybrid
```

### Memory Strategy Configuration

```bash
# Optimized (default) — semantic relevance filtering
MEMORY_TYPE=optimized
MEMORY_MAX_RECENT=3
MEMORY_RELEVANCE_THRESHOLD=0.5
MEMORY_TOKEN_BUDGET=4096

# Sliding Window (fast, low retention)
MEMORY_TYPE=sliding_window
WINDOW_SIZE=10

# Summarization (slower, high retention)
MEMORY_TYPE=summarization
SUMMARY_THRESHOLD=20
SUMMARY_INTERVAL=10

# Hybrid (adaptive)
MEMORY_TYPE=hybrid
HYBRID_THRESHOLD=30
```

### Guardrails Configuration

```bash
# Enable/disable guardrails
GUARDRAILS_ENABLED=true
GUARDRAILS_INPUT_ENABLED=true
GUARDRAILS_OUTPUT_ENABLED=true

# Prompt hardening
GUARDRAILS_HARDENING_ENABLED=true
```

### Retrieval Configuration

```bash
# Vector search settings
VECTOR_WEIGHT=0.7
TOP_K=3

# Reranking
USE_RERANKING=true
RERANKER_TOP_N=5

# GraphRAG fusion (when enabled)
GRAPH_RAG_FUSION_WEIGHT=0.3
```

### LLM Provider Selection

The chatbot auto-selects the first available provider: GLM → OpenAI → Anthropic.

**Option 1: GLM (Zhipu AI) — Recommended for Chinese**
```bash
GLM_API_KEY=your-glm-api-key
GLM_MODEL=glm-4.5-air
```

**Option 2: OpenAI**
```bash
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4-turbo-preview
```

**Option 3: Anthropic**
```bash
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-3-opus-20240229
```

## Deployment

### Docker

```bash
# Build image
docker build -t rag-chatbot:latest .

# Run container
docker run -d \
  -p 8000:8000 \
  -e DATABASE_URL="postgresql+asyncpg://..." \
  -e GLM_API_KEY="..." \
  -e SECRET_KEY="..." \
  rag-chatbot:latest
```

### Docker Compose (Production)

```bash
# Use production profile with monitoring
docker-compose --profile monitoring up -d

# Scale API instances
docker-compose up -d --scale api=5
```

### Kubernetes

```bash
# Create namespace
kubectl create namespace rag-chatbot

# Create secrets
kubectl create secret generic rag-chatbot-secrets \
  --from-literal=database-url="postgresql+asyncpg://..." \
  --from-literal=redis-url="redis://..." \
  --from-literal=glm-api-key="..." \
  --from-literal=secret-key="..." \
  -n rag-chatbot

# Deploy
kubectl apply -f deploy/k8s/

# Check status
kubectl get pods -n rag-chatbot
kubectl get svc -n rag-chatbot
```

## Monitoring

### Health Checks

```bash
# Liveness probe
curl http://localhost:8000/health

# Readiness probe
curl http://localhost:8000/ready
```

### Metrics

```bash
# Prometheus metrics
curl http://localhost:8000/metrics
```

Access Grafana at http://localhost:3001 (admin/admin) when monitoring profile is enabled.

## Development

### Running Tests

```bash
# Run all tests with coverage (80% minimum enforced)
pytest

# Run specific test file
pytest app/tests/unit/services/chat/test_chat_service.py

# Run specific test with verbose output
pytest app/tests/unit/services/chat/test_chat_service.py::test_specific -v

# Run intent tests
pytest app/tests/unit/services/intent/ -v

# Run dialogue tests
pytest app/tests/unit/services/chat/ -v
```

### Code Quality

```bash
# Run all checks (lint + format + type check)
ruff check app/
ruff format app/
mypy app/
```

### Project Structure

```
rag-chatbot/
├── app/
│   ├── api/
│   │   ├── deps/           # Authentication dependencies
│   │   ├── middleware/     # Request ID, rate limiting, error handling
│   │   └── v1/             # API v1 endpoints (chat, docs, auth, feedback, graph)
│   ├── config/             # Pydantic settings
│   ├── core/               # Core utilities
│   ├── middleware/         # FastAPI middleware
│   ├── models/
│   │   ├── database/       # SQLAlchemy ORM models
│   │   ├── enums/          # Enumerations (business intents)
│   │   └── schemas/        # Pydantic schemas
│   ├── repositories/       # Async data access layer
│   ├── services/           # Business logic
│   │   ├── chat/           # Chat orchestration (ChatService + Factory)
│   │   ├── dialogue/       # LangGraph dialogue engine
│   │   │   ├── graph.py    # StateGraph construction + compilation
│   │   │   ├── nodes.py    # 9 dialogue nodes + conditional edges
│   │   │   ├── state.py    # DialogueState TypedDict
│   │   │   └── tools.py    # ToolRegistry + 5 mock handlers
│   │   ├── documents/      # Chunking, ingestion, preprocessing
│   │   ├── embeddings/     # Multi-provider embeddings (local, GLM, OpenAI)
│   │   ├── graph/          # GraphRAG (Neo4j)
│   │   │   ├── community/  # Community detection & global search
│   │   │   ├── extraction/ # Entity/relation extraction
│   │   │   └── retrieval/  # Text-to-Cypher, graph embedding search, fusion
│   │   ├── guardrails/     # Input/output safety checks
│   │   ├── intent/         # Business intent detection (rule + LLM + hybrid)
│   │   ├── llm/            # Multi-provider LLM clients
│   │   ├── memory/         # Memory management strategies
│   │   ├── observability/  # OpenTelemetry tracing
│   │   ├── retrieval/      # Hybrid search, reranking, Qdrant
│   │   └── slot_filling/   # Slot schemas + rule/LLM extraction
│   ├── tests/              # Test suites
│   │   ├── e2e/            # End-to-end tests
│   │   ├── integration/    # Integration tests
│   │   └── unit/           # Unit tests
│   └── main.py             # Application factory
├── deploy/                 # Deployment configs (Docker, K8s)
├── docs/                   # Documentation
├── pyproject.toml          # Dependencies & tool config
└── docker-compose*.yml     # Docker Compose configs
```

## Performance Tuning

### Database

- Use connection pooling (default: 20 connections)
- Enable prepared statements
- Add indexes on frequently queried fields

### Redis

- Use Redis for session storage
- Enable persistence (AOF)
- Configure max memory policy

### Vector Database

- Tune search parameters (`top_k`, `vector_weight`)
- Use quantization for large collections
- Enable HNSW indexing

### LLM

- Use streaming for faster time-to-first-token
- Cache embeddings via `cached_embeddings.py`
- Batch requests when possible

### LangGraph

- Use `MemorySaver` for development, `PostgresSaver` for production
- Tune intent confidence threshold to reduce false positives
- Adjust slot fallback message length threshold (default: 30 chars)

## Security

### Production Checklist

- [ ] Change default `SECRET_KEY`
- [ ] Use strong password policy
- [ ] Enable HTTPS
- [ ] Configure CORS properly (`CORS_ORIGINS`)
- [ ] Set up rate limiting
- [ ] Use read-only database credentials for API
- [ ] Enable audit logging
- [ ] Regular security updates
- [ ] Rotate API keys
- [ ] Enable guardrails (`GUARDRAILS_ENABLED=true`)

### Rate Limiting

```bash
RATE_LIMIT_REQUESTS_PER_MINUTE=60
RATE_LIMIT_BUCKET_SIZE=10
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests first (TDD)
4. Run code quality checks (`ruff check && ruff format && mypy`)
5. Commit changes (`git commit -m 'feat: add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open Pull Request

## Project Statistics

```
Total Python Files: ~200
Total Test Files: ~58
Test Coverage: 80%+ (enforced in CI)
```

## License

MIT License — see LICENSE file for details.

## Support

- Documentation: See `docs/` directory
- GitHub Issues: https://github.com/example/rag-chatbot/issues
- Email: luopengllpp@yahoo.com
