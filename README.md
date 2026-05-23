# RAG Chatbot

Production-grade Retrieval-Augmented Generation (RAG) chatbot with intelligent intent detection, memory management, vector search, GraphRAG, and comprehensive safety guardrails.

## Features

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
- **Hybrid Intent Detection**: Rule-based + LLM-powered intent classification with confidence scores
- **Slot Filling**: Extracts structured entities from queries for precise search filtering
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

```
User Query
    |
    v
+------------------+------------------+------------------+
| Input Guardrail  | Intent Detection |  Slot Filling    |
| (Safety Check)   | (Query Type)     | (Entity Extract) |
+------------------+------------------+------------------+
    |                       |                  |
    v                       v                  v
+-------------------------------------------------------+
|              Memory Strategy (Context)                |
|     optimized / sliding_window / summarization        |
+-------------------------------------------------------+
    |
    v
+-------------------------------------------------------+
|              Document Retrieval (Knowledge)           |
|  +----------------+  +-------------------------------+|
|  | Vector Search  |  | GraphRAG (optional)           ||
|  | - Semantic     |  | - Text-to-Cypher              ||
|  | - BM25 + RRF   |  | - Graph Embedding Search      ||
|  +----------------+  | - Community/Global Search     ||
|         |            +-------------------------------+|
|         v                        |                    |
|  +----------------+             |                    |
|  | Reranking      | <----------+                    |
|  +----------------+                                  |
+-------------------------------------------------------+
    |
    v
+-------------------------------------------------------+
|              LLM Generation (Streaming)               |
|         GLM (default) / OpenAI / Anthropic            |
+-------------------------------------------------------+
    |
    v
+------------------+------------------+
| Output Guardrail |  Feedback Store  |
| (PII Redaction)  |  (Rating/Stats)  |
+------------------+------------------+
    |
    v
  Response
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
git clone https://github.com/example/rag-chatbot.git
cd rag-chatbot

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

## RAG Pipeline

### Document Management

**Upload documents:**
```bash
# Upload text document
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -H "Content-Type: application/json" \
  -d '{
    "title": "My Document",
    "content": "Full document text..."
  }'

# Upload file (PDF, TXT, MD)
curl -X POST http://localhost:8000/api/v1/documents/upload/file \
  -F "file=@document.pdf"
```

**Search documents:**
```bash
curl -X POST http://localhost:8000/api/v1/documents/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "search query",
    "top_k": 5
  }'
```

### Embedding Provider Switching

Switch between providers in `.env`:

```bash
# Local BGE-M3 (FREE, default)
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=bge-m3-v2-zh

# GLM API (pay-per-use)
EMBEDDING_PROVIDER=glm
GLM_API_KEY=your-api-key

# OpenAI (pay-per-use)
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

### GraphRAG (Optional)

Enable knowledge graph capabilities:

```bash
# .env
GRAPH_RAG_ENABLED=true
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
GRAPH_RAG_FUSION_WEIGHT=0.3
```

GraphRAG endpoints (available when enabled):
```bash
# Query graph
curl -X POST http://localhost:8000/api/v1/graph/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What projects is Alice working on?"}'

# Import structured data
curl -X POST http://localhost:8000/api/v1/graph/import/structured \
  -H "Content-Type: application/json" \
  -d '{
    "nodes": [{"id": "Alice", "label": "Person", "properties": {"role": "Engineer"}}],
    "relationships": [{"source": "Alice", "target": "ProjectX", "type": "WORKS_ON"}]
  }'

# Detect communities
curl -X POST http://localhost:8000/api/v1/graph/communities/detect

# Graph health check
curl http://localhost:8000/api/v1/graph/health
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

# 3. Send chat message
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is Python?",
    "session_id": 1
  }'

# 4. Stream response
curl -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Tell me a joke",
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

# 6. View feedback stats
curl http://localhost:8000/api/v1/feedback/stats \
  -H "Authorization: Bearer $TOKEN"
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

### Intent Detection Configuration

```bash
# Hybrid (default) — adaptive rule + LLM
INTENT_TYPE=hybrid
CONFIDENCE_THRESHOLD=0.7

# Rule-based (fast, less accurate)
INTENT_TYPE=rule_based

# LLM-based (slower, more accurate)
INTENT_TYPE=llm_based
```

### Slot Filling Configuration

```bash
# Hybrid (default)
SLOT_FILLING_TYPE=hybrid

# Rule-based only
SLOT_FILLING_TYPE=rule_based

# LLM-based only
SLOT_FILLING_TYPE=llm_based
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

### Logging

Logs are structured JSON:

```json
{
  "timestamp": "2024-01-29T10:00:00Z",
  "level": "info",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Request processed",
  "path": "/api/v1/chat",
  "method": "POST",
  "status_code": 200,
  "duration_ms": 150
}
```

## Development

### Running Tests

```bash
# Run all tests with coverage (80% minimum enforced)
pytest

# Run specific test file
pytest app/tests/unit/services/chat/test_chat_service.py

# Run specific test with verbose output
pytest app/tests/unit/services/chat/test_chat_service.py::test_specific -v

# Run integration tests
pytest app/tests/integration/
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
│   │   ├── enums/          # Enumerations
│   │   └── schemas/        # Pydantic schemas
│   ├── repositories/       # Async data access layer
│   ├── services/           # Business logic
│   │   ├── chat/           # Chat orchestration (pipeline)
│   │   ├── documents/      # Chunking, ingestion, preprocessing
│   │   ├── embeddings/     # Multi-provider embeddings (local, GLM, OpenAI)
│   │   ├── graph/          # GraphRAG (Neo4j)
│   │   │   ├── community/  # Community detection & global search
│   │   │   ├── extraction/ # Entity/relation extraction
│   │   │   └── retrieval/  # Text-to-Cypher, graph embedding search, fusion
│   │   ├── guardrails/     # Input/output safety checks
│   │   ├── intent/         # Intent detection strategies
│   │   ├── llm/            # Multi-provider LLM clients
│   │   ├── memory/         # Memory management strategies
│   │   ├── observability/  # OpenTelemetry tracing
│   │   ├── retrieval/      # Hybrid search, reranking, Qdrant
│   │   └── slot_filling/   # Entity extraction from queries
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

### GraphRAG

- Start with `GRAPH_RAG_ENABLED=false` for baseline performance
- Enable after document ingestion for advanced relationship queries
- Tune `GRAPH_RAG_FUSION_WEIGHT` to balance vector vs. graph results

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

### Input Validation

All inputs validated with Pydantic:

```python
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)
    session_id: int = Field(..., gt=0)
```

## Troubleshooting

### Common Issues

**Database connection failed**
```
Solution: Check DATABASE_URL format
postgresql+asyncpg://user:pass@host:port/dbname
```

**Qdrant connection timeout**
```
Solution: Verify Qdrant is running
curl http://localhost:6333/health
```

**Neo4j connection failed (GraphRAG)**
```
Solution: Verify Neo4j is running and GRAPH_RAG_ENABLED matches setup
curl http://localhost:8000/api/v1/graph/health
```

**Rate limit errors**
```
Solution: Increase limits or use Redis for distributed rate limiting
```

**Memory issues**
```
Solution: Adjust memory strategy or token budget
MEMORY_TYPE=sliding_window
WINDOW_SIZE=5
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
