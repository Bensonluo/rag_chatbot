# RAG Chatbot

Production-grade Retrieval-Augmented Generation (RAG) chatbot with intelligent intent detection, memory management, and vector search.

## ✨ Features

### Core RAG Capabilities
- 🔍 **Vector Search**: Qdrant vector database with semantic search
- 🧠 **Smart Embeddings**: BGE-M3 local embeddings (free) + GLM API option
- 📄 **Document Management**: Upload, search, and manage documents (PDF, TXT, MD)
- 🎯 **Hybrid Search**: Combines semantic vector search + BM25 keyword search
- 🔄 **Semantic Chunking**: Multiple strategies (fixed, semantic, recursive)
- 📊 **LLM Reranking**: Advanced reranking for better results

### Chat Features
- 🤖 **Hybrid Intent Detection**: Rule-based + LLM-powered intent classification
- 💬 **Memory Management**: Sliding window, summarization, and hybrid strategies
- 🌊 **Streaming Responses**: Real-time server-sent events
- 🌐 **Multilingual**: Excellent Chinese + English support

### System Features
- 🔐 **JWT Authentication**: Secure user authentication
- 🚦 **Rate Limiting**: Token bucket algorithm
- 📊 **Monitoring**: Prometheus metrics and health checks
- 🐳 **Docker Support**: Containerized deployment
- ☸️ **Kubernetes Ready**: Production manifests included
- 🚀 **Easy Provider Switching**: Switch between embedding providers in seconds

## Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│         API Gateway (FastAPI)        │
│  ┌──────────────────────────────┐  │
│  │   Middleware Stack          │  │
│  │  - Request ID               │  │
│  │  - Rate Limiting            │  │
│  │  - Error Handling           │  │
│  │  - Metrics                  │  │
│  └──────────────────────────────┘  │
│  ┌──────────────────────────────┐  │
│  │      Chat Service            │  │
│  │  - Intent Detection          │  │
│  │  - Memory Management         │  │
│  │  - Vector Retrieval          │  │
│  │  - LLM Generation            │  │
│  └──────────────────────────────┘  │
└─────────────────────────────────────┘
       │           │           │
       ▼           ▼           ▼
┌──────────┐  ┌─────────┐  ┌──────────┐
│PostgreSQL│  │ Redis   │  │ Qdrant   │
└──────────┘  └─────────┘  └──────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 16+
- Redis 7+
- Qdrant 1.7+ (vector database)
- GLM API key (recommended) or OpenAI/Anthropic API key
- Docker & Docker Compose (for deployment)

### Using Docker Compose (Recommended)

```bash
# Clone repository
git clone https://github.com/example/rag-chatbot.git
cd rag-chatbot

# Create environment file
cat > .env << EOF
# LLM Configuration
GLM_API_KEY=your-glm-api-key-here
GLM_MODEL=glm-4.5-air

# Embedding Configuration (local = free, glm = API)
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=bge-m3-v2-zh

# Security
SECRET_KEY=your-secret-key-here
EOF

# Start all services
docker-compose up -d

# Check health
curl http://localhost:8000/health

# View logs
docker-compose logs -f api
```

### Manual Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export DATABASE_URL="postgresql+asyncpg://user:pass@localhost/rag_chatbot"
export REDIS_URL="redis://localhost:6379/0"
export QDRANT_URL="http://localhost:6333"
export OPENAI_API_KEY="sk-..."
export SECRET_KEY="your-secret-key"

# Run database migrations
alembic upgrade head

# Start development server
uvicorn app.main:create_app --factory --reload --host 0.0.0.0 --port 8000
```

## RAG Features

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

**Switch between providers in `.env`:**

```bash
# Local BGE-M3 (FREE, default)
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=bge-m3-v2-zh

# GLM API (pay-per-use)
EMBEDDING_PROVIDER=glm
GLM_API_KEY=your-api-key
```

**That's it!** Same code, different providers.

### Vector Database

**Qdrant Dashboard:**
- URL: http://localhost:6333/dashboard
- View collections, vectors, and perform searches
- Built-in visualization tools

**Direct API access:**
```bash
# Get collection info
curl http://localhost:6333/collections/documents

# Search vectors
curl -X POST http://localhost:6333/collections/documents/points/search \
  -H "Content-Type: application/json" \
  -d '{...}'
```

## API Documentation

Once running, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

### Quick API Example

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
```

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `DATABASE_URL` | PostgreSQL connection string | - | Yes |
| `REDIS_URL` | Redis connection string | - | Yes |
| `QDRANT_URL` | Qdrant vector DB URL | - | Yes |
| `OPENAI_API_KEY` | OpenAI API key | - | Yes* |
| `ANTHROPIC_API_KEY` | Anthropic API key | - | No |
| `GLM_API_KEY` | Zhipu AI GLM API key | - | No |
| `SECRET_KEY` | JWT secret key | - | Yes |
| `ENVIRONMENT` | Environment (development/production) | development | No |
| `LOG_LEVEL` | Logging level | INFO | No |
| `CORS_ORIGINS` | Allowed CORS origins | * | No |

*Required if using OpenAI LLM (at least one LLM provider required: OpenAI, Anthropic, or GLM)

### Memory Strategy Configuration

Choose memory strategy based on your use case:

```python
# Sliding Window (fast, low retention)
MEMORY_TYPE=sliding_window
WINDOW_SIZE=10

# Summarization (slower, high retention)
MEMORY_TYPE=summarization
SUMMARY_THRESHOLD=20
SUMMARY_INTERVAL=10

# Hybrid (adaptive, recommended)
MEMORY_TYPE=hybrid
HYBRID_THRESHOLD=30
```

### Intent Detection Configuration

```python
# Rule-based (fast, less accurate)
INTENT_TYPE=rule_based

# LLM-based (slower, more accurate)
INTENT_TYPE=llm_based

# Hybrid (adaptive, recommended)
INTENT_TYPE=hybrid
CONFIDENCE_THRESHOLD=0.7
```

### Retrieval Configuration

```python
# Vector search settings
VECTOR_WEIGHT=0.7
TOP_K=3

# Reranking
USE_RERANKING=true
RERANKER_TOP_N=5

# Metadata enrichment
USE_METADATA_ENRICHMENT=true
```

### LLM Provider Selection

The chatbot supports multiple LLM providers:

**Option 1: GLM (Zhipu AI) - Recommended for Chinese**
```bash
GLM_API_KEY=your-glm-api-key
GLM_MODEL=glm-4-plus
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

For detailed GLM setup instructions, see [docs/glm_setup.md](docs/glm_setup.md)

## Deployment

### Docker

```bash
# Build image
docker build -t rag-chatbot:latest .

# Run container
docker run -d \
  -p 8000:8000 \
  -e DATABASE_URL="postgresql+asyncpg://..." \
  -e OPENAI_API_KEY="sk-..." \
  -e SECRET_KEY="..." \
  rag-chatbot:latest
```

### Kubernetes

```bash
# Create namespace
kubectl create namespace rag-chatbot

# Create secrets
kubectl create secret generic rag-chatbot-secrets \
  --from-literal=database-url="postgresql+asyncpg://..." \
  --from-literal=redis-url="redis://..." \
  --from-literal=openai-api-key="sk-..." \
  --from-literal=secret-key="..." \
  -n rag-chatbot

# Deploy
kubectl apply -f deploy/k8s/

# Check status
kubectl get pods -n rag-chatbot
kubectl get svc -n rag-chatbot
```

### Docker Compose (Production)

```bash
# Use production profile
docker-compose -f docker-compose.yml --profile monitoring up -d

# Scale API
docker-compose up -d --scale api=5
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

Access Grafana at http://localhost:3001 (admin/admin)

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
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test
pytest tests/unit/services/chat/test_chat_service.py

# Run integration tests
pytest tests/integration/
```

### Code Quality

```bash
# Type checking
mypy app/

# Linting
ruff check app/

# Format code
ruff format app/
```

### Project Structure

```
rag-chatbot/
├── app/
│   ├── api/              # API endpoints
│   │   └── v1/          # API v1 endpoints
│   ├── core/            # Core utilities
│   ├── models/          # Database models
│   ├── repositories/    # Data access layer
│   ├── services/        # Business logic
│   │   ├── chat/       # Chat orchestration
│   │   ├── intent/     # Intent detection
│   │   ├── llm/        # LLM integration
│   │   ├── memory/     # Memory strategies
│   │   └── retrieval/  # Vector search
│   ├── middleware/      # FastAPI middleware
│   └── main.py         # Application factory
├── tests/               # Test suites
├── deploy/              # Deployment configs
├── docs/                # Documentation
└── pyproject.toml      # Dependencies
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

- Tune search parameters (top_k, vector_weight)
- Use quantization for large collections
- Enable HNSW indexing

### LLM

- Use streaming for faster time-to-first-token
- Cache embeddings
- Batch requests when possible

## Security

### Production Checklist

- [ ] Change default `SECRET_KEY`
- [ ] Use strong password policy
- [ ] Enable HTTPS
- [ ] Configure CORS properly
- [ ] Set up rate limiting
- [ ] Use read-only database credentials for API
- [ ] Enable audit logging
- [ ] Regular security updates
- [ ] Rotate API keys
- [ ] Enable request signing

### Rate Limiting

Protect against abuse:

```python
# In middleware or config
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

**Rate limit errors**
```
Solution: Increase limits or whitelist IPs
```

**Memory issues**
```
Solution: Adjust memory strategy window size
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests first (TDD)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open Pull Request

## Project Statistics

```
Total Phases: 9
Total Files: 80+
Total Test Cases: 335+
Total Lines of Code: 10000+
Test Coverage: 80%+
```

## License

MIT License - see LICENSE file for details

## Support

- Documentation: See `docs/` directory
- GitHub Issues: https://github.com/example/rag-chatbot/issues
- Email: luopengllpp@yahoo.com
