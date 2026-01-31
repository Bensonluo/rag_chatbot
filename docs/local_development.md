# Local Development Setup Guide

This guide explains how to set up and run the RAG chatbot locally for development.

## Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Git
- 8GB RAM minimum (16GB recommended)

## Quick Start

### Option 1: Docker Compose (Recommended)

**Easiest way to get started - everything runs in containers.**

```bash
# Clone repository
git clone https://github.com/yourusername/rag-chatbot.git
cd rag-chatbot

# Configure environment
cp .env.example .env
nano .env  # Edit as needed

# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api
```

**Access services:**
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Qdrant Dashboard: http://localhost:6333/dashboard
- Grafana (if enabled): http://localhost:3001

### Option 2: Local Python Development

**For active development with hot reload.**

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install sentence-transformers (for local embeddings)
pip install sentence-transformers

# Install pypdf (for PDF support)
pip install pypdf

# Start services with Docker
docker-compose up -d postgres redis qdrant

# Run migrations
alembic upgrade head

# Start API server (with auto-reload)
uvicorn app.main:create_app --factory --reload --host 0.0.0.0 --port 8000
```

## Configuration

### .env File Setup

Create a `.env` file in the project root:

```bash
# Environment
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# Database (Docker)
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/ragchatbot

# Redis (Docker)
REDIS_URL=redis://localhost:6379/0

# Vector DB (Docker)
VECTOR_DB_URL=http://localhost:6333
VECTOR_COLLECTION_NAME=documents

# LLM Configuration
GLM_API_KEY=your-glm-api-key-here
GLM_MODEL=glm-4.5-air

# Embedding Configuration
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=bge-m3-v2-zh
EMBEDDING_CACHE_TTL=604800
EMBEDDING_DEVICE=cpu

# Security
SECRET_KEY=development-secret-key
JWT_ALGORITHM=HS256
```

### Switching Embedding Providers

**Use Local BGE-M3 (Free):**
```bash
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=bge-m3-v2-zh
```

**Use GLM Embeddings API:**
```bash
EMBEDDING_PROVIDER=glm
GLM_API_KEY=your-glm-api-key
```

**Use OpenAI Embeddings:**
```bash
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=your-openai-api-key
```

## Development Workflow

### 1. Start Development Environment

```bash
# Start all services
docker-compose up -d

# Run database migrations
docker-compose exec api alembic upgrade head

# Check health
curl http://localhost:8000/health
```

### 2. Upload Documents

```bash
# Upload a text document
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Test Document",
    "content": "This is a test document for development."
  }'
```

### 3. Search Documents

```bash
# Search for relevant content
curl -X POST http://localhost:8000/api/v1/documents/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "test document",
    "top_k": 5
  }'
```

### 4. Test Chat Endpoint

```bash
# Send a chat message
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is RAG?",
    "session_id": 1
  }'
```

## Testing

### Run Unit Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test
pytest tests/unit/services/embeddings/test_local_embeddings.py
```

### Run Integration Tests

```bash
# Start services first
docker-compose up -d

# Run integration tests
pytest tests/integration/
```

### Test Embeddings

```bash
# Test embedding service
python test_embeddings.py
```

## Development Tools

### API Documentation

**Swagger UI:**
- URL: http://localhost:8000/docs
- Interactive API documentation
- Try out endpoints directly

**ReDoc:**
- URL: http://localhost:8000/redoc
- Alternative documentation format

### Qdrant Dashboard

**Web UI:**
- URL: http://localhost:6333/dashboard
- View collections
- Search vectors
- Manage data

### Database Access

**Connect to PostgreSQL:**
```bash
docker exec -it rag-chatbot-db psql -U postgres -d rag_chatbot
```

**Common queries:**
```sql
-- List all tables
\dt

-- View documents
SELECT * FROM documents LIMIT 10;

-- View messages
SELECT * FROM messages ORDER BY created_at DESC LIMIT 10;

-- Count chunks in Qdrant (via API)
curl http://localhost:6333/collections/documents
```

### Redis CLI

```bash
# Connect to Redis
docker exec -it rag-chatbot-redis redis-cli

# View all keys
KEYS *

# View embedding cache
KEYS embedding:*

# Clear cache
FLUSHDB
```

## Debugging

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api
docker-compose logs -f qdrant

# Last 100 lines
docker-compose logs --tail=100 api
```

### Common Issues

**Port already in use:**
```bash
# Check what's using the port
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or change port in .env
API_PORT=8001
```

**Out of memory:**
```bash
# Check Docker stats
docker stats

# Increase Docker memory limit (Docker Desktop)
# Settings → Resources → Memory → 8GB+
```

**Embedding model not found:**
```bash
# Clear cache and restart
docker-compose down
docker volume rm rag-chatbot-qdrant_data
docker-compose up -d

# Model will download on first use
```

### Hot Reload

**For Python development:**

```bash
# Run with auto-reload
uvicorn app.main:create_app --factory --reload --host 0.0.0.0 --port 8000
```

**For frontend development:**

```bash
# In a separate terminal
cd frontend
npm install
npm run dev
```

## Performance Tuning

### Local Embedding Optimization

```bash
# Use smaller model for faster development
EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2

# Increase cache TTL
EMBEDDING_CACHE_TTL=864000  # 10 days

# Disable cache during development
EMBEDDING_PROVIDER=local
# (set use_cache=False in code)
```

### Database Optimization

```bash
# Connection pooling
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=10

# Query optimization
# Create indexes in migrations
```

## Useful Scripts

### Reset Everything

```bash
#!/bin/bash
# reset-dev.sh

echo "Stopping services..."
docker-compose down

echo "Removing volumes..."
docker volume rm rag-chatbot_postgres_data
docker volume rm rag-chatbot_redis_data
docker volume rm rag-chatbot_qdrant_data

echo "Starting fresh..."
docker-compose up -d

echo "Running migrations..."
sleep 5
docker-compose exec api alembic upgrade head

echo "Done!"
```

### Backup Development Data

```bash
#!/bin/bash
# backup-dev.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="./backups"

mkdir -p $BACKUP_DIR

# Backup PostgreSQL
docker exec rag-chatbot-db pg_dump -U postgres rag_chatbot > $BACKUP_DIR/postgres_$DATE.sql

# Backup Qdrant
docker run --rm \
  --volumes-from rag-chatbot-qdrant \
  -v $BACKUP_DIR:/backup \
  ubuntu tar czf /backup/qdrant_$DATE.tar.gz /qdrant/storage

echo "Backup completed: $BACKUP_DIR"
```

### Monitor Resources

```bash
#!/bin/bash
# monitor.sh

while true; do
  clear
  echo "=== Docker Stats ==="
  docker stats --no-stream

  echo -e "\n=== Disk Usage ==="
  df -h

  echo -e "\n=== Memory ==="
  free -h

  sleep 5
done
```

## IDE Setup

### VS Code

**Recommended extensions:**
- Python
- Pylance
- Docker
- REST Client
- GitLens

**Settings (.vscode/settings.json):**
```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true
}
```

### PyCharm

1. Open project
2. Configure Python interpreter (use venv)
3. Mark directories as Sources Root
4. Enable auto-reload for development server

## Next Steps

1. ✅ Set up local development environment
2. ✅ Upload test documents
3. ✅ Test search functionality
4. ✅ Integrate with chat endpoint
5. ✅ Write tests for new features
6. ✅ Deploy to Aliyun

## Troubleshooting

### Can't connect to services

```bash
# Check if services are running
docker-compose ps

# Restart services
docker-compose restart

# Check network
docker network inspect rag-network
```

### Tests failing

```bash
# Check test configuration
cat pytest.ini

# Run with verbose output
pytest -v

# Check test logs
cat tests/test.log
```

### Import errors

```bash
# Install in development mode
pip install -e .

# Check Python path
python -c "import sys; print(sys.path)"
```

## Resources

- **Project README:** [README.md](../README.md)
- **API Documentation:** http://localhost:8000/docs
- **Qdrant Docs:** https://qdrant.tech/documentation/
- **GLM Docs:** https://open.bigmodel.cn/dev/api

---

**Happy coding!** 🚀
