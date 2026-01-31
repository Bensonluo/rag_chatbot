#!/bin/bash
set -e

echo "🚀 Starting RAG Chatbot container..."

# Function to wait for service
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3

    echo "⏳ Waiting for $service_name at $host:$port..."

    for i in {1..30}; do
        if nc -z "$host" "$port" 2>/dev/null; then
            echo "✅ $service_name is ready!"
            return 0
        fi
        echo "   $service_name not ready yet, retrying... ($i/30)"
        sleep 2
    done

    echo "❌ ERROR: $service_name did not become ready in time"
    exit 1
}

# Wait for PostgreSQL
if [ -n "$DATABASE_HOST" ]; then
    wait_for_service "$DATABASE_HOST" "${DATABASE_PORT:-5432}" "PostgreSQL"
fi

# Wait for Redis
if [ -n "$REDIS_HOST" ]; then
    wait_for_service "$REDIS_HOST" "${REDIS_PORT:-6379}" "Redis"
fi

# Wait for Qdrant (if enabled)
if [ -n "$VECTOR_DB_URL" ]; then
    qdrant_host=$(echo "$VECTOR_DB_URL" | sed -E 's|https?://([^/:]+).*|\1|')
    qdrant_port=$(echo "$VECTOR_DB_URL" | sed -E 's|https?://[^:]+:([0-9]+).*|\1|')
    wait_for_service "$qdrant_host" "${qdrant_port:-6333}" "Qdrant"
fi

# Initialize database on first startup
echo "🔧 Initializing database..."
python scripts/init_db.py

# Start the application
echo "🎯 Starting uvicorn server..."
exec uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000
