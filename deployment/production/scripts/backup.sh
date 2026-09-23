#!/bin/bash
# ==========================================
# RAG Chatbot Database Backup Script
# ==========================================
# This script creates automated backups of the PostgreSQL database
# Usage: ./backup.sh [retention_days]

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BACKUP_DIR="$PROJECT_ROOT/backups"
RETENTION_DAYS=${1:-7}
DB_USER="${POSTGRES_USER:-postgres}"
DB_NAME="${POSTGRES_DB:-rag_chatbot}"

# Docker Compose v2 ships as a plugin (`docker compose`); v1 was a
# standalone binary. Support both so the script works on modern hosts.
if docker compose version >/dev/null 2>&1; then
    COMPOSE_CMD=(docker compose)
else
    COMPOSE_CMD=(docker-compose)
fi
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.yml"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  RAG Chatbot Database Backup${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# Generate backup filename with timestamp
BACKUP_FILE="$BACKUP_DIR/db_backup_$(date +%Y%m%d_%H%M%S).sql"

echo "Creating database backup..."
echo "Backup file: $BACKUP_FILE"
echo ""

# Backup database
ERR_FILE="$BACKUP_DIR/pg_dump_error.log"
if "${COMPOSE_CMD[@]}" -f "$COMPOSE_FILE" exec -T postgres pg_dump -U "$DB_USER" "$DB_NAME" > "$BACKUP_FILE" 2>"$ERR_FILE"; then
    # Compress backup
    gzip "$BACKUP_FILE"
    BACKUP_FILE="${BACKUP_FILE}.gz"

    echo -e "${GREEN}✓ Database backup created successfully${NC}"
    echo "Compressed backup: $BACKUP_FILE"
    echo "Size: $(du -h "$BACKUP_FILE" | cut -f1)"
else
    echo -e "${RED}✗ Database backup failed${NC}"
    echo "pg_dump stderr:"
    cat "$ERR_FILE" 2>/dev/null || echo "(no stderr captured)"
    rm -f "$ERR_FILE"
    rm -f "$BACKUP_FILE"
fi

echo ""
echo "Cleaning up old backups (keeping last $RETENTION_DAYS days)..."

# Remove old backups
find "$BACKUP_DIR" -name "db_backup_*.sql.gz" -type f -mtime +$RETENTION_DAYS -delete

OLD_BACKUPS=$(find "$BACKUP_DIR" -name "db_backup_*.sql.gz" -type f | wc -l)
echo "Remaining backups: $OLD_BACKUPS"

echo ""
echo -e "${GREEN}Backup completed successfully!${NC}"
