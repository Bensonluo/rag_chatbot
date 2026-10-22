#!/bin/bash
# PostgreSQL logical backup only; Qdrant/Neo4j require separate recovery plans.
# Usage: backup.sh [retention_days]
# COMPOSE_FILE, DB_SERVICE, BACKUP_DIR and optional ENV_FILE select the deployment.
# Optional PROJECT_DIRECTORY preserves the caller's Compose project identity.
set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
COMPOSE_FILE="${COMPOSE_FILE:-$PROJECT_ROOT/docker-compose.yml}"
DB_SERVICE="${DB_SERVICE:-postgres}"
BACKUP_DIR="${BACKUP_DIR:-$PROJECT_ROOT/backups}"
RETENTION_DAYS="${1-7}"
if [[ $# -gt 1 || ! "$RETENTION_DAYS" =~ ^[0-9]+$ || ${#RETENTION_DAYS} -gt 6 ]]; then
    echo "Retention must be a nonnegative whole number of days (up to 999999)." >&2
    exit 2
fi

if docker compose version >/dev/null 2>&1; then
    COMPOSE_CMD=(docker compose)
else
    COMPOSE_CMD=(docker-compose)
fi
if [[ -n "${PROJECT_DIRECTORY:-}" ]]; then
    COMPOSE_CMD+=(--project-directory "$PROJECT_DIRECTORY")
fi
if [[ -n "${ENV_FILE:-}" ]]; then
    COMPOSE_CMD+=(--env-file "$ENV_FILE")
fi
COMPOSE_CMD+=(-f "$COMPOSE_FILE")

mkdir -p "$BACKUP_DIR"
WORK_DIR="$(mktemp -d "$BACKUP_DIR/.backup.XXXXXXXX")"
trap 'rm -rf -- "$WORK_DIR"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
BACKUP_FILE="$BACKUP_DIR/db_backup_$(date +%Y%m%d_%H%M%S)_${WORK_DIR##*.}.sql.gz"

echo "Creating database backup..."
# The running container owns these settings; host .env variables are not proof
# of the database actually being backed up. Never print connection credentials.
if ! "${COMPOSE_CMD[@]}" exec -T "$DB_SERVICE" sh -c '
    user=${POSTGRES_USER:-postgres}
    database=${POSTGRES_DB:-$user}
    exec pg_dump -U "$user" -- "$database"
' > "$WORK_DIR/dump.sql" 2> "$WORK_DIR/error.log"; then
    ERROR_LOG="${BACKUP_FILE%.sql.gz}.error.log"
    mv "$WORK_DIR/error.log" "$ERROR_LOG"
    echo "Database backup failed; no backup published and retention skipped."
    echo "Diagnostic log (restricted permissions): $ERROR_LOG"
    exit 1
fi
if [[ ! -s "$WORK_DIR/dump.sql" ]]; then
    echo "Database backup failed: pg_dump returned an empty file." >&2
    exit 1
fi
gzip -c "$WORK_DIR/dump.sql" > "$WORK_DIR/dump.sql.gz"
gzip -t "$WORK_DIR/dump.sql.gz"
# Publish only a complete, validated archive on the same filesystem.
mv "$WORK_DIR/dump.sql.gz" "$BACKUP_FILE"
echo "Database backup created: $BACKUP_FILE"

# Quote the validated value; a malformed retention expression must never become
# additional find predicates. Failed dumps never enter this cleanup path.
# Nested deployment snapshots and other environments own their own retention.
find "$BACKUP_DIR" -maxdepth 1 -type f -name 'db_backup_*.sql.gz' -mtime "+$RETENTION_DAYS" -delete
echo "Backup completed successfully!"
