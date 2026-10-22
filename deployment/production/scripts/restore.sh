#!/bin/bash
# Restore trusted plain SQL produced by backup.sh into a NEW database only.
# Usage: restore.sh backup.sql.gz --database recovery_database
# Or: RESTORE_DB=recovery_database restore.sh backup.sql.gz
# Supports COMPOSE_FILE, DB_SERVICE, ENV_FILE, PROJECT_DIRECTORY like backup.sh.
# Never changes the
# application's DATABASE_URL, drops a database, or restores other datastores.
set -euo pipefail
umask 077

usage() {
    echo "Usage: restore.sh trusted.sql.gz --database NEW_DATABASE (or explicit RESTORE_DB)" >&2
    exit 2
}
[[ $# -ge 1 ]] || usage
ARCHIVE="$1"
shift
TARGET_DB="${RESTORE_DB:-}"
if [[ $# -gt 0 ]]; then
    [[ $# -eq 2 && "$1" == "--database" ]] || usage
    TARGET_DB="$2"
fi
# psql treats dbname containing '=' as a connection string; PostgreSQL silently
# truncates long identifiers. Accept neither route to a different target.
if [[ ! "$TARGET_DB" =~ ^[a-zA-Z_][a-zA-Z0-9_]*$ || ${#TARGET_DB} -gt 63 ]]; then
    echo "An explicit new database name is required (ASCII identifier, at most 63 characters)." >&2
    exit 2
fi
[[ -f "$ARCHIVE" && -r "$ARCHIVE" ]] || { echo "Backup is not a readable file." >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
COMPOSE_FILE="${COMPOSE_FILE:-$PROJECT_ROOT/docker-compose.yml}"
DB_SERVICE="${DB_SERVICE:-postgres}"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/rag-restore.XXXXXXXX")"
trap 'rm -rf -- "$WORK_DIR"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Finish decompression BEFORE opening a database transaction. A broken gzip pipe
# can otherwise give psql valid SQL followed by EOF, allowing a partial commit.
gzip -t -- "$ARCHIVE"
gzip -dc -- "$ARCHIVE" > "$WORK_DIR/dump.sql"
[[ -s "$WORK_DIR/dump.sql" ]] || { echo "Backup contains no SQL." >&2; exit 1; }
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

# createdb is the atomic existence check. Even an existing empty database is
# rejected, so there is no check-empty-then-write race and no destructive flag.
if ! "${COMPOSE_CMD[@]}" exec -T "$DB_SERVICE" sh -c '
    user=${POSTGRES_USER:-postgres}
    source_database=${POSTGRES_DB:-$user}
    if [ "$1" = "$source_database" ]; then
        echo "Refusing to restore into the configured source database." >&2
        exit 1
    fi
    exec createdb -U "$user" --template=template0 -- "$1"
' sh "$TARGET_DB"; then
    echo "Restore refused: target must be a new database; no SQL restored." >&2
    exit 1
fi

# -X ignores any psqlrc; ON_ERROR_STOP and a single transaction ensure SQL
# failures roll back all restored objects. Input is already fully validated.
if ! "${COMPOSE_CMD[@]}" exec -T "$DB_SERVICE" sh -c '
    exec psql -X -U "${POSTGRES_USER:-postgres}" --dbname="$1" \
        --set=ON_ERROR_STOP=1 --single-transaction --file=-
' sh "$TARGET_DB" < "$WORK_DIR/dump.sql"; then
    echo "Restore failed. Target database retained for inspection; do not use it until verified. Nothing was dropped." >&2
    exit 1
fi
echo "Restore completed successfully into new database: $TARGET_DB"
echo "Application configuration was not changed. Verify restored data before switching traffic."
