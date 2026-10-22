#!/bin/bash
# Deploy a Compose stack. Service replacement can interrupt requests.
# Usage: ./deploy.sh [dev|prod] (default: dev)
# An initial prod deployment without a DB container requires ALLOW_INITIAL_DEPLOY=1.
# COMPOSE_FILE, ENV_FILE, DB_SERVICE and APP_SERVICE may select an explicit stack.
# PROJECT_DIRECTORY is forwarded to backups to preserve Compose project identity.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
STATE_DIR=""
PHASE="configuration"

configure_deployment() {
    DEPLOY_ENV="$1"
    case "$DEPLOY_ENV" in
        dev)
            COMPOSE_FILE="${COMPOSE_FILE:-$PROJECT_ROOT/docker-compose.yml}"
            DB_SERVICE="${DB_SERVICE:-postgres}"
            APP_SERVICE="${APP_SERVICE:-api}"
            ;;
        prod)
            COMPOSE_FILE="${COMPOSE_FILE:-$PROJECT_ROOT/docker-compose.prod.yml}"
            DB_SERVICE="${DB_SERVICE:-db}"
            APP_SERVICE="${APP_SERVICE:-app}"
            ;;
        *) echo "Invalid environment: use dev or prod." >&2; return 2 ;;
    esac
    ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/.env}"
    PROJECT_DIRECTORY="${PROJECT_DIRECTORY:-$PROJECT_ROOT}"
    READINESS_ATTEMPTS="${READINESS_ATTEMPTS:-60}"
    READINESS_INTERVAL="${READINESS_INTERVAL:-2}"
    if ! [[ "$READINESS_ATTEMPTS" =~ ^[1-9][0-9]*$ && "$READINESS_INTERVAL" =~ ^[0-9]+$ ]]; then
        echo "Readiness attempts must be positive and interval nonnegative integers." >&2
        return 2
    fi
    [[ -f "$COMPOSE_FILE" && -f "$ENV_FILE" && -d "$PROJECT_DIRECTORY" ]] || {
        echo "Compose configuration or environment file missing; configure them before deployment." >&2
        return 1
    }
    # Resolve caller-relative selectors before switching to the checkout directory.
    COMPOSE_FILE="$(cd "$(dirname "$COMPOSE_FILE")" && pwd)/$(basename "$COMPOSE_FILE")"
    ENV_FILE="$(cd "$(dirname "$ENV_FILE")" && pwd)/$(basename "$ENV_FILE")"
    PROJECT_DIRECTORY="$(cd "$PROJECT_DIRECTORY" && pwd)"
    command -v docker >/dev/null
    if docker compose version >/dev/null 2>&1; then
        COMPOSE_CMD=(docker compose)
    elif command -v docker-compose >/dev/null 2>&1; then
        COMPOSE_CMD=(docker-compose)
    else
        echo "Docker Compose is required." >&2
        return 1
    fi
    COMPOSE_ARGS=(--project-directory "$PROJECT_DIRECTORY" --env-file "$ENV_FILE" -f "$COMPOSE_FILE")
    export COMPOSE_FILE DB_SERVICE ENV_FILE PROJECT_DIRECTORY
    cd "$PROJECT_ROOT"
    # Do not print the resolved configuration: it may contain credentials.
    compose config --quiet
}

compose() { "${COMPOSE_CMD[@]}" "${COMPOSE_ARGS[@]}" "$@"; }

record_failure() {
    local code=$?
    trap - ERR
    if [[ -n "$STATE_DIR" ]]; then
        printf 'failed\nphase=%s\nexit=%s\n' "$PHASE" "$code" > "$STATE_DIR/result.txt"
        echo "Deployment failed during $PHASE. Protected recovery state: $STATE_DIR" >&2
    else
        echo "Deployment configuration failed." >&2
    fi
    echo "No automatic rollback or teardown performed. Inspect the failure before retrying." >&2
    exit "$code"
}

save_deployment_state() {
    PHASE="save-state"
    umask 077
    mkdir -p "$PROJECT_ROOT/backups"
    STATE_DIR=$(mktemp -d "$PROJECT_ROOT/backups/deployment_${DEPLOY_ENV}_$(date +%Y%m%d_%H%M%S)_XXXXXX")
    cp "$ENV_FILE" "$STATE_DIR/environment.env"
    cp "$COMPOSE_FILE" "$STATE_DIR/compose.yml"
    compose config > "$STATE_DIR/resolved-compose.yml"
    compose images -q > "$STATE_DIR/image-ids.txt"
    compose ps -a -q > "$STATE_DIR/container-ids.txt"
    # Inspect records the prior containers' actual environment and immutable image IDs.
    # These files contain credentials; the directory and all files are private.
    local container
    while IFS= read -r container; do
        [[ -n "$container" ]] || continue
        docker inspect "$container" > "$STATE_DIR/container_${container}.json"
    done < "$STATE_DIR/container-ids.txt"
    git -C "$PROJECT_ROOT" rev-parse HEAD > "$STATE_DIR/git-revision.txt" 2>/dev/null ||
        printf 'unavailable\n' > "$STATE_DIR/git-revision.txt"
    printf 'environment=%s\ncompose_file=%s\ndb_service=%s\napp_service=%s\n' \
        "$DEPLOY_ENV" "$COMPOSE_FILE" "$DB_SERVICE" "$APP_SERVICE" > "$STATE_DIR/deployment-context.txt"
    printf 'pending\n' > "$STATE_DIR/result.txt"
    echo "Protected deployment state saved: $STATE_DIR"
}

verify_backup_target() {
    PHASE="verify-backup-target"
    local db_container="$1" metadata database port aliases addresses app_containers app_container match
    if ! metadata=$(compose exec -T "$DB_SERVICE" sh -c '
        user=${POSTGRES_USER:-postgres}
        database=${POSTGRES_DB:-$user}
        printf "%s\n" "$database"
        exec psql -X -A -t -U "$user" -d "$database" -c "SHOW port"
    ' 2> "$STATE_DIR/database-target.error.log"); then
        echo "Cannot verify the selected database target; backup/update stopped." >&2
        return 1
    fi
    database="${metadata%%$'\n'*}"
    port="${metadata#*$'\n'}"
    [[ -n "$database" && "$port" =~ ^[0-9]+$ ]] || return 1
    aliases=$(docker inspect --format '{{println .Name}}{{range .NetworkSettings.Networks}}{{range .Aliases}}{{println .}}{{end}}{{end}}' "$db_container")
    addresses=$(docker inspect --format '{{range .NetworkSettings.Networks}}{{println .IPAddress}}{{println .GlobalIPv6Address}}{{end}}' "$db_container")
    aliases="$DB_SERVICE"$'\n'"$aliases"
    app_containers=$(compose ps -a -q "$APP_SERVICE")
    if [[ -z "$app_containers" ]]; then
        printf 'local-database-only: no application container to verify\n' > "$STATE_DIR/backup-target.txt"
        echo "No application container: only the selected local database backup can be verified."
        return
    fi
    while IFS= read -r app_container; do
        if [[ "$(docker inspect --format '{{.State.Running}}' "$app_container")" != true ]]; then
            echo "Cannot read effective settings from a stopped application; backup/update stopped." >&2
            return 1
        fi
        # Read effective settings inside each running application. Only a result
        # marker leaves the container: DSNs and settings errors must not be logged.
        if ! match=$(docker exec -i "$app_container" python -c '
# BACKUP_TARGET_CHECK
import contextlib
import io
import ipaddress
import socket
import sys
from urllib.parse import parse_qs, unquote, urlsplit

try:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from app.config.settings import get_settings
        settings = get_settings()
    database, port, aliases, addresses = sys.argv[1:]
    aliases = {value.strip().lstrip("/").lower() for value in aliases.splitlines() if value.strip()}
    addresses = {ipaddress.ip_address(value.strip()) for value in addresses.splitlines() if value.strip()}

    def matches(value):
        url = urlsplit(value)
        if not (url.scheme == "postgresql" or url.scheme.startswith("postgresql+")):
            return False
        target_port = url.port if url.port is not None else 5432
        if url.hostname not in aliases or unquote(url.path[1:]) != database or target_port != int(port):
            return False
        # Libpq/driver query parameters can override the authority target.
        if {key.lower() for key in parse_qs(url.query, keep_blank_values=True)} & {"host", "hostaddr", "port", "dbname", "database", "service"}:
            return False
        resolved = {ipaddress.ip_address(item[4][0]) for item in socket.getaddrinfo(url.hostname, target_port, type=socket.SOCK_STREAM)}
        return bool(resolved) and resolved <= addresses

    valid = matches(settings.DATABASE_URL)
    if settings.DIALOGUE_CHECKPOINTER_TYPE == "postgres":
        valid = valid and matches(settings.DIALOGUE_CHECKPOINTER_DB_URL or settings.DATABASE_URL)
    print("match" if valid else "mismatch")
    sys.exit(0 if valid else 1)
except Exception:
    print("unverified")
    sys.exit(1)
' "$database" "$port" "$aliases" "$addresses" 2>/dev/null) || [[ "$match" != match ]]; then
            echo "Application/database targets could not be matched. External or separate checkpoint databases require an independent consistent backup plan; deployment stopped." >&2
            return 1
        fi
    done <<< "$app_containers"
    printf 'matched-running-application-database-and-checkpoint\n' > "$STATE_DIR/backup-target.txt"
}

backup_current_database() {
    PHASE="database-backup"
    local existing running
    existing=$(compose ps -a -q "$DB_SERVICE")
    if [[ -z "$existing" ]]; then
        if [[ "${1:-deploy}" == deploy && ( "$DEPLOY_ENV" == dev || "${ALLOW_INITIAL_DEPLOY:-0}" == 1 ) ]]; then
            printf 'No database container; initial deployment explicitly selected.\n' > "$STATE_DIR/database-backup-skipped.txt"
            echo "No database backup: initial deployment. Existing volumes have not been backed up."
            return
        fi
        echo "No database container found. Updates require a backup; for first prod deployment set ALLOW_INITIAL_DEPLOY=1." >&2
        return 1
    fi
    if [[ "$existing" == *$'\n'* ]]; then
        echo "Multiple database containers found; select one database service before deployment." >&2
        return 1
    fi
    running=$(docker inspect --format '{{.State.Running}}' "$existing")
    if [[ "$running" != true ]]; then
        echo "The database container is stopped. Start it and obtain a backup before deployment." >&2
        return 1
    fi
    verify_backup_target "$existing"
    PHASE="database-backup"
    BACKUP_DIR="$STATE_DIR/database" "$SCRIPT_DIR/backup.sh"
}

wait_for_readiness() {
    PHASE="readiness"
    local attempt
    for ((attempt=1; attempt<=READINESS_ATTEMPTS; attempt++)); do
        # Use the selected DB container's own credentials, not host defaults.
        # Redis credentials remain inside the selected application container.
        # /ready confirms chat initialization; the separate probes test DB/Redis reachability.
        if compose exec -T "$DB_SERVICE" sh -c \
            'pg_isready -U "${POSTGRES_USER:-postgres}" -d "${POSTGRES_DB:-${POSTGRES_USER:-postgres}}"' >/dev/null 2>&1 &&
            compose exec -T "$APP_SERVICE" python -c \
            'import os; from redis import Redis; client = Redis.from_url(os.environ["REDIS_URL"], socket_connect_timeout=3, socket_timeout=3); assert client.ping()' >/dev/null 2>&1 &&
            compose exec -T "$APP_SERVICE" curl --fail --silent --show-error --max-time 5 \
            http://localhost:8000/ready >/dev/null 2>&1; then
            echo "PostgreSQL/Redis probes passed; application chat initialization is ready."
            return 0
        fi
        if ((attempt < READINESS_ATTEMPTS)); then sleep "$READINESS_INTERVAL"; fi
    done
    echo "Readiness timed out for the selected stack. Deployment is not successful." >&2
    return 1
}

finish_deployment() {
    printf 'successful\n' > "$STATE_DIR/result.txt"
    echo "Deployment completed after readiness checks. Service replacement may have interrupted requests."
    echo "Recovery state retained at $STATE_DIR. Rollback requires an explicit image/schema recovery plan."
}

main() {
    trap record_failure ERR
    configure_deployment "${1:-dev}"
    save_deployment_state
    backup_current_database deploy
    PHASE="build"
    compose build --no-cache
    PHASE="start-services"
    # Compose pulls missing dependency images; do not pull the locally built app
    # from a registry or silently upgrade all existing database/cache images.
    compose up -d
    wait_for_readiness
    finish_deployment
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then main "$@"; fi
