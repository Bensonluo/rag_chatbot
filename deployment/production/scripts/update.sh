#!/bin/bash
# Update the selected Compose application. This is not a zero-downtime rollout.
# Usage: ./update.sh [prod|dev] (default: prod)
# A failed update preserves diagnostics and never tears down dependencies.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Reuse exactly the same stack selection, backup and readiness behavior as deploy.
source "$SCRIPT_DIR/deploy.sh"

main() {
    trap record_failure ERR
    configure_deployment "${1:-prod}"
    echo "This replaces the application container and may interrupt requests. No automatic rollback is provided."
    read -r -p "Continue with the update? (y/n) " reply
    if [[ ! "$reply" =~ ^[Yy]$ ]]; then
        echo "Update cancelled."
        return 0
    fi

    save_deployment_state
    backup_current_database update
    PHASE="pull-code"
    git -C "$PROJECT_ROOT" pull --ff-only origin main
    PHASE="validate-updated-configuration"
    compose config --quiet
    PHASE="build"
    compose build "$APP_SERVICE"
    PHASE="replace-application"
    compose up -d --no-deps "$APP_SERVICE"
    wait_for_readiness
    finish_deployment
}

main "$@"
