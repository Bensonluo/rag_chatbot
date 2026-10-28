#!/usr/bin/env bash
# One-command deploy of the rag_chatbot api to the Tencent demo box.
#
# Why this file exists: an ad-hoc deploy once rsynced a stale local
# docker-compose.override.yml (8001 dev remap) over the server's port
# pin, and the api container then fought the stock agent for :8001.
# The procedure is codified here — same steps, same exclusions, and a
# pre-flight gate that refuses to start when the resolved api port is
# anything but 127.0.0.1:8000.
#
# Usage: bash deploy/deploy-server.sh   (from the repo root)
set -euo pipefail

SERVER=ubuntu@101.43.97.91
SSH_KEY="$HOME/.ssh/ssh_tencent.pem"
REMOTE_DIR=/opt/rag_chatbot
# The api's ONE and only host port on this shared box. 8001/8002/8003
# are reserved for other services / scale-out replicas — never here.
API_PORT=8000
SSH_CMD="ssh -i $SSH_KEY $SERVER"

echo "==> 1/5 rsync source tree (never carries .env, port overrides, caches)"
rsync -az -e "ssh -i $SSH_KEY" \
  --exclude='.git' --exclude='.env' --exclude='.venv' \
  --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='.pytest_cache' --exclude='.ruff_cache' --exclude='.mypy_cache' \
  --exclude='htmlcov' --exclude='coverage.xml' \
  --exclude='.omc' --exclude='.claude' --exclude='.DS_Store' \
  --exclude='node_modules' \
  --exclude='docker-compose.override.yml' \
  --exclude='docker-compose.local.yml' \
  ./ "$SERVER:$REMOTE_DIR/"

echo "==> 2/5 pre-flight: resolved api port must be exactly 127.0.0.1:$API_PORT"
RESOLVED=$($SSH_CMD "cd $REMOTE_DIR && docker compose config --format json" | python3 -c '
import json, sys
ports = json.load(sys.stdin)["services"]["api"]["ports"]
parts = [str(p.get("host_ip", "")) + ":" + str(p["published"]) for p in ports]
print(";".join(parts))
')
if [ "$RESOLVED" != "127.0.0.1:$API_PORT" ]; then
  echo "FATAL: api ports resolve to [$RESOLVED], expected [127.0.0.1:$API_PORT]."
  echo "Fix $REMOTE_DIR/docker-compose.override.yml on the server (pin with !override) and retry."
  exit 1
fi
echo "    ok: $RESOLVED"

echo "==> 3/6 build api and commerce-provider images (pip via Tsinghua mirror)"
$SSH_CMD "cd $REMOTE_DIR && PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple docker compose build api commerce-provider"

echo "==> 4/6 restart commerce provider, then api"
$SSH_CMD "cd $REMOTE_DIR && docker compose up -d commerce-provider api"

echo "==> 5/6 wait for provider health"
PROVIDER_HEALTHY=false
for i in $(seq 1 20); do
  PROVIDER_STATUS=$($SSH_CMD "cd $REMOTE_DIR && docker inspect --format '{{.State.Health.Status}}' \$(docker compose ps -q commerce-provider)" 2>/dev/null || true)
  if [ "$PROVIDER_STATUS" = "healthy" ]; then
    PROVIDER_HEALTHY=true
    echo "    commerce-provider: healthy"
    break
  fi
  sleep 5
done
if [ "$PROVIDER_HEALTHY" != "true" ]; then
  echo "FATAL: commerce-provider did not become healthy; check: $SSH_CMD 'cd $REMOTE_DIR && docker compose logs commerce-provider --tail 50'"
  exit 1
fi

echo "==> 6/6 api health check"
for i in $(seq 1 20); do
  if $SSH_CMD "curl -sf http://localhost:$API_PORT/health" >/dev/null 2>&1; then
    echo "    healthy: http://localhost:$API_PORT/health -> 200"
    echo "DEPLOY OK"
    exit 0
  fi
  sleep 5
done
echo "FATAL: api did not become healthy within 100s; check: $SSH_CMD 'cd $REMOTE_DIR && docker compose logs api --tail 50'"
exit 1
