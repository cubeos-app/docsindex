#!/usr/bin/env bash
# =============================================================================
# CubeOS Docsindex — Pi-side deploy script (executed via SSH from GPU VM)
# =============================================================================
# Usage: GHCR_TOKEN=... GHCR_USER=... bash /tmp/ci-deploy-docsindex.sh
# =============================================================================
set -euo pipefail

COMPOSE_FILE="/cubeos/coreapps/cubeos-docsindex/appconfig/docker-compose.yml"

echo "=== Docsindex Deploy ==="

# --- Pre-flight ---
if [ ! -f "$COMPOSE_FILE" ]; then
  echo "ERROR: docsindex compose file not found"
  exit 1
fi

# --- GHCR login ---
echo "$GHCR_TOKEN" | docker login ghcr.io -u "$GHCR_USER" --password-stdin

# --- Deploy ---
echo "Removing existing stack..."
docker stack rm cubeos-docsindex 2>/dev/null || true
sleep 5

echo "Deploying stack..."
docker stack deploy \
  -c "$COMPOSE_FILE" \
  --resolve-image=never \
  cubeos-docsindex

# --- Health check: wait for replicas ---
sleep 5
for i in $(seq 1 12); do
  REPLICAS=$(docker stack services cubeos-docsindex --format "{{.Replicas}}" 2>/dev/null | head -1 || echo "0/0")
  RUNNING=$(echo "$REPLICAS" | cut -d'/' -f1)
  DESIRED=$(echo "$REPLICAS" | cut -d'/' -f2)
  if [ "$RUNNING" = "$DESIRED" ] && [ "$RUNNING" != "0" ]; then
    echo "docsindex running ($REPLICAS)"
    break
  fi
  [ "$i" -eq 12 ] && echo "docsindex may still be starting ($REPLICAS)"
  sleep 5
done

echo "Deploy complete"
