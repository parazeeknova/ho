#!/usr/bin/env bash
set -euo pipefail

RED="\033[31m"
GREEN="\033[32m"
YELLOW="\033[33m"
BOLD="\033[1m"
RESET="\033[0m"

DOCKER_COMPOSE="docker compose -f docker-compose.yaml"

cleanup() {
    echo "Shutting down..."
    $DOCKER_COMPOSE down 2>/dev/null || true
    podman rm -f firecrawl_rabbitmq_1 2>/dev/null || true
    killall llama-server 2>/dev/null || true
}

status_line() {
    local name="$1" url="$2"
    if curl -sf --max-time 3 "$url" >/dev/null 2>&1; then
        echo -e "  ${GREEN}RUNNING${RESET} $name"
        return 0
    else
        echo -e "  ${RED}DOWN${RESET}   $name"
        return 1
    fi
}

status_tcp() {
    local name="$1" host="$2" port="$3"
    if curl -sf --max-time 3 "http://$host:$port" >/dev/null 2>&1; then
        echo -e "  ${GREEN}RUNNING${RESET} $name"
        return 0
    else
        echo -e "  ${RED}DOWN${RESET}   $name"
        return 1
    fi
}

echo -e "${BOLD}=== ho dev ===${RESET}"
echo ""

# 1. Cleanup
echo "Cleaning up..."
cleanup
sleep 1

# 2. llama-server
echo "Starting llama-server..."
./scripts/serve.sh &
LLAMA_PID=$!
sleep 2

# 3. Firecrawl infra
echo "Starting firecrawl infra..."
$DOCKER_COMPOSE up -d redis playwright-service nuq-postgres 2>&1 | grep -v "^$" || true

# 4. Rabbitmq (podman)
echo "Starting rabbitmq..."
podman rm -f firecrawl_rabbitmq_1 2>/dev/null || true
podman run -d --name firecrawl_rabbitmq_1 \
    --network firecrawl_default \
    --network-alias rabbitmq \
    --entrypoint /bin/bash \
    rabbitmq:3-management \
    -c "rm -f /var/lib/rabbitmq/.erlang.cookie; exec docker-entrypoint.sh rabbitmq-server" >/dev/null 2>&1
sleep 3

# 5. Firecrawl api
echo "Starting firecrawl api (this takes ~20s)..."
$DOCKER_COMPOSE up -d api 2>&1 | grep -v "^$" || true

# Wait for api to be ready
for i in $(seq 1 15); do
    if curl -sf --max-time 2 http://localhost:3002 >/dev/null 2>&1; then
        break
    fi
    sleep 2
done

# 6. Health check
echo ""
echo -e "${BOLD}Status${RESET}"
echo "-------"
status_line "llama-server  :8899" "http://localhost:8899/health"
status_tcp  "firecrawl api :3002" localhost 3002

# Check containers
for c in redis nuq-postgres playwright-service rabbitmq; do
    if podman ps --format '{{.Names}}' --filter "name=firecrawl_${c}_1" 2>/dev/null | grep -q .; then
        echo -e "  ${GREEN}RUNNING${RESET} $c"
    else
        echo -e "  ${RED}DOWN${RESET}   $c"
    fi
done

echo ""
echo "Ready. llama-server PID: $LLAMA_PID"
echo "Press Ctrl+C to stop all services."
echo ""

# Wait for background process
wait $LLAMA_PID
