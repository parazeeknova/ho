#!/usr/bin/env bash
set -uo pipefail

GREEN="\033[32m"
RED="\033[31m"
RESET="\033[0m"
FAILS=0

check() {
    local name="$1"
    if eval "$2" >/dev/null 2>&1; then
        echo -e "  ${GREEN}OK${RESET}   $name"
    else
        echo -e "  ${RED}DOWN${RESET} $name"
        FAILS=$((FAILS + 1))
    fi
}

check_http() {
    local name="$1" url="$2"
    check "$name" "curl -sf --max-time 3 '$url'"
}

check_container() {
    local name="$1" pattern="$2"
    check "$name" "podman ps --filter name='$pattern' --filter status=running | grep -q ."
}

echo "LLM"
check_http "llama-server :8899" "http://localhost:8899/health"

echo ""
echo "Firecrawl"
check_http  "api              :3002" "http://localhost:3002"
check_container "redis"             "firecrawl_redis"
check_container "rabbitmq"          "firecrawl_rabbitmq"
check_container "playwright"        "firecrawl_playwright"
check_container "nuq-postgres"      "firecrawl_nuq-postgres"

echo ""
echo "Metasearch"
check_http "searxng          :8080"  "http://localhost:8080"

echo ""
echo "Agent Memory (pgvector)"
check_container "agent-memory-db" "firecrawl_agent-memory-db"
check "pgvector :5433" "pg_isready -h localhost -p 5433 -U postgres -d agent_memory"

exit $FAILS
