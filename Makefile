.PHONY: check check-types test serve run match health fc-up fc-down fc-logs dev dev-down start overnight clean-volumes start-daemon stop-daemon graph graph-stop graph-shell graph-reset

check:
	uv run ruff format . && uv run ruff check . --fix

check-types:
	uv run mypy . --ignore-missing-imports --exclude 'refs/' --explicit-package-bases

test:
	uv run python -m pytest . -v --ignore=refs

serve:
	uv run python scripts/serve.py

run: health
	@mkdir -p logs
	uv run python -m src.radar.orchestrator 2>&1 | tee logs/run.log

match: health
	uv run python -m src.radar.orchestrator

overnight: health
	OVERNIGHT_LOOP=true uv run python -m src.radar.orchestrator

fc-up:
	@docker compose -f docker-compose.yaml up -d redis playwright-service nuq-postgres searxng neo4j; \
	podman rm -f firecrawl_rabbitmq_1 2>/dev/null; \
	podman run -d --name firecrawl_rabbitmq_1 \
		--network firecrawl_default \
		--network-alias rabbitmq \
		--entrypoint /bin/bash \
		rabbitmq:3-management \
		-c "rm -f /var/lib/rabbitmq/.erlang.cookie; exec docker-entrypoint.sh rabbitmq-server"; \
	echo "Waiting for rabbitmq..."; sleep 5; \
	docker compose -f docker-compose.yaml up -d api

fc-down:
	docker compose -f docker-compose.yaml down
	podman rm -f firecrawl_rabbitmq_1 2>/dev/null; true

fc-logs:
	docker compose -f docker-compose.yaml logs -f

dev:
	uv run python scripts/dev.py

dev-down:
	docker compose -f docker-compose.yaml down 2>/dev/null; \
	podman rm -f firecrawl_rabbitmq_1 2>/dev/null; \
	killall llama-server 2>/dev/null; \
	true

health:
	uv run python scripts/health.py

clean-volumes:
	docker compose -f docker-compose.yaml down -v 2>/dev/null; \
	podman rm -f firecrawl_rabbitmq_1 2>/dev/null; \
	rm -rf storage/ 2>/dev/null; \
	echo "All container volumes and local storage cleared."

start-daemon:
	@echo "Starting ho in background with nohup..."
	@OVERNIGHT_LOOP=true nohup uv run python -m src.radar.orchestrator > pipeline.log 2>&1 &
	@echo "Pipeline running in background. View logs with: tail -f pipeline.log"

stop-daemon:
	@pkill -f "python -m src.radar.orchestrator" || true
	@echo "Daemon stopped."

graph:
	docker compose -f docker-compose.yaml up -d neo4j

graph-stop:
	docker compose -f docker-compose.yaml stop neo4j

graph-shell:
	docker compose -f docker-compose.yaml exec neo4j cypher-shell -u neo4j -p password

graph-reset:
	docker compose -f docker-compose.yaml down -v neo4j 2>/dev/null; \
	docker compose -f docker-compose.yaml up -d neo4j; \
	echo "Neo4j reset."

