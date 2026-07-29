.PHONY: check check-types test serve run match health fc-up fc-down fc-logs dev dev-down start overnight

check:
	uv run ruff format . && uv run ruff check . --fix

check-types:
	uv run mypy . --ignore-missing-imports --exclude 'refs/' --explicit-package-bases

test:
	uv run python -m pytest . -v --ignore=refs

serve:
	uv run python scripts/serve.py

run: health
	uv run python -m src.pipeline.orchestrator

match: health
	uv run python -m src.pipeline.orchestrator

overnight: health
	OVERNIGHT_LOOP=true uv run python -m src.pipeline.orchestrator

fc-up:
	@docker compose -f docker-compose.yaml up -d redis playwright-service nuq-postgres searxng; \
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
