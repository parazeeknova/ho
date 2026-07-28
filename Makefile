.PHONY: check check-types test serve run fc-up fc-down fc-logs dev dev-down

check:
	uv run ruff format . && uv run ruff check . --fix

check-types:
	uv run mypy . --ignore-missing-imports --exclude 'refs/'

test:
	uv run python -m pytest . -v --ignore=refs

serve:
	./scripts/serve.sh

run:
	uv run python -m pipeline.orchestrator

fc-up:
	@docker compose -f docker-compose.yaml up -d redis playwright-service nuq-postgres; \
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
	./scripts/serve.sh & \
	make fc-up; \
	wait

dev-down:
	docker compose -f docker-compose.yaml down; \
	pkill -f llama-server; \
	wait
