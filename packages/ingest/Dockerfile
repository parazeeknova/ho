# syntax=docker/dockerfile:1
# ho-ingest: full-stack runner (radar pipeline + bridge + autofill worker).
# Mirrors `bun run run` (packages/ingest/scripts/run_all.py), which brings up
# its own infra via `docker compose` — so the container needs the docker CLI
# plus the host socket mounted at runtime:
#   docker run -v /var/run/docker.sock:/var/run/docker.sock ho-ingest
# DB URLs, secrets and proxy config come from the orchestrator environment.
# Build from the repo root:  docker build -f packages/ingest/Dockerfile -t ho-ingest .

FROM python:3.14-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
  && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY packages ./packages
RUN uv run --frozen --no-dev playwright install chromium \
  && uv run --frozen --no-dev playwright install-deps chromium

FROM python:3.14-slim AS runner
WORKDIR /app
ENV PYTHONPATH=/app:/app/packages/ingest:/app/packages \
  PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
  PATH="/app/.venv/bin:${PATH}"

# docker CLI: run_all.py manages its own compose stack (needs the host socket).
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg \
  && install -m 0755 -d /etc/apt/keyrings \
  && curl -fsSL https://download.docker.com/linux/debian/gpg \
    | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
  && . /etc/os-release && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian ${VERSION_CODENAME} stable" \
    > /etc/apt/sources.list.d/docker.list \
  && apt-get update && apt-get install -y --no-install-recommends docker-ce-cli xvfb \
  && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /ms-playwright /ms-playwright
COPY --from=builder /app/pyproject.toml /app/uv.lock ./
COPY packages ./packages

RUN useradd -m -u 1001 hosvc \
  && chown -R hosvc:hosvc /app /ms-playwright /home/hosvc
USER hosvc

# No HTTP surface: progress flows through STATUS_EVENT lines, logs and the DB.
CMD ["xvfb-run", "-a", "python", "packages/ingest/scripts/run_all.py"]
