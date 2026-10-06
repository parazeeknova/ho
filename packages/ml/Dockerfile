# syntax=docker/dockerfile:1
# ho-ml: outcomes sidecar — Gmail Pub/Sub push watcher resolving application
# outcomes into rewards (`python -m ml.src.outcomes.gmail_push`).
# Train/eval one-shots reuse this image with an overridden command, e.g.:
#   docker run ho-ml python packages/ml/scripts/train_model.py
# Build from the repo root:  docker build -f packages/ml/Dockerfile -t ho-ml .

FROM python:3.14-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY packages ./packages

FROM python:3.14-slim AS runner
WORKDIR /app
ENV PYTHONPATH=/app:/app/packages/ingest:/app/packages \
  PATH="/app/.venv/bin:${PATH}"

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/pyproject.toml /app/uv.lock ./
COPY packages ./packages

RUN useradd -m -u 1001 homl \
  && chown -R homl:homl /app /home/homl
USER homl

# No HTTP surface: outcomes flow through Gmail push events into Postgres.
CMD ["python", "-m", "ml.src.outcomes.gmail_push"]
