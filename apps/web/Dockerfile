# syntax=docker/dockerfile:1

FROM oven/bun:latest AS builder
WORKDIR /app

COPY package.json bun.lock ./
COPY apps/web/package.json apps/web/
COPY packages/node/package.json packages/node/
RUN bun install --frozen-lockfile --ignore-scripts

COPY apps/web ./apps/web
RUN bun run --filter 'ho-web' build

FROM node:25-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production PORT=3000 HOSTNAME=0.0.0.0

RUN apk add --no-cache curl \
  && addgroup --system --gid 1001 ho \
  && adduser --system --uid 1001 --ingroup ho howeb

COPY --from=builder --chown=howeb:ho /app/apps/web/.output ./
USER howeb

EXPOSE 3000
HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
  CMD curl -fsS http://localhost:3000/ || exit 1
CMD ["node", "server/index.mjs"]
