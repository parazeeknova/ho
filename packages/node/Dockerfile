# syntax=docker/dockerfile:1
# ho-node-service: persistent autofill runner daemon (Stagehand + Chrome).
# Stagehand launches headed Chrome via chrome-launcher, so the image ships
# Google Chrome stable plus Xvfb, and wraps the daemon in xvfb-run.
# The daemon spawns `npx tsx`, hence nodejs/npm and the workspace .bin on PATH.
# Runtime config (AUTOFILL_*, DB, proxy) comes from the orchestrator
# environment, not this image.
# Build from the repo root:  docker build -f packages/node/Dockerfile -t ho-node-service .

FROM oven/bun:latest AS deps
WORKDIR /app
COPY package.json bun.lock ./
COPY apps/web/package.json apps/web/
COPY packages/node/package.json packages/node/
RUN bun install --frozen-lockfile --ignore-scripts

FROM oven/bun:latest AS runner
WORKDIR /app
ENV NODE_ENV=production \
  HOME=/home/hosvc \
  PATH="/app/packages/node/node_modules/.bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends wget gnupg \
  && wget -q -O - https://dl.google.com/linux/linux_signing_key.pub \
    | gpg --dearmor -o /usr/share/keyrings/google.gpg \
  && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google.gpg] http://dl.google.com/linux/chrome/deb/ stable main" \
    > /etc/apt/sources.list.d/google-chrome.list \
  && apt-get update && apt-get install -y --no-install-recommends \
    google-chrome-stable nodejs npm xvfb \
  && rm -rf /var/lib/apt/lists/*

COPY --from=deps /app/package.json /app/bun.lock ./
COPY --from=deps /app/node_modules ./node_modules
COPY packages/node ./packages/node
COPY --from=deps /app/packages/node/node_modules ./packages/node/node_modules

RUN useradd -m -u 1001 hosvc \
  && chown -R hosvc:hosvc /home/hosvc /app
USER hosvc

# No HTTP surface: the daemon speaks over stdin/stdout (STATUS_EVENT/RPC lines).
CMD ["xvfb-run", "-a", "bun", "packages/node/runner_daemon.ts"]
