// Root dev orchestrator: `bun run dev`.
//
//  1. Loads .env.development, then .env (gitignored secrets win). Real
//     environment variables win over both files. Children inherit the merge,
//     so every app (web, ingest, node) sees one consistent env.
//  2. Preflights docker/docker-compose.yml: starts missing services, waits
//     until every container is running and its TCP port answers.
//  3. Spawns the web frontend and the ingest pipeline side by side with
//     prefixed logs. Ctrl+C stops everything.
import { spawn, spawnSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import net from "node:net";
import path from "node:path";

const ROOT = path.dirname(import.meta.dirname);
const COMPOSE = path.join(ROOT, "docker", "docker-compose.yml");

// TCP health targets for the compose stack (torproxy is host-networked and
// has no published port, so only its running state is checked).
const TCP_CHECKS = [
  { port: 8080, service: "searxng" },
  { port: 5443, service: "agent-memory-db" },
  { port: 6380, service: "redis" },
  { port: 7474, service: "neo4j" },
];
const HEALTH_TIMEOUT_MS = 120_000;
const POLL_MS = 2000;

function loadEnvFile(file) {
  if (!existsSync(file)) {
    return {};
  }
  const env = {};
  for (const line of readFileSync(file, "utf-8").split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) {
      continue;
    }
    const eq = trimmed.indexOf("=");
    if (eq === -1) {
      continue;
    }
    const key = trimmed.slice(0, eq).trim();
    let value = trimmed.slice(eq + 1).trim();
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }
    env[key] = value;
  }
  return env;
}

function loadEnv() {
  const merged = {
    ...loadEnvFile(path.join(ROOT, ".env.development")),
    ...loadEnvFile(path.join(ROOT, ".env")),
    ...process.env,
  };
  console.log("[dev] env: .env.development + .env merged");
  return merged;
}

function compose(args, env) {
  const r = spawnSync("docker", ["compose", "-f", COMPOSE, ...args], {
    cwd: ROOT,
    encoding: "utf-8",
    env,
  });
  if (r.error) {
    throw new Error(`docker compose ${args[0]} failed: ${r.error.message}`);
  }
  return r;
}

function runningServices(env) {
  const r = compose(["ps", "--format", "json"], env);
  if (r.status !== 0) {
    throw new Error(`docker compose ps failed: ${(r.stderr || "").trim()}`);
  }
  const out = r.stdout.trim();
  if (!out) {
    return [];
  }
  // Real docker prints one object per line; podman-compose prints a single
  // JSON array with podman-style keys (service in compose labels).
  let containers;
  try {
    const parsed = JSON.parse(out);
    containers = Array.isArray(parsed) ? parsed : [parsed];
  } catch {
    containers = out.split("\n").map((line) => JSON.parse(line));
  }
  return containers
    .filter((c) => String(c.State || c.state || "").toLowerCase() === "running")
    .map(
      (c) =>
        c.Service ||
        c.service ||
        ((c.Labels || {})["com.docker.compose.service"] ?? "")
    )
    .filter((s) => s !== "");
}

function sleep(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

function tcpOpen(port) {
  return new Promise((resolve) => {
    const socket = new net.Socket();
    const done = (ok) => {
      socket.destroy();
      resolve(ok);
    };
    socket.setTimeout(2000);
    socket.on("connect", () => {
      done(true);
    });
    socket.on("timeout", () => {
      done(false);
    });
    socket.on("error", () => {
      done(false);
    });
    socket.connect(port, "127.0.0.1");
  });
}

async function closedServices() {
  const probes = await Promise.all(TCP_CHECKS.map((c) => tcpOpen(c.port)));
  return TCP_CHECKS.filter((_, i) => !probes[i]).map((c) => c.service);
}

function missingServices(expected, running) {
  return expected.filter((s) => !running.includes(s));
}

async function preflight(env) {
  const expected = ["searxng", "torproxy", "agent-memory-db", "redis", "neo4j"];
  let running = [];
  try {
    running = runningServices(env);
  } catch (error) {
    console.log(`[infra] ${error.message}`);
  }
  const missing = expected.filter((s) => !running.includes(s));
  if (missing.length > 0) {
    console.log(`[infra] starting: ${missing.join(", ")}`);
    const r = compose(["up", "-d", ...missing], env);
    if (r.status !== 0) {
      throw new Error(
        `docker compose up failed: ${(r.stderr || "").trim()} ` +
          "(hint: host ports may be held by another stack — check `ss -ltn` " +
          "for :5443/:6380, stop it or remap ports in docker/docker-compose.yml)"
      );
    }
  } else {
    console.log("[infra] all containers already running, checking health");
  }
  const deadline = Date.now() + HEALTH_TIMEOUT_MS;
  for (;;) {
    running = runningServices(env);
    const down = missingServices(expected, running);
    const closed = await closedServices();
    if (down.length === 0 && closed.length === 0) {
      console.log(
        "[infra] healthy: searxng :8080, postgres :5443, redis :6380"
      );
      console.log("[infra] healthy: neo4j :7474");
      return;
    }
    if (Date.now() > deadline) {
      throw new Error(
        `infra unhealthy after 120s (down: ${down.join(",") || "none"}; ports closed: ${closed.join(",") || "none"})`
      );
    }
    await sleep(POLL_MS);
  }
}

function spawnTagged(tag, cmd, args, env) {
  const child = spawn(cmd, args, {
    cwd: ROOT,
    env,
    stdio: ["ignore", "pipe", "pipe"],
  });
  const prefix = (data, stream) => {
    for (const line of data.toString().split("\n")) {
      if (line.trim()) {
        console.log(`[${tag}${stream === "err" ? ":err" : ""}] ${line}`);
      }
    }
  };
  child.stdout.on("data", (d) => prefix(d, "out"));
  child.stderr.on("data", (d) => prefix(d, "err"));
  return child;
}

async function main() {
  const env = {
    ...loadEnv(),
    NODE_ENV: "development",
    PYTHONPATH: [".", "packages/ingest", "packages"].join(path.delimiter),
  };
  await preflight(env);
  const children = [
    spawnTagged("web", "bun", ["run", "--filter", "ho-web", "dev"], env),
    spawnTagged(
      "ingest",
      "uv",
      ["run", "python", "packages/ingest/scripts/run_all.py"],
      env
    ),
  ];
  let stopping = false;
  const stop = (signal) => {
    if (stopping) {
      return;
    }
    stopping = true;
    console.log(`\n[dev] ${signal}, stopping services`);
    for (const child of children) {
      child.kill("SIGTERM");
    }
  };
  process.on("SIGINT", () => stop("SIGINT"));
  process.on("SIGTERM", () => stop("SIGTERM"));
  const codes = await Promise.all(
    children.map(
      (child) =>
        new Promise((resolve) => {
          child.on("exit", (code) => resolve(code ?? 1));
          child.on("error", () => resolve(1));
        })
    )
  );
  const failed = codes.some((c) => c !== 0 && c !== 143 && c !== 130);
  if (!stopping && failed) {
    stop("child failed");
  }
  process.exit(failed ? 1 : 0);
}

main().catch((error) => {
  console.error(`[dev] fatal: ${error.message}`);
  process.exit(1);
});
