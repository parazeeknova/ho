/**
 * Persistent autofill runner daemon.
 *
 * The single-job runner (`runner.ts`) is spawned per application, which costs
 * a cold Node boot + tsx transpile of ~11k lines every job. This daemon keeps
 * ONE Node/tsx process alive and runs `runner.ts` as a child for each job,
 * reusing the warm module cache and the same process. The browser is still
 * launched fresh per job (per-job fingerprint/proxy/session require it), but
 * the interpreter/transpile overhead is paid once.
 *
 * Protocol (same as runner.ts):
 *   stdin:   one JSON JobPayload per line
 *   stdout:  STATUS_EVENT:... and RPC_REQUEST:... lines (forwarded verbatim)
 *
 * Exit: closes after EOF, or after AUTOFILL_DAEMON_MAX_JOBS jobs (default 50)
 * so a long-lived daemon cannot accumulate memory/leaked resources.
 */
import { spawn } from "node:child_process";
import * as path from "node:path";
import * as readline from "node:readline";

const RUNNER = path.resolve(__dirname, "runner.ts");

async function main(): Promise<void> {
  const rl = readline.createInterface({ input: process.stdin, terminal: false });
  const maxJobs = parseInt(process.env.AUTOFILL_DAEMON_MAX_JOBS || "50", 10);
  let jobsDone = 0;

  const runOne = (payloadRaw: string): Promise<number> =>
    new Promise<number>((resolve) => {
      const child = spawn("npx", ["tsx", RUNNER], {
        stdio: ["pipe", "pipe", "inherit"],
        env: process.env,
      });
      // Forward the job payload to the child.
      child.stdin?.write(payloadRaw + "\n");
      child.stdin?.end();

      // Forward the child's stdout (STATUS_EVENT / RPC_REQUEST lines) verbatim.
      child.stdout?.pipe(process.stdout);

      child.on("close", (code) => resolve(code ?? 0));
    });

  for await (const line of rl) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    try {
      JSON.parse(trimmed); // validate it's a real payload line
    } catch {
      console.error(`[RunnerDaemon] Skipping non-JSON line: ${trimmed.slice(0, 80)}`);
      continue;
    }
    await runOne(trimmed);
    jobsDone += 1;
    if (maxJobs > 0 && jobsDone >= maxJobs) {
      console.error(`[RunnerDaemon] Reached ${jobsDone} jobs; exiting.`);
      break;
    }
  }
  process.exit(0);
}

main().catch((err) => {
  console.error("[RunnerDaemon] Fatal:", err);
  process.exit(1);
});
