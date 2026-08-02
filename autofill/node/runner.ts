import { Stagehand } from "@browserbasehq/stagehand";
import * as fs from "fs";
import * as path from "path";
import * as readline from "readline";
import { JobPayloadSchema, ActionCallbackSchema, StatusEvent } from "./types.js";
import { ATSAdapter, RpcHelper } from "./ats/base.js";
import { GreenhouseAdapter } from "./ats/greenhouse.js";
import { AshbyAdapter } from "./ats/ashby.js";
import { LeverAdapter } from "./ats/lever.js";
import { WorkdayAdapter } from "./ats/workday.js";
import { GenericAdapter } from "./ats/generic.js";
import { ActivityWatchdog } from "./utils/activity.js";
import { getDeferredFieldCount, resetDeferredFieldCount } from "./ats/shared/screener.js";

interface AdapterRegistration {
  pattern: RegExp;
  factory: (stagehand: Stagehand) => ATSAdapter;
}

const adapterRegistry: AdapterRegistration[] = [
  { pattern: /greenhouse\.io/, factory: (s) => new GreenhouseAdapter(s) },
  { pattern: /jobs\.ashbyhq\.com/, factory: (s) => new AshbyAdapter(s) },
  { pattern: /jobs\.lever\.co/, factory: (s) => new LeverAdapter(s) },
  { pattern: /myworkdayjobs\.com/, factory: (s) => new WorkdayAdapter(s) },
  // The GenericAdapter is the intelligent fallback for ANY unknown URL: it
  // classifies the form's shape (single form / wizard / JD-page / gate) and
  // drives the shared fill machinery. Known ATS regexes above always win.
  { pattern: /.*/, factory: (s) => new GenericAdapter(s) },
];

function getAdapterForUrl(url: string, stagehand: Stagehand): ATSAdapter {
  // Debug override to force the generic adapter against a known platform.
  if (process.env.AUTOFILL_FORCE_GENERIC === "1") {
    console.warn("[Runner] AUTOFILL_FORCE_GENERIC=1 — forcing GenericAdapter.");
    return new GenericAdapter(stagehand);
  }
  const entry = adapterRegistry.find((reg) => reg.pattern.test(url));
  if (!entry) {
    throw new Error(`Unsupported ATS platform for URL: ${url}`);
  }
  if (entry.pattern.source === ".*") {
    console.log(`[Runner] No specialized adapter for ${url}; using GenericAdapter.`);
  }
  return entry.factory(stagehand);
}

function emitStatus(statusEvent: StatusEvent) {
  console.log(`STATUS_EVENT:${JSON.stringify(statusEvent)}`);
}

async function main() {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    terminal: false
  });

  // Read initial payload line
  const payloadRaw = await new Promise<any>((resolve, reject) => {
    rl.once("line", (line) => {
      try {
        resolve(JSON.parse(line.trim()));
      } catch (err) {
        reject(err);
      }
    });
  }).catch((err) => {
    console.error("[Runner] Error reading initial JSON payload from stdin:", err);
    process.exit(1);
  });

  const parseResult = JobPayloadSchema.safeParse(payloadRaw);
  if (!parseResult.success) {
    console.error("[Runner] Invalid JobPayload schema:", parseResult.error.format());
    process.exit(1);
  }

  const payload = parseResult.data;
  console.log(`[Runner] Received job ${payload.jobId} for ${payload.url}`);

  const apiKey = process.env.GENERALCOMPUTE_API_KEY || process.env.OPENAI_API_KEY;
  if (!apiKey) {
    console.error("[Runner] Missing API key. Set GENERALCOMPUTE_API_KEY or OPENAI_API_KEY in environment.");
    emitStatus({
      jobId: payload.jobId,
      status: "failed",
      error: "Missing API key. Set GENERALCOMPUTE_API_KEY or OPENAI_API_KEY."
    });
    process.exit(1);
  }

  const modelName = process.env.GENERALCOMPUTE_MODEL || "deepseek-v3.2";
  const genericModel = modelName.includes("/") ? modelName : `openai/${modelName}`;
  if (modelName.includes("/")) {
    console.warn(`[Runner] GENERALCOMPUTE_MODEL "${modelName}" includes a provider prefix; using as-is.`);
  }

  console.log(`[Runner] Initializing Stagehand LOCAL environment with model ${genericModel}...`);

  // Stagehand v3 unified model config: modelClientOptions was removed, and the
  // OpenAI AI SDK defaults custom baseURL endpoints to the Responses API. GeneralCompute
  // exposes an OpenAI-compatible Chat Completions endpoint, so we must opt into chat format.
  const stagehandConfig: any = {
    env: "LOCAL",
    model: {
      modelName: genericModel,
      apiKey: apiKey,
      baseURL: "https://api.generalcompute.com/v1",
      openaiEndpointFormat: "chat"
    },
    // Deterministic act(action) fills must never self-heal via an LLM — a
    // self-heal can silently re-target a DIFFERENT element (and the committed
    // value check would then read the wrong field). The generic adapter's
    // observe fallback relies on exact selector execution.
    selfHeal: false,
    localBrowserLaunchOptions: {
      headless: false,
      args: ["--disable-blink-features=AutomationControlled"]
    }
  };
  const stagehand = new Stagehand(stagehandConfig);

  // Unified Single Readline Dispatcher for RPC responses and Action Callbacks
  const pendingRpcPromises = new Map<string, { resolve: (val: any) => void; reject: (err: any) => void; timer: NodeJS.Timeout }>();
  let actionCallbackResolver: ((action: "submit" | "skip") => void) | null = null;

  rl.on("line", (line) => {
    const lineStr = line.trim();
    if (!lineStr.startsWith("{")) return;

    try {
      const parsed = JSON.parse(lineStr);

      // Handle RPC Responses
      if (parsed.type === "RPC_RESPONSE" && parsed.id) {
        const pending = pendingRpcPromises.get(parsed.id);
        if (pending) {
          clearTimeout(pending.timer);
          pendingRpcPromises.delete(parsed.id);
          if (parsed.error) {
            pending.reject(new Error(parsed.error));
          } else {
            pending.resolve(parsed.result);
          }
        }
      }
      // Handle Action Callbacks (Submit / Skip)
      else if (parsed.action && actionCallbackResolver) {
        const callbackParse = ActionCallbackSchema.safeParse(parsed);
        if (callbackParse.success) {
          actionCallbackResolver(callbackParse.data.action);
          actionCallbackResolver = null;
        }
      }
    } catch (_) {}
  });

  const rpcTimeoutMs = parseInt(process.env.AUTOFILL_RPC_TIMEOUT_MS || "1800000", 10);

  const askPythonRpc: RpcHelper = (method: string, args: Record<string, any>): Promise<any> => {
    return new Promise((resolve, reject) => {
      const id = `rpc-${Math.random().toString(36).substring(2, 9)}`;

      const timer = setTimeout(() => {
        pendingRpcPromises.delete(id);
        reject(new Error(`RPC timeout for method "${method}" after ${rpcTimeoutMs / 1000} seconds`));
      }, rpcTimeoutMs);

      pendingRpcPromises.set(id, { resolve, reject, timer });
      console.log(`RPC_REQUEST:${JSON.stringify({ id, method, args })}`);
    });
  };

  // Overnight safety: an activity watchdog kills a run whose browser stops
  // making progress (stuck fill). The worker enables it (overnight only) via
  // AUTOFILL_ACTIVITY_TIMEOUT_MS; anything that counts as observable progress
  // — status events, RPC traffic, adapter logs — resets the idle timer, so a
  // healthy run is never aborted. A hung Stagehand act/observe emits nothing,
  // so the run dies here instead of hanging until the hour-long DB lease.
  const activityTimeoutMs = parseInt(process.env.AUTOFILL_ACTIVITY_TIMEOUT_MS || "0", 10);
  let watchdog: ActivityWatchdog | null = null;
  if (activityTimeoutMs > 0) {
    watchdog = new ActivityWatchdog(activityTimeoutMs, () => {
      const err = new Error(
        `STUCK_TIMEOUT: no runner/browser activity for ${Math.round(activityTimeoutMs / 1000)}s`
      );
      console.error("[Runner] Activity watchdog fired; aborting stuck run:", err.message);
      emitStatus({
        jobId: payload.jobId,
        status: "failed",
        error: err.message,
      });
      watchdog?.stop();
      try {
        rl.close();
      } catch (_) {}
      // Best-effort browser close with a hard fallback: a hung browser must
      // not prevent the process from dying.
      const forceExit = setTimeout(() => process.exit(1), 5000);
      Promise.resolve(stagehand.close())
        .catch(() => {})
        .then(() => {
          clearTimeout(forceExit);
          process.exit(1);
        });
    });
    // Any console output from the runner or the adapters is observable
    // progress, so treat every log line as activity.
    const touch = () => watchdog?.touch();
    const origLog = console.log;
    const origWarn = console.warn;
    const origError = console.error;
    console.log = (...args: any[]) => {
      touch();
      origLog(...args);
    };
    console.warn = (...args: any[]) => {
      touch();
      origWarn(...args);
    };
    console.error = (...args: any[]) => {
      touch();
      origError(...args);
    };
    watchdog.start();
  }

  try {
    await stagehand.init();
    console.log("[Runner] Stagehand initialized successfully.");

    const adapter = getAdapterForUrl(payload.url, stagehand);

    // Anti-bot watchdog: if a captcha/challenge blocks the form, abort loudly
    // (status failed, error CAPTCHA_DETECTED) instead of silently grinding
    // through fill/RPC timeouts. The Python worker turns this into a Telegram
    // notification to the user. Polled while the fill runs. A captcha-blocked
    // fill does NOT throw on its own (the walk just finds no fields), so the
    // fill is raced against an abort signal: detection rejects the race even
    // when the adapter would otherwise "complete" with a blank form.
    const captchaWatchMs = parseInt(process.env.AUTOFILL_CAPTCHA_WATCH_MS || "8000", 10);
    let captchaMessage: string | null = null;
    let rejectCaptcha: ((err: Error) => void) | null = null;
    const captchaAbort = new Promise<never>((_, reject) => {
      rejectCaptcha = reject;
    });
    const captchaTimer = setInterval(async () => {
      if (captchaMessage) return;
      try {
        const hit = await adapter.detectCaptcha();
        if (hit) {
          captchaMessage = hit;
          clearInterval(captchaTimer);
          rejectCaptcha?.(
            new Error(`CAPTCHA_DETECTED: ${hit} blocked the application form`)
          );
        }
      } catch (_) {}
    }, captchaWatchMs);

    // Execute filling process with RPC helper
    resetDeferredFieldCount();
    const fillPromise = adapter.fill(payload, askPythonRpc);
    try {
      await Promise.race([fillPromise, captchaAbort]);
    } catch (fillErr: any) {
      if (captchaMessage) {
        const err = new Error(`CAPTCHA_DETECTED: ${captchaMessage} blocked the application form`);
        console.error("[Runner] Captcha detected:", err.message);
        emitStatus({
          jobId: payload.jobId,
          status: "failed",
          error: err.message,
        });
        // The abandoned fill is still running in the background; stop its
        // unhandled-rejection noise before tearing the browser down.
        fillPromise.catch(() => {});
        try {
          rl.close();
          await stagehand.close();
        } catch (_) {}
        process.exit(1);
      }
      throw fillErr;
    } finally {
      clearInterval(captchaTimer);
    }

    // Save screenshot safely
    const screenshotDir = path.resolve("./artifacts/screenshots");
    if (!fs.existsSync(screenshotDir)) {
      fs.mkdirSync(screenshotDir, { recursive: true });
    }
    const screenshotPath = path.join(screenshotDir, `${payload.jobId}.png`);

    const pages = stagehand.context.pages();
    if (pages.length === 0) {
      throw new Error("No active browser pages available for taking screenshot.");
    }
    const activePage = adapter.getActivePage();
    await activePage.screenshot({ path: screenshotPath, fullPage: true });
    console.log(`[Runner] Screenshot saved to ${screenshotPath}`);

    if (!payload.submitAllowed) {
      // No-apply phase: the form is filled and verified, but the application
      // is never submitted. The browser stays open (bounded by
      // AUTOFILL_REVIEW_HOLD_MS) so a human can inspect the filled answers;
      // any action — or the hold timeout — closes without submitting.
      watchdog?.stop();
      console.log(
        "[Runner] Submission disabled — browser window remaining open for review. " +
          "Any action (or the hold timeout) closes without submitting."
      );
      emitStatus({
        jobId: payload.jobId,
        status: "awaiting_review",
        screenshotPath: screenshotPath,
        message: "Form filled successfully. Submission is disabled in this phase."
      });
      const holdMs = parseInt(process.env.AUTOFILL_REVIEW_HOLD_MS || "600000", 10);
      await new Promise<"submit" | "skip">((resolve) => {
        actionCallbackResolver = resolve;
        if (holdMs > 0) {
          setTimeout(() => resolve("skip"), holdMs);
        }
      });
      rl.close();
      await stagehand.close();
      emitStatus({
        jobId: payload.jobId,
        status: "skipped",
        screenshotPath: screenshotPath,
        message: "Application filled but not submitted (submission disabled)."
      });
      process.exit(0);
    }

    if (payload.mode === "auto") {
      if (getDeferredFieldCount() > 0) {
        // A question was deferred (overnight, no human): its field is blank.
        // Never submit an incomplete application — abort for the morning
        // digest / resume flow instead. The worker marks the job deferred
        // (the skipped event is kept deferred when questions are pending).
        console.log(
          "[Runner] Deferred question(s) remain; not submitting. " +
            "Job stays deferred for the morning digest / resume flow."
        );
        emitStatus({
          jobId: payload.jobId,
          status: "skipped",
          screenshotPath: screenshotPath,
          message: "Deferred questions remain; application not submitted."
        });
        watchdog?.stop();
        rl.close();
        await stagehand.close();
        process.exit(0);
      }
      console.log("[Runner] Auto mode enabled. Submitting application immediately...");
      // The watchdog stays armed through submit so a stuck submit is also
      // killed, then disarms once the outcome is reported.
      await adapter.submit();
      emitStatus({
        jobId: payload.jobId,
        status: "submitted",
        screenshotPath: screenshotPath,
        message: "Application filled and submitted automatically."
      });
      watchdog?.stop();
      rl.close();
      await stagehand.close();
      process.exit(0);
    } else {
      // Review mode: Emit awaiting_review and await action from single dispatcher
      watchdog?.stop();
      emitStatus({
        jobId: payload.jobId,
        status: "awaiting_review",
        screenshotPath: screenshotPath,
        message: "Form filled successfully. Awaiting review command."
      });

      console.log("[Runner] Browser window remaining open for review. Waiting for callback on stdin...");

      const action = await new Promise<"submit" | "skip">((resolve) => {
        actionCallbackResolver = resolve;
      });

      if (action === "submit") {
        console.log("[Runner] Callback 'submit' received. Submitting...");
        await adapter.submit();
        emitStatus({
          jobId: payload.jobId,
          status: "submitted",
          screenshotPath: screenshotPath,
          message: "Application submitted after user review."
        });
      } else {
        console.log("[Runner] Callback 'skip' received. Skipping submission...");
        emitStatus({
          jobId: payload.jobId,
          status: "skipped",
          screenshotPath: screenshotPath,
          message: "Application skipped by user."
        });
      }

      rl.close();
      await stagehand.close();
      process.exit(0);
    }
  } catch (err: any) {
    watchdog?.stop();
    console.error("[Runner] Execution error:", err);
    emitStatus({
      jobId: payload.jobId,
      status: "failed",
      error: err?.message || String(err)
    });
    try {
      rl.close();
      await stagehand.close();
    } catch (_) {}
    process.exit(1);
  }
}

main();
