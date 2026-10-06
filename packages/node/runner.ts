import * as fs from "node:fs";
import path from "node:path";
import * as readline from "node:readline";

import { Stagehand } from "@browserbasehq/stagehand";

import { AshbyAdapter } from "./ats/ashby";
import type { ATSAdapter, RpcHelper } from "./ats/base";
import { GenericAdapter } from "./ats/generic";
import { GreenhouseAdapter } from "./ats/greenhouse";
import { LeverAdapter } from "./ats/lever";
import {
  getBlankedRequiredCount,
  getDeferredFieldCount,
  resetBlankedRequiredCount,
  resetDeferredFieldCount,
} from "./ats/shared/screener";
import { WorkdayAdapter } from "./ats/workday";
import { JobPayloadSchema, ActionCallbackSchema } from "./types";
import type { StatusEvent } from "./types";
import { ActivityWatchdog } from "./utils/activity";
import { waitForAtsEmail } from "./utils/atsEmail";
import type { AtsEmailResult } from "./utils/atsEmail";
import { applyFingerprint, loadFingerprint } from "./utils/fingerprint";

// Rejects when a promise has not settled within `ms`. Used so a best-effort
// captcha attempt can never stall the run indefinitely.
function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(
      () => reject(new Error(`operation timed out after ${ms}ms`)),
      ms
    );
    promise
      .then((v) => {
        clearTimeout(timer);
        resolve(v);
      })
      .catch((error) => {
        clearTimeout(timer);
        reject(error);
      });
  });
}

interface AdapterRegistration {
  pattern: RegExp;
  factory: (stagehand: Stagehand) => ATSAdapter;
}

const adapterRegistry: AdapterRegistration[] = [
  { factory: (s) => new GreenhouseAdapter(s), pattern: /greenhouse\.io/ },
  // Greenhouse custom domains: companies host their greenhouse board on their
  // own domain (careers.airbnb.com, mongodb.com/careers?gh_jid=, abnormal.ai/
  // careers/jobs/...?gh_jid=). The gh_jid query param is the greenhouse
  // posting id — the form is the greenhouse form and needs the greenhouse
  // adapter (consent checkboxes, OTP handling, submit verification). Without
  // this they fell to the GenericAdapter and failed submit verification.
  { factory: (s) => new GreenhouseAdapter(s), pattern: /[?&]gh_jid=\d+/ },
  { factory: (s) => new AshbyAdapter(s), pattern: /jobs\.ashbyhq\.com/ },
  { factory: (s) => new LeverAdapter(s), pattern: /jobs\.lever\.co/ },
  { factory: (s) => new WorkdayAdapter(s), pattern: /myworkdayjobs\.com/ },
  // The GenericAdapter is the intelligent fallback for ANY unknown URL: it
  // classifies the form's shape (single form / wizard / JD-page / gate) and
  // drives the shared fill machinery. Known ATS regexes above always win.
  { factory: (s) => new GenericAdapter(s), pattern: /.*/ },
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
    console.log(
      `[Runner] No specialized adapter for ${url}; using GenericAdapter.`
    );
  }
  return entry.factory(stagehand);
}

function emitStatus(statusEvent: StatusEvent) {
  console.log(`STATUS_EVENT:${JSON.stringify(statusEvent)}`);
}

/**
 * Soft post-submit email feedback: after the ATS confirmation page is reached,
 * briefly poll Gmail for the ATS's reply email and classify it. Never blocks —
 * a missing config, timeout, or IMAP failure just returns null so the page
 * confirmation alone stands. The result (confirmation/rejection/screening/otp)
 * rides along on the "submitted" status event for the worker to surface.
 */
async function softAtsEmailFeedback(
  payload: any
): Promise<StatusEvent["emailStatus"]> {
  if (
    !process.env.AUTOFILL_EMAIL_FEEDBACK ||
    process.env.AUTOFILL_EMAIL_FEEDBACK === "0"
  ) {
    return undefined;
  }
  try {
    // Bounded window (default 45s) so it never meaningfully delays the run.
    const ms = Number.parseInt(
      process.env.AUTOFILL_EMAIL_FEEDBACK_TIMEOUT_MS || "45000",
      10
    );
    const result: AtsEmailResult | null = await withTimeout(
      waitForAtsEmail({
        context: (payload.company || "").toLowerCase() || undefined,
        log: (m) => console.log(`[Runner] ${m}`),
        timeoutMs: ms,
      }),
      ms + 5000
    );
    if (!result) {
      return undefined;
    }
    return {
      from: result.from,
      kind: result.kind,
      snippet: result.snippet,
      subject: result.subject,
    };
  } catch (error: any) {
    console.warn(`[Runner] Email feedback skipped: ${error?.message || error}`);
    return undefined;
  }
}

async function main() {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    terminal: false,
  });

  // Read initial payload line
  const payloadRaw = await new Promise<any>((resolve, reject) => {
    rl.once("line", (line) => {
      try {
        resolve(JSON.parse(line.trim()));
      } catch (error) {
        reject(error);
      }
    });
  }).catch((error) => {
    console.error(
      "[Runner] Error reading initial JSON payload from stdin:",
      error
    );
    process.exit(1);
  });

  const parseResult = JobPayloadSchema.safeParse(payloadRaw);
  if (!parseResult.success) {
    console.error(
      "[Runner] Invalid JobPayload schema:",
      parseResult.error.format()
    );
    process.exit(1);
  }

  const payload = parseResult.data;
  console.log(`[Runner] Received job ${payload.jobId} for ${payload.url}`);
  console.log(
    `[Runner] DEPLOY-MARKER: consent-fix-v3 (checkCheckboxViaCdpClick)`
  );

  const apiKey =
    process.env.GENERALCOMPUTE_API_KEY || process.env.OPENAI_API_KEY;
  if (!apiKey) {
    console.error(
      "[Runner] Missing API key. Set GENERALCOMPUTE_API_KEY or OPENAI_API_KEY in environment."
    );
    emitStatus({
      error: "Missing API key. Set GENERALCOMPUTE_API_KEY or OPENAI_API_KEY.",
      jobId: payload.jobId,
      status: "failed",
    });
    process.exit(1);
  }

  const modelName = process.env.GENERALCOMPUTE_MODEL || "deepseek-v3.2";
  const genericModel = modelName.includes("/")
    ? modelName
    : `openai/${modelName}`;
  if (modelName.includes("/")) {
    console.warn(
      `[Runner] GENERALCOMPUTE_MODEL "${modelName}" includes a provider prefix; using as-is.`
    );
  }

  // Per-job browser fingerprint (UA/platform/viewport/cores/memory/languages,
  // India-consistent locale+timezone). Seeded by AUTOFILL_FINGERPRINT_SEED
  // (set per job by the worker) so a batch of applications never presents as
  // a single device — the "many apps from one device" fraud signal.
  const fingerprint = loadFingerprint();

  console.log(
    `[Runner] Initializing Stagehand (direct launch) with model ${genericModel}...`
  );

  // Stagehand v3 unified model config: modelClientOptions was removed, and the
  // OpenAI AI SDK defaults custom baseURL endpoints to the Responses API. GeneralCompute
  // exposes an OpenAI-compatible Chat Completions endpoint, so we must opt into chat format.
  const stagehandConfig: any = {
    env: "LOCAL",
    localBrowserLaunchOptions: {
      args: ["--disable-blink-features=AutomationControlled"],
      headless: false,
      // Drop the test-harness flags that scream "automation" to a fingerprint
      // scanner. A real user's Chrome has no --metrics-recording-only,
      // --propagate-iph-for-testing, disabled sync/extensions/background
      // networking, or the Stagehand MCP feature flag (our act()/observe()
      // fallbacks use the a11y snapshot, not WebMCP, so it is safe to drop).
      ignoreDefaultArgs: [
        "--metrics-recording-only",
        "--use-mock-keychain",
        "--propagate-iph-for-testing",
        "--disable-sync",
        "--disable-extensions",
        "--disable-component-extensions-with-background-pages",
        "--disable-background-networking",
        "--disable-hang-monitor",
        "--enable-features=WebMCPTesting,DevToolsWebMCPSupport",
      ],
      // Reuse a persistent Chrome profile (one per worker slot, assigned by the
      // worker via AUTOFILL_USER_DATA_DIR) so the browser accumulates cookies,
      // storage, and history like a real lived-in browser instead of starting
      // as a brand-new profile on every run — the "fresh browser every submit"
      // session shape is itself an automation signal.
      ...(process.env.AUTOFILL_USER_DATA_DIR
        ? { userDataDir: process.env.AUTOFILL_USER_DATA_DIR }
        : {}),
      // Randomized device attributes applied at launch: locale -> --lang,
      // viewport -> --window-size, deviceScaleFactor ->
      // --force-device-scale-factor. UA + timezone are applied post-init via
      // CDP (see applyFingerprint below) since they need the page session.
      deviceScaleFactor: fingerprint.deviceScaleFactor,
      locale: fingerprint.locale,
      viewport: fingerprint.viewport,
      // Route the whole browser session through a proxy when AUTOFILL_PROXY is
      // set (either the legacy Tor SOCKS5 proxy, or — with a proxy template —
      // a per-job residential IP URL substituted by the worker). Stagehand maps
      // proxy.server -> --proxy-server=... at launch.
      ...(process.env.AUTOFILL_PROXY
        ? { proxy: { server: process.env.AUTOFILL_PROXY } }
        : {}),
    },
    model: {
      apiKey,
      baseURL: "https://api.generalcompute.com/v1",
      modelName: genericModel,
      openaiEndpointFormat: "chat",
    },
    // Deterministic act(action) fills must never self-heal via an LLM — a
    // self-heal can silently re-target a DIFFERENT element (and the committed
    // value check would then read the wrong field). The generic adapter's
    // observe fallback relies on exact selector execution.
    selfHeal: false,
  };
  const stagehand = new Stagehand(stagehandConfig);

  // Unified Single Readline Dispatcher for RPC responses and Action Callbacks
  const pendingRpcPromises = new Map<
    string,
    {
      resolve: (val: any) => void;
      reject: (err: any) => void;
      timer: NodeJS.Timeout;
    }
  >();
  let actionCallbackResolver:
    | ((cb: {
        action: "submit" | "skip" | "correct";
        corrections?: Record<string, string>;
      }) => void)
    | null = null;

  const waitForDecision = () =>
    new Promise<{
      action: "submit" | "skip" | "correct";
      corrections?: Record<string, string>;
    }>((resolve) => {
      actionCallbackResolver = resolve;
    });

  rl.on("line", (line) => {
    const lineStr = line.trim();
    if (!lineStr.startsWith("{")) {
      return;
    }

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
      // Handle Action Callbacks (Submit / Skip / Correct)
      else if (parsed.action && actionCallbackResolver) {
        const callbackParse = ActionCallbackSchema.safeParse(parsed);
        if (callbackParse.success) {
          actionCallbackResolver(callbackParse.data);
          actionCallbackResolver = null;
        }
      }
    } catch {
      // best-effort cleanup; ignore failures
    }
  });

  const rpcTimeoutMs = Number.parseInt(
    process.env.AUTOFILL_RPC_TIMEOUT_MS || "1800000",
    10
  );

  const askPythonRpc: RpcHelper = (
    method: string,
    args: Record<string, any>
  ): Promise<any> =>
    new Promise((resolve, reject) => {
      const id = `rpc-${Math.random().toString(36).slice(2, 9)}`;

      const timer = setTimeout(() => {
        pendingRpcPromises.delete(id);
        reject(
          new Error(
            `RPC timeout for method "${method}" after ${rpcTimeoutMs / 1000} seconds`
          )
        );
      }, rpcTimeoutMs);

      pendingRpcPromises.set(id, { reject, resolve, timer });
      console.log(`RPC_REQUEST:${JSON.stringify({ args, id, method })}`);
    });

  // Overnight safety: an activity watchdog kills a run whose browser stops
  // making progress (stuck fill). The worker enables it (overnight only) via
  // AUTOFILL_ACTIVITY_TIMEOUT_MS; anything that counts as observable progress
  // — status events, RPC traffic, adapter logs — resets the idle timer, so a
  // healthy run is never aborted. A hung Stagehand act/observe emits nothing,
  // so the run dies here instead of hanging until the hour-long DB lease.
  const activityTimeoutMs = Number.parseInt(
    process.env.AUTOFILL_ACTIVITY_TIMEOUT_MS || "0",
    10
  );
  let watchdog: ActivityWatchdog | null = null;
  if (activityTimeoutMs > 0) {
    watchdog = new ActivityWatchdog(activityTimeoutMs, () => {
      const err = new Error(
        `STUCK_TIMEOUT: no runner/browser activity for ${Math.round(activityTimeoutMs / 1000)}s`
      );
      console.error(
        "[Runner] Activity watchdog fired; aborting stuck run:",
        err.message
      );
      emitStatus({
        error: err.message,
        jobId: payload.jobId,
        status: "failed",
      });
      watchdog?.stop();
      try {
        rl.close();
      } catch {
        // best-effort cleanup; ignore failures
      }
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

    // Apply the per-job fingerprint before the adapter navigates anywhere: CDP
    // Emulation overrides persist on the page session and the document-start
    // init script runs on every future navigation.
    await applyFingerprint(stagehand, fingerprint);

    const adapter = getAdapterForUrl(payload.url, stagehand);

    // Anti-bot watchdog: if a captcha/challenge blocks the form, do NOT abort
    // the moment it is detected — attempt to solve it once (click the visible
    // checkbox/challenge), and only fail the run (status failed, error
    // CAPTCHA_DETECTED) if it is STILL blocking after that attempt. The Python
    // worker turns this into a Telegram notification to the user. Polled while
    // the fill runs and stays armed through submit, so a captcha blocking the
    // submit button is caught the same way. A captcha-blocked fill does NOT
    // throw on its own (the walk just finds no fields), so the fill and submit
    // are raced against an abort signal: an unresolved challenge rejects the
    // race even when the adapter would otherwise "complete" with a blank form.
    const captchaWatchMs = Number.parseInt(
      process.env.AUTOFILL_CAPTCHA_WATCH_MS || "8000",
      10
    );
    const captchaAttemptTimeoutMs = Number.parseInt(
      process.env.AUTOFILL_CAPTCHA_ATTEMPT_TIMEOUT_MS || "20000",
      10
    );
    let captchaMessage: string | null = null;
    let rejectCaptcha: ((err: Error) => void) | null = null;
    const captchaAbort = new Promise<never>((_, reject) => {
      rejectCaptcha = reject;
    });
    // The watchdog stays armed through fill AND submit; once nothing races
    // captchaAbort any more (e.g. after a successful submit), a late rejection
    // must not crash the process as an unhandled rejection.
    captchaAbort.catch(() => {});
    let captchaAttempted = false;
    const captchaTimer = setInterval(async () => {
      if (captchaMessage) {
        return;
      }
      try {
        const hit = await adapter.detectCaptcha();
        if (!hit) {
          // Nothing blocking now (or a previous attempt cleared it).
          captchaAttempted = false;
          return;
        }
        // A concurrent tick is already attempting; wait for it to settle.
        if (captchaAttempted) {
          return;
        }
        // First sighting of a challenge: try to solve it once before declaring
        // failure. The fill keeps running in parallel; only a challenge that
        // survives the attempt aborts the run.
        captchaAttempted = true;
        try {
          const attempted = await withTimeout(
            adapter.attemptCaptcha(),
            captchaAttemptTimeoutMs
          );
          console.log(
            `[Runner] Captcha attempt (clicked: ${attempted}); re-checking...`
          );
        } catch (error: any) {
          console.warn(
            "[Runner] Captcha attempt failed:",
            error?.message || String(error)
          );
        }
        const after = await adapter.detectCaptcha();
        if (!after) {
          // The attempt cleared the challenge — keep watching, and allow a
          // future challenge its own single attempt.
          captchaAttempted = false;
          return;
        }
        captchaMessage = after;
        clearInterval(captchaTimer);
        rejectCaptcha?.(
          new Error(`CAPTCHA_DETECTED: ${after} blocked the application form`)
        );
      } catch {
        // best-effort cleanup; ignore failures
      }
    }, captchaWatchMs);

    // Execute filling process with RPC helper
    resetDeferredFieldCount();
    resetBlankedRequiredCount();
    const fillPromise = adapter.fill(payload, askPythonRpc);
    try {
      await Promise.race([fillPromise, captchaAbort]);
    } catch (error: any) {
      if (captchaMessage) {
        const err = new Error(
          `CAPTCHA_DETECTED: ${captchaMessage} blocked the application form`
        );
        console.error("[Runner] Captcha detected:", err.message);
        emitStatus({
          error: err.message,
          jobId: payload.jobId,
          status: "failed",
        });
        // The abandoned fill is still running in the background; stop its
        // unhandled-rejection noise before tearing the browser down.
        fillPromise.catch(() => {});
        try {
          rl.close();
          await stagehand.close();
        } catch {
          // best-effort cleanup; ignore failures
        }
        process.exit(1);
      }
      throw error;
    }

    // Save screenshot safely
    const screenshotDir = path.resolve("./artifacts/screenshots");
    if (!fs.existsSync(screenshotDir)) {
      fs.mkdirSync(screenshotDir, { recursive: true });
    }
    const screenshotPath = path.join(screenshotDir, `${payload.jobId}.png`);
    let pages: any[] = [];
    try {
      pages = stagehand.context?.pages?.() ?? [];
    } catch {
      pages = [];
    }

    if (pages.length === 0) {
      // The browser context is gone (rare, e.g. the remote session died between
      // fill and screenshot). This is not a fill failure — emit a skipped status
      // so the worker records it as such instead of crashing with
      // "Cannot read properties of null (reading 'pages')" and burning a retry.
      console.warn(
        "[Runner] No browser context for screenshot; treating fill as skipped."
      );
      emitStatus({
        jobId: payload.jobId,
        message: "Browser context lost after fill; nothing submitted.",
        status: "skipped",
      });
      watchdog?.stop();
      rl.close();
      await stagehand.close();
      process.exit(0);
    }
    const activePage = adapter.getActivePage();
    await activePage.screenshot({ fullPage: true, path: screenshotPath });
    console.log(`[Runner] Screenshot saved to ${screenshotPath}`);

    // Expired/removed posting detection: if the page says the posting is gone
    // (page-not-found, position-filled, etc.), mark the job `expired` — a
    // terminal, non-retryable state the worker records so the queue stops
    // retrying a dead listing. This runs BEFORE the deferred/blank gate: a
    // dead page shows no form, so every field reads "blank" and the fill
    // would otherwise fail as "no form" instead of "expired posting".
    try {
      const expiredReason = await adapter.detectExpired();
      if (expiredReason) {
        console.log(`[Runner] Posting appears expired: ${expiredReason}`);
        emitStatus({
          error: expiredReason,
          jobId: payload.jobId,
          screenshotPath,
          status: "expired",
        });
        watchdog?.stop();
        clearInterval(captchaTimer);
        rl.close();
        await stagehand.close();
        process.exit(0);
      }
    } catch (error: any) {
      console.warn(
        `[Runner] Expired-posting check failed: ${error?.message || error}`
      );
    }

    if (!payload.submitAllowed) {
      // No-apply phase: the form is filled and verified, but the application
      // is never submitted. The browser stays open (bounded by
      // AUTOFILL_REVIEW_HOLD_MS) so a human can inspect the filled answers;
      // any action — or the hold timeout — closes without submitting.
      watchdog?.stop();
      clearInterval(captchaTimer);
      console.log(
        "[Runner] Submission disabled — browser window remaining open for review. " +
          "Any action (or the hold timeout) closes without submitting."
      );
      emitStatus({
        filledFields: adapter.filledValues,
        jobId: payload.jobId,
        message:
          "Form filled successfully. Submission is disabled in this phase.",
        screenshotPath,
        status: "awaiting_review",
      });
      const holdMs = Number.parseInt(
        process.env.AUTOFILL_REVIEW_HOLD_MS || "360000",
        10
      );
      await new Promise<"submit" | "skip">((resolve) => {
        actionCallbackResolver = (cb) =>
          resolve(cb.action === "submit" ? "submit" : "skip");
        if (holdMs > 0) {
          setTimeout(() => resolve("skip"), holdMs);
        }
      });
      rl.close();
      await stagehand.close();
      emitStatus({
        jobId: payload.jobId,
        message: "Application filled but not submitted (submission disabled).",
        screenshotPath,
        status: "skipped",
      });
      process.exit(0);
    }

    if (payload.mode === "auto") {
      if (getDeferredFieldCount() > 0 || getBlankedRequiredCount() > 0) {
        // A required field is blank (or a question was deferred). Before
        // giving up, GO BACK and re-fill the blank required fields via the
        // adapter's recheck (it re-asks unresolved questions through the RPC
        // bridge — which now includes Discord prompting for unknown fields).
        // Only if blanks genuinely remain does the job defer instead of
        // submitting an incomplete application.
        console.log(
          `[Runner] ${getDeferredFieldCount()} deferred + ${getBlankedRequiredCount()} required-blank question(s) remain; re-filling before deciding...`
        );
        try {
          const stillBlank = await adapter.recheckMissingFields(askPythonRpc);
          if (stillBlank === 0) {
            console.log(
              "[Runner] Re-fill resolved all blank required fields; proceeding to submit."
            );
          } else {
            console.warn(
              `[Runner] ${stillBlank} required field(s) still blank after re-fill.`
            );
          }
        } catch (error: any) {
          console.warn("[Runner] Re-fill failed:", error?.message || error);
        }
        if (getBlankedRequiredCount() > 0 || getDeferredFieldCount() > 0) {
          // Still incomplete: never submit a half-filled application — abort
          // for the morning digest / resume flow instead.
          console.log(
            `[Runner] ${getDeferredFieldCount()} deferred + ${getBlankedRequiredCount()} required-blank question(s) remain; not submitting. ` +
              "Job stays deferred for the morning digest / resume flow."
          );
          emitStatus({
            jobId: payload.jobId,
            message:
              "Deferred/blank required questions remain; application not submitted.",
            screenshotPath,
            status: "skipped",
          });
          watchdog?.stop();
          rl.close();
          await stagehand.close();
          process.exit(0);
        }
      }
      console.log(
        "[Runner] Auto mode enabled. Submitting application immediately..."
      );

      // Non-LLM consistency gate: when enabled, emit the filled fields first
      // and WAIT for the worker's go/no-go before submitting. The worker runs
      // the persona cross-check and replies submit/skip via the action
      // callback — so a wrong value (e.g. location guessed as "United
      // Kingdom") is caught before the application is sent.
      if (process.env.AUTOFILL_CONSISTENCY_GATE === "1") {
        watchdog?.stop();
        clearInterval(captchaTimer);
        // Gate loop: emit filled fields, await the worker's decision, apply
        // any corrections (re-fill wrong fields), re-read, and re-ask until
        // the worker approves or skips (bounded rounds so a bad field can't
        // loop forever).
        let decision: {
          action: "submit" | "skip" | "correct";
          corrections?: Record<string, string>;
        } = {
          action: "submit",
        };
        for (let round = 0; round < 3; round++) {
          emitStatus({
            filledFields: adapter.filledValues,
            jobId: payload.jobId,
            message:
              round === 0
                ? "Form filled; awaiting consistency gate before submission."
                : `Consistency gate round ${round + 1}: awaiting decision.`,
            screenshotPath,
            status: "awaiting_review",
          });
          decision = await waitForDecision();
          if (decision.action === "submit") {
            break;
          }
          if (decision.action === "skip") {
            console.log(
              "[Runner] Consistency gate declined submission; skipping."
            );
            rl.close();
            await stagehand.close();
            emitStatus({
              jobId: payload.jobId,
              message: "Application skipped by pre-submit consistency gate.",
              screenshotPath,
              status: "skipped",
            });
            process.exit(0);
          }
          // "correct": apply the worker's corrected values and re-read.
          const corrections = decision.corrections ?? {};
          const keys = Object.keys(corrections);
          if (keys.length === 0) {
            console.warn(
              "[Runner] Gate requested corrections but none supplied; skipping."
            );
            rl.close();
            await stagehand.close();
            emitStatus({
              jobId: payload.jobId,
              message: "Application skipped: gate sent no corrections.",
              screenshotPath,
              status: "skipped",
            });
            process.exit(0);
          }
          let allApplied = true;
          for (const [label, value] of Object.entries(corrections)) {
            const ok = await adapter.correctField(label, value);
            if (!ok) {
              allApplied = false;
            }
          }
          if (!allApplied) {
            console.warn(
              "[Runner] One or more corrections could not be applied; skipping rather than submitting wrong data."
            );
            rl.close();
            await stagehand.close();
            emitStatus({
              jobId: payload.jobId,
              message: "Application skipped: corrections could not be applied.",
              screenshotPath,
              status: "skipped",
            });
            process.exit(0);
          }
        }
        if (decision.action !== "submit") {
          console.log("[Runner] Consistency gate did not approve; skipping.");
          rl.close();
          await stagehand.close();
          emitStatus({
            jobId: payload.jobId,
            message: "Application skipped by pre-submit consistency gate.",
            screenshotPath,
            status: "skipped",
          });
          process.exit(0);
        }
        console.log("[Runner] Consistency gate approved; submitting...");
        // Re-arm the watchdog + captcha timer for the actual submit.
        if (activityTimeoutMs > 0) {
          watchdog = new ActivityWatchdog(activityTimeoutMs, () => {
            const err = new Error(
              `STUCK_TIMEOUT: no runner/browser activity for ${Math.round(activityTimeoutMs / 1000)}s`
            );
            console.error(
              "[Runner] Activity watchdog fired; aborting stuck run:",
              err.message
            );
            emitStatus({
              error: err.message,
              jobId: payload.jobId,
              status: "failed",
            });
            watchdog?.stop();
            try {
              rl.close();
            } catch {
              // best-effort cleanup; ignore failures
            }
            const forceExit = setTimeout(() => process.exit(1), 5000);
            Promise.resolve(stagehand.close())
              .catch(() => {})
              .then(() => {
                clearTimeout(forceExit);
                process.exit(1);
              });
          });
          const touch = () => watchdog?.touch();
          const origLog = console.log;
          console.log = (...a) => {
            touch();
            origLog(...a);
          };
          const origError = console.error;
          console.error = (...a) => {
            touch();
            origError(...a);
          };
          watchdog.start();
        }
      }
      // The watchdog stays armed through submit so a stuck submit is also
      // killed, and a captcha blocking the submit button is attempted once and
      // then aborts the run; it disarms once the outcome is reported.
      //
      // Verified-submit retry model (user-specified):
      //   submit attempt 1
      //     -> retryable failure (validation banner / submit button visible)
      //        -> recheck missing fields (ONCE)
      //        -> submit attempt 2
      //           -> retryable failure -> submit attempt 3
      //              -> still failing -> FAILED with the last error banner
      //   Only a CONFIRMATION state (success-page redirect, or the submit form
      //   gone with a success phrase) yields `submitted`.
      const MAX_SUBMIT_ATTEMPTS = 3;
      // ATS server-side spam flag (Ashby "flagged as possible spam"). We do NOT
      // resubmit from the same session (it is sticky to IP+fingerprint); the
      // worker re-queues for a fresh-session relaunch instead.
      const SPAM_FLAG_RE =
        /flagged as possible spam|possible spam|submitted.*spam|spam/i;
      let lastError: string | undefined;
      let rechecked = false;
      for (let attempt = 1; attempt <= MAX_SUBMIT_ATTEMPTS; attempt++) {
        let outcome: any;
        try {
          outcome = await Promise.race([adapter.submit(), captchaAbort]);
        } catch (error: any) {
          if (captchaMessage) {
            clearInterval(captchaTimer);
            throw new Error(
              `CAPTCHA_DETECTED: ${captchaMessage} blocked the application form`,
              {
                cause: error,
              }
            );
          }
          throw error;
        }

        if (outcome?.confirmed) {
          // Confirmation page reached — the application is truly submitted.
          clearInterval(captchaTimer);
          // Capture post-submit evidence: the confirmation page.
          let confirmShot = screenshotPath;
          try {
            const confirmPage = adapter.getActivePage();
            confirmShot = screenshotPath.replace(/\.png$/, "-confirm.png");
            await confirmPage.screenshot({ fullPage: true, path: confirmShot });
            console.log(
              `[Runner] Confirmation screenshot saved to ${confirmShot}`
            );
          } catch {
            console.warn("[Runner] Could not capture confirmation screenshot.");
          }
          emitStatus({
            emailStatus: await softAtsEmailFeedback(payload),
            filledFields: adapter.filledValues,
            jobId: payload.jobId,
            message:
              "Application filled and submitted automatically (confirmation verified).",
            screenshotPath: confirmShot,
            status: "submitted",
          });
          watchdog?.stop();
          rl.close();
          await stagehand.close();
          process.exit(0);
        }

        // Not confirmed.
        lastError = outcome?.error;

        // Spam flag: do NOT do the Overview->back->resubmit dance. Ashby's
        // flag is server-side and sticky to the session (IP + device
        // fingerprint); resubmitting from the same flagged session fails every
        // time and just wastes minutes. Surface the spam error so the worker
        // re-queues the job for a FRESH session (new residential IP + new
        // fingerprint seed) — that is what actually clears the flag.
        if (lastError && SPAM_FLAG_RE.test(lastError)) {
          clearInterval(captchaTimer);
          console.warn(
            `[Runner] ATS flagged submit as possible spam (attempt ${attempt}). ` +
              "Not resubmitting from the same session; worker will relaunch with a fresh IP."
          );
          emitStatus({
            error: `Submit not confirmed: ${lastError}`,
            jobId: payload.jobId,
            screenshotPath,
            status: "failed",
          });
          watchdog?.stop();
          rl.close();
          await stagehand.close();
          process.exit(1);
        }

        // Normal retryable failure (validation): recheck missing fields once.
        if (outcome?.retryable) {
          console.warn(
            `[Runner] Submit attempt ${attempt}/${MAX_SUBMIT_ATTEMPTS} not confirmed: ${lastError ?? "validation likely failed"}`
          );
          if (attempt < MAX_SUBMIT_ATTEMPTS) {
            if (!rechecked) {
              rechecked = true;
              console.log(
                "[Runner] Rechecking missing required fields before retry..."
              );
              try {
                const stillBlank =
                  await adapter.recheckMissingFields(askPythonRpc);
                if (stillBlank > 0) {
                  console.warn(
                    `[Runner] Recheck found ${stillBlank} required field(s) still blank.`
                  );
                }
              } catch (error: any) {
                console.warn(
                  "[Runner] Recheck failed:",
                  error?.message || error
                );
              }
            }
            continue;
          }
        }

        // Exhausted attempts or non-retryable: mark failed with the reason.
        clearInterval(captchaTimer);
        const failMsg =
          lastError ||
          `submit not confirmed after ${MAX_SUBMIT_ATTEMPTS} attempts (no success or error outcome)`;
        console.error(`[Runner] Submission failed: ${failMsg}`);
        emitStatus({
          error: `Submit not confirmed: ${failMsg}`,
          jobId: payload.jobId,
          screenshotPath,
          status: "failed",
        });
        watchdog?.stop();
        rl.close();
        await stagehand.close();
        process.exit(1);
      }
      // Unreachable safety net.
      clearInterval(captchaTimer);
      emitStatus({
        error: "submit loop exited without a terminal outcome",
        jobId: payload.jobId,
        screenshotPath,
        status: "failed",
      });
      watchdog?.stop();
      rl.close();
      await stagehand.close();
      process.exit(1);
    } else {
      // Review mode: Emit awaiting_review and await action from single dispatcher
      watchdog?.stop();
      clearInterval(captchaTimer);
      emitStatus({
        jobId: payload.jobId,
        message: "Form filled successfully. Awaiting review command.",
        screenshotPath,
        status: "awaiting_review",
      });

      console.log(
        "[Runner] Browser window remaining open for review. Waiting for callback on stdin..."
      );

      const action = await new Promise<"submit" | "skip">((resolve) => {
        actionCallbackResolver = (cb) =>
          resolve(cb.action === "submit" ? "submit" : "skip");
      });

      if (action === "submit") {
        console.log("[Runner] Callback 'submit' received. Submitting...");
        let outcome: any;
        try {
          outcome = await Promise.race([adapter.submit(), captchaAbort]);
        } catch (error) {
          if (captchaMessage) {
            clearInterval(captchaTimer);
            throw new Error(
              `CAPTCHA_DETECTED: ${captchaMessage} blocked the application form`,
              {
                cause: error,
              }
            );
          }
          throw error;
        }
        clearInterval(captchaTimer);
        if (!outcome?.confirmed) {
          // The ATS did not reach a confirmation page — never report submitted.
          const failMsg = outcome?.error ?? "submit not confirmed";
          console.error(
            `[Runner] Submit not confirmed after user review: ${failMsg}`
          );
          emitStatus({
            error: `Submit not confirmed: ${failMsg}`,
            jobId: payload.jobId,
            screenshotPath,
            status: "failed",
          });
          rl.close();
          await stagehand.close();
          process.exit(1);
        }
        emitStatus({
          emailStatus: await softAtsEmailFeedback(payload),
          jobId: payload.jobId,
          message:
            "Application submitted after user review (confirmation verified).",
          screenshotPath,
          status: "submitted",
        });
      } else {
        clearInterval(captchaTimer);
        console.log(
          "[Runner] Callback 'skip' received. Skipping submission..."
        );
        emitStatus({
          jobId: payload.jobId,
          message: "Application skipped by user.",
          screenshotPath,
          status: "skipped",
        });
      }

      rl.close();
      await stagehand.close();
      process.exit(0);
    }
  } catch (error: any) {
    watchdog?.stop();
    console.error("[Runner] Execution error:", error);
    emitStatus({
      error: error?.message || String(error),
      jobId: payload.jobId,
      status: "failed",
    });
    try {
      rl.close();
      await stagehand.close();
    } catch {
      // best-effort cleanup; ignore failures
    }
    process.exit(1);
  }
}

main();
