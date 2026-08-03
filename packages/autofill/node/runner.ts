import { Stagehand } from "@browserbasehq/stagehand";
import * as fs from "fs";
import * as path from "path";
import * as readline from "readline";
import { JobPayloadSchema, ActionCallbackSchema, StatusEvent } from "./types";
import { ATSAdapter, RpcHelper } from "./ats/base";
import { GreenhouseAdapter } from "./ats/greenhouse";
import { AshbyAdapter } from "./ats/ashby";
import { LeverAdapter } from "./ats/lever";
import { WorkdayAdapter } from "./ats/workday";
import { GenericAdapter } from "./ats/generic";
import { ActivityWatchdog } from "./utils/activity";
import { applyFingerprint, loadFingerprint } from "./utils/fingerprint";
import {
  getBlankedRequiredCount,
  getDeferredFieldCount,
  resetBlankedRequiredCount,
  resetDeferredFieldCount,
} from "./ats/shared/screener";

// Rejects when a promise has not settled within `ms`. Used so a best-effort
// captcha attempt can never stall the run indefinitely.
function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`operation timed out after ${ms}ms`)), ms);
    promise.then(
      (v) => {
        clearTimeout(timer);
        resolve(v);
      },
      (e) => {
        clearTimeout(timer);
        reject(e);
      },
    );
  });
}

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
    terminal: false,
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
    console.error(
      "[Runner] Missing API key. Set GENERALCOMPUTE_API_KEY or OPENAI_API_KEY in environment.",
    );
    emitStatus({
      jobId: payload.jobId,
      status: "failed",
      error: "Missing API key. Set GENERALCOMPUTE_API_KEY or OPENAI_API_KEY.",
    });
    process.exit(1);
  }

  const modelName = process.env.GENERALCOMPUTE_MODEL || "deepseek-v3.2";
  const genericModel = modelName.includes("/") ? modelName : `openai/${modelName}`;
  if (modelName.includes("/")) {
    console.warn(
      `[Runner] GENERALCOMPUTE_MODEL "${modelName}" includes a provider prefix; using as-is.`,
    );
  }

  console.log(`[Runner] Initializing Stagehand LOCAL environment with model ${genericModel}...`);

  // Per-job browser fingerprint (UA/platform/viewport/cores/memory/languages,
  // India-consistent locale+timezone). Seeded by AUTOFILL_FINGERPRINT_SEED
  // (set per job by the worker) so a batch of applications never presents as
  // a single device — the "many apps from one device" fraud signal.
  const fingerprint = loadFingerprint();

  // Stagehand v3 unified model config: modelClientOptions was removed, and the
  // OpenAI AI SDK defaults custom baseURL endpoints to the Responses API. GeneralCompute
  // exposes an OpenAI-compatible Chat Completions endpoint, so we must opt into chat format.
  const stagehandConfig: any = {
    env: "LOCAL",
    model: {
      modelName: genericModel,
      apiKey: apiKey,
      baseURL: "https://api.generalcompute.com/v1",
      openaiEndpointFormat: "chat",
    },
    // Deterministic act(action) fills must never self-heal via an LLM — a
    // self-heal can silently re-target a DIFFERENT element (and the committed
    // value check would then read the wrong field). The generic adapter's
    // observe fallback relies on exact selector execution.
    selfHeal: false,
    localBrowserLaunchOptions: {
      headless: false,
      args: ["--disable-blink-features=AutomationControlled"],
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
      locale: fingerprint.locale,
      viewport: fingerprint.viewport,
      deviceScaleFactor: fingerprint.deviceScaleFactor,
      // Route the whole browser session through a proxy when AUTOFILL_PROXY is
      // set (either the legacy Tor SOCKS5 proxy, or — with a proxy template —
      // a per-job residential IP URL substituted by the worker). Stagehand maps
      // proxy.server -> --proxy-server=... at launch.
      ...(process.env.AUTOFILL_PROXY ? { proxy: { server: process.env.AUTOFILL_PROXY } } : {}),
    },
  };
  const stagehand = new Stagehand(stagehandConfig);

  // Unified Single Readline Dispatcher for RPC responses and Action Callbacks
  const pendingRpcPromises = new Map<
    string,
    { resolve: (val: any) => void; reject: (err: any) => void; timer: NodeJS.Timeout }
  >();
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
    } catch {}
  });

  const rpcTimeoutMs = parseInt(process.env.AUTOFILL_RPC_TIMEOUT_MS || "1800000", 10);

  const askPythonRpc: RpcHelper = (method: string, args: Record<string, any>): Promise<any> => {
    return new Promise((resolve, reject) => {
      const id = `rpc-${Math.random().toString(36).substring(2, 9)}`;

      const timer = setTimeout(() => {
        pendingRpcPromises.delete(id);
        reject(
          new Error(`RPC timeout for method "${method}" after ${rpcTimeoutMs / 1000} seconds`),
        );
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
        `STUCK_TIMEOUT: no runner/browser activity for ${Math.round(activityTimeoutMs / 1000)}s`,
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
      } catch {}
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
    const captchaWatchMs = parseInt(process.env.AUTOFILL_CAPTCHA_WATCH_MS || "8000", 10);
    const captchaAttemptTimeoutMs = parseInt(
      process.env.AUTOFILL_CAPTCHA_ATTEMPT_TIMEOUT_MS || "20000",
      10,
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
      if (captchaMessage) return;
      try {
        const hit = await adapter.detectCaptcha();
        if (!hit) {
          // Nothing blocking now (or a previous attempt cleared it).
          captchaAttempted = false;
          return;
        }
        // A concurrent tick is already attempting; wait for it to settle.
        if (captchaAttempted) return;
        // First sighting of a challenge: try to solve it once before declaring
        // failure. The fill keeps running in parallel; only a challenge that
        // survives the attempt aborts the run.
        captchaAttempted = true;
        try {
          const attempted = await withTimeout(adapter.attemptCaptcha(), captchaAttemptTimeoutMs);
          console.log(`[Runner] Captcha attempt (clicked: ${attempted}); re-checking...`);
        } catch (attemptErr: any) {
          console.warn(
            "[Runner] Captcha attempt failed:",
            attemptErr?.message || String(attemptErr),
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
        rejectCaptcha?.(new Error(`CAPTCHA_DETECTED: ${after} blocked the application form`));
      } catch {}
    }, captchaWatchMs);

    // Execute filling process with RPC helper
    resetDeferredFieldCount();
    resetBlankedRequiredCount();
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
        } catch {}
        process.exit(1);
      }
      throw fillErr;
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
      clearInterval(captchaTimer);
      console.log(
        "[Runner] Submission disabled — browser window remaining open for review. " +
          "Any action (or the hold timeout) closes without submitting.",
      );
      emitStatus({
        jobId: payload.jobId,
        status: "awaiting_review",
        screenshotPath: screenshotPath,
        message: "Form filled successfully. Submission is disabled in this phase.",
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
        message: "Application filled but not submitted (submission disabled).",
      });
      process.exit(0);
    }

    if (payload.mode === "auto") {
      if (getDeferredFieldCount() > 0 || getBlankedRequiredCount() > 0) {
        // A question was deferred (overnight, no human) or a required field
        // failed to commit. Never submit an incomplete application — abort
        // for the morning digest / resume flow instead. The worker keeps the
        // job retryable (skipped/deferred), never submitted.
        console.log(
          `[Runner] ${getDeferredFieldCount()} deferred + ${getBlankedRequiredCount()} required-blank question(s) remain; not submitting. ` +
            "Job stays deferred for the morning digest / resume flow.",
        );
        emitStatus({
          jobId: payload.jobId,
          status: "skipped",
          screenshotPath: screenshotPath,
          message: "Deferred/blank required questions remain; application not submitted.",
        });
        watchdog?.stop();
        rl.close();
        await stagehand.close();
        process.exit(0);
      }
      console.log("[Runner] Auto mode enabled. Submitting application immediately...");
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
      // Ashby flags a submission as possible spam and offers "submit again". A
      // human clicks the posting's Overview, returns to the application form
      // (fields preserved), and submits again. `retryAfterSpamFlag` does that
      // navigation; up to MAX_SPAM_RETRIES such round-trips are attempted
      // before giving up. The normal field-recheck path is only for client-side
      // validation failures (submit button still visible), not spam flags.
      const SPAM_FLAG_RE = /flagged as possible spam|possible spam|submitted.*spam|spam/i;
      const MAX_SPAM_RETRIES = 2;
      let spamRetries = 0;
      let lastError: string | undefined;
      let rechecked = false;
      for (let attempt = 1; attempt <= MAX_SUBMIT_ATTEMPTS; attempt++) {
        let outcome: any;
        try {
          outcome = await Promise.race([adapter.submit(), captchaAbort]);
        } catch (submitErr: any) {
          if (captchaMessage) {
            clearInterval(captchaTimer);
            throw new Error(`CAPTCHA_DETECTED: ${captchaMessage} blocked the application form`);
          }
          throw submitErr;
        }

        if (outcome?.confirmed) {
          // Confirmation page reached — the application is truly submitted.
          clearInterval(captchaTimer);
          // Capture post-submit evidence: the confirmation page.
          let confirmShot = screenshotPath;
          try {
            const activePage = adapter.getActivePage();
            confirmShot = screenshotPath.replace(/\.png$/, "-confirm.png");
            await activePage.screenshot({ path: confirmShot, fullPage: true });
            console.log(`[Runner] Confirmation screenshot saved to ${confirmShot}`);
          } catch {
            console.warn("[Runner] Could not capture confirmation screenshot.");
          }
          emitStatus({
            jobId: payload.jobId,
            status: "submitted",
            screenshotPath: confirmShot,
            message: "Application filled and submitted automatically (confirmation verified).",
          });
          watchdog?.stop();
          rl.close();
          await stagehand.close();
          process.exit(0);
        }

        // Not confirmed.
        lastError = outcome?.error;

        // Spam flag: navigate Overview -> back to the form -> resubmit without
        // touching any field. Takes priority over the field-recheck path.
        if (lastError && SPAM_FLAG_RE.test(lastError) && spamRetries < MAX_SPAM_RETRIES) {
          spamRetries += 1;
          console.warn(
            `[Runner] Ashby flagged the submit as possible spam (attempt ${attempt}); ` +
              "navigating to Overview and back, then resubmitting...",
          );
          try {
            const back = await adapter.retryAfterSpamFlag();
            if (!back) {
              console.warn("[Runner] Could not return to the application form after spam flag.");
            }
          } catch (spamErr: any) {
            console.warn(
              "[Runner] Spam-flag retry navigation failed:",
              spamErr?.message || spamErr,
            );
          }
          // Resubmit on the next loop iteration, even past the normal attempt cap
          // (spam retries are a distinct budget).
          if (attempt >= MAX_SUBMIT_ATTEMPTS && spamRetries < MAX_SPAM_RETRIES) {
            attempt = MAX_SUBMIT_ATTEMPTS - 1;
          }
          continue;
        }

        // Normal retryable failure (validation): recheck missing fields once.
        if (outcome?.retryable) {
          console.warn(
            `[Runner] Submit attempt ${attempt}/${MAX_SUBMIT_ATTEMPTS} not confirmed: ${lastError ?? "validation likely failed"}`,
          );
          if (attempt < MAX_SUBMIT_ATTEMPTS) {
            if (!rechecked) {
              rechecked = true;
              console.log("[Runner] Rechecking missing required fields before retry...");
              try {
                const stillBlank = await adapter.recheckMissingFields(askPythonRpc);
                if (stillBlank > 0) {
                  console.warn(
                    `[Runner] Recheck found ${stillBlank} required field(s) still blank.`,
                  );
                }
              } catch (recheckErr: any) {
                console.warn("[Runner] Recheck failed:", recheckErr?.message || recheckErr);
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
          jobId: payload.jobId,
          status: "failed",
          screenshotPath: screenshotPath,
          error: `Submit not confirmed: ${failMsg}`,
        });
        watchdog?.stop();
        rl.close();
        await stagehand.close();
        process.exit(1);
      }
      // Unreachable safety net.
      clearInterval(captchaTimer);
      emitStatus({
        jobId: payload.jobId,
        status: "failed",
        screenshotPath: screenshotPath,
        error: "submit loop exited without a terminal outcome",
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
        status: "awaiting_review",
        screenshotPath: screenshotPath,
        message: "Form filled successfully. Awaiting review command.",
      });

      console.log(
        "[Runner] Browser window remaining open for review. Waiting for callback on stdin...",
      );

      const action = await new Promise<"submit" | "skip">((resolve) => {
        actionCallbackResolver = resolve;
      });

      if (action === "submit") {
        console.log("[Runner] Callback 'submit' received. Submitting...");
        let outcome: any;
        try {
          outcome = await Promise.race([adapter.submit(), captchaAbort]);
        } catch (submitErr) {
          if (captchaMessage) {
            clearInterval(captchaTimer);
            throw new Error(`CAPTCHA_DETECTED: ${captchaMessage} blocked the application form`);
          }
          throw submitErr;
        }
        clearInterval(captchaTimer);
        if (!outcome?.confirmed) {
          // The ATS did not reach a confirmation page — never report submitted.
          const failMsg = outcome?.error ?? "submit not confirmed";
          console.error(`[Runner] Submit not confirmed after user review: ${failMsg}`);
          emitStatus({
            jobId: payload.jobId,
            status: "failed",
            screenshotPath: screenshotPath,
            error: `Submit not confirmed: ${failMsg}`,
          });
          rl.close();
          await stagehand.close();
          process.exit(1);
        }
        emitStatus({
          jobId: payload.jobId,
          status: "submitted",
          screenshotPath: screenshotPath,
          message: "Application submitted after user review (confirmation verified).",
        });
      } else {
        clearInterval(captchaTimer);
        console.log("[Runner] Callback 'skip' received. Skipping submission...");
        emitStatus({
          jobId: payload.jobId,
          status: "skipped",
          screenshotPath: screenshotPath,
          message: "Application skipped by user.",
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
      error: err?.message || String(err),
    });
    try {
      rl.close();
      await stagehand.close();
    } catch {}
    process.exit(1);
  }
}

main();
