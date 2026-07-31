import { Stagehand } from "@browserbasehq/stagehand";
import * as fs from "fs";
import * as path from "path";
import * as readline from "readline";
import { JobPayloadSchema, ActionCallbackSchema, StatusEvent } from "./types.js";
import { ATSAdapter } from "./ats/base.js";
import { GreenhouseAdapter } from "./ats/greenhouse.js";

// Extensible ATS Adapter Registry Pattern
interface AdapterRegistration {
  pattern: RegExp;
  factory: (stagehand: Stagehand) => ATSAdapter;
}

const adapterRegistry: AdapterRegistration[] = [
  { pattern: /greenhouse\.io/, factory: (s) => new GreenhouseAdapter(s) },
];

function getAdapterForUrl(url: string, stagehand: Stagehand): ATSAdapter {
  const entry = adapterRegistry.find((reg) => reg.pattern.test(url));
  if (!entry) {
    throw new Error(`Unsupported ATS platform for URL: ${url}`);
  }
  return entry.factory(stagehand);
}

function emitStatus(statusEvent: StatusEvent) {
  // Output JSON status message on stdout for Python to parse
  console.log(`STATUS_EVENT:${JSON.stringify(statusEvent)}`);
}

async function readInitialPayload(): Promise<any> {
  return new Promise((resolve, reject) => {
    const rl = readline.createInterface({
      input: process.stdin,
      terminal: false
    });
    rl.once("line", (line) => {
      try {
        const parsed = JSON.parse(line.trim());
        rl.close();
        resolve(parsed);
      } catch (err) {
        rl.close();
        reject(err);
      }
    });
  });
}

async function main() {
  let payloadRaw;
  try {
    payloadRaw = await readInitialPayload();
  } catch (err) {
    console.error("[Runner] Error reading or parsing initial JSON payload from stdin:", err);
    process.exit(1);
  }

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

  process.env.OPENAI_API_KEY = apiKey;
  process.env.OPENAI_BASE_URL = "https://api.generalcompute.com/v1";
  
  const modelName = process.env.GENERALCOMPUTE_MODEL || "deepseek-v3.2";
  const stagehandModel = modelName.includes("/") ? modelName : `openai/${modelName}`;

  console.log(`[Runner] Initializing Stagehand LOCAL environment with model ${stagehandModel}...`);

  const stagehand = new Stagehand({
    env: "LOCAL",
    model: stagehandModel,
    modelClientOptions: {
      apiKey: apiKey,
      baseURL: "https://api.generalcompute.com/v1"
    },
    localBrowserLaunchOptions: {
      headless: false,
      args: ["--disable-blink-features=AutomationControlled"]
    }
  } as any);

  try {
    await stagehand.init();
    console.log("[Runner] Stagehand initialized successfully.");

    const adapter = getAdapterForUrl(payload.url, stagehand);

    // Execute filling process
    await adapter.fill(payload);

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
    const activePage = pages[0];
    await activePage.screenshot({ path: screenshotPath, fullPage: true });
    console.log(`[Runner] Screenshot saved to ${screenshotPath}`);

    if (payload.mode === "auto") {
      console.log("[Runner] Auto mode enabled. Submitting application immediately...");
      await adapter.submit();
      emitStatus({
        jobId: payload.jobId,
        status: "submitted",
        screenshotPath: screenshotPath,
        message: "Application filled and submitted automatically."
      });
      await stagehand.close();
      process.exit(0);
    } else {
      // Review mode: Emit awaiting_review and listen on stdin for submit/skip command
      emitStatus({
        jobId: payload.jobId,
        status: "awaiting_review",
        screenshotPath: screenshotPath,
        message: "Form filled successfully. Awaiting review command."
      });

      console.log("[Runner] Browser window remaining open for review. Waiting for callback on stdin...");
      
      const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
        terminal: false
      });

      rl.on("line", async (line) => {
        try {
          const callbackRaw = JSON.parse(line.trim());
          const callbackParse = ActionCallbackSchema.safeParse(callbackRaw);
          if (callbackParse.success) {
            const { action } = callbackParse.data;
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
          }
        } catch (e) {
          console.error("[Runner] Error processing stdin callback:", e);
        } finally {
          rl.close();
          await stagehand.close();
          process.exit(0);
        }
      });
    }
  } catch (err: any) {
    console.error("[Runner] Execution error:", err);
    emitStatus({
      jobId: payload.jobId,
      status: "failed",
      error: err?.message || String(err)
    });
    try {
      await stagehand.close();
    } catch (_) {}
    process.exit(1);
  }
}

main();
