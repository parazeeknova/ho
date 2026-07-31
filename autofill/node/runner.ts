import { Stagehand } from "@browserbasehq/stagehand";
import * as fs from "fs";
import * as path from "path";
import * as readline from "readline";
import { JobPayloadSchema, ActionCallbackSchema, StatusEvent } from "./types.js";
import { ATSAdapter, RpcHelper } from "./ats/base.js";
import { GreenhouseAdapter } from "./ats/greenhouse.js";

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

  process.env.OPENAI_API_KEY = apiKey;
  process.env.OPENAI_BASE_URL = "https://api.generalcompute.com/v1";

  const modelName = process.env.GENERALCOMPUTE_MODEL || "deepseek-v3.2";
  const stagehandModel = modelName.includes("/") ? modelName : `openai/${modelName}`;

  console.log(`[Runner] Initializing Stagehand LOCAL environment with model ${stagehandModel}...`);

  const stagehandConfig: any = {
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

  const askPythonRpc: RpcHelper = (method: string, args: Record<string, any>): Promise<any> => {
    return new Promise((resolve, reject) => {
      const id = `rpc-${Math.random().toString(36).substring(2, 9)}`;
      
      const timer = setTimeout(() => {
        pendingRpcPromises.delete(id);
        reject(new Error(`RPC timeout for method "${method}" after 30 seconds`));
      }, 30000);

      pendingRpcPromises.set(id, { resolve, reject, timer });
      console.log(`RPC_REQUEST:${JSON.stringify({ id, method, args })}`);
    });
  };

  try {
    await stagehand.init();
    console.log("[Runner] Stagehand initialized successfully.");

    const adapter = getAdapterForUrl(payload.url, stagehand);

    // Execute filling process with RPC helper
    await adapter.fill(payload, askPythonRpc);

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
      rl.close();
      await stagehand.close();
      process.exit(0);
    } else {
      // Review mode: Emit awaiting_review and await action from single dispatcher
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
