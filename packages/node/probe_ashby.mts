/**
 * Manual anti-bot probe: runs the SAME browser stack the runner uses
 * (Stagehand LOCAL + fingerprint) against an Ashby posting, then dumps:
 *   - the page's detection surface (webdriver, plugins, CDP leaks, UA)
 *   - what requests Ashby's page makes that could signal automation
 *   - the submit/apply button state
 *
 * Run: npx tsx probe_ashby.mjs <posting-url>
 */
import { Stagehand } from "@browserbasehq/stagehand";
import { loadFingerprint, applyFingerprint } from "./utils/fingerprint";

const url = process.argv[2] || "https://jobs.ashbyhq.com/replit/2fa2d079";

const fingerprint = loadFingerprint();
const stagehand = new Stagehand({
  env: "LOCAL",
  selfHeal: false,
  model: {
    modelName: "openai/deepseek-v3.2",
    apiKey: process.env.GENERALCOMPUTE_API_KEY || process.env.OPENAI_API_KEY,
    baseURL: "https://api.generalcompute.com/v1",
    openaiEndpointFormat: "chat",
  },
  localBrowserLaunchOptions: {
    // Headless when PROBE_HEADLESS=1 (CI / no display); the real worker runs
    // headful on the user's machine.
    headless: process.env.PROBE_HEADLESS === "1",
    executablePath: process.env.PROBE_CHROME || undefined,
    args: ["--disable-blink-features=AutomationControlled"],
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
    locale: fingerprint.locale,
    viewport: fingerprint.viewport,
    deviceScaleFactor: fingerprint.deviceScaleFactor,
  },
});

await stagehand.init();
await applyFingerprint(stagehand, fingerprint);

const page = stagehand.context.pages()[0];
const consoleLog: string[] = [];

page.on("console", (m: any) => {
  const t = (m.text && m.text()) || "";
  if (/error|warn|spam|automation|bot|block/i.test(t)) consoleLog.push(t.slice(0, 160));
});

console.log("[Probe] Navigating to", url);
await page.goto(url, { waitUntil: "domcontentloaded" });
await new Promise((r) => setTimeout(r, 4000));

// Anti-bot surface
const surface = await page.evaluate(() => {
  const out: Record<string, any> = {
    webdriver: navigator.webdriver,
    userAgent: navigator.userAgent,
    platform: navigator.platform,
    languages: navigator.languages,
    hardwareConcurrency: navigator.hardwareConcurrency,
    deviceMemory: (navigator as any).deviceMemory,
    plugins: Array.from(navigator.plugins).map((p) => p.name),
    chrome: !!(window as any).chrome,
    hasCdp: typeof (window as any).cdp !== "undefined",
  };
  return out;
});

// Network requests via the Resource Timing API (works headless, no CDP hook).
const resources = await page.evaluate(() =>
  (performance.getEntriesByType("resource") as PerformanceResourceTiming[])
    .map((e) => e.name)
    .filter((u) => /ashby|segment|posthog|heap|amplitude|clarity|sentry|fingerprint|fingerprin/i.test(u))
    .slice(0, 60),
);

// Apply / submit button presence
const applyState = await page.evaluate(() => {
  const btns = Array.from(document.querySelectorAll("button, a[href*='/application']"));
  const text = btns.map((b) => ((b as HTMLElement).textContent || "").trim()).filter(Boolean);
  return { buttons: text.slice(0, 15), url: location.href };
});

console.log("\n=== ANTI-BOT SURFACE ===");
console.log(JSON.stringify(surface, null, 2));
console.log("\n=== APPLY/SUBMIT STATE ===");
console.log(JSON.stringify(applyState, null, 2));
console.log("\n=== NETWORK RESOURCES (tracking/analytics/anti-bot) ===");
console.log((resources || []).join("\n"));
console.log("\n=== CONSOLE (spam/error/bot) ===");
console.log((consoleLog || []).slice(0, 20).join("\n"));

await stagehand.close();
process.exit(0);
