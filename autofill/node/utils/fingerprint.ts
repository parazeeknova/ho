import { Stagehand } from "@browserbasehq/stagehand";

/**
 * Per-job browser fingerprint randomization.
 *
 * Ashby's ATS-integrated fraud detection flags "multiple applications from
 * one device": every run launched with Stagehand's defaults shares the same
 * UA/platform/viewport, so a batch of applications fingerprints as one
 * device. This module derives a STABLE random fingerprint from a per-job
 * seed (AUTOFILL_FINGERPRINT_SEED set by the worker, e.g. the proxy session
 * id) and applies it:
 *   - at launch: locale (--lang), viewport (--window-size), deviceScaleFactor
 *     via localBrowserLaunchOptions,
 *   - post-init via CDP on the page's own session:
 *     Emulation.setUserAgentOverride (UA + platform + accept-language) and
 *     Emulation.setTimezoneOverride,
 *   - via a document-start init script: navigator.webdriver=false, and the
 *     randomized hardwareConcurrency / deviceMemory / languages getters.
 *
 * Locale, timezone and languages stay India-consistent (en-IN, Asia/Kolkata)
 * so the browser profile never contradicts the Indian residential egress IP
 * or the persona's stated location.
 */

export interface BrowserFingerprint {
  seed: string;
  userAgent: string;
  platform: string;
  acceptLanguage: string;
  locale: string;
  timezoneId: string;
  viewport: { width: number; height: number };
  deviceScaleFactor: number;
  hardwareConcurrency: number;
  deviceMemory: number;
  languages: string[];
}

/** Deterministic PRNG (mulberry32) so a given seed always yields the same fingerprint. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function hashString(str: string): number {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function pick<T>(rng: () => number, arr: readonly T[]): T {
  return arr[Math.floor(rng() * arr.length)];
}

const VIEWPORTS: readonly { width: number; height: number }[] = [
  { width: 1920, height: 1080 },
  { width: 1536, height: 864 },
  { width: 1440, height: 900 },
  { width: 1366, height: 768 },
  { width: 1280, height: 800 },
  { width: 1600, height: 900 },
  { width: 1280, height: 720 },
  { width: 1680, height: 1050 },
];

const MAC_VERSIONS = ["10_15_7", "11_7_10", "12_7_6", "13_6_9", "14_7_1"];
const CHROME_MAJORS = [124, 125, 126, 127, 128, 129, 130, 131];
const CHROME_BUILDS = [0, 4334, 5129, 5989];
const CHROME_PATCHES = [0, 88, 99, 122];

const LOCALE = "en-IN";
const TIMEZONE = "Asia/Kolkata";
const LANGUAGE_SETS: readonly string[][] = [
  ["en-IN", "en-US", "hi"],
  ["en-IN", "hi", "en-US"],
  ["en-IN", "en-GB", "hi"],
  ["en-IN", "hi-IN", "en-US"],
];

function buildUserAgent(rng: () => number): { userAgent: string; platform: string } {
  const version = `${pick(rng, CHROME_MAJORS)}.0.${pick(rng, CHROME_BUILDS)}.${pick(rng, CHROME_PATCHES)}`;
  const os = pick(rng, ["windows", "macos", "linux"] as const);
  if (os === "windows") {
    return {
      userAgent: `Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/${version} Safari/537.36`,
      platform: "Win32",
    };
  }
  if (os === "macos") {
    return {
      userAgent: `Mozilla/5.0 (Macintosh; Intel Mac OS X ${pick(rng, MAC_VERSIONS)}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/${version} Safari/537.36`,
      platform: "MacIntel",
    };
  }
  return {
    userAgent: `Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/${version} Safari/537.36`,
    platform: "Linux x86_64",
  };
}

export function loadFingerprint(seedOverride?: string): BrowserFingerprint {
  const raw = seedOverride ?? process.env.AUTOFILL_FINGERPRINT_SEED;
  const seed = raw && raw.trim() ? raw.trim() : `${Math.random().toString(36).slice(2)}${Date.now().toString(36)}`;
  const rng = mulberry32(hashString(seed));
  const { userAgent, platform } = buildUserAgent(rng);
  const languages = pick(rng, LANGUAGE_SETS);
  const fp: BrowserFingerprint = {
    seed,
    userAgent,
    platform,
    acceptLanguage: `en-IN,${languages[1] || "en-US"};q=0.9,hi;q=0.8,en;q=0.7`,
    locale: LOCALE,
    timezoneId: TIMEZONE,
    viewport: pick(rng, VIEWPORTS),
    deviceScaleFactor: platform === "MacIntel" && rng() < 0.35 ? 2 : 1,
    hardwareConcurrency: pick(rng, [4, 8, 8, 12, 16]),
    deviceMemory: pick(rng, [4, 8, 8, 16]),
    languages,
  };
  // Hard overrides for human-in-the-loop runs (e.g. the user manually submits
  // a filled form, so the window must render at a NORMAL scale and size — a
  // randomized dpr=2 / odd viewport would make the submit button look wrong).
  // AUTOFILL_DEVICE_SCALE_FACTOR ("1") and AUTOFILL_VIEWPORT ("1440x900")
  // win over the seeded values when set.
  const dsf = process.env.AUTOFILL_DEVICE_SCALE_FACTOR;
  if (dsf && /^\d+(\.\d+)?$/.test(dsf.trim())) {
    fp.deviceScaleFactor = parseFloat(dsf.trim());
  }
  const vp = process.env.AUTOFILL_VIEWPORT;
  if (vp) {
    const m = vp.trim().match(/^(\d{3,5})\s*[xX]\s*(\d{3,5})$/);
    if (m) {
      const w = parseInt(m[1], 10);
      const h = parseInt(m[2], 10);
      if (w >= 800 && w <= 2560 && h >= 600 && h <= 1600) {
        fp.viewport = { width: w, height: h };
      }
    }
  }
  return fp;
}

/**
 * Document-start script installing the property getters that CDP Emulation
 * cannot reach: navigator.webdriver suppression plus the randomized
 * hardwareConcurrency / deviceMemory / languages / language. Property getters
 * are installed on Navigator.prototype so every navigator instance (and every
 * future navigation) reports the randomized values.
 */
function initScriptSource(fp: BrowserFingerprint): string {
  return `(() => {
  const langs = ${JSON.stringify(fp.languages)};
  const def = Object.defineProperty.bind(Object);
  try {
    def(Navigator.prototype, 'webdriver', { get: () => undefined, configurable: true });
    def(Navigator.prototype, 'hardwareConcurrency', { get: () => ${fp.hardwareConcurrency}, configurable: true });
    def(Navigator.prototype, 'deviceMemory', { get: () => ${fp.deviceMemory}, configurable: true });
    def(Navigator.prototype, 'languages', { get: () => langs.slice(), configurable: true });
    def(Navigator.prototype, 'language', { get: () => langs[0], configurable: true });
  } catch (e) {}
})();`;
}

/**
 * Apply the fingerprint to the running Stagehand browser. Safe to call right
 * after `stagehand.init()` and before the adapter navigates: the CDP
 * Emulation overrides persist for the page's session and the init script is
 * re-run on every document start. Every failure is logged and swallowed so
 * fingerprinting can never abort a fill.
 */
export async function applyFingerprint(stagehand: any, fp: BrowserFingerprint): Promise<void> {
  let page: any = stagehand?.context?.pages?.()[0];
  if (!page && typeof stagehand?.context?.newPage === "function") {
    page = await stagehand.context.newPage();
  }
  if (!page) {
    console.warn("[Fingerprint] No page available to apply fingerprint.");
    return;
  }

  // Stagehand v3's understudy Page exposes its top-level CDP session here.
  const session = page?.mainSession ?? page?.session ?? null;
  if (session && typeof session.send === "function") {
    try {
      await session.send("Emulation.setUserAgentOverride", {
        userAgent: fp.userAgent,
        platform: fp.platform,
        acceptLanguage: fp.acceptLanguage,
      });
      await session.send("Emulation.setTimezoneOverride", {
        timezoneId: fp.timezoneId,
      });
      console.log(
        `[Fingerprint] seed=${fp.seed} platform=${fp.platform} tz=${fp.timezoneId} ` +
          `viewport=${fp.viewport.width}x${fp.viewport.height} dpr=${fp.deviceScaleFactor} ` +
          `cores=${fp.hardwareConcurrency} mem=${fp.deviceMemory} langs=${fp.languages.join(",")}`
      );
    } catch (err: any) {
      console.warn("[Fingerprint] CDP emulation failed (continuing):", err?.message || err);
    }
  } else {
    console.warn("[Fingerprint] No CDP session on the initial page.");
  }

  if (typeof page.registerInitScript === "function") {
    try {
      await page.registerInitScript(initScriptSource(fp));
    } catch (err: any) {
      console.warn("[Fingerprint] init script failed (continuing):", err?.message || err);
    }
  }
}
