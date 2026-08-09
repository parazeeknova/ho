import Steel from "steel-sdk";

import type { BrowserFingerprint } from "./fingerprint";

export interface SteelSessionHandle {
  client: Steel;
  sessionId: string;
  websocketUrl: string;
  /** The per-job proxy URL handed to Steel, if any (for logging). */
  proxyUrl?: string;
}

/**
 * Create a local Steel browser session and return the CDP websocket the
 * runner's Stagehand instance should attach to via `localBrowserLaunchOptions.
 * cdpUrl`. Returns null when Steel is not configured (STEEL_BASE_URL unset) or
 * a session could not be created — callers then fall back to the direct
 * chrome-launcher path.
 *
 * The session is configured from the per-job fingerprint: viewport + UA come
 * from the seeded fingerprint so a Steel-backed run presents the same device
 * variety as a direct launch. The optional proxy URL (residential template or
 * Tor relay) is handed to Steel natively — Steel accepts credentials in the
 * URL, unlike Chrome, so the per-job residential creds can go straight in.
 *
 * A session that is never released self-expires after `timeout` ms (set long
 * enough to cover the longest fill+submit; the runner releases it explicitly
 * on every close path via the patched stagehand.close).
 */
export async function createSteelSession(
  fingerprint: BrowserFingerprint,
  opts?: { proxyUrl?: string },
): Promise<SteelSessionHandle | null> {
  const baseURL = (process.env.STEEL_BASE_URL || "").trim();
  if (!baseURL) return null;

  let client: Steel;
  try {
    client = new Steel({ baseURL });
  } catch (err) {
    console.warn("[Steel] Client init failed (falling back to direct launch):", err);
    return null;
  }

  try {
    const session = await client.sessions.create({
      // Steel's browser-level viewport from the seeded fingerprint.
      dimensions: { width: fingerprint.viewport.width, height: fingerprint.viewport.height },
      userAgent: fingerprint.userAgent,
      proxyUrl: opts?.proxyUrl,
      // Long enough to cover a full fill+submit; released early on clean close.
      timeout: 60 * 60 * 1000,
    });
    if (!session || !session.websocketUrl) {
      console.warn(
        "[Steel] Session created without a websocket URL; falling back to direct launch.",
      );
      return null;
    }
    // Steel's self-hosted API reports the websocket bound to 0.0.0.0 (or the
    // container's own host), which a client on the runner host cannot reach.
    // Rebuild the URL against the configured STEEL_BASE_URL so the runner
    // attaches to the reachable endpoint. The path/query (a proxied CDP
    // endpoint under Steel) is preserved.
    let wsUrl = session.websocketUrl;
    try {
      const parsed = new URL(wsUrl);
      const base = new URL(baseURL);
      parsed.protocol = base.protocol === "https:" ? "wss:" : "ws:";
      parsed.host = base.host;
      wsUrl = parsed.toString();
    } catch {
      // Unparseable websocket URL — fall back to the raw value.
    }
    const handle: SteelSessionHandle = {
      client,
      sessionId: session.id,
      websocketUrl: wsUrl,
      proxyUrl: opts?.proxyUrl,
    };
    console.log(
      `[Steel] Session ${handle.sessionId} ready (viewport ${fingerprint.viewport.width}x` +
        `${fingerprint.viewport.height}, proxy: ${opts?.proxyUrl ? "yes" : "none"}).`,
    );
    return handle;
  } catch (err) {
    console.warn("[Steel] Session create failed (falling back to direct launch):", err);
    return null;
  }
}

/**
 * Best-effort release of a Steel session. Steel browsers are ephemeral and
 * self-expire, but releasing immediately frees the browser process and the
 * per-job residential IP so the next job can reuse it. Never throws.
 */
export async function releaseSteelSession(handle: SteelSessionHandle | null): Promise<void> {
  if (!handle) return;
  try {
    await handle.client.sessions.release(handle.sessionId);
    console.log(`[Steel] Session ${handle.sessionId} released.`);
  } catch (err) {
    console.warn(`[Steel] Session ${handle.sessionId} release failed (will self-expire):`, err);
  }
}
