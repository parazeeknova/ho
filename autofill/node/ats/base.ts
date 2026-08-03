import { Stagehand } from "@browserbasehq/stagehand";
import { JobPayload } from "../types.js";
import type { SubmitOutcome } from "./shared/audit.js";

export type RpcHelper = (method: string, args: Record<string, any>) => Promise<any>;

export abstract class ATSAdapter {
  protected stagehand: Stagehand;

  constructor(stagehand: Stagehand) {
    this.stagehand = stagehand;
  }

  /** The page to screenshot after fill. Defaults to the first context page;
   *  adapters that switch tabs (e.g. a form opened in a new tab) override. */
  getActivePage(): any {
    return this.stagehand.context.pages()[0];
  }

  /**
   * Detect an anti-bot challenge blocking the current page. Returns a
   * human-readable description when a VISIBLE captcha widget or challenge
   * interstitial is present, else null. Only visible elements count — an
   * invisible recaptcha v3 badge that every site loads must not false-positive.
   */
  async detectCaptcha(): Promise<string | null> {
    let page: any;
    try {
      page = this.getActivePage();
      if (!page) return null;
    } catch {
      return null;
    }
    try {
      // WARNING: only anonymous arrows here (tsx keepNames wraps inferred-name
      // arrows in __name(), which throws in page context). Destructure so the
      // visibility helpers never gain a name.
      const hit = (await page.evaluate(() => {
        // A captcha only BLOCKS a form when it is a large interactive
        // challenge widget that is actually on screen. The invisible reCAPTCHA
        // v3 badge (~256x60, bottom-right) and auto-solving Turnstile/checkbox
        // frames (~300x65) load on nearly every page. And many forms embed a
        // recaptcha frame that is hidden by an ancestor or parked off-screen —
        // it still reports layout size, so a bare size check false-positives.
        // Only count a frame that is big AND within the viewport AND not
        // hidden by itself or an ancestor.
        const blocking = (void 0, (el: Element): boolean => {
          const r = el.getBoundingClientRect();
          if (r.width < 200 || r.height < 100) return false;
          // Must intersect the visible viewport.
          if (r.bottom < 0 || r.right < 0 || r.top > window.innerHeight || r.left > window.innerWidth) {
            return false;
          }
          let node: Element | null = el;
          while (node) {
            const st = window.getComputedStyle(node);
            if (st.display === "none" || st.visibility === "hidden") return false;
            if (node === document.body) break;
            node = node.parentElement;
          }
          return true;
        });
        for (const fr of Array.from(
          document.querySelectorAll(
            'iframe[src*="recaptcha"], iframe[src*="hcaptcha"], ' +
              'iframe[src*="turnstile"], iframe[src*="challenges.cloudflare.com"]'
          )
        )) {
          if (blocking(fr)) return "captcha challenge iframe";
        }
        for (const el of Array.from(
          document.querySelectorAll(
            ".g-recaptcha, .h-captcha, .cf-turnstile, " +
              "#challenge-stage, .cf-challenge, [class*='challenge-error']"
          )
        )) {
          if (blocking(el)) return "captcha challenge widget";
        }
        // FunCaptcha / puzzle-style challenges are rendered inline (not in a
        // recaptcha/turnstile iframe) with a distinctive prompt. Only a LARGE
        // visible widget counts — never the invisible v3 badge. Match the
        // prompt text so legitimate form instructions can't false-positive.
        for (const el of Array.from(document.querySelectorAll("div, section, form, iframe"))) {
          if (!blocking(el)) continue;
          const text = (el.textContent || "").replace(/\s+/g, " ").trim().slice(0, 300);
          if (
            /place the correct (animal|shape|object|image|character)/i.test(text) ||
            /complete the pattern/i.test(text) ||
            /select all (images|pictures|squares|photos) (that contain|with)/i.test(text) ||
            /enter the text (you see|shown)|type the (characters|letters|text) you see/i.test(text)
          ) {
            return "fun captcha challenge";
          }
        }
        const body = (document.body?.textContent || "").slice(0, 4000);
        if (/verify you are human|attention required|checking your browser before accessing/i.test(body)) {
          return "challenge interstitial page";
        }
        return null;
      })) as string | null;
      return hit;
    } catch {
      return null;
    }
  }

  /**
   * Best-effort attempt to solve a detected captcha once: clicks the visible
   * checkbox / submit control inside a matching frame. Returns true when a
   * click was attempted; the caller re-runs detectCaptcha afterward to decide
   * whether the challenge actually cleared. Cross-origin frames (reCAPTCHA
   * anchor, Turnstile, hCaptcha, Cloudflare) are reachable via Playwright's
   * frame handles, which the page-context evaluate in detectCaptcha cannot
   * touch. Never throws — an unclickable or absent challenge just returns
   * false so the run can fail cleanly instead of hanging.
   */
  async attemptCaptcha(): Promise<boolean> {
    let page: any;
    try {
      page = this.getActivePage();
      if (!page) return false;
    } catch {
      return false;
    }
    try {
      const frames = page.frames();
      const captchaFrames = frames.filter((f: any) =>
        /recaptcha|hcaptcha|turnstile|challenges\.cloudflare\.com/i.test(f.url() || "")
      );
      if (captchaFrames.length === 0) return false;

      const clickSelectors = [
        "#recaptcha-anchor",
        ".recaptcha-checkbox-border",
        ".recaptcha-checkbox",
        "#checkbox",
        "[role='checkbox']",
        ".button-submit",
        ".prompt-submit-button",
      ];

      let attempted = false;
      for (const frame of captchaFrames) {
        for (const sel of clickSelectors) {
          try {
            const loc = frame.locator(sel).first();
            if (!(await loc.isVisible())) continue;
            await loc.click({ timeout: 3000 });
            attempted = true;
            console.log(`[Adapter] Captcha attempt: clicked ${sel} in ${frame.url()}`);
            break;
          } catch (_) {
            continue;
          }
        }
        if (attempted) break;
      }

      if (attempted) {
        // Give the challenge a moment to process before the caller re-detects.
        await page.waitForTimeout(2500);
      }
      return attempted;
    } catch {
      return false;
    }
  }

  abstract fill(payload: JobPayload, rpc?: RpcHelper): Promise<void>;

  /**
   * Submit the filled application. MUST return a verified SubmitOutcome — a
   * submission is only `confirmed` when the ATS reached a confirmation state
   * (success-URL redirect or inline confirmation text). Never throw for a
   * normal validation failure; return `{confirmed:false, retryable:true}` so
   * the runner can recheck missing fields and retry. Throw only for fatal
   * adapter/browser errors that cannot be retried.
   */
  abstract submit(): Promise<SubmitOutcome>;

  /**
   * Recheck and re-fill any required field that is still blank after a
   * retryable submit failure, returning how many remain blank. Defaults to a
   * no-op (0) so adapters that do not implement field rechecking still work;
   * the runner only calls this on a retryable failure.
   */
  async recheckMissingFields(_rpc?: RpcHelper): Promise<number> {
    return 0;
  }
}
