import * as fs from "fs";

import { Stagehand } from "@browserbasehq/stagehand";

import { type JobPayload } from "../types";
import type { SubmitOutcome } from "./shared/audit";

export type RpcHelper = (method: string, args: Record<string, any>) => Promise<any>;

export abstract class ATSAdapter {
  protected stagehand: Stagehand;

  constructor(stagehand: Stagehand) {
    this.stagehand = stagehand;
  }

  /**
   * The filled form fields as label -> committed value, captured before
   * submit. The runner emits this in the status event so the worker can run a
   * non-LLM consistency check against the persona before marking submitted.
   * Adapters populate this during their pre-submit readback.
   */
  filledValues: Record<string, string> = {};

  /**
   * Resolve the resume path to attach at the END of the fill (mirrors the
   * cover-letter flow): asks the worker for the JD-tailored resume PDF, which
   * is generated in the background while the form walk proceeds. Returns the
   * path when a file exists on disk, else null (caller keeps whatever resume
   * was already attached, or attaches none).
   */
  async resolveTailoredResume(rpc?: RpcHelper): Promise<string | null> {
    if (!rpc) return null;
    try {
      const res = await rpc("tailored_resume", {});
      const p = (res && (res.pdf_path as string | undefined)) || "";
      if (p && fs.existsSync(p)) return p;
      return null;
    } catch {
      return null;
    }
  }

  /**
   * Re-fill a single field (found by its label) with a corrected value and
   * verify it committed. Used by the pre-submit consistency gate to fix a
   * wrong value (e.g. location guessed as "United Kingdom") instead of
   * blocking the application. Returns true when the corrected value is
   * committed and readable back.
   */
  async correctField(_label: string, _value: string): Promise<boolean> {
    return false;
  }

  /** The page to screenshot after fill. Defaults to the first context page;
   *  adapters that switch tabs (e.g. a form opened in a new tab) override. */
  getActivePage(): any {
    try {
      return this.stagehand.context?.pages?.()[0];
    } catch {
      return undefined;
    }
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
        const blocking =
          (void 0,
          (el: Element): boolean => {
            const r = el.getBoundingClientRect();
            if (r.width < 200 || r.height < 100) return false;
            // Must intersect the visible viewport.
            if (
              r.bottom < 0 ||
              r.right < 0 ||
              r.top > window.innerHeight ||
              r.left > window.innerWidth
            ) {
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
              'iframe[src*="turnstile"], iframe[src*="challenges.cloudflare.com"]',
          ),
        )) {
          if (blocking(fr)) return "captcha challenge iframe";
        }
        for (const el of Array.from(
          document.querySelectorAll(
            ".g-recaptcha, .h-captcha, .cf-turnstile, " +
              "#challenge-stage, .cf-challenge, [class*='challenge-error']",
          ),
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
        if (
          /verify you are human|attention required|checking your browser before accessing/i.test(
            body,
          )
        ) {
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
        /recaptcha|hcaptcha|turnstile|challenges\.cloudflare\.com/i.test(f.url() || ""),
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
          } catch {
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
   * (success-page redirect, or the submit form gone with a success phrase).
   * Never throw for a normal validation failure; return
   * `{confirmed:false, retryable:true}` so the runner can recheck missing
   * fields and retry. Throw only for fatal adapter/browser errors that cannot
   * be retried.
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

  /**
   * Some ATS flag a submission as possible spam and offer to "submit again";
   * a human resolves it by navigating back to the posting and reopening the
   * form (fields preserved), then resubmitting. Adapters that support that
   * return true when the form is back on screen; the default is a no-op that
   * returns false (no navigation attempted).
   */
  async retryAfterSpamFlag(): Promise<boolean> {
    return false;
  }

  /**
   * Detect whether the current page indicates the posting was removed/expired
   * (a 404 job board page, "this position is no longer available", an expired
   * redirect, etc.). Returns a human-readable reason when expired, else null.
   * Called before the fill walk: an expired posting must be marked `expired`
   * (a terminal, non-retryable state) instead of failing a fill that will
   * never find a form.
   */
  async detectExpired(): Promise<string | null> {
    let page: any;
    try {
      page = this.getActivePage();
      if (!page) return null;
    } catch {
      return null;
    }
    try {
      const url = typeof page.url === "function" ? await page.url() : "";
      const hit = (await page.evaluate(() => {
        const text = (document.body?.innerText || "")
          .slice(0, 8000)
          .replace(/\s+/g, " ")
          .toLowerCase();
        const markers = [
          "no longer available",
          "position has been filled",
          "this position is no longer",
          "this job is no longer",
          "this posting is no longer",
          "job has been closed",
          "this job has been filled",
          "position is no longer accepting",
          "job posting has expired",
          "we are no longer accepting",
          "this role is no longer",
          "this opportunity is no longer",
          "404 not found",
          "page not found",
          "job not found",
          "position not found",
        ];
        for (const m of markers) {
          if (text.includes(m)) return m;
        }
        return null;
      })) as string | null;
      if (hit) return `expired posting (${hit})`;
      if (url && /\/404\/?$/i.test(url)) return `expired posting (404 url)`;
      return null;
    } catch {
      return null;
    }
  }
}
