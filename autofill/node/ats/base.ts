import { Stagehand } from "@browserbasehq/stagehand";
import { JobPayload } from "../types.js";

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
      // visibility helper never gains a name.
      const hit = (await page.evaluate(() => {
        const [vis] = [
          (el: Element): boolean => {
            const r = el.getBoundingClientRect();
            return r.width > 0 && r.height > 0;
          },
        ];
        for (const fr of Array.from(
          document.querySelectorAll(
            'iframe[src*="recaptcha"], iframe[src*="hcaptcha"], ' +
              'iframe[src*="turnstile"], iframe[src*="challenges.cloudflare.com"]'
          )
        )) {
          if (vis(fr)) return "captcha challenge iframe";
        }
        for (const el of Array.from(
          document.querySelectorAll(
            ".g-recaptcha, .h-captcha, .cf-turnstile, " +
              "#challenge-stage, .cf-challenge, [class*='challenge-error']"
          )
        )) {
          if (vis(el)) return "captcha challenge widget";
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

  abstract fill(payload: JobPayload, rpc?: RpcHelper): Promise<void>;
  abstract submit(): Promise<void>;
}
