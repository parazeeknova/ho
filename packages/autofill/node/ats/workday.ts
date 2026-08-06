import * as fs from "fs";

import { Stagehand } from "@browserbasehq/stagehand";

import { type JobPayload, type Profile } from "../types.js";
import { randomSleep } from "../utils/evasion.js";
import { ATSAdapter, type RpcHelper } from "./base.js";
import { auditBlanks, finalReverify, type SubmitOutcome } from "./shared/audit.js";
import { FormControls } from "./shared/controls.js";
import {
  chooseOption,
  cssEscape,
  cssIdLocator,
  escapePromptValue,
  normalizeOptionText,
  pickLocationOption,
  selectCandidates,
} from "./shared/matching.js";
import {
  fieldKey,
  type FormField,
  isLocationAutocomplete,
  PRE_FILLED_LABELS,
} from "./shared/model.js";
import { Screener, setBlankedRequiredCount } from "./shared/screener.js";

/** Workday system automation-ids the walker must never treat as a question. */
const SYSTEM_SKIP = new Set([
  "beecatcher", // honeypot — MUST never be filled
  "createAccountCheckbox", // sign-in gate consent
]);

/** Profile field automation-ids overwritten deterministically AFTER Workday's
 *  resume parse (Workday is notorious for mis-attributing parsed resume data). */
const IDENTITY_IDS: Array<[string, keyof Profile]> = [
  ["firstName", "firstName"],
  ["lastName", "lastName"],
  ["emailAddress", "email"],
  ["email", "email"],
  ["phoneNumber", "phone"],
  ["mobilePhone", "phone"],
  ["phone", "phone"],
];

/** Workday step-1 personal-info fields keyed by their DOM input id (the inputs
 *  carry plain ids, not data-automation-id). Overwritten deterministically so
 *  resume-parse misattribution and account defaults are corrected. */
const IDENTITY_IDS_BY_ID: Array<[string, keyof Profile]> = [
  ["legalName--firstName", "firstName"],
  ["legalName--lastName", "lastName"],
  ["legalName--firstNameLocal", "firstName"],
  ["legalName--lastNameLocal", "lastName"],
  ["emailAddress", "email"],
  ["phoneNumber--phoneNumber", "phone"],
];

/**
 * Workday adapter.
 *
 * Workday career sites are multi-step wizards behind a sign-in/account gate:
 *   JD page (`/job/...`) → Apply → `/apply/applyManually` → Sign In / Create
 *   Account gate → up to 7 sequential screens (Continue between them) →
 *   Submit Application on the final review screen.
 *
 * Verified against live portals (Intel, Salesforce):
 *  - the JD page is server-rendered; title in `[data-automation-id=
 *    "jobPostingHeader"]`, description in `jobPostingDescription`, locations in
 *    `locations`, Apply button `adventureButton`;
 *  - the landing screen (optional) offers "Autofill with Resume",
 *    "Apply Manually" (`applyManually`) and "Use My Last Application";
 *  - the gate defaults to a Create Account tab with `email`/`password`/
 *    `verifyPassword` + a `createAccountCheckbox` consent box, a `signInLink` /
 *    `createAccountLink` tab switch and a `signInSubmitButton`; because Workday
 *    accounts are per-tenant, the adapter CREATES an account on this tenant
 *    with WORKDAY_EMAIL/WORKDAY_PASSWORD and falls back to sign-in when one
 *    already exists;
 *  - a hidden `beecatcher` honeypot input must never be touched;
 *  - step navigation is the stable platform ids `bottom-navigation-continue-`
 *    and `bottom-navigation-submit-button`;
 *  - field inputs carry `data-automation-id`; comboboxes are async dropdowns
 *    whose options render as `li[role="option"]` in a portal.
 *
 * Anti-mis-parse: after the walk completes, identity fields are overwritten
 * from the profile so any resume-parsed garbage is corrected before submit.
 */
/** Company slug from a `*.wd<N>.myworkdayjobs.com` host ("intel.wd1" → intel). */
export function workdayCompanyFromHostname(host: string): string {
  const m = (host || "").match(/^([^.]+)\.wd\d+\.myworkdayjobs\.com$/i);
  if (m) return m[1];
  return (host || "").split(".")[0] || "";
}

/** Strip a `/apply`/`/apply/applyManually` suffix (and any query/hash) off a
 *  posting URL. */
export function workdayPostingUrl(url: string): string {
  const clean = (url || "").replace(/[?#].*$/, "");
  return clean.replace(/\/apply\/applyManually(\/)?$/, "").replace(/\/apply(\/)?$/, "");
}

/** Derive the deterministic manual-apply URL from a posting URL. */
export function workdayApplyManuallyUrl(url: string): string {
  return `${workdayPostingUrl(url).replace(/\/+$/, "")}/apply/applyManually`;
}

/** True when a screen's visible text marks it as a voluntary disclosure step
 *  (EEOC-style survey). Such steps are legally optional — never guessed. */
export function isVoluntaryStepText(text: string): boolean {
  return /voluntary|self[- ]identif|demographic|eeoc|diversity|equal opportunity/i.test(text || "");
}

/** The class of a Workday control from its DOM attributes. Pure so it is
 *  unit-testable in isolation. */
export function classifyWorkdayControl(attrs: {
  tag: string;
  type?: string;
  role?: string;
  ariaAutocomplete?: boolean;
}): FormField["kind"] {
  if (attrs.tag === "SELECT") return "select";
  if (attrs.role === "combobox" || attrs.ariaAutocomplete) return "combobox";
  if (attrs.type === "radio") return "radio";
  if (attrs.type === "checkbox") return "checkbox";
  return "text";
}

export class WorkdayAdapter extends ATSAdapter {
  protected controls!: WorkdayControlStack;
  protected profile!: Profile;
  private jobCtx: {
    title: string;
    company: string;
    location: string;
    description: string;
  } | null = null;

  constructor(stagehand: Stagehand) {
    super(stagehand);
    this.controls = new WorkdayControlStack(stagehand, "WorkdayAdapter");
  }

  protected getPage(): any {
    return this.controls.getPage();
  }

  /** Workday may navigate/re-render within one tab; the adopted page is always
   *  the one to screenshot. */
  getActivePage(): any {
    return this.controls.getPage();
  }

  // --------------------------------------------------------------------------
  // Job context (title/company/location/description)
  // --------------------------------------------------------------------------

  /**
   * Read the job posting context. Prefers the live DOM (JD page); falls back to
   * fetching the server-rendered posting HTML (Workday SSRs the JD for SEO, so
   * og:title / og:description are present even on the apply view) and to the
   * hostname for the company slug.
   */
  private async readJobContext(): Promise<{
    title: string;
    company: string;
    location: string;
    description: string;
  }> {
    const page = this.getPage();
    try {
      const info: any = await page.evaluate(() => {
        const [txt] = [
          (sel: string) => {
            const el = document.querySelector(sel);
            return el ? (el.textContent || "").replace(/\s+/g, " ").trim() : "";
          },
        ];
        return {
          title: txt('[data-automation-id="jobPostingHeader"], h1'),
          location: txt('[data-automation-id="locations"]'),
          description: txt('[data-automation-id="jobPostingDescription"]'),
          ogTitle:
            document.querySelector('meta[property="og:title"]')?.getAttribute("content") || "",
          ogDesc:
            document.querySelector('meta[property="og:description"]')?.getAttribute("content") ||
            "",
        };
      });
      let title = (info?.title || info?.ogTitle || "").replace(/\s+/g, " ").trim();
      let description = (info?.description || info?.ogDesc || "")
        .replace(/<[^>]+>/g, " ")
        .replace(/\s+/g, " ")
        .trim()
        .slice(0, 6000);
      let location = (info?.location || "").replace(/\s+/g, " ").trim();

      // On the apply view the JD DOM is absent — fetch the server-rendered
      // posting page (og tags) and the company from the hostname.
      if (!title || !description) {
        try {
          const fetched = await fetch(this.postingUrl(page.url()), {
            headers: { "user-agent": "Mozilla/5.0" },
          });
          const html = await fetched.text();
          const og = (p: string) =>
            (html.match(new RegExp(`<meta[^>]*property="${p}"[^>]*content="([^"]*)"`, "i")) ||
              [])[1] || "";
          title = title || og("og:title").replace(/\s+/g, " ").trim();
          description =
            description ||
            og("og:description")
              .replace(/<[^>]+>/g, " ")
              .replace(/\s+/g, " ")
              .trim()
              .slice(0, 6000);
        } catch {
          // Best-effort; title/description may stay empty.
        }
      }
      const company = workdayCompanyFromHostname(new URL(page.url()).hostname);
      return { title, company, location, description };
    } catch (err: any) {
      console.warn(`[Workday] readJobContext failed: ${err?.message || err}`);
      return { title: "", company: "", location: "", description: "" };
    }
  }

  /** Strip a `/apply`/`/apply/applyManually` suffix off a posting URL. */
  private postingUrl(url: string): string {
    return workdayPostingUrl(url);
  }

  /** Derive the deterministic manual-apply URL from a posting URL. */
  private applyManuallyUrl(url: string): string {
    return workdayApplyManuallyUrl(url);
  }

  // --------------------------------------------------------------------------
  // Navigation: land on the application view and pass the sign-in gate
  // --------------------------------------------------------------------------

  private async ensureApplicationView(): Promise<void> {
    const page = this.getPage();
    const cur = page.url();
    if (!/\/apply\b/.test(cur)) {
      // JD page: capture the context FIRST (the apply view drops the JD), then
      // navigate to the deterministic apply route.
      this.jobCtx = await this.readJobContext();
      await page
        .goto(this.applyManuallyUrl(cur), { waitUntil: "domcontentloaded" })
        .catch(() => {});
    }
    // Optional landing screen (Salesforce): pick "Apply Manually" over
    // "Autofill with Resume"/"Use My Last Application" — we upload our own
    // resume and never trust Workday's parse.
    await this.clickApplyManuallyIfPresent();
  }

  private async clickApplyManuallyIfPresent(): Promise<boolean> {
    const page = this.getPage();
    const btn = page.locator('[data-automation-id="applyManually"]').first();
    if (await btn.isVisible().catch(() => false)) {
      await btn.click();
      await randomSleep(2500, 3500);
      return true;
    }
    return false;
  }

  /** True once the actual application wizard (stepper/nav buttons or fields)
   *  is on screen — i.e. we are past the sign-in/account gate. The gate is
   *  detected FIRST: its own `email`/`password`/`verifyPassword` fields and the
   *  consent checkbox all carry `data-automation-id`, so a bare "any field
   *  visible" test would mistake the gate for the form. */
  private async isApplicationFormReady(): Promise<boolean> {
    const page = this.getPage();
    const debug = process.env.DEBUG_WORKDAY === "1";
    const gate = page
      .locator(
        '[data-automation-id="signInSubmitButton"], ' +
          '[data-automation-id="createAccountSubmitButton"], ' +
          '[data-automation-id="signInLink"], ' +
          '[data-automation-id="createAccountLink"], ' +
          '[data-automation-id="forgotPasswordLink"], ' +
          'input[data-automation-id="verifyPassword"], ' +
          'input[data-automation-id="createAccountCheckbox"]',
      )
      .first();
    const gateVis = await gate.isVisible().catch(() => false);
    if (debug) console.log(`[DEBUG_WORKDAY] gateVisible=${gateVis}`);
    if (gateVis) return false;
    const nav = page
      .locator(
        '[data-automation-id="bottom-navigation-continue-button"], ' +
          '[data-automation-id="bottom-navigation-submit-button"], ' +
          '[data-automation-id="bottom-navigation-back-button"], ' +
          '[data-automation-id="pageFooterNextButton"], ' +
          '[data-automation-id="pageFooterSubmitButton"], ' +
          '[data-automation-id="pageFooterBackButton"], ' +
          '[data-automation-id="submitButton"]',
      )
      .first();
    const navVis = await nav.isVisible().catch(() => false);
    if (debug) console.log(`[DEBUG_WORKDAY] navVisible=${navVis}`);
    if (navVis) return true;
    // A Workday wizard step container (e.g. applyFlowMyInfoPage) proves we are
    // past the gate even before any button is interactive.
    const stepPage = page
      .locator('[data-automation-id^="applyFlow"]:not([data-automation-id="applyFlowPage"])')
      .first();
    const stepPageVis = await stepPage.isVisible().catch(() => false);
    if (debug) console.log(`[DEBUG_WORKDAY] stepPageVisible=${stepPageVis}`);
    if (stepPageVis) return true;
    // Some tenants label the step buttons differently ("Save and Continue").
    const textNav = page
      .locator(
        'button:has-text("Save and Continue"), ' +
          'button:has-text("Submit Application"), ' +
          'button:has-text("Continue"), ' +
          'button:has-text("Next")',
      )
      .first();
    const textNavVis = await textNav.isVisible().catch(() => false);
    if (debug) console.log(`[DEBUG_WORKDAY] textNavVisible=${textNavVis} url=${page.url()}`);
    if (textNavVis) return true;
    // No gate, no nav — a FIELD proves the wizard only if it is not one of the
    // gate's own inputs (which also carry automation-ids).
    const field = page
      .locator(
        'input[data-automation-id]:not([data-automation-id="email"]):not([data-automation-id="password"]):not([data-automation-id="verifyPassword"]):not([data-automation-id="createAccountCheckbox"]):not([data-automation-id="beecatcher"]), ' +
          'textarea[data-automation-id]:not([data-automation-id="email"]):not([data-automation-id="password"])',
      )
      .first();
    const fieldVis = await field.isVisible().catch(() => false);
    if (debug) console.log(`[DEBUG_WORKDAY] fieldVisible=${fieldVis}`);
    return fieldVis;
  }

  /**
   * Pass the sign-in/account gate. Strategy (in order):
   *   1. any guest path ("Apply Manually", "Continue without signing in", …);
   *   2. CREATE an account on this tenant with WORKDAY_EMAIL/WORKDAY_PASSWORD
   *      (Workday accounts are per-tenant — the same credentials won't already
   *      exist on most companies' sites);
   *   3. sign in with the same credentials (the account may already exist, in
   *      which case creation failed);
   *   4. abort cleanly.
   */
  private async handleGate(): Promise<void> {
    const hasCreds = !!(process.env.WORKDAY_EMAIL && process.env.WORKDAY_PASSWORD);
    let attemptedCreate = false;
    let attemptedSignIn = false;
    // Phase A: wait for the gate (or the form) to hydrate before attempting
    // anything — running too early makes every check "not visible" and bails.
    for (let i = 0; i < 12; i++) {
      if (await this.isApplicationFormReady()) return;
      if (await this.isGatePresent()) break;
      await randomSleep(1000, 1500);
    }
    // Phase B: guest path → create account (per-tenant) → sign-in → abort.
    for (let i = 0; i < 12; i++) {
      if (await this.isApplicationFormReady()) return;
      if (await this.tryGuestContinue()) continue;

      if (hasCreds) {
        if (!attemptedCreate) {
          // Ensure we are on the "Create Account" tab, then create the account.
          if (await this.switchToCreateAccount()) continue;
          if (await this.tryCreateAccount()) {
            attemptedCreate = true;
            continue;
          }
          // Creation failed (or the tenant has no create form / already has the
          // account and redirected to /login). Fall through to sign-in.
          console.warn("[Workday] Account creation not possible on this gate; trying sign-in.");
          attemptedCreate = true;
        }
        if (!attemptedSignIn) {
          if (await this.switchToSignIn()) continue;
          if (await this.trySignIn()) {
            attemptedSignIn = true;
            continue;
          }
          console.warn("[Workday] Sign-in not possible on this gate.");
          attemptedSignIn = true;
        }
      } else {
        console.warn(
          "[Workday] Gate requires an account, but WORKDAY_EMAIL/WORKDAY_PASSWORD are not set.",
        );
      }
      await randomSleep(2000, 3000);
    }
    await this.dumpGateState();
    throw new Error(
      "Workday: could not reach the application form after account creation/sign-in. " +
        "Set WORKDAY_EMAIL/WORKDAY_PASSWORD (and make sure WORKDAY_PASSWORD meets the " +
        "portal's password requirements: ≥8 chars with upper/lower/special/numeric).",
    );
  }

  /** True once a gate element (create/sign-in form) is actually rendered. */
  private async isGatePresent(): Promise<boolean> {
    const page = this.getPage();
    return await page
      .locator(
        '[data-automation-id="signInSubmitButton"], ' +
          '[data-automation-id="createAccountSubmitButton"], ' +
          '[data-automation-id="signInLink"], ' +
          '[data-automation-id="createAccountLink"], ' +
          'input[data-automation-id="verifyPassword"]',
      )
      .first()
      .isVisible()
      .catch(() => false);
  }

  /** Poll a locator until it is visible (or the timeout elapses). Workday's
   *  SPA hydrates lazily — the gate fields appear after the first frames. */
  private async waitVisible(locator: any, timeoutMs = 9000): Promise<boolean> {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
      if (await locator.isVisible().catch(() => false)) return true;
      await randomSleep(400, 600);
    }
    return false;
  }

  /** Diagnostic dump of the gate screen when the walker cannot get past it. */
  private async dumpGateState(): Promise<void> {
    const page = this.getPage();
    try {
      const s: any = await page.evaluate(() => {
        // WARNING: only anonymous arrows (array-destructured) — tsx keepNames.
        const [txt] = [
          (el: Element | null) =>
            (el ? (el.textContent || "").replace(/\s+/g, " ").trim() : "").slice(0, 80),
        ];
        const aids = Array.from(document.querySelectorAll("[data-automation-id]"))
          .filter((e) => (e as HTMLElement).offsetParent !== null)
          .map((e) => e.getAttribute("data-automation-id"))
          .slice(0, 50);
        const overlays = Array.from(
          document.querySelectorAll("[data-automation-id='click_filter'], [role='button']"),
        )
          .filter((e) => (e as HTMLElement).offsetParent !== null)
          .map((e) => ({ text: txt(e), aid: e.getAttribute("data-automation-id") }));
        const alerts = Array.from(
          document.querySelectorAll("[role='alert'], .error, .error-message, [class*='error']"),
        )
          .filter((e) => (e as HTMLElement).offsetParent !== null)
          .map((e) => txt(e))
          .filter(Boolean);
        return { url: location.href, aids, overlays: overlays.slice(0, 8), alerts };
      });
      console.warn("[Workday] GATE DIAGNOSTIC:", JSON.stringify(s, null, 1));
    } catch (err: any) {
      console.warn(`[Workday] GATE DIAGNOSTIC failed: ${err?.message || err}`);
    }
  }

  private async tryGuestContinue(): Promise<boolean> {
    const page = this.getPage();
    const manual = page.locator('[data-automation-id="applyManually"]').first();
    if (await manual.isVisible().catch(() => false)) {
      await manual.click();
      await randomSleep(2500, 3500);
      return true;
    }
    const guest = page
      .locator(
        'button:has-text("Continue without signing in"), ' +
          'button:has-text("Continue as guest"), ' +
          'button:has-text("Apply without signing in"), ' +
          'button:has-text("Skip for now")',
      )
      .first();
    if (await guest.isVisible().catch(() => false)) {
      await guest.click();
      await randomSleep(2000, 3000);
      return true;
    }
    return false;
  }

  private async switchToCreateAccount(): Promise<boolean> {
    const page = this.getPage();
    const link = page.locator('[data-automation-id="createAccountLink"]').first();
    if (await link.isVisible().catch(() => false)) {
      await link.click();
      await randomSleep(1500, 2000);
      return true;
    }
    return false;
  }

  private async switchToSignIn(): Promise<boolean> {
    const page = this.getPage();
    const link = page.locator('[data-automation-id="signInLink"]').first();
    if (await link.isVisible().catch(() => false)) {
      await link.click();
      await randomSleep(1500, 2000);
      return true;
    }
    return false;
  }

  /**
   * Create an account on THIS tenant with the env credentials. Workday career
   * sites are per-tenant, so the same email/password usually has no account yet
   * — creating one is the only way past the gate. Fills email + password +
   * verify-password, accepts the consent checkbox, and submits. Returns true
   * when it clicked submit (regardless of outcome — the caller falls back to
   * sign-in if the account already existed).
   */
  private async tryCreateAccount(): Promise<boolean> {
    const page = this.getPage();
    const user = process.env.WORKDAY_EMAIL;
    const pw = process.env.WORKDAY_PASSWORD;
    if (!user || !pw) return false;
    // Confirm this is the CREATE form (has verify-password), never the sign-in tab.
    const verify = page.locator('input[data-automation-id="verifyPassword"]').first();
    const submit = page.locator('[data-automation-id="createAccountSubmitButton"]').first();
    if (!(await this.waitVisible(verify))) {
      console.warn("[Workday] tryCreateAccount: no verify-password field (not on the create tab).");
      return false;
    }
    if (!(await this.waitVisible(submit))) {
      console.warn("[Workday] tryCreateAccount: no create-account submit button.");
      return false;
    }

    const email = page.locator('input[data-automation-id="email"]').first();
    const password = page.locator('input[data-automation-id="password"]').first();
    if (!(await this.waitVisible(email))) {
      console.warn("[Workday] tryCreateAccount: email field not visible.");
      return false;
    }
    if (!(await this.waitVisible(password))) {
      console.warn("[Workday] tryCreateAccount: password field not visible.");
      return false;
    }

    console.log("[Workday] Creating an account on this tenant...");
    await email.fill(user);
    await password.fill(pw);
    await verify.fill(pw);
    await randomSleep(200, 400);
    // Consent to the privacy/terms checkbox — required to proceed.
    const consent = page.locator('input[data-automation-id="createAccountCheckbox"]').first();
    if (await consent.isVisible().catch(() => false)) {
      await consent.check({ force: true }).catch(() => {});
    }
    await randomSleep(300, 600);
    await this.clickSubmitAction(page, submit);
    console.log("[Workday] Account creation submitted.");
    // Wait for the outcome: either we land on the application wizard (new
    // account, auto-signed-in) or we are redirected to /login to sign in
    // (account already existed).
    for (let i = 0; i < 12; i++) {
      if (await this.isApplicationFormReady()) return true;
      if (/\/login(\?|$)/.test(page.url())) return true;
      await randomSleep(1500, 2000);
    }
    return true;
  }

  private async trySignIn(): Promise<boolean> {
    const page = this.getPage();
    const email = page.locator('input[data-automation-id="email"]').first();
    const submit = page.locator('[data-automation-id="signInSubmitButton"]').first();
    if (!(await this.waitVisible(email))) {
      console.warn("[Workday] trySignIn: no email field visible.");
      return false;
    }
    if (!(await this.waitVisible(submit))) {
      console.warn("[Workday] trySignIn: no sign-in submit button.");
      return false;
    }
    const user = process.env.WORKDAY_EMAIL;
    const pw = process.env.WORKDAY_PASSWORD;
    if (!user || !pw) {
      console.warn("[Workday] Sign-in required but WORKDAY_EMAIL/WORKDAY_PASSWORD are not set.");
      return false;
    }
    await email.fill(user);
    const password = page.locator('input[data-automation-id="password"]').first();
    if (await password.isVisible().catch(() => false)) {
      await password.fill(pw);
    }
    await randomSleep(300, 600);
    await this.clickSubmitAction(page, submit);
    console.log("[Workday] Signed in.");
    await randomSleep(3500, 4500);
    return true;
  }

  /**
   * Click a Workday gate submit action. The real submit button is wrapped in an
   * invisible `click_filter` overlay (Workday's anti-bot interception): the
   * button itself is `tabindex="-2"` and a human actually clicks the overlay,
   * which forwards to the button. A synthetic mousedown/mouseup/click sequence
   * on the overlay reliably triggers submission (verified live); a plain
   * Playwright trusted click on the overlay does NOT forward.
   */
  private async clickSubmitAction(page: any, submitBtn: any): Promise<void> {
    const filter = page.locator('[data-automation-id="click_filter"]').first();
    const hasOverlay = await filter.isVisible().catch(() => false);
    if (hasOverlay) {
      // Dispatch the synthetic sequence the overlay's handler expects.
      const ok = await page
        .evaluate(() => {
          const f = document.querySelector('[data-automation-id="click_filter"]');
          if (!f) return false;
          const r = f.getBoundingClientRect();
          const cx = r.left + r.width / 2;
          const cy = r.top + r.height / 2;
          for (const t of ["mousedown", "mouseup", "click"]) {
            f.dispatchEvent(
              new MouseEvent(t, {
                bubbles: true,
                cancelable: true,
                view: window,
                clientX: cx,
                clientY: cy,
                button: 0,
              }),
            );
          }
          return true;
        })
        .catch(() => false);
      if (ok) return;
    }
    await submitBtn.click().catch(() => {});
  }

  // --------------------------------------------------------------------------
  // DOM inventory
  // --------------------------------------------------------------------------

  private async collectQuestions(): Promise<FormField[]> {
    const page = this.getPage();
    try {
      const rows = await page.evaluate(
        (skipIds: string[]) => {
          const out: Array<{
            label: string;
            id: string;
            name: string;
            kind: string;
            required: boolean;
            options: string[];
            targets: Array<{ text: string; name: string; value: string; id?: string }>;
          }> = [];
          // WARNING: only anonymous arrows may be defined inside this evaluate
          // (tsx keepNames wraps inferred-name arrows in __name()). Destructure
          // helpers into an array so none gains a name.
          const [norm, visible, inNav, labelOf, hasAsterisk, qesc, push] = [
            (t: string) =>
              (t || "")
                .replace(/\s+/g, " ")
                .trim()
                .replace(/^\*+|\*+$/g, ""),
            (el: Element): boolean => {
              const e = el as HTMLElement;
              const r = e.getBoundingClientRect();
              if (r.width === 0 && r.height === 0) return false;
              const cs = getComputedStyle(e);
              if (cs.display === "none" || cs.visibility === "hidden" || cs.opacity === "0")
                return false;
              return true;
            },
            (el: Element): boolean => {
              let n: Element | null = el;
              while (n && n !== document.body) {
                const tag = n.tagName.toLowerCase();
                const role = n.getAttribute && n.getAttribute("role");
                if (
                  tag === "header" ||
                  tag === "nav" ||
                  tag === "footer" ||
                  role === "banner" ||
                  role === "navigation"
                ) {
                  return true;
                }
                n = n.parentElement;
              }
              return false;
            },
            (el: Element): string => {
              const labelledby = el.getAttribute && el.getAttribute("aria-labelledby");
              if (labelledby) {
                const l = document.getElementById(labelledby);
                if (l) {
                  const t = norm(l.textContent || "");
                  if (t) return t;
                }
              }
              const aria = el.getAttribute && el.getAttribute("aria-label");
              if (aria) {
                const t = norm(aria);
                if (t && !/robots only/i.test(t)) return t;
              }
              const wrap = el.closest("label");
              if (wrap) {
                const t = norm(wrap.textContent || "");
                if (t) return t;
              }
              const id = el.getAttribute && el.getAttribute("id");
              if (id) {
                const fl = document.querySelector(`label[for="${qesc(id)}"]`);
                if (fl) {
                  const t = norm(fl.textContent || "");
                  if (t) return t;
                }
              }
              let n = el.parentElement;
              for (let i = 0; n && i < 4; i++, n = n.parentElement) {
                const cand = n.querySelector(
                  ':scope > label, :scope > legend, :scope > [data-automation-label], :scope > h1, :scope > h2, :scope > h3, :scope > span[class*="label"]',
                );
                if (cand) {
                  const t = norm(cand.textContent || "");
                  if (t && t.length < 120) return t;
                }
              }
              return "";
            },
            (el: Element): boolean => {
              // A visible required asterisk is the ONLY required marker on some
              // Workday fields (no aria-required/required attribute). The label
              // text is normalized (asterisk stripped) elsewhere, so check the
              // raw label sources here.
              const test = (t: string | null): boolean => !!t && /\*/.test(t);
              const wrap = el.closest("label");
              if (test(wrap ? wrap.textContent : "")) return true;
              const id = el.getAttribute && el.getAttribute("id");
              if (id) {
                const fl = document.querySelector(`label[for="${qesc(id)}"]`);
                if (test(fl ? fl.textContent : "")) return true;
              }
              if (test(el.getAttribute && el.getAttribute("aria-label"))) return true;
              const p = el.parentElement;
              if (p) {
                const l = p.querySelector(
                  ":scope > label, :scope > legend, :scope > [data-automation-label]",
                );
                if (test(l ? l.textContent : "")) return true;
              }
              return false;
            },
            (s: string): string => (s || "").replace(/\\/g, "\\\\").replace(/"/g, '\\"'),
            (
              label: string,
              id: string,
              name: string,
              kind: string,
              required: boolean,
              options: string[] = [],
              targets: Array<{ text: string; name: string; value: string; id?: string }> = [],
            ): void => {
              if (!label) return;
              out.push({ label, id, name, kind, required, options, targets });
            },
          ];

          const seenText = new Set<string>();
          const textSel =
            'input[type="text"], input[type="email"], input[type="tel"], input[type="url"], input[type="number"], input[type="date"], input:not([type]), textarea';
          for (const el of Array.from(document.querySelectorAll(textSel))) {
            const e = el as HTMLInputElement;
            const aid = e.getAttribute("data-automation-id") || "";
            if (aid === "beecatcher") continue;
            if (e.type === "password") continue;
            if (!visible(e)) continue;
            if (inNav(e)) continue;
            if (aid && skipIds.includes(aid)) continue;
            const label = labelOf(e);
            if (!label) continue;
            const combo =
              e.getAttribute("role") === "combobox" || !!e.getAttribute("aria-autocomplete");
            const kind = combo ? "combobox" : "text";
            const key = norm(label).toLowerCase() + "|" + kind;
            if (seenText.has(key)) continue;
            seenText.add(key);
            const required =
              !!e.getAttribute("aria-required") || e.hasAttribute("required") || hasAsterisk(e);
            push(label, e.getAttribute("id") || aid || norm(label), aid, kind, required);
          }

          // Radio/checkbox groups grouped by input name.
          const seenGroups = new Set<string>();
          for (const el of Array.from(
            document.querySelectorAll('input[type="radio"], input[type="checkbox"]'),
          )) {
            const e = el as HTMLInputElement;
            if (inNav(e)) continue;
            const aid = e.getAttribute("data-automation-id") || "";
            if (aid && skipIds.includes(aid)) continue;
            const name = e.name || "";
            if (!name || seenGroups.has(name)) continue;
            seenGroups.add(name);
            const type = e.type;
            const group = Array.from(
              document.querySelectorAll(`input[type="${type}"][name="${CSS.escape(name)}"]`),
            ) as HTMLInputElement[];
            const targets: Array<{ text: string; name: string; value: string; id?: string }> = [];
            const options: string[] = [];
            for (const g of group) {
              if (
                !visible(g) &&
                !(g.closest("label") && visible(g.closest("label") as Element)) &&
                !(
                  g.closest("[role='radio'], [role='checkbox'], [class*='option']") &&
                  visible(
                    g.closest("[role='radio'], [role='checkbox'], [class*='option']") as Element,
                  )
                )
              )
                continue;
              const gid = g.getAttribute("id");
              const labFor = gid
                ? document.querySelector(`label[for="${qesc(gid)}"]`)?.textContent || ""
                : "";
              const wrapLabel = g.closest("label");
              const row = g.closest("[role='radio'], [role='checkbox'], [class*='option']");
              const text = norm(
                wrapLabel
                  ? wrapLabel.textContent || ""
                  : labFor || g.getAttribute("aria-label") || (row ? row.textContent || "" : ""),
              );
              if (!text) continue;
              if (!targets.some((t) => t.text === text)) {
                targets.push({ text, name, value: g.value || "", id: gid || "" });
              }
              if (!options.includes(text)) options.push(text);
            }
            if (!targets.length) continue;
            // Group label: the container's label/legend/heading, never an option.
            const container = e.closest(
              '[data-automation-id], fieldset, [role="radiogroup"], [role="group"], [class*="form-control"], [class*="field"]',
            );
            let groupLabel = "";
            if (container) {
              const cb = container.getAttribute && container.getAttribute("aria-labelledby");
              if (cb) {
                const l = document.getElementById(cb);
                if (l) groupLabel = norm(l.textContent || "");
              }
              if (!groupLabel) {
                for (const cand of Array.from(
                  container.querySelectorAll(
                    ":scope > legend, :scope > [data-automation-label], :scope > label, :scope > h1, :scope > h2, :scope > h3, :scope > span",
                  ),
                )) {
                  const t = norm(cand.textContent || "");
                  if (!t || t.length > 150) continue;
                  if (targets.some((tg) => tg.text === t)) continue;
                  if (cand.querySelector('input[type="radio"], input[type="checkbox"]')) continue;
                  groupLabel = t;
                  break;
                }
              }
            }
            if (!groupLabel) groupLabel = labelOf(e);
            if (!groupLabel) continue;
            const kind = type === "radio" ? "radio" : "checkbox";
            const required =
              group.some(
                (g) => g.hasAttribute("required") || g.getAttribute("aria-required") === "true",
              ) ||
              /\*/.test(groupLabel) ||
              hasAsterisk(e);
            push(groupLabel, name, name, kind, required, options, targets);
          }

          // Native selects (rare on Workday, but handled).
          for (const el of Array.from(document.querySelectorAll("select"))) {
            const e = el as HTMLSelectElement;
            if (!visible(e)) continue;
            if (inNav(e)) continue;
            const aid = e.getAttribute("data-automation-id") || "";
            if (aid && skipIds.includes(aid)) continue;
            const label = labelOf(e);
            if (!label) continue;
            const options = Array.from(e.options)
              .map((o) => norm(o.textContent || ""))
              .filter(Boolean);
            const required =
              !!e.getAttribute("aria-required") || e.hasAttribute("required") || /\*/.test(label);
            push(
              label,
              e.getAttribute("id") || aid || norm(label),
              aid,
              "select",
              required,
              options,
            );
          }

          const uniq: typeof out = [];
          const seen = new Set<string>();
          for (const r of out) {
            const key = norm(r.label).toLowerCase() + "|" + r.kind;
            if (seen.has(key)) continue;
            seen.add(key);
            uniq.push(r);
          }
          return uniq;
        },
        [...SYSTEM_SKIP],
      );

      return (rows ?? []).map((r: any): FormField => ({
        label: r.label,
        id: r.id,
        kind: r.kind as FormField["kind"],
        required: !!r.required,
        options: r.options ?? [],
        optionTargets: (r.targets ?? []).map((t: any) => ({
          text: t.text,
          name: t.name,
          value: t.value,
          id: t.id ?? "",
        })),
        name: r.name,
      }));
    } catch (err: any) {
      console.warn(`[Workday] collectQuestions failed: ${err?.message || err}`);
      return [];
    }
  }

  // --------------------------------------------------------------------------
  // Deterministic fills (identity overwrite + resume + cover letter)
  // --------------------------------------------------------------------------

  /** Overwrite identity fields from the profile — repairs any mis-parse that
   *  Workday's resume importer made into the wrong fields. Idempotent. Targets
   *  both the automation-id inputs (gate/system fields) and the step-1 fields
   *  by their plain DOM ids. */
  private async fillIdentityFields(profile: Profile): Promise<void> {
    for (const [aid, key] of IDENTITY_IDS) {
      const value = (profile as any)?.[key];
      if (!value) continue;
      await this.controls.fillByAutomationId(aid, String(value));
    }
    for (const [id, key] of IDENTITY_IDS_BY_ID) {
      const value = (profile as any)?.[key];
      if (!value) continue;
      await this.controls.fillById(id, String(value));
    }
  }

  private async uploadResumeIfVisible(resumePath: string): Promise<boolean> {
    if (!resumePath || !fs.existsSync(resumePath)) return false;
    const page = this.getPage();
    // Only the resume file input — never a bare catch-all that could grab a
    // transcript/other attachment on the same step.
    const input = page
      .locator(
        'input[type="file"][data-automation-id="resume"], input[type="file"][name="resume"], [data-automation-id="resume"] input[type="file"]',
      )
      .first();
    if ((await input.count()) === 0) return false;
    for (let attempt = 0; attempt < 3; attempt++) {
      if (await this.controls.isResumeAttached()) return true;
      try {
        await input.setInputFiles(resumePath);
      } catch (err: any) {
        console.warn(
          `[Workday] Resume setInputFiles threw (attempt ${attempt + 1}): ${err?.message || err}`,
        );
      }
      await randomSleep(2500, 3500);
      if (await this.controls.isResumeAttached()) {
        console.log(`[Workday] Resume uploaded and registered (attempt ${attempt + 1}).`);
        return true;
      }
      // Some Workday flows need an explicit "Upload" button after attach.
      const uploadBtn = page
        .locator('[data-automation-id="uploadButton"], button:has-text("Upload")')
        .first();
      if (await uploadBtn.isVisible().catch(() => false)) {
        await uploadBtn.click();
        await randomSleep(2000, 3000);
      }
      console.warn(`[Workday] Resume upload not confirmed (attempt ${attempt + 1}); retrying...`);
    }
    return false;
  }

  /** Fill an LLM-generated cover letter into a cover-letter/additional-info
   *  textarea, if the step renders one. */
  private async fillCoverLetter(
    rpc: RpcHelper,
    filled: string[],
    blanked: Array<{ label: string; reason: string }>,
  ): Promise<void> {
    const page = this.getPage();
    const candidates: Array<{ index: number; label: string; filled: boolean }> = await page
      .evaluate(() => {
        const out: Array<{ index: number; label: string; filled: boolean }> = [];
        const areas = Array.from(document.querySelectorAll("textarea"));
        areas.forEach((el, i) => {
          const e = el as HTMLTextAreaElement;
          if (e.offsetParent === null) return;
          const aria = e.getAttribute("aria-label") || "";
          const id = e.getAttribute("id") || "";
          const forLabel = id
            ? document.querySelector(`label[for="${id}"]`)?.textContent || ""
            : "";
          const wrap = e.closest("label")?.textContent || "";
          const label = (aria || forLabel || wrap).replace(/\s+/g, " ").trim();
          out.push({ index: i, label, filled: !!e.value.trim() });
        });
        return out;
      })
      .catch(() => []);
    const target = candidates.find(
      (c) =>
        /cover letter|additional information|anything else you|more about you|tell us about yourself|anything you would like/i.test(
          c.label,
        ) && !c.filled,
    );
    if (!target) {
      console.log("[Workday] No cover-letter textarea on this step; skipping generation.");
      return;
    }
    const result = await rpc("cover_letter", {});
    const pdfPath = result?.pdf_path;
    let attached = false;

    if (pdfPath) {
      // Look for a file input specifically for cover letter or just any file input in a generic document upload section.
      const clFileInputs = [
        'input[type="file"][data-automation-id="coverLetter"]',
        'input[type="file"][name*="coverLetter" i]',
        'input[type="file"][data-automation-id="file-upload-input"]',
      ];
      for (const sel of clFileInputs) {
        const fileInput = page.locator(sel).first();
        if ((await fileInput.isVisible().catch(() => false)) || (await fileInput.count()) > 0) {
          try {
            await fileInput.setInputFiles(pdfPath);
            console.log("[Workday] Cover letter PDF uploaded successfully.");
            attached = true;
            break;
          } catch {
            // Ignore and try next
          }
        }
      }
    }

    if (!attached) {
      const coverLetter = (result?.answer ?? "").toString().trim();
      if (!coverLetter) return;
      const ta = page.locator("textarea").nth(target.index);
      await ta.fill(coverLetter);
      await randomSleep(200, 400);
      const committed = await ta.inputValue().catch(() => "");
      if (committed) {
        filled.push(target.label || "Cover Letter");
        console.log("[Workday] Cover letter filled (LLM-generated, JD-personalized).");
      } else {
        blanked.push({
          label: target.label || "Cover Letter",
          reason: "cover letter did not commit",
        });
      }
    }
  }

  // --------------------------------------------------------------------------
  // Step machine
  // --------------------------------------------------------------------------

  private async readStepNumber(): Promise<number> {
    const page = this.getPage();
    return (await page
      .evaluate(() => {
        for (const el of Array.from(
          document.querySelectorAll("[class*='step'], [class*='Step'], [data-automation-id]"),
        )) {
          const t = (el.textContent || "").replace(/\s+/g, " ").trim();
          const m = t.match(/step\s+(\d+)\s+of\s+\d+/i);
          if (m) return parseInt(m[1], 10);
        }
        return 0;
      })
      .catch(() => 0)) as number;
  }

  private async hasSubmitButton(): Promise<boolean> {
    const page = this.getPage();
    const b = page
      .locator(
        '[data-automation-id="bottom-navigation-submit-button"], ' +
          '[data-automation-id="submitButton"], ' +
          '[data-automation-id="pageFooterSubmitButton"], ' +
          'button:has-text("Submit Application"), ' +
          'button:has-text("Submit")',
      )
      .first();
    return await b.isVisible().catch(() => false);
  }

  private async clickContinue(): Promise<boolean> {
    const page = this.getPage();
    const btn = page
      .locator(
        '[data-automation-id="bottom-navigation-continue-button"], ' +
          '[data-automation-id="pageFooterNextButton"], ' +
          'button:has-text("Save and Continue"), ' +
          'button:has-text("Continue"), ' +
          'button:has-text("Next")',
      )
      .first();
    if (await btn.isVisible().catch(() => false)) {
      await btn.click();
      return true;
    }
    return false;
  }

  private async advanceStep(): Promise<boolean> {
    const before = await this.readStepNumber();
    const clicked = await this.clickContinue();
    if (!clicked) return false;
    await randomSleep(2000, 3000);
    for (let i = 0; i < 20; i++) {
      const after = await this.readStepNumber();
      if (before > 0 && after > before) return true;
      if (await this.isApplicationFormReady()) return true;
      await randomSleep(800, 1200);
    }
    if (before > 0) {
      // Continue was clicked but the step never advanced (e.g. client-side
      // validation error). Stop the walk instead of spinning on the same screen.
      console.warn(`[Workday] Step advance stalled on step ${before}; stopping the walk.`);
      return false;
    }
    return true;
  }

  /** True when the visible step is a voluntary disclosure (EEOC-style) step:
   *  all optional fields AND the fields/on-screen text carry a survey marker.
   *  Such steps are legally optional — we never guess, we just advance. */
  private async isVoluntaryStep(fields: FormField[]): Promise<boolean> {
    if (fields.some((f) => f.required)) return false;
    const surveyLabels = fields.some((f) =>
      /race|ethnic|gender|sex|veteran|disability|orientation|military|self[- ]identif|voluntary|diversity/i.test(
        f.label,
      ),
    );
    const page = this.getPage();
    const text: string = await page
      .evaluate(() => {
        const main = document.querySelector("main, [role='main'], #wd-content");
        const scope = main || document.body;
        return (scope.textContent || "").replace(/\s+/g, " ").slice(0, 2000);
      })
      .catch(() => "");
    return surveyLabels || isVoluntaryStepText(text);
  }

  private async hasValue(f: FormField): Promise<boolean> {
    return !!(await this.controls.readFieldValue(f));
  }

  // --------------------------------------------------------------------------
  // fill / submit
  // --------------------------------------------------------------------------

  async fill(payload: JobPayload, rpc?: RpcHelper): Promise<void> {
    const { url, profile } = payload;
    this.profile = profile;
    console.log(`[Workday] Navigating to ${url}...`);
    const page = this.getPage();
    await page.goto(url);
    await randomSleep(300, 600);

    await this.ensureApplicationView();
    await this.handleGate();

    // Deterministic identity overwrite up front (repairs resume-parse errors
    // even before the walk sees the fields) and an early resume attempt — both
    // run regardless of whether an RPC resolver is wired.
    await this.fillIdentityFields(profile);
    let resumeAttached = await this.uploadResumeIfVisible(profile.resumePath ?? "");

    if (rpc) {
      // Job context first so open-ended answers are personalized to the role.
      const jobCtx = this.jobCtx ?? (await this.readJobContext());
      await rpc("job_context", jobCtx);
      console.log(
        `[Workday] Job context: ${jobCtx.title || "?"} @ ${jobCtx.company || "?"}` +
          (jobCtx.location ? ` (${jobCtx.location})` : ""),
      );

      const screener = new Screener(this.controls, "WorkdayAdapter", profile, rpc, true);
      const filled: string[] = [];
      const blanked: Array<{ label: string; reason: string }> = [];
      const processedKeys = new Set<string>();
      const userSkippedKeys = new Set<string>();

      const MAX_STEPS = 12;
      for (let step = 0; step < MAX_STEPS; step++) {
        // Let the step hydrate before enumerating.
        await randomSleep(1200, 1800);

        const inventory = await this.collectQuestions();
        if (inventory.length === 0) {
          console.log(`[Workday] Step ${step + 1}: no visible questions.`);
        }

        if (await this.isVoluntaryStep(inventory)) {
          console.log("[Workday] Voluntary disclosure step detected (all-optional); skipping.");
          for (const f of inventory) {
            blanked.push({ label: f.label, reason: "voluntary disclosure step (left unchecked)" });
          }
        } else {
          // Converging re-scan walk over the visible fields (conditional
          // questions appear only after an interaction).
          for (let pass = 0; pass < 30; pass++) {
            const fields = await this.collectQuestions();
            const fresh = fields.filter((f) => !processedKeys.has(fieldKey(f)));
            if (pass === 0) {
              console.log(`[Workday] Step ${step + 1} inventory: ${fields.length} question(s).`);
            }
            if (fresh.length === 0) {
              console.log(`[Workday] Step ${step + 1} walk converged after ${pass + 1} pass(es).`);
              break;
            }
            for (const f of fresh) {
              processedKeys.add(fieldKey(f));
              try {
                await screener.process(f, filled, blanked, userSkippedKeys);
              } catch (err: any) {
                // A single unruly field must never abort the whole fill.
                console.warn(
                  `[Workday] Skipping "${escapePromptValue(f.label)}" (${err?.message || err})`,
                );
                blanked.push({ label: f.label, reason: `fill threw: ${err?.message || err}` });
              }
            }
            await this.controls.closeMenu().catch(() => {});
            await randomSleep(900, 1400);
          }

          // Uploads / cover letter on this step.
          if (!resumeAttached) {
            resumeAttached = await this.uploadResumeIfVisible(profile.resumePath ?? "");
          }
          await this.fillCoverLetter(rpc, filled, blanked);

          // Zero-blank audit for the visible step.
          const stepFields = await this.collectQuestions();
          const requiredBlanks = await auditBlanks({
            fields: stepFields,
            readValue: (f) => this.controls.readFieldValue(f),
            transcript: blanked,
          });
          if (requiredBlanks.length > 0) {
            console.warn(
              `[Workday] ${requiredBlanks.length} REQUIRED field(s) blank after step ${step + 1}:`,
            );
            for (const rb of requiredBlanks) {
              console.warn(
                `[Workday]   REQUIRED blank: ${escapePromptValue(rb.label)} (${rb.reason})`,
              );
            }
          }
        }

        if (await this.hasSubmitButton()) {
          console.log("[Workday] Final review step reached (Submit button visible).");
          break;
        }
        const advanced = await this.advanceStep();
        if (!advanced) {
          console.warn(
            "[Workday] No Continue/Submit button found, or the step did not advance after clicking. " +
              "If required fields above were left blank, Workday blocks progression until they are answered.",
          );
          break;
        }
      }

      // Final sweep over the last visible step.
      const sweepFilled: string[] = [];
      const sweepBlanks: Array<{ label: string; reason: string }> = [];
      for (let pass = 0; pass < 3; pass++) {
        const swept = await this.collectQuestions();
        let touched = 0;
        for (const f of swept) {
          if (PRE_FILLED_LABELS.has(`${f.label}`.replace(/\s+/g, " ").toLowerCase())) continue;
          if (userSkippedKeys.has(fieldKey(f))) continue;
          if (await this.hasValue(f)) continue;
          touched += 1;
          try {
            await screener.process(f, sweepFilled, sweepBlanks, userSkippedKeys);
          } catch (err: any) {
            console.warn(
              `[Workday] Sweep skip "${escapePromptValue(f.label)}" (${err?.message || err})`,
            );
            sweepBlanks.push({ label: f.label, reason: `fill threw: ${err?.message || err}` });
          }
        }
        if (touched === 0) break;
      }
      if (sweepFilled.length) {
        console.log(`[Workday] Final sweep filled ${sweepFilled.length} field(s):`);
        for (const l of sweepFilled) console.log(`[Workday]   filled: ${escapePromptValue(l)}`);
      }

      // Definitive identity overwrite — correct any resume-parse misattribution
      // that only surfaced on later steps.
      await this.fillIdentityFields(profile);

      const stillBlank = await finalReverify({
        tag: "WorkdayAdapter",
        collect: () => this.collectQuestions(),
        isEmpty: async (f) => !(await this.hasValue(f)),
        skippedKeys: userSkippedKeys,
        reasons: [...blanked, ...sweepBlanks],
      });
      // Surface how many required fields are still blank so the runner can
      // gate auto-submit on an incomplete form.
      setBlankedRequiredCount(stillBlank.length);

      if (profile.resumePath && !resumeAttached && !(await this.controls.isResumeAttached())) {
        console.warn("[Workday] REVERIFY: resume is NOT attached after the final pass.");
      } else if (profile.resumePath) {
        console.log("[Workday] REVERIFY: resume is attached.");
      }
    }

    console.log("[Workday] Form filling completed.");
  }

  async submit(): Promise<SubmitOutcome> {
    const page = this.getPage();
    console.log("[Workday] Submitting application form...");
    const submitBtn = page
      .locator(
        '[data-automation-id="bottom-navigation-submit-button"], ' +
          '[data-automation-id="submitButton"], ' +
          '[data-automation-id="pageFooterSubmitButton"], ' +
          'button:has-text("Submit Application")',
      )
      .first();
    if (await submitBtn.isVisible().catch(() => false)) {
      // The final review screen can wrap Submit in the same invisible
      // click_filter overlay that swallows trusted Playwright clicks on the
      // wizard's nav buttons — drive it through the same synthetic dispatch.
      await this.clickSubmitAction(page, submitBtn);
    } else {
      await this.stagehand.act("Click the Submit Application button");
    }
    await randomSleep(1500, 2500);

    // Verify a success/error outcome like Lever.
    for (let i = 0; i < 10; i++) {
      const url = page.url();
      if (/thanks|submitted|confirmation|success/i.test(url)) {
        console.log("[Workday] Submitted: redirect confirmed.");
        return { confirmed: true, retryable: false };
      }
      // Never use `:visible` inside a comma-combined selector (Playwright fails
      // to parse it and matches nothing). Check alert roles and error blocks
      // with separate locators.
      const err =
        (await page
          .locator('[role="alert"]')
          .first()
          .innerText()
          .catch(() => "")) ||
        (await page
          .locator('.error, .error-message, [class*="error"]')
          .first()
          .innerText()
          .catch(() => ""));
      if (err && !/exceeds? the maximum upload size|too large|100MB/i.test(err)) {
        console.error(`[Workday] Submit error banner: ${escapePromptValue(err)}`);
        return {
          confirmed: false,
          error: `Workday submit failed (form re-rendered): ${escapePromptValue(err)}`,
          retryable: true,
        };
      }
      await randomSleep(1500, 2000);
    }
    console.warn(`[Workday] Submit outcome not detected at ${page.url()}; treating as failed.`);
    return {
      confirmed: false,
      error: "Workday submit: no success or error outcome detected after clicking submit",
      retryable: false,
    };
  }

  /**
   * Recheck and re-fill any required field that is still blank, then report
   * how many remain blank. Called by the runner after a retryable submit
   * failure (validation blocked by unfilled fields).
   */
  async recheckMissingFields(rpc?: RpcHelper): Promise<number> {
    console.log("[Workday] Rechecking missing required fields...");
    const stillBlank: string[] = [];
    const fields = await this.collectQuestions();
    for (const f of fields) {
      if (!f.required) continue;
      if (await this.hasValue(f)) continue;
      if (PRE_FILLED_LABELS.has(normalizeOptionText(f.label))) continue;
      const screener = new Screener(
        this.controls,
        "WorkdayAdapter",
        this.profile,
        rpc ?? (async () => ({ answer: "" })),
        true,
      );
      const filled: string[] = [];
      const blanked: { label: string; reason: string }[] = [];
      const skipped = new Set<string>();
      await screener.process(f, filled, blanked, skipped);
      if (filled.length === 0) stillBlank.push(f.label);
    }
    const remaining = stillBlank.length;
    setBlankedRequiredCount(remaining);
    console.log(`[Workday] Recheck complete: ${remaining} required field(s) still blank.`);
    for (const l of stillBlank) {
      console.warn(`[Workday]   still blank: ${escapePromptValue(l)}`);
    }
    return remaining;
  }
}

// ---------------------------------------------------------------------------
// Workday interaction layer — subclasses shared FormControls so the shared
// Screener / audit machinery runs unchanged. Adds Workday's combobox handling
// (options are `li[role="option"]` in a portal) and scoped value reads.
// ---------------------------------------------------------------------------

export class WorkdayControlStack extends FormControls {
  constructor(stagehand: Stagehand, tag: string) {
    super(stagehand, {
      tagName: tag,
      optionSelector: 'li[role="option"], div[role="option"], button[role="option"]',
      optionTag: "*",
    });
  }

  private scope(f: FormField): string {
    return f.name ? `[data-automation-id="${cssEscape(f.name)}"]` : cssIdLocator(f.id);
  }

  /** Committed value of a question (verification + audit). Comboboxes read the
   *  selected text from the combobox input; radio/checkbox groups read the
   *  checked option's label; text fields fall back to the automation-id scope
   *  (the input itself OR a descendant — Workday puts the automation-id on the
   *  control directly) when the input carries no DOM id; else delegates. */
  override async readFieldValue(field: FormField): Promise<string> {
    if (field.kind === "text") {
      const page = this.getPage();
      try {
        return (await page.evaluate(
          (fid: string, fname: string) => {
            const byId = document.getElementById(fid);
            if (byId && (byId instanceof HTMLInputElement || byId instanceof HTMLTextAreaElement)) {
              return (byId.value || "").trim();
            }
            if (fname) {
              const scope = document.querySelector(`[data-automation-id="${fname}"]`);
              if (scope) {
                const c = scope.matches(
                  'input[type="text"], input[type="email"], input[type="tel"], input[type="url"], input[type="date"], input:not([type]), textarea',
                )
                  ? scope
                  : scope.querySelector(
                      'input[type="text"], input[type="email"], input[type="tel"], input[type="url"], input[type="date"], input:not([type]), textarea',
                    );
                if (c) return ((c as HTMLInputElement).value || "").trim();
              }
            }
            return "";
          },
          field.id,
          field.name || "",
        )) as string;
      } catch {
        return "";
      }
    }
    if (field.kind === "combobox") {
      const page = this.getPage();
      try {
        const v = await page.evaluate((fid: string) => {
          const byId = document.getElementById(fid) as HTMLInputElement | null;
          if (byId && byId instanceof HTMLInputElement && byId.value) {
            return (byId.value || "").trim();
          }
          const scope = document.querySelector(`[data-automation-id="${fid}"]`);
          if (scope) {
            const c = scope.matches('input[role="combobox"], input[aria-autocomplete]')
              ? scope
              : scope.querySelector('input[role="combobox"], input[aria-autocomplete]');
            if (c) return ((c as HTMLInputElement).value || "").trim();
          }
          return "";
        }, field.id);
        return (v as string) || "";
      } catch {
        return "";
      }
    }
    if (field.kind === "radio" || field.kind === "checkbox") {
      const page = this.getPage();
      try {
        return (await page.evaluate(
          (gname: string) => {
            const checked = document.querySelector(
              `input[type="radio"][name="${gname}"]:checked, input[type="checkbox"][name="${gname}"]:checked`,
            ) as HTMLInputElement | null;
            if (!checked) return "";
            const row =
              checked.closest("label") ||
              checked.closest("[role='radio'], [role='checkbox'], [class*='option']");
            const rowText = row ? (row.textContent || "").replace(/\s+/g, " ").trim() : "";
            const lab = checked.id
              ? document.querySelector(`label[for="${CSS.escape(checked.id)}"]`)?.textContent || ""
              : "";
            return (
              rowText ||
              lab.trim() ||
              checked.getAttribute("aria-label") ||
              checked.value ||
              ""
            ).trim();
          },
          cssEscape(field.optionTargets[0]?.name || field.name || field.id),
        )) as string;
      } catch {
        return "";
      }
    }
    if (field.kind === "select") {
      const page = this.getPage();
      try {
        return (await page.evaluate((id: string) => {
          const sel = document.getElementById(id) as HTMLSelectElement | null;
          if (!sel) return "";
          const idx = sel.selectedIndex;
          if (idx < 0) return "";
          return (sel.options[idx]?.textContent || "").replace(/\s+/g, " ").trim();
        }, field.id)) as string;
      } catch {
        return "";
      }
    }
    return super.readFieldValue(field);
  }

  /** Kind-aware fill. Workday question dropdowns are comboboxes answered by
   *  picking a suggestion (never raw typed text); plain text fields fall back
   *  to the automation-id scope when the input carries no DOM id; the rest use
   *  the shared machinery unchanged. */
  override async fillByKind(
    field: FormField,
    answer: string,
    optionTexts?: string[],
  ): Promise<boolean> {
    if (field.kind === "combobox") {
      return this.fillWorkdayCombobox(field, answer, optionTexts ?? []);
    }
    if (field.kind === "text") {
      return this.fillWorkdayText(field, answer);
    }
    return super.fillByKind(field, answer, optionTexts);
  }

  /** Fill a plain text/date field. Prefers the field's automation-id scope
   *  (the control itself OR a descendant), then the input's real DOM id — the
   *  automation-id is unique across steps and is the more reliable anchor.
   *  Refuses to type into combobox/select elements (those are answered by
   *  picking options, never by typing free text). */
  async fillWorkdayText(field: FormField, answer: string): Promise<boolean> {
    const page = this.getPage();
    const textSel =
      'input[type="text"], input[type="email"], input[type="tel"], input[type="url"], input[type="date"], input:not([type]), textarea';
    const isComboboxLike = (scopeSel: string): Promise<boolean> =>
      page
        .evaluate((sel: string) => {
          // The first text-like control that matches, or the scope itself when
          // it IS a control. If it is a combobox/select we must not type into it.
          const scope = document.querySelector(sel);
          if (!scope) return false;
          const cand = scope.matches("input, textarea, select")
            ? scope
            : scope.querySelector("input, textarea, select");
          if (!cand) return false;
          if (cand.tagName === "SELECT") return true;
          return (
            cand.getAttribute("role") === "combobox" || !!cand.getAttribute("aria-autocomplete")
          );
        }, scopeSel)
        .catch(() => false);
    try {
      if (field.name) {
        const base = this.scope(field);
        const scopeSel = `${base} ${textSel}`;
        const scoped = page.locator(`${base}:is(input, textarea), ${scopeSel}`).first();
        if (await scoped.isVisible().catch(() => false)) {
          if (await isComboboxLike(base)) return false;
          await scoped.fill(String(answer ?? "")).catch(() => {});
          await randomSleep(200, 500);
          return true;
        }
      }
      const byId = page.locator(cssIdLocator(field.id)).first();
      if (await byId.isVisible().catch(() => false)) {
        if (await isComboboxLike(cssIdLocator(field.id))) return false;
        await byId.fill(String(answer ?? "")).catch(() => {});
        await randomSleep(200, 500);
        return true;
      }
      return false;
    } catch (err: any) {
      console.warn(`[${this.tagName}] fillWorkdayText failed: ${err?.message || err}`);
      return false;
    }
  }

  /**
   * Fill a Workday combobox. Typing is only a trigger to load/filter options;
   * the committed value MUST be a picked suggestion — raw typed text is never
   * accepted as an answer. Options come from the field's own DOM (read via the
   * open menu) when the walker did not collect them.
   */
  async fillWorkdayCombobox(
    field: FormField,
    answer: string,
    optionTexts: string[] = [],
  ): Promise<boolean> {
    const page = this.getPage();
    try {
      let input: any;
      if (field.name) {
        const base = this.scope(field);
        input = page
          .locator(
            `${base}:is(input[role="combobox"], input[aria-autocomplete]), ` +
              `${base} input[role="combobox"], ${base} input[aria-autocomplete]`,
          )
          .first();
        if (!(await input.isVisible().catch(() => false))) {
          input = page.locator(cssIdLocator(field.id)).first();
        }
      } else {
        input = page.locator(cssIdLocator(field.id)).first();
      }
      if (!(await input.isVisible().catch(() => false))) return false;
      await this.closeMenu();
      await randomSleep(150, 300);
      await input.click();
      await randomSleep(200, 350);

      let opts = optionTexts.length ? optionTexts.slice() : [];
      if (opts.length === 0) {
        await input.fill(answer);
        for (let i = 0; i < 8 && opts.length === 0; i++) {
          await randomSleep(900, 1200);
          opts = await this.readVisibleOptionTexts();
        }
      }
      if (opts.length === 0) {
        const short = answer.split(/[\s,]+/).find((t) => t && t.length > 1);
        if (short && short !== answer.trim()) {
          await input.fill(short);
          for (let i = 0; i < 6 && opts.length === 0; i++) {
            await randomSleep(900, 1200);
            opts = await this.readVisibleOptionTexts();
          }
        }
      }
      if (opts.length) {
        // Only a genuine async location autocomplete may fall back to a ranked
        // first suggestion (pickLocationOption). For any real question a blind
        // first-option guess is never acceptable — an unmatched answer stays
        // blank and is surfaced by the reverify instead of mis-committed.
        const isLocation = isLocationAutocomplete(field);
        const picked =
          chooseOption(selectCandidates(answer), opts) ??
          (isLocation ? pickLocationOption(answer, opts) : null);
        if (picked && (await this.clickVisibleOption(picked))) {
          await this.closeMenu();
          await randomSleep(300, 500);
          if (await this.readFieldValue(field)) return true;
          return false;
        }
      }
      await this.closeMenu();
      console.warn(
        `[${this.tagName}] No selectable suggestion for "${answer}" (${escapePromptValue(field.label)}).`,
      );
      return false;
    } catch (err: any) {
      console.warn(`[${this.tagName}] fillWorkdayCombobox failed: ${err?.message || err}`);
      return false;
    }
  }

  /** Fill an identity/short text input by its data-automation-id. */
  async fillByAutomationId(aid: string, value: string): Promise<void> {
    if (!value) return;
    const page = this.getPage();
    const input = page
      .locator(`input[data-automation-id="${aid}"], textarea[data-automation-id="${aid}"]`)
      .first();
    if (await input.isVisible().catch(() => false)) {
      await input.fill(value).catch(() => {});
      await randomSleep(150, 300);
    }
  }

  /** Fill an identity/short text input by its plain DOM id (Workday step-1
   *  fields like `legalName--firstName` carry ids, not data-automation-id).
   *  Identity fields are text/email/tel — never file inputs. */
  async fillById(id: string, value: string): Promise<void> {
    if (!value) return;
    const page = this.getPage();
    const input = page.locator(cssIdLocator(id)).first();
    if (await input.isVisible().catch(() => false)) {
      await input.fill(value).catch(() => {});
      await randomSleep(150, 300);
    }
  }

  /** Whether the resume is attached (input consumed, has files, or the
   *  upload zone shows the file name / an upload state). */
  async isResumeAttached(): Promise<boolean> {
    const page = this.getPage();
    return page
      .evaluate(() => {
        const input = document.querySelector(
          'input[type="file"][data-automation-id="resume"], input[type="file"][name="resume"]',
        ) as HTMLInputElement | null;
        if (!input) return true; // consumed by the board = attached
        if (input.files && input.files.length > 0) return true;
        const zone = document.querySelector('[data-automation-id="resume"], [class*="resume"]');
        const text = zone ? zone.textContent || "" : "";
        return /attached|uploaded|✓|added|done/i.test(text);
      })
      .catch(() => false);
  }
}
