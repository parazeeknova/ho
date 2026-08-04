import { Stagehand, type Action } from "@browserbasehq/stagehand";
import * as fs from "fs";
import { ATSAdapter, RpcHelper } from "./base.js";
import { JobPayload, Profile } from "../types.js";
import { randomSleep } from "../utils/evasion.js";
import { auditBlanks, finalReverify, SubmitOutcome, verifySubmitOutcome } from "./shared/audit.js";
import { FormControls, sanitizeNumberAnswer } from "./shared/controls.js";
import {
  chooseOption,
  cssEscape,
  cssIdLocator,
  escapePromptValue,
  pickLocationOption,
  selectCandidates,
} from "./shared/matching.js";
import {
  fieldKey,
  FormField,
  isLocationAutocomplete,
  JsonFieldSource,
  mergeFormInventory,
  PRE_FILLED_LABELS,
} from "./shared/model.js";
import { BlankEntry, Screener, setBlankedRequiredCount } from "./shared/screener.js";

/**
 * GenericAdapter — the intelligent fallback for ANY job application form.
 *
 * It is not a board-specific adapter; it classifies the *shape* of whatever
 * page it lands on and drives the shared fill/audit/reverify machinery through
 * it. The layered strategy (deterministic-first, LLM only as a scoped
 * fallback) is what makes unknown forms reliable:
 *
 *   1. FLOW CLASSIFICATION — a cheap DOM probe decides whether the page is a
 *      JD page (click Apply), a single form, a multi-step wizard, or a
 *      sign-in/account gate, then drives the right navigation.
 *   2. GENERIC DOM WALKER — enumerates questions from standard accessibility
 *      semantics (label[for], wrapping labels, aria-labelledby/aria-label,
 *      fieldset/legend, name-grouped radio/checkbox, native selects,
 *      role=combobox/aria-autocomplete dropdowns). Controls without an id are
 *      given a synthetic `data-field-path` so the SHARED scope-based fill and
 *      verification machinery works unchanged.
 *   3. EMBEDDED JSON PROBE — when the page ships a question model in a
 *      script blob (window.__remixContext, __CONFIG, __NEXT_DATA__, or any
 *      JSON containing a "questions" array) it is merged into the DOM
 *      inventory via the shared mergeFormInventory.
 *   4. WIZARD / GATE MACHINE — text/aria-driven Continue/Next/Submit stepping
 *      and best-effort gate handling (guest path, env credentials, else the
 *      gate is deferred for a human and surfaced by finalReverify).
 *   5. OBSERVE FALLBACK — only when the DOM walker finds ZERO fields (a truly
 *      non-standard renderer) does it fall back to Stagehand observe() to
 *      locate fillable text fields and fill them deterministically via act().
 *
 * JD extraction and reverification are non-negotiable: the job context is read
 * from every available source (live DOM, embedded JSON, fetched HTML og-tags)
 * and fed to rpc("job_context", …) before any question is answered, and every
 * empty field is reported through auditBlanks + finalReverify so nothing is
 * silently missed.
 */

export type FlowKind = "form" | "apply" | "wizard" | "gate";

/** The shape-of-the-page decision a DOM probe feeds into classifyFlow. */
export interface FlowProbe {
  formDetected: boolean;
  applyDetected: boolean;
  wizardDetected: boolean;
  gateDetected: boolean;
}

/** Pure flow classification — unit-testable in isolation. Gate wins, then
 *  wizard (multi-step), then a live form, then a JD page asking us to apply. */
export function classifyFlow(p: FlowProbe): FlowKind {
  if (p.gateDetected) return "gate";
  if (p.wizardDetected) return "wizard";
  if (p.formDetected) return "form";
  if (p.applyDetected) return "apply";
  return "form"; // nothing detected — the walker's zero-field result decides
}

/** Strip leading verbs and trailing scaffolding from an observe() description
 *  so it reads as a question label ("fill the First Name field" → "First
 *  Name"). Pure and unit-testable. */
export function cleanObserveLabel(description: string): string {
  let d = (description || "").replace(/\s+/g, " ").trim();
  d = d.replace(
    /^(?:please\s+)?(?:fill\s+(?:(?:in|out|the)\s+)+|fill\s+|type\s+|enter\s+|input\s+|set\s+|provide\s+)/i,
    ""
  );
  d = d.replace(/\s+(?:field|input|box|dropdown|textbox)\s*$/i, "");
  d = d.replace(/^(?:the|your|my|a|an)\s+/i, "").trim();
  return d;
}

/** Pure test of voluntary-disclosure (EEOC-style) screen text. */
export function isVoluntaryText(text: string): boolean {
  return /voluntary|self[- ]identif|demographic|eeoc|diversity|equal opportunity/i.test(
    text || ""
  );
}

/**
 * Extract the first balanced `{...}` JSON object from `html` starting at the
 * first occurrence of `marker`. Returns the raw JSON substring or null.
 */
export function extractBalancedObject(html: string, marker: RegExp): string | null {
  const m = html.match(marker);
  if (!m || m.index === undefined) return null;
  const start = html.indexOf("{", m.index + m[0].length);
  if (start < 0) return null;
  let depth = 0;
  let inStr = false;
  let esc = false;
  for (let i = start; i < html.length; i++) {
    const c = html[i];
    if (inStr) {
      if (esc) esc = false;
      else if (c === "\\") esc = true;
      else if (c === '"') inStr = false;
      continue;
    }
    if (c === '"') inStr = true;
    else if (c === "{") depth++;
    else if (c === "}") {
      depth--;
      if (depth === 0) return html.slice(start, i + 1);
    }
  }
  return null;
}

/**
 * Recursively walk a parsed JSON object and collect every board-style question
 * model it contains: objects with a `questions` array whose items carry
 * `label` (and optionally `fields`/`values` — Greenhouse's remix shape) and
 * `eeoc_sections[].questions`. File/resume/cover-letter entries are dropped
 * (uploads are handled by dedicated paths). Pure and unit-testable.
 */
export function extractQuestionsFromJsonObject(root: any): JsonFieldSource[] {
  const out: JsonFieldSource[] = [];
  const seen = new Set<string>();
  const visit = (obj: any): void => {
    if (!obj || typeof obj !== "object") return;
    if (Array.isArray(obj)) {
      for (const it of obj) visit(it);
      return;
    }
    const qs = Array.isArray(obj.questions) ? obj.questions : null;
    if (qs && qs.length) {
      const looksLikeQuestions = qs.every(
        (q: any) => q && typeof q === "object" && (q.label !== undefined || q.name !== undefined)
      );
      if (looksLikeQuestions) {
        for (const q of qs) {
          const label = String(q?.label || "").replace(/\s+/g, " ").trim();
          if (!label) continue;
          const fields =
            Array.isArray(q?.fields) && q.fields.length ? q.fields : [q];
          for (const f of fields) {
            const name = String(f?.name || q?.name || "");
            const kind = String(f?.type || q?.type || "input_text");
            if (/^(input_file|file|signature)$/i.test(kind)) continue;
            if (/^resume(_text)?$|^cover_letter(_text)?$/.test(name)) continue;
            const key = `${name}|${label}`;
            if (seen.has(key)) continue;
            seen.add(key);
            out.push({
              name,
              label,
              kind,
              required: !!q?.required,
              options: (f?.values ?? [])
                .map((v: any) =>
                  (v?.label ?? v?.name ?? v?.value ?? "").toString().trim()
                )
                .filter(Boolean),
            });
          }
        }
      }
    }
    for (const k of Object.keys(obj)) {
      if (k === "questions" && qs?.length) continue; // already handled above
      visit(obj[k]);
    }
  };
  visit(root);
  return out;
}

const JSON_MARKERS: RegExp[] = [
  /window\.__remixContext\s*=\s*/,
  /window\.__CONFIG\s*=\s*/,
  /__NEXT_DATA__\s*=\s*/,
  /window\.__INITIAL_STATE__\s*=\s*/,
  /window\.__INITIAL_DATA__\s*=\s*/,
  /window\.__APP_DATA__\s*=\s*/i,
  /window\.__appData\s*=\s*/,
];

/** Raw JSON blobs found in the page: marker-assigned globals plus the inline
 *  JSON payload of a `<script id="__NEXT_DATA__">` (which is the script's own
 *  text, not an assignment). */
function collectJsonBlobs(html: string): string[] {
  const blobs: string[] = [];
  for (const marker of JSON_MARKERS) {
    const raw = extractBalancedObject(html, marker);
    if (raw) blobs.push(raw);
  }
  const script = html.match(
    /<script[^>]*__NEXT_DATA__[^>]*>([\s\S]*?)<\/script>/i
  );
  if (script && script[1] && script[1].trim()) blobs.push(script[1].trim());
  return blobs;
}

/**
 * Parse every embedded JSON blob we know how to find and pull the question
 * model out of it. Returns a deduplicated list (possibly empty).
 */
export function parseJsonQuestions(html: string): JsonFieldSource[] {
  const out: JsonFieldSource[] = [];
  const seenKey = new Set<string>();
  const add = (list: JsonFieldSource[]) => {
    for (const f of list) {
      const key = `${f.name}|${f.label}`;
      if (seenKey.has(key)) continue;
      seenKey.add(key);
      out.push(f);
    }
  };
  for (const raw of collectJsonBlobs(html)) {
    try {
      add(extractQuestionsFromJsonObject(JSON.parse(raw)));
    } catch {
      // Malformed blob — skip and keep probing other blobs.
    }
  }
  return out;
}

/** Best-effort job context (title/company/location/description) from embedded
 *  JSON blobs. Pure and unit-testable. */
export function extractJsonJobContext(
  html: string
): { title: string; company: string; location: string; description: string } | null {
  for (const raw of collectJsonBlobs(html)) {
    try {
      const found: { title: string; company: string; location: string; description: string } | null =
        (() => {
          const walk = (obj: any): any => {
            if (!obj || typeof obj !== "object") return null;
            if (obj.jobPost && (obj.jobPost.title || obj.jobPost.company_name)) return obj.jobPost;
            if (obj.posting && obj.posting.title) return obj.posting;
            if (obj.job && obj.job.title) return obj.job;
            for (const k of Object.keys(obj)) {
              const r = walk(obj[k]);
              if (r) return r;
            }
            return null;
          };
          const jp = walk(JSON.parse(raw));
          if (!jp) return null;
          const htmlDesc = String(jp?.descriptionHtml || jp?.description || "").replace(
            /<[^>]+>/g,
            " "
          );
          return {
            title: String(jp?.title || "").replace(/\s+/g, " ").trim(),
            company: String(jp?.company_name || jp?.company || jp?.companyName || "")
              .replace(/\s+/g, " ")
              .trim(),
            location: String(jp?.job_post_location || jp?.location || jp?.locationName || "")
              .replace(/\s+/g, " ")
              .trim(),
            description: htmlDesc.replace(/\s+/g, " ").trim().slice(0, 6000),
          };
        })();
      if (found && (found.title || found.description)) return found;
    } catch {
      // keep probing
    }
  }
  return null;
}

export class GenericAdapter extends ATSAdapter {
  protected controls!: GenericControls;
  private jobCtx: {
    title: string;
    company: string;
    location: string;
    description: string;
  } | null = null;
  private gateDeferred = false;
  private _jsonModel: JsonFieldSource[] | null = null;
  protected profile!: Profile;

  constructor(stagehand: Stagehand) {
    super(stagehand);
    this.controls = new GenericControls(stagehand, "GenericAdapter");
  }

  protected getPage(): any {
    return this.controls.getPage();
  }

  /** The generic adapter navigates within one tab (or adopts a form tab); the
   *  adopted page is the one to screenshot. */
  getActivePage(): any {
    return this.controls.getPage();
  }

  private warn(msg: string): void {
    console.warn(`[Generic] ${msg}`);
  }

  // --------------------------------------------------------------------------
  // Flow classification & navigation
  // --------------------------------------------------------------------------

  /** Cheap DOM probe that decides the shape of the current page. */
  private async probeFlow(): Promise<FlowProbe> {
    const page = this.getPage();
    try {
      // WARNING: only anonymous arrows (tsx keepNames stringifies the callback
      // into the page; a named arrow's __name wrapper would throw).
      return (await page.evaluate(() => {
        const [visible, hasText] = [
          (el: Element | null): boolean => {
            if (!el) return false;
            const e = el as HTMLElement;
            const r = e.getBoundingClientRect();
            if (r.width === 0 && r.height === 0) return false;
            const cs = getComputedStyle(e);
            if (cs.display === "none" || cs.visibility === "hidden" || cs.opacity === "0")
              return false;
            return true;
          },
          (re: RegExp) => buttons.some((t) => re.test(t)),
        ];
        const buttons = Array.from(
          document.querySelectorAll(
            "a, button, [role='button'], input[type='button'], input[type='submit']"
          )
        )
          .filter((el) => visible(el))
          .map((el) => {
            const t = ((el as HTMLElement).textContent || (el as HTMLInputElement).value || "")
              .replace(/\s+/g, " ")
              .trim()
              .toLowerCase();
            return t;
          })
          .filter(Boolean);
        // Only inputs inside a real <form> or the <main> content region count
        // as form controls. A JD page's header/nav search box is neither — if
        // it counted, formDetected would suppress the Apply click and the walk
        // would treat the JD page as the form.
        const controls = Array.from(
          document.querySelectorAll(
            "form input, form select, form textarea, " +
              "main input, main select, main textarea, " +
              "[role='main'] input, [role='main'] select, [role='main'] textarea"
          )
        ).filter((el) => {
          const e = el as HTMLInputElement;
          if (e.type === "hidden" || e.type === "file") return false;
          if (!visible(el)) return false;
          return true;
        });
        const passwordFields = Array.from(
          document.querySelectorAll('input[type="password"]')
        ).filter((el) => visible(el)).length;
        const formDetected = controls.length > 0;
        const stepIndicator = /step\s*\d+\s+of\s+\d+/i.test(
          (document.body?.innerText || "").slice(0, 3000)
        );
        const gateDetected =
          passwordFields > 0 &&
          (hasText(/sign\s*in|log\s*in|create\s*account|account/) ||
            controls.length === 1);
        const applyText = hasText(
          /^(apply|apply now|apply for this job|apply for this position|start application|apply here|apply online)$/i
        ) || hasText(/apply\s*(now|for this job|for this position|online|here|to this job)$/i);
        const applyHref =
          !formDetected &&
          !!document.querySelector("a[href*='apply' i], [role='button'][href*='apply' i]");
        const wizardDetected =
          stepIndicator ||
          hasText(/^(continue|next|next step|save and continue|proceed|continue application)$/i) ||
          !!document.querySelector("[role='tablist'], [class*='stepper'], [class*='progress']");
        return {
          formDetected,
          applyDetected: !formDetected && (applyText || applyHref),
          wizardDetected,
          gateDetected,
        };
      })) as FlowProbe;
    } catch {
      return {
        formDetected: false,
        applyDetected: false,
        wizardDetected: false,
        gateDetected: false,
      };
    }
  }

  private async waitForFormOrWizard(): Promise<void> {
    const page = this.getPage();
    for (let i = 0; i < 40; i++) {
      const probe = await this.probeFlow();
      const kind = classifyFlow(probe);
      if (kind === "form" || kind === "wizard" || kind === "gate") return;
      await randomSleep(800, 1200);
    }
  }

  private async clickApply(): Promise<void> {
    const page = this.getPage();
    const apply = page
      .locator(
        "a[href*='apply' i], button:has-text('Apply'), a:has-text('Apply'), " +
          "[role='button']:has-text('Apply'), input[value*='Apply' i]"
      )
      .first();
    if (await apply.isVisible().catch(() => false)) {
      await apply.click().catch(() => {});
      await randomSleep(1500, 2500);
      return;
    }
    await this.controls.safeAct(
      "click the 'Apply', 'Apply now', or 'Apply for this job' button or link to open the application form"
    );
  }

  /**
   * Land on the application form. JD page → capture the JD (the form view
   * usually drops it) → click Apply → adopt a form tab if one opened → wait.
   * A sign-in/account gate is then handled (or deferred for a human).
   */
  private async ensureApplicationView(): Promise<void> {
    let probe = await this.probeFlow();
    if (classifyFlow(probe) === "apply") {
      this.jobCtx = await this.readJobContext();
      console.log("[Generic] JD page detected; clicking the Apply button...");
      await this.clickApply();
      await this.waitForFormOrWizard();
      probe = await this.probeFlow();
      if (classifyFlow(probe) === "apply") {
        // The click did not navigate — adopt a form tab if one opened. Probe
        // each candidate WITHOUT permanently retargeting the active page until
        // one qualifies; a page that is still loading or a modal must not
        // strand the walk on the wrong tab.
        let adopted = false;
        const previous = this.getPage();
        for (const p of this.stagehand.context.pages()) {
          if (p === previous) continue;
          this.controls.adoptPage(p);
          const kind = classifyFlow(await this.probeFlow());
          if (kind === "form" || kind === "wizard" || kind === "gate") {
            adopted = true;
            break;
          }
        }
        if (!adopted) {
          this.controls.adoptPage(previous);
          console.warn("[Generic] Apply click did not reveal a form; proceeding to the walk.");
        }
      }
    }
    probe = await this.probeFlow();
    if (classifyFlow(probe) === "gate") {
      console.log("[Generic] Sign-in/account gate detected; attempting to pass it.");
      await this.handleGate();
      await this.waitForFormOrWizard();
    }
  }

  /** Best-effort gate handling: guest path → env credentials → defer to human.
   *  A deferred gate is recorded and surfaced by finalReverify. */
  private async handleGate(): Promise<void> {
    const page = this.getPage();
    const guest = page
      .locator(
        "button:has-text('Continue without signing in'), button:has-text('Continue as guest'), " +
          "button:has-text('Apply without signing in'), button:has-text('Skip for now'), " +
          "a:has-text('Continue without signing in')"
      )
      .first();
    if (await guest.isVisible().catch(() => false)) {
      await guest.click();
      await randomSleep(2000, 3000);
      return;
    }
    const user = process.env.WORKDAY_EMAIL;
    const pw = process.env.WORKDAY_PASSWORD;
    if (user && pw) {
      const email = page
        .locator('input[type="email"], input[name*="email" i]')
        .first();
      if (await email.isVisible().catch(() => false)) {
        await email.fill(user);
        const password = page.locator('input[type="password"]').first();
        await password.fill(pw).catch(() => {});
        const submit = page
          .locator("button[type='submit'], button:has-text('Sign in'), button:has-text('Log in')")
          .first();
        if (await submit.isVisible().catch(() => false)) {
          await submit.click();
          await randomSleep(2000, 3000);
          return;
        }
      }
    }
    this.gateDeferred = true;
    this.warn(
      "Sign-in/account gate cannot be passed automatically (no guest path and " +
        "WORKDAY_EMAIL/WORKDAY_PASSWORD not usable). Form fields will be reported " +
        "as deferred for manual completion."
    );
  }

  private async hasSubmitButton(): Promise<boolean> {
    const page = this.getPage();
    const b = page
      .locator(
        "button:has-text('Submit Application'), button:has-text('Submit'), " +
          "input[type='submit'], button[type='submit'], a:has-text('Submit Application'), " +
          "[data-automation-id*='submit' i]"
      )
      .first();
    return await b.isVisible().catch(() => false);
  }

  private async advanceStep(): Promise<boolean> {
    const page = this.getPage();
    const btn = page
      .locator(
        "button:has-text('Save and Continue'), button:has-text('Continue'), " +
          "button:has-text('Next Step'), button:has-text('Next'), button:has-text('Proceed'), " +
          "[role='button']:has-text('Continue'), [data-automation-id*='continue' i]"
      )
      .first();
    if (!(await btn.isVisible().catch(() => false))) return false;
    await btn.click();
    await randomSleep(2000, 3000);
    return true;
  }

  /** A step is a voluntary disclosure (EEOC-style survey) when every field is
   *  optional AND either its field labels OR its own heading carry a survey
   *  marker. Such steps are legally optional — never guessed, just advanced
   *  past. A body-wide phrase is never enough (ubiquitous "Equal Opportunity
   *  Employer" footers would wrongly skip ordinary steps). */
  private async isVoluntaryStep(fields: FormField[]): Promise<boolean> {
    if (fields.some((f) => f.required)) return false;
    const surveyLabels = fields.some((f) =>
      /race|ethnic|gender|sex|veteran|disability|orientation|military|self[- ]identif|voluntary|diversity/i.test(
        f.label
      )
    );
    if (surveyLabels) return true;
    // Heading/legend-level marker only — never body footer text.
    const page = this.getPage();
    const heading: string = await page
      .evaluate(() => {
        const main = document.querySelector("main, [role='main']");
        const scope = main || document.body;
        const cand = scope.querySelector(
          "h1, h2, h3, legend, [data-automation-label], [class*='title']"
        );
        return (cand ? cand.textContent || "" : "").replace(/\s+/g, " ").trim();
      })
      .catch(() => "");
    return isVoluntaryText(heading);
  }

  // --------------------------------------------------------------------------
  // Job context (JD extraction — wherever possible)
  // --------------------------------------------------------------------------

  /**
   * Extract the job posting context from every available source: live DOM
   * (h1, title/location/description selectors, og tags), embedded JSON blobs,
   * and fetched server-rendered HTML og tags. Used to personalize open-ended
   * answers via rpc("job_context", …).
   */
  private async readJobContext(): Promise<{
    title: string;
    company: string;
    location: string;
    description: string;
  }> {
    const page = this.getPage();
    try {
      // WARNING: only anonymous arrows (tsx keepNames stringifies the callback
      // into the page; a named arrow's __name wrapper would throw).
      const live: any = await page.evaluate(() => {
        const [txt, meta] = [
          (sel: string): string => {
            const el = document.querySelector(sel);
            return el ? (el.textContent || "").replace(/\s+/g, " ").trim() : "";
          },
          (p: string): string =>
            document.querySelector(`meta[property="${p}"]`)?.getAttribute("content") || "",
        ];
        return {
          h1: txt("h1"),
          title: txt(
            "[class*='job-title'], [class*='jobTitle'], [data-automation-id='jobPostingHeader']"
          ),
          location: txt(
            "[data-automation-id='locations'], [class*='location'], [class*='job-location'], [class*='posting-location']"
          ),
          desc: txt(
            "#job-description, [class*='job-description'], [class*='job__description'], " +
              "[data-automation-id='jobPostingDescription']"
          ),
          ogTitle: meta("og:title"),
          ogDesc: meta("og:description"),
        };
      });
      let title = (live?.title || live?.ogTitle || live?.h1 || "").replace(/\s+/g, " ").trim();
      let description = (live?.desc || live?.ogDesc || "").slice(0, 6000);
      let location = (live?.location || "").replace(/\s+/g, " ").trim();

      try {
        const fetched = await fetch(page.url(), {
          headers: { "user-agent": "Mozilla/5.0" },
        });
        const html = await fetched.text();
        const json = extractJsonJobContext(html);
        if (json) {
          title = title || json.title;
          location = location || json.location;
          description = description || json.description;
        }
        const ogTitle =
          (html.match(/<meta[^>]*property="og:title"[^>]*content="([^"]*)"/i) || [])[1] || "";
        const ogDesc =
          (html.match(/<meta[^>]*property="og:description"[^>]*content="([^"]*)"/i) || [])[1] || "";
        title = title || ogTitle;
        description = description || ogDesc;
      } catch {
        // Best-effort; the live DOM values stand.
      }

      let company = "";
      try {
        const u = new URL(page.url());
        company =
          u.hostname.replace(/^(www|careers|jobs)\./, "").split(".")[0] || "";
        const pathToken = u.pathname.split("/").filter(Boolean)[0] || "";
        if (!company) company = pathToken;
      } catch {
        // fall through
      }

      return {
        title: title.replace(/\s+/g, " ").trim(),
        company: company.replace(/[-_]+/g, " ").trim(),
        location,
        description: description.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim().slice(0, 6000),
      };
    } catch (err: any) {
      this.warn(`readJobContext failed: ${err?.message || err}`);
      return { title: "", company: "", location: "", description: "" };
    }
  }

  /** Pull the embedded JSON question model from the fetched page HTML. */
  private async fetchJsonQuestions(): Promise<JsonFieldSource[] | null> {
    try {
      const res = await fetch(this.getPage().url(), {
        headers: { "user-agent": "Mozilla/5.0" },
      });
      const html = await res.text();
      const out = parseJsonQuestions(html);
      if (out.length) {
        console.log(`[Generic] Embedded JSON question model found (${out.length} question(s)).`);
      }
      return out.length ? out : null;
    } catch (err: any) {
      this.warn(`fetchJsonQuestions failed: ${err?.message || err}`);
      return null;
    }
  }

  // --------------------------------------------------------------------------
  // DOM inventory
  // --------------------------------------------------------------------------

  /**
   * Generic DOM walker. Enumerates questions from standard accessibility
   * semantics (label[for], wrapping labels, aria-labelledby/aria-label,
   * fieldset/legend, name-grouped radio/checkbox, native selects,
   * role=combobox/aria-autocomplete dropdowns). Controls without an id are
   * given a synthetic `data-field-path` so the shared scope-based fill and
   * verification machinery works unchanged.
   */
  private async collectQuestions(): Promise<FormField[]> {
    const page = this.getPage();
    try {
      const rows = await page.evaluate(() => {
        const out: Array<{
          label: string;
          id: string;
          name: string;
          kind: string;
          required: boolean;
          options: string[];
          targets: Array<{ text: string; name: string; value: string; id?: string }>;
        }> = [];
        // WARNING: anonymous arrows only (tsx keepNames stringifies functions
        // into the page; a named arrow's __name wrapper would throw).
        // `nextGenericId` is monotonic and persisted on <html> so synthetic
        // data-field-path ids stay UNIQUE and STABLE across the re-scan walks
        // (a conditional field revealed in pass 2 must not collide with a
        // field tagged in pass 1 — that would mis-fill/mis-read).
        const [norm, visible, inNav, labelOf, hasAsterisk, qesc, nextGenericId, singlePath, push] = [
          (t: string) =>
            (t || "").replace(/\s+/g, " ").trim().replace(/^\*+|\*+$/g, ""),
          (el: Element): boolean => {
            const e = el as HTMLElement;
            const r = e.getBoundingClientRect();
            if (r.width === 0 && r.height === 0) return false;
            // Honeypot guard: off-viewport-to-the-top/left (position:absolute;
            // left:-9999px) fields are traps, never real questions.
            if (r.right < 0 || r.bottom < 0) return false;
            if (e.getAttribute && e.getAttribute("tabindex") === "-1") return false;
            const cs = getComputedStyle(e);
            if (cs.display === "none" || cs.visibility === "hidden" || cs.opacity === "0")
              return false;
            let n: Element | null = e;
            while (n && n !== document.body) {
              if (n.getAttribute && n.getAttribute("hidden") != null) return false;
              n = n.parentElement;
            }
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
                ':scope > label, :scope > legend, :scope > [class*="label"], ' +
                  ':scope > [data-automation-label], :scope > h1, :scope > h2, :scope > h3'
              );
              if (cand) {
                const t = norm(cand.textContent || "");
                if (t && t.length < 160) return t;
              }
            }
            return "";
          },
          (el: Element): boolean => {
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
              const l = p.querySelector(":scope > label, :scope > legend, :scope > [data-automation-label]");
              if (test(l ? l.textContent : "")) return true;
            }
            return false;
          },
          (s: string): string =>
            (s || "").replace(/\\/g, "\\\\").replace(/"/g, '\\"'),
          (): string => {
            // Monotonic AND persisted on <html> so synthetic ids never collide
            // between re-scan walks (a conditional field revealed in pass 2
            // must not reuse an id already assigned in pass 1).
            const root = document.documentElement;
            const n =
              (parseInt(root.getAttribute("data-generic-id") || "0", 10) || 0) + 1;
            root.setAttribute("data-generic-id", String(n));
            return "generic-field-" + n;
          },
          (el: Element): string => {
            // A stable per-control scope id. Reuses the nearest existing
            // [data-field-path] (self-inclusive), else tags THIS element —
            // never a shared ancestor container (two id-less inputs in one
            // row must not collapse onto a single scope, which would
            // mis-fill/mis-read the second field).
            if (el.matches && el.matches("[data-field-path]")) {
              return el.getAttribute("data-field-path") || "";
            }
            const existing = el.closest("[data-field-path]");
            if (existing) return existing.getAttribute("data-field-path") || "";
            const fresh = nextGenericId();
            el.setAttribute("data-field-path", fresh);
            return fresh;
          },
          (
            label: string,
            id: string,
            name: string,
            kind: string,
            required: boolean,
            options: string[] = [],
            targets: Array<{ text: string; name: string; value: string; id?: string }> = []
          ): void => {
            if (!label) return;
            out.push({ label, id, name, kind, required, options, targets });
          },
        ];

        // Text-like controls (input[type=text/email/tel/number/url/date],
        // bare inputs, textareas). Comboboxes (role=combobox / aria-autocomplete
        // / react-select shells) are classified separately.
        const textSel =
          'input[type="text"], input[type="email"], input[type="tel"], input[type="number"], ' +
          'input[type="url"], input[type="date"], input:not([type]), textarea';
        const seenText = new Set<string>();
        for (const el of Array.from(document.querySelectorAll(textSel))) {
          const e = el as HTMLInputElement;
          if (e.type === "password" || e.type === "hidden" || e.type === "file") continue;
          if (!visible(e) || inNav(e)) continue;
          const label = labelOf(e);
          if (!label) continue;
          const combo =
            e.getAttribute("role") === "combobox" ||
            !!e.getAttribute("aria-autocomplete") ||
            !!e.closest('[class*="select-shell"], [class*="react-select"]');
          const dateish = !!e.closest(
            ".react-datepicker-wrapper, .react-datepicker, [class*='datepicker']"
          );
          let kind: string;
          if (combo) kind = "combobox";
          else if (dateish) kind = "date";
          else kind = "text";
          const key = norm(label).toLowerCase() + "|" + kind;
          if (seenText.has(key)) continue;
          seenText.add(key);
          const required =
            !!e.getAttribute("aria-required") ||
            e.hasAttribute("required") ||
            hasAsterisk(e);
          push(label, e.id || singlePath(e), e.name || "", kind, required);
        }

        // Native selects.
        const seenSel = new Set<string>();
        for (const el of Array.from(document.querySelectorAll("select"))) {
          const e = el as HTMLSelectElement;
          if (!visible(e) || inNav(e)) continue;
          const label = labelOf(e);
          if (!label) continue;
          const options = Array.from(e.options)
            .map((o) => norm(o.textContent || ""))
            .filter(Boolean);
          const key = norm(label).toLowerCase() + "|select";
          if (seenSel.has(key)) continue;
          seenSel.add(key);
          const required =
            !!e.getAttribute("aria-required") || e.hasAttribute("required") || /\*/.test(label);
          push(label, e.id || e.name || singlePath(e), e.name || "", "select", required, options);
        }

        // Radio/checkbox groups grouped by input name. Unnamed radios/checks
        // (very common for a lone consent checkbox like "I agree to the privacy
        // policy") must NOT be dropped — each is treated as its own singleton
        // group keyed by its element path, so the structural accept/leave logic
        // still sees it instead of silently losing a required gate.
        const seenGroups = new Set<string>();
        for (const el of Array.from(
          document.querySelectorAll('input[type="radio"], input[type="checkbox"]')
        )) {
          const e = el as HTMLInputElement;
          if (inNav(e)) continue;
          const type = e.type;
          const name = e.name || "";
          if (name && seenGroups.has(name)) continue;
          if (name) seenGroups.add(name);
          const group: HTMLInputElement[] = name
            ? (Array.from(
                document.querySelectorAll(`input[type="${type}"][name="${qesc(name)}"]`)
              ) as HTMLInputElement[])
            : [e];
          const targets: Array<{ text: string; name: string; value: string; id?: string }> = [];
          const options: string[] = [];
          let anyVisible = false;
          for (const g of group) {
            const wrapLabel = g.closest("label");
            const row = g.closest("[role='radio'], [role='checkbox'], [class*='option'], li");
            const gVisible =
              visible(g) ||
              (wrapLabel != null && visible(wrapLabel)) ||
              (row != null && visible(row));
            if (!gVisible) continue;
            anyVisible = true;
            const gid = g.getAttribute("id") || "";
            const labFor = gid
              ? (document.querySelector(`label[for="${qesc(gid)}"]`)?.textContent || "")
              : "";
            const text = norm(
              wrapLabel
                ? wrapLabel.textContent || ""
                : labFor || g.getAttribute("aria-label") || (row ? row.textContent || "" : "")
            );
            if (!text) continue;
            if (!targets.some((t) => t.text === text)) {
              targets.push({ text, name, value: g.value || "", id: gid });
            }
            if (!options.includes(text)) options.push(text);
          }
          if (!anyVisible || !targets.length) continue;

          // Group label from the container's legend/aria/heading — never an
          // option's own text.
          const container = e.closest(
            "[data-automation-id], fieldset, [role='radiogroup'], [role='group'], " +
              "[role='checkbox'], [class*='form-control'], [class*='field']"
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
                  ':scope > legend, :scope > [class*="label"], :scope > label, ' +
                    ':scope > h1, :scope > h2, :scope > h3, :scope > span'
                )
              )) {
                const t = norm(cand.textContent || "");
                if (!t || t.length > 160) continue;
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
              (g) => g.hasAttribute("required") || g.getAttribute("aria-required") === "true"
            ) ||
            /\*/.test(groupLabel) ||
            hasAsterisk(e);
          // Group scope: prefer the container (scoped reads find checked
          // inputs inside it), else the first option input itself.
          const groupScope = container ? singlePath(container) : singlePath(e);
          push(groupLabel, name || groupScope, name, kind, required, options, targets);
        }

        // De-dup by normalized label + kind, keep the first.
        const uniq: typeof out = [];
        const seen = new Set<string>();
        for (const r of out) {
          const key = norm(r.label).toLowerCase() + "|" + r.kind;
          if (seen.has(key)) continue;
          seen.add(key);
          uniq.push(r);
        }
        return uniq;
      });

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
        name: r.name || undefined,
      }));
    } catch (err: any) {
      this.warn(`collectQuestions failed: ${err?.message || err}`);
      return [];
    }
  }

  // --------------------------------------------------------------------------
  // Uploads / cover letter
  // --------------------------------------------------------------------------

  /**
   * Upload the resume to the first file input that clearly reads as a resume
   * (label/aria/name/accept containing resume/cv), falling back to a lone
   * unlabeled file input ONLY when it has no contradictory label. Never
   * guesses an unrelated upload (transcript, portfolio). Verified after every
   * attempt.
   */
  private async uploadResumeIfVisible(resumePath: string): Promise<boolean> {
    if (!resumePath || !fs.existsSync(resumePath)) return false;
    const page = this.getPage();
    const baseName = resumePath.split(/[\\/]/).pop() || "";
    const target = await page
      .evaluate(() => {
        const inputs = Array.from(
          document.querySelectorAll('input[type="file"]')
        ) as HTMLInputElement[];
        if (!inputs.length) return { index: -1, label: "" };
        let resume: { index: number; label: string } | null = null;
        for (let i = 0; i < inputs.length; i++) {
          const e = inputs[i];
          const aria = e.getAttribute("aria-label") || "";
          const id = e.id || "";
          const forLabel = id
            ? (document.querySelector(`label[for="${id}"]`)?.textContent || "")
            : "";
          const wrap = e.closest("label")?.textContent || "";
          const txt = `${aria} ${forLabel} ${wrap}`.toLowerCase();
          if (/cover letter|cover_letter|transcript|portfolio/.test(txt)) continue;
          if (/resume|cv|curriculum|attach your|upload your/.test(txt) || /resume|cv/i.test(e.name)) {
            resume = { index: i, label: txt.replace(/\s+/g, " ").trim().slice(0, 80) };
            break;
          }
        }
        // A lone unlabeled file input is likely the resume — but only when its
        // label really is empty; a transcript/portfolio-only form must not get
        // the resume jammed into the wrong upload.
        if (!resume && inputs.length === 1) {
          const e = inputs[0];
          const id = e.id || "";
          const txt = `${e.getAttribute("aria-label") || ""} ${
            id ? (document.querySelector(`label[for="${id}"]`)?.textContent || "") : ""
          } ${e.closest("label")?.textContent || ""}`.toLowerCase();
          if (!/transcript|portfolio|cover letter|id card|identity/i.test(txt)) {
            resume = { index: 0, label: "" };
          }
        }
        return resume ?? { index: -1, label: "" };
      })
      .catch(() => ({ index: -1, label: "" }));
    if (target.index < 0) return false;

    const input = page.locator('input[type="file"]').nth(target.index);
    for (let attempt = 0; attempt < 3; attempt++) {
      if (await this.controls.isResumeAttached()) return true;
      try {
        await input.setInputFiles(resumePath);
      } catch (err: any) {
        this.warn(`Resume setInputFiles threw (attempt ${attempt + 1}): ${err?.message || err}`);
      }
      await randomSleep(2000, 3000);
      const attached = await this.controls.isResumeAttached();
      if (attached) {
        console.log(`[Generic] Resume uploaded and registered (attempt ${attempt + 1}).`);
        return true;
      }
      // Some forms need an explicit Upload button after attach.
      const uploadBtn = page.locator("button:has-text('Upload')").first();
      if (await uploadBtn.isVisible().catch(() => false)) {
        await uploadBtn.click();
        await randomSleep(1500, 2200);
        if (await this.controls.isResumeAttached()) {
          console.log(`[Generic] Resume uploaded and registered after Upload (attempt ${attempt + 1}).`);
          return true;
        }
      }
      this.warn(`Resume upload not confirmed for ${baseName} (attempt ${attempt + 1}); retrying...`);
    }
    return false;
  }

  /** Fill an LLM-generated cover letter into a cover-letter/additional-info
   *  textarea, if the current view renders one. */
  private async fillCoverLetter(
    rpc: RpcHelper,
    filled: string[],
    blanked: BlankEntry[]
  ): Promise<void> {
    const page = this.getPage();
    const target: { index: number; label: string } | null = await page
      .evaluate(() => {
        const out: Array<{ index: number; label: string }> = [];
        const areas = Array.from(document.querySelectorAll("textarea"));
        areas.forEach((el, i) => {
          const e = el as HTMLTextAreaElement;
          if (e.offsetParent === null) return;
          const aria = e.getAttribute("aria-label") || "";
          const id = e.getAttribute("id") || "";
          const forLabel = id
            ? (document.querySelector(`label[for="${id}"]`)?.textContent || "")
            : "";
          const wrap = e.closest("label")?.textContent || "";
          const label = (aria || forLabel || wrap).replace(/\s+/g, " ").trim();
          out.push({ index: i, label });
        });
        const match = out.find(
          (c) =>
            /cover letter|additional information|anything else you|more about you|tell us about yourself|anything you would like/i.test(
              c.label
            ) && !(areas[c.index] as HTMLTextAreaElement).value.trim()
        );
        return match ?? null;
      })
      .catch(() => null);
    // Allow it to proceed even if target (textarea) is not found so PDF can be uploaded.
    const result = await rpc("cover_letter", {});
    const pdfPath = result?.pdf_path;
    let attached = false;

    if (pdfPath) {
      const clFileInputs = [
        'input[type="file"][name*="cover" i]',
        'input[type="file"][id*="cover" i]',
        'input[type="file"][aria-label*="cover" i]',
      ];
      for (const sel of clFileInputs) {
        const fileInput = page.locator(sel).first();
        if (await fileInput.isVisible().catch(() => false) || await fileInput.count() > 0) {
          try {
            await fileInput.setInputFiles(pdfPath);
            console.log("[Generic] Cover letter PDF uploaded successfully.");
            attached = true;
            break;
          } catch (e) {
            // Ignore
          }
        }
      }
    }

    if (!attached && target) {
      const coverLetter = (result?.answer ?? "").toString().trim();
      if (!coverLetter) return;
      const ta = page.locator("textarea").nth(target.index);
      await ta.fill(coverLetter);
      await randomSleep(200, 400);
      const committed = await ta.inputValue().catch(() => "");
      if (committed) {
        filled.push(target.label || "Cover Letter");
        console.log("[Generic] Cover letter filled (LLM-generated, JD-personalized).");
      } else {
        blanked.push({ label: target.label || "Cover Letter", reason: "cover letter did not commit" });
      }
    }
  }

  // --------------------------------------------------------------------------
  // Observe fallback (Tier-2) — only when the DOM walker finds nothing
  // --------------------------------------------------------------------------

  /**
   * Stagehand observe() fallback for forms the DOM walker cannot enumerate
   * (canvas-rendered, unusual custom components). observe() returns Actions
   * carrying selectors that act() executes deterministically — no extra LLM.
   * ONLY free-text fields are filled here: a selection made from a
   * guessed option list is never safe, so non-text observed controls are left
   * for the reverify report and a human.
   */
  private async observeFallback(
    screener: Screener,
    filled: string[],
    blanked: BlankEntry[],
    userSkippedKeys: Set<string>,
    processedKeys: Set<string>
  ): Promise<number> {
    let filledCount = 0;
    try {
      const actions: Action[] = (await this.stagehand.observe(
        "Find every fillable free-text input field in the application form: " +
          "text inputs, textareas, and email/phone/number/url inputs. " +
          "Return one Action per field with a selector that identifies exactly that field " +
          "and a description naming the field's label (e.g. 'fill the First Name field'). " +
          "Do NOT include buttons, links, dropdowns, checkboxes, radio buttons, or file uploads.",
        { page: this.getPage() }
      )) as Action[];
      let counter = 0;
      for (const action of actions ?? []) {
        const label = cleanObserveLabel(action.description);
        if (!label) continue;
        const id = `observe:${counter++}`;
        this.controls.registerObservedAction(id, action);
        const field: FormField = {
          label,
          id,
          kind: "text",
          required: false,
          options: [],
          optionTargets: [],
        };
        processedKeys.add(fieldKey(field));
        await screener.process(field, filled, blanked, userSkippedKeys);
        if (await this.controls.readFieldValue(field)) filledCount += 1;
      }
    } catch (err: any) {
      this.warn(`observeFallback failed: ${err?.message || err}`);
    }
    return filledCount;
  }

  // --------------------------------------------------------------------------
  // fill / submit
  // --------------------------------------------------------------------------

  async fill(payload: JobPayload, rpc?: RpcHelper): Promise<void> {
    const { url, profile } = payload;
    console.log(`[Generic] Navigating to ${url}...`);
    const page = this.getPage();
    await page.goto(url);
    await randomSleep(300, 600);

    await this.ensureApplicationView();

    let resumeAttached = false;
    if (profile.resumePath && fs.existsSync(profile.resumePath)) {
      resumeAttached = await this.uploadResumeIfVisible(profile.resumePath);
    }

    if (!rpc) {
      console.log("[Generic] Form filling completed (no resolver wired).");
      return;
    }

    // JD context first so open-ended answers are personalized to the role.
    const jobCtx = this.jobCtx ?? (await this.readJobContext());
    await rpc("job_context", jobCtx);
    console.log(
      `[Generic] Job context: ${jobCtx.title || "?"} @ ${jobCtx.company || "?"}` +
        (jobCtx.location ? ` (${jobCtx.location})` : "")
    );

    const jsonModel = await this.fetchJsonQuestions();
    this._jsonModel = jsonModel;
    this.profile = profile;
    const screener = new Screener(this.controls, "GenericAdapter", profile, rpc, true);
    const filled: string[] = [];
    const blanked: BlankEntry[] = [];
    const processedKeys = new Set<string>();
    const userSkippedKeys = new Set<string>();

    const MAX_STEPS = 12;
    for (let step = 0; step < MAX_STEPS; step++) {
      await randomSleep(1200, 1800);

      const probe = await this.probeFlow();
      if (classifyFlow(probe) === "gate") {
        console.log("[Generic] Sign-in gate reached mid-flow; attempting to pass it.");
        await this.handleGate();
        await randomSleep(1500, 2500);
      }

      const inventory = await this.collectQuestions();
      if (inventory.length === 0) {
        console.log(`[Generic] Step ${step + 1}: no visible questions.`);
      }

      if (await this.isVoluntaryStep(inventory)) {
        console.log("[Generic] Voluntary disclosure step detected (all-optional); skipping.");
        for (const f of inventory) {
          // Mark skipped so the final reverify never flags these as unfilled.
          userSkippedKeys.add(fieldKey(f));
          processedKeys.add(fieldKey(f));
          blanked.push({ label: f.label, reason: "voluntary disclosure step (left unchecked)" });
        }
      } else {
        // Converging re-scan walk over the visible fields (conditional
        // questions appear only after an interaction).
        for (let pass = 0; pass < 30; pass++) {
          const domFields = await this.collectQuestions();
          const fields = mergeFormInventory(jsonModel, domFields);
          const fresh = fields.filter((f) => !processedKeys.has(fieldKey(f)));
          if (pass === 0) {
            console.log(
              `[Generic] Step ${step + 1} inventory: ${fields.length} question(s) ` +
                `(json: ${jsonModel?.length ?? 0}, dom: ${domFields.length}).`
            );
          }
          if (fresh.length === 0) {
            console.log(`[Generic] Step ${step + 1} walk converged after ${pass + 1} pass(es).`);
            break;
          }
          for (const f of fresh) {
            processedKeys.add(fieldKey(f));
            await screener.process(f, filled, blanked, userSkippedKeys);
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
            `[Generic] ${requiredBlanks.length} REQUIRED field(s) blank after step ${step + 1}:`
          );
          for (const rb of requiredBlanks) {
            console.warn(`[Generic]   REQUIRED blank: ${escapePromptValue(rb.label)} (${rb.reason})`);
          }
        }
      }

      if (await this.hasSubmitButton()) {
        console.log("[Generic] Submit button visible — final step reached.");
        break;
      }
      const advanced = await this.advanceStep();
      if (!advanced) {
        console.warn(
          "[Generic] No Continue/Submit button found, or the step did not advance after " +
            "clicking. If required fields were left blank, the form may block progression."
        );
        break;
      }
    }

    // Tier-2: observe fallback ONLY when the DOM walker never found a single
    // recognizable field (a truly non-standard form renderer).
    if (processedKeys.size === 0) {
      console.warn("[Generic] DOM walker found no recognizable fields — falling back to observe().");
      const observed = await this.observeFallback(
        screener,
        filled,
        blanked,
        userSkippedKeys,
        processedKeys
      );
      console.log(`[Generic] Observe fallback filled ${observed} free-text field(s).`);
    }

    // Final sweep over visible fields.
    const sweepFilled: string[] = [];
    const sweepBlanks: BlankEntry[] = [];
    for (let pass = 0; pass < 3; pass++) {
      const sweptDom = await this.collectQuestions();
      const swept = mergeFormInventory(jsonModel, sweptDom);
      let touched = 0;
      for (const f of swept) {
        if (PRE_FILLED_LABELS.has(normalizeLabel(f.label))) continue;
        if (userSkippedKeys.has(fieldKey(f))) continue;
        if (await this.controls.readFieldValue(f)) continue;
        touched += 1;
        await screener.process(f, sweepFilled, sweepBlanks, userSkippedKeys);
      }
      if (touched === 0) break;
    }
    if (sweepFilled.length) {
      console.log(`[Generic] Final sweep filled ${sweepFilled.length} field(s):`);
      for (const l of sweepFilled) console.log(`[Generic]   filled: ${escapePromptValue(l)}`);
    }

    if (this.gateDeferred) {
      blanked.push({
        label: "Sign-in / account gate",
        reason: "blocked by an account gate (no guest path, no credentials); needs manual completion",
      });
    }

    // Definitive reverify of every still-empty field (required/optional, minus
    // identity fields and manual skips). This is the pre-completion checkpoint.
    const stillBlank = await finalReverify({
      tag: "GenericAdapter",
      collect: async () => {
        const dom = await this.collectQuestions();
        return mergeFormInventory(jsonModel, dom);
      },
      isEmpty: async (f) => !(await this.controls.readFieldValue(f)),
      skippedKeys: userSkippedKeys,
      reasons: [...blanked, ...sweepBlanks],
    });
    // Surface how many required fields are still blank so the runner can gate
    // auto-submit on an incomplete form. finalReverify excludes manual skips,
    // so a field in the report is a real unfilled field; conservatively count
    // it as required (the runner must not submit with any unknown blank).
    setBlankedRequiredCount(stillBlank.length);

    if (profile.resumePath && !resumeAttached && !(await this.controls.isResumeAttached())) {
      console.warn("[Generic] REVERIFY: resume is NOT attached after the final pass.");
    } else if (profile.resumePath) {
      console.log("[Generic] REVERIFY: resume is attached.");
    }

    console.log("[Generic] Form filling completed.");
  }

  async submit(): Promise<SubmitOutcome> {
    const page = this.getPage();
    console.log("[Generic] Submitting application form...");
    const submitBtn = page
      .locator(
        "button[type='submit'], input[type='submit'], button:has-text('Submit Application'), " +
          "button:has-text('Submit'), a:has-text('Submit Application'), [data-automation-id*='submit' i]"
      )
      .first();
    if (await submitBtn.isVisible().catch(() => false)) {
      await submitBtn.click();
    } else {
      await this.controls.safeAct("click the 'Submit Application' or 'Submit' button");
    }
    await randomSleep(1500, 2500);

    return verifySubmitOutcome(page, {
      tag: "Generic",
      submitButtonSelector:
        "button[type='submit'], input[type='submit'], button:has-text('Submit Application'), " +
        "button:has-text('Submit'), a:has-text('Submit Application')",
    });
  }

  /**
   * Recheck and re-fill any required field that is still blank, then report
   * how many remain blank. Called by the runner after a retryable submit
   * failure (validation blocked by unfilled fields).
   */
  async recheckMissingFields(rpc?: RpcHelper): Promise<number> {
    console.log("[Generic] Rechecking missing required fields...");
    const stillBlank: string[] = [];
    const dom = await this.collectQuestions();
    const fields = mergeFormInventory(this._jsonModel ?? null, dom);
    for (const f of fields) {
      if (!f.required) continue;
      if (await this.controls.readFieldValue(f)) continue;
      if (PRE_FILLED_LABELS.has(normalizeLabel(f.label))) continue;
      const screener = new Screener(
        this.controls,
        "GenericAdapter",
        this.profile,
        rpc ?? (async () => ({ answer: "" })),
        true
      );
      const filled: string[] = [];
      const blanked: { label: string; reason: string }[] = [];
      const skipped = new Set<string>();
      await screener.process(f, filled, blanked, skipped);
      if (filled.length === 0) stillBlank.push(f.label);
    }
    const remaining = stillBlank.length;
    setBlankedRequiredCount(remaining);
    console.log(`[Generic] Recheck complete: ${remaining} required field(s) still blank.`);
    for (const l of stillBlank) {
      console.warn(`[Generic]   still blank: ${escapePromptValue(l)}`);
    }
    return remaining;
  }
}

function normalizeLabel(label: string): string {
  return label.replace(/\s+/g, " ").trim().toLowerCase();
}

// ---------------------------------------------------------------------------
// Generic interaction layer — subclasses shared FormControls so the shared
// Screener / audit machinery runs unchanged. Adds observe()-backed fills and
// combobox handling for the generic walker's controls.
// ---------------------------------------------------------------------------

export class GenericControls extends FormControls {
  private observedActions = new Map<string, Action>();

  constructor(stagehand: Stagehand, tag = "GenericAdapter") {
    // Generic boards render dropdown options as ANY tag (div/li/button) with
    // role="option" — a workday-style li menu must be visible to the shared
    // option reader, not hard-wired to div[role="option"].
    super(stagehand, {
      tagName: tag,
      optionSelector: '[role="option"]',
      optionTag: "*",
    });
  }

  /** Record the observe() Action behind a synthetic field id. */
  registerObservedAction(id: string, action: Action): void {
    this.observedActions.set(id, action);
  }

  /** Kind-aware fill. observe()-backed fields fill deterministically through
   *  their recorded Action; comboboxes pick a suggestion (never raw typed
   *  text); native selects select an option; everything else uses the shared
   *  machinery unchanged. */
  override async fillByKind(
    field: FormField,
    answer: string,
    optionTexts?: string[]
  ): Promise<boolean> {
    if (field.id.startsWith("observe:")) {
      const action = this.observedActions.get(field.id);
      if (!action) return false;
      return this.fillObserved(action, answer);
    }
    if (field.kind === "combobox") {
      return this.fillGenericCombobox(field, answer, optionTexts ?? []);
    }
    if (field.kind === "select") {
      return this.fillGenericSelect(field, answer, optionTexts ?? []);
    }
    if (field.kind === "text" || field.kind === "date") {
      const ok = await super.fillByKind(field, answer, optionTexts);
      if (ok) return true;
      // Fallback: name-based fill for inputs the scope machinery could not
      // resolve (no id, no data-field-path wrapper).
      if (field.name) {
        const page = this.getPage();
        const input = page
          .locator(
            `input[name="${cssEscape(field.name)}"], textarea[name="${cssEscape(field.name)}"]`
          )
          .first();
        if (await input.isVisible().catch(() => false)) {
          const tagType = await page
            .evaluate(
              (n: string) =>
                document.querySelector(`input[name="${n}"]`)?.getAttribute("type") ??
                null,
              field.name
            )
            .catch(() => null);
          let value = String(answer ?? "");
          if (tagType === "number") {
            value = sanitizeNumberAnswer(value);
            if (!value) return false;
          }
          await input.fill(value).catch(() => {});
          await randomSleep(200, 400);
          return true;
        }
      }
      return false;
    }
    return super.fillByKind(field, answer, optionTexts);
  }

  /** Committed value of a field after filling, for verification/audit. */
  override async readFieldValue(field: FormField): Promise<string> {
    if (field.id.startsWith("observe:")) {
      const action = this.observedActions.get(field.id);
      return action ? this.readObservedValue(action.selector) : "";
    }
    if (field.kind === "combobox") {
      const page = this.getPage();
      try {
        const v = await page.evaluate((fid: string) => {
          // WARNING: only anonymous arrows here (tsx keepNames wraps any arrow
          // with an inferred name in __name(), which throws in page context).
          // Destructure the helper so it never gains a name.
          const [find] = [
            (root: Element | null): string => {
              if (!root) return "";
              const sv = root.querySelector('[class*="select__single-value"]');
              if (sv && (sv.textContent || "").trim()) {
                return (sv.textContent || "").replace(/\s+/g, " ").trim();
              }
              const multi = Array.from(root.querySelectorAll('[class*="select__multi-value"]'));
              if (multi.length) {
                return multi
                  .map((m) => (m.textContent || "").replace(/\s+/g, " ").trim())
                  .filter(Boolean)
                  .join(", ");
              }
              const selected = root.querySelector('[aria-selected="true"]');
              if (selected && (selected.textContent || "").trim()) {
                return (selected.textContent || "").replace(/\s+/g, " ").trim();
              }
              // NO raw input.value fallback: a dropdown is only "committed" when
              // a real suggestion was picked (which renders a committed marker).
              // Typed-but-unselected text must read as empty so the sweep/reverify
              // re-resolves it instead of accepting an uncommitted value.
              return "";
            },
          ];
          const scope = document.querySelector(`[data-field-path="${fid}"]`);
          if (scope) {
            const direct = find(scope);
            if (direct) return direct;
            // The scope may BE the combobox input (walker tags id-less inputs
            // with the path on themselves) — its committed value lives in the
            // surrounding select shell.
            if (scope.matches && scope.matches("input, textarea")) {
              const shell = scope.closest(
                '[class*="select-shell"], [class*="react-select"], [class*="select__"]'
              );
              if (shell) return find(shell);
            }
            return "";
          }
          const byId = document.getElementById(fid) as HTMLInputElement | null;
          if (byId) {
            const shell = byId.closest(
              '[class*="select-shell"], [class*="react-select"], [class*="select__"]'
            );
            if (shell) return find(shell);
            return "";
          }
          return "";
        }, field.id);
        return (v as string) || "";
      } catch {
        return "";
      }
    }
    if (field.kind === "select") {
      // Native selects may be addressed by name (no id) or by the synthetic
      // data-field-path tagged on the element itself; read the selected
      // option's visible text so verification/audit see the committed answer.
      const page = this.getPage();
      try {
        const v = await page.evaluate(
          (args: { id: string; name: string }) => {
            const byId = document.getElementById(args.id) as HTMLSelectElement | null;
            const byName = document.querySelector(
              `select[name="${args.name}"]`
            ) as HTMLSelectElement | null;
            const byPath = document.querySelector(
              `[data-field-path="${args.id}"]`
            ) as HTMLSelectElement | null;
            const sel = byId || byName || byPath;
            if (sel) {
              const idx = sel.selectedIndex;
              if (idx >= 0) {
                return (sel.options[idx]?.textContent || "")
                  .replace(/\s+/g, " ")
                  .trim();
              }
            }
            return "";
          },
          { id: field.id, name: field.name || "" }
        );
        if (v) return (v as string) || "";
      } catch {
        // fall through to the shared machinery
      }
    }
    return super.readFieldValue(field);
  }

  /**
   * Fill a combobox (react-select / role=combobox / aria-autocomplete).
   * Typing is only a trigger to load/filter options; the committed value MUST
   * be a picked suggestion — raw typed text is never accepted as an answer.
   * Multi answers (comma-separated) are filled one pick at a time: each pick
   * is typed and matched against the menu it produces, so a later pick is
   * never matched against an earlier pick's filtered options.
   */
  async fillGenericCombobox(
    field: FormField,
    answer: string,
    optionTexts: string[] = []
  ): Promise<boolean> {
    const page = this.getPage();
    // Only a genuine async location autocomplete may fall back to a ranked
    // first suggestion (pickLocationOption). For any real question a blind
    // first-option guess is never acceptable — an unmatched answer stays
    // blank and is surfaced by the reverify instead of being mis-committed.
    const isLocation = isLocationAutocomplete(field);
    try {
      // The walker tags an id-less combobox input with data-field-path on the
      // input ITSELF — the locator must self-match (not require a container
      // div), mirroring the select fill.
      const input = page
        .locator(
          `[data-field-path="${cssEscape(field.id)}"]:is(input[role="combobox"], input[aria-autocomplete], input), ` +
            `[data-field-path="${cssEscape(field.id)}"] input[role="combobox"], ` +
            `[data-field-path="${cssEscape(field.id)}"] input[aria-autocomplete], ` +
            `[data-field-path="${cssEscape(field.id)}"] input, ` +
            cssIdLocator(field.id)
        )
        .first();
      if (!(await input.isVisible().catch(() => false))) return false;

      const pickFor = (pick: string, opts: string[]): string | null => {
        const chosen = chooseOption(selectCandidates(pick), opts);
        if (chosen) return chosen;
        return isLocation ? pickLocationOption(pick, opts) : null;
      };

      // Load the menu for ONE pick (typed individually — never the whole
      // comma-joined answer). With static options we just reopen the menu.
      const loadOptsFor = async (query: string): Promise<string[]> => {
        if (optionTexts.length) {
          await this.ensureGenericMenuOpen(input);
          return optionTexts.slice();
        }
        await this.closeMenu();
        await randomSleep(150, 300);
        await input.click();
        await randomSleep(200, 350);
        await input.fill(query);
        let opts: string[] = [];
        for (let i = 0; i < 8 && opts.length === 0; i++) {
          await randomSleep(900, 1200);
          opts = await this.readVisibleOptionTexts();
        }
        if (opts.length === 0) {
          const short = query.split(/[\s,]+/).filter((t) => t && t.length > 1)[0];
          if (short && short !== query.trim()) {
            await input.fill(short);
            for (let i = 0; i < 6 && opts.length === 0; i++) {
              await randomSleep(900, 1200);
              opts = await this.readVisibleOptionTexts();
            }
          }
        }
        return opts;
      };

      const picks = answer
        .split(",")
        .map((p) => p.trim())
        .filter(Boolean);
      let clicked = 0;
      for (const pick of picks) {
        const opts = await loadOptsFor(pick);
        if (!opts.length) continue;
        const picked = pickFor(pick, opts);
        if (picked && (await this.clickVisibleOption(picked))) {
          clicked += 1;
          await randomSleep(200, 400);
          await this.ensureGenericMenuOpen(input);
        } else if (picked) {
          console.warn(`[${this.tagName}] Option "${picked}" not visible for #${field.id}`);
        } else {
          console.warn(
            `[${this.tagName}] No confident option for pick "${pick}" (${escapePromptValue(field.label)}); leaving it blank.`
          );
        }
      }
      await this.closeMenu();
      await randomSleep(300, 500);
      if (clicked > 0 && (await this.readFieldValue(field))) return true;
      console.warn(
        `[${this.tagName}] Could not commit a suggestion for "${answer}" (${escapePromptValue(field.label)}).`
      );
      return false;
    } catch (err: any) {
      console.warn(`[${this.tagName}] fillGenericCombobox failed: ${err?.message || err}`);
      return false;
    }
  }

  /** Reopen a combobox menu after a pick closed it (multi-selects). */
  private async ensureGenericMenuOpen(input: any): Promise<void> {
    for (let i = 0; i < 3; i++) {
      if (await this.hasVisibleOption()) return;
      await input.click().catch(() => {});
      await randomSleep(300, 500);
    }
  }

  /** Fill a native <select> by its id or name. */
  async fillGenericSelect(
    field: FormField,
    answer: string,
    optionTexts: string[] = []
  ): Promise<boolean> {
    const page = this.getPage();
    try {
      const sel = page
        .locator(
          `select${cssIdLocator(field.id)}, select[name="${cssEscape(field.name || field.id)}"], ` +
            `[data-field-path="${cssEscape(field.id)}"]`
        )
        .first();
      if (!(await sel.isVisible().catch(() => false))) {
        // Possibly a styled dropdown mis-walked as native — delegate to base.
        return super.fillByKind({ ...field, kind: "select" }, answer, optionTexts);
      }
      const isNative =
        (await sel.evaluate((el: Element) => el.tagName).catch(() => "DIV")) === "SELECT";
      if (!isNative) {
        // The data-field-path scope resolved to a container (or the styled
        // dropdown) rather than the <select> — delegate to the shared machinery.
        return super.fillByKind({ ...field, kind: "select" }, answer, optionTexts);
      }
      const opts =
        optionTexts.length > 0
          ? optionTexts
          : ((await sel.locator("option").allTextContents().catch(() => [])) as string[]);
      const picked = chooseOption(selectCandidates(answer), opts);
      if (!picked) {
        console.warn(
          `[${this.tagName}] No matching option for #${field.id} (answer "${escapePromptValue(answer)}"); leaving blank.`
        );
        return false;
      }
      await sel.selectOption(picked);
      await randomSleep(200, 400);
      const committed = !!(await page
        .evaluate((fid: string) => {
          const el = document.getElementById(fid) as HTMLSelectElement | null;
          return el ? String(el.value ?? "") : "";
        }, field.id)
        .catch(() => ""));
      if (!committed) {
        console.warn(
          `[${this.tagName}] Native select #${field.id} did not commit "${picked}"`
        );
      }
      return committed;
    } catch (err: any) {
      console.warn(`[${this.tagName}] fillGenericSelect failed for #${field.id}: ${err?.message || err}`);
      return false;
    }
  }

  /** Whether the resume is attached (input consumed, has files, or an upload
   *  zone shows an attached/uploaded state). */
  async isResumeAttached(): Promise<boolean> {
    const page = this.getPage();
    return page
      .evaluate(() => {
        const inputs = Array.from(
          document.querySelectorAll('input[type="file"]')
        ) as HTMLInputElement[];
        if (inputs.length === 0) return true; // consumed by the form = attached
        if (inputs.some((i) => i.files && i.files.length > 0)) return true;
        const zone = document.querySelector(
          "[class*='resume'], [class*='upload'], [id*='resume' i]"
        );
        const text = zone ? (zone.textContent || "") : "";
        return /attached|uploaded|added|done|✓/i.test(text);
      })
      .catch(() => false);
  }
}
