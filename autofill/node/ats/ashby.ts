import { Stagehand } from "@browserbasehq/stagehand";
import * as fs from "fs";
import { ATSAdapter, RpcHelper } from "./base.js";
import { JobPayload, Profile } from "../types.js";
import { randomSleep } from "../utils/evasion.js";
import { auditBlanks, finalReverify, SubmitOutcome, verifySubmitOutcome } from "./shared/audit.js";
import { FormControls } from "./shared/controls.js";
import {
  chooseOption,
  escapePromptValue,
  pickLocationOption,
  selectCandidates,
} from "./shared/matching.js";
import {
  fieldKey,
  FormField,
  PRE_FILLED_LABELS,
} from "./shared/model.js";
import { Screener, setBlankedRequiredCount } from "./shared/screener.js";

const SYSTEM_SKIP = new Set([
  "_systemfield_name",
  "_systemfield_email",
  "_systemfield_phone",
  "_systemfield_resume",
  "_systemfield_location",
]);

/**
 * Ashby adapter.
 *
 * Ashby job pages are client-rendered SPAs. The fetched HTML only carries
 * `window.__appData` (posting metadata — NO question definitions), the form is
 * hydrated from the JS bundle, and questions must be enumerated from the live
 * DOM after hydration.
 *
 * Two views: the posting page (job description) and the application form at
 * the posting URL + "/application". When a bare posting URL is given, the JD
 * is captured from the posting view FIRST (it feeds rpc("job_context", ...)),
 * then the Apply button is clicked — which may open the form in a NEW TAB —
 * and that tab is adopted as the active page for the rest of the fill.
 *
 * DOM shapes (verified against live postings):
 *  - every question lives in a div carrying `data-field-path="<questionId>"`;
 *  - short answers are `input#<fieldId>` and long answers `textarea#<fieldId>`
 *    (the field path is also the element id);
 *  - option groups (radio / single yes-no / multi-select) are option rows that
 *    toggle an underlying radio/checkbox input; the underlying name may be the
 *    question path or the option text itself;
 *  - dropdowns are comboboxes (`input[role="combobox"]`) — suggestions appear
 *    as visible `div[role="option"]` after typing; the picked value commits
 *    into the combobox input. System location is filled deterministically;
 *  - a voluntary Diversity Survey (`.ashby-survey-form-container`) is skipped;
 *  - `_systemfield_*` identity/resume/location fields are filled by
 *    deterministic routines in fill() and never walked.
 */
export class AshbyAdapter extends ATSAdapter {
  protected controls!: AshbyControlStack;
  private jobCtx: { title: string; company: string; location: string; description: string } | null = null;
  protected profile!: Profile;

  constructor(stagehand: Stagehand) {
    super(stagehand);
    this.controls = new AshbyControlStack(stagehand, "AshbyAdapter");
  }

  protected getPage(): any {
    return this.controls.getPage();
  }

  /** The Ashby form may live in a second tab; screenshot the adopted page. */
  getActivePage(): any {
    return this.controls.getPage();
  }

  /**
   * Extract the job posting context (title, company, location, description)
   * from the posting view. Prefers the SPA hydration blob (`window.__appData`,
   * present on both the posting and application views) and falls back to DOM
   * selectors and the company slug from the URL. Used to personalize open-ended
   * answers, matching GreenhouseAdapter's job_context RPC.
   */
  private async readJobContext(): Promise<{
    title: string;
    company: string;
    location: string;
    description: string;
  }> {
    const page = this.getPage();
    try {
      const appData: any = await page
        .evaluate(() => (window as any).__appData ?? null)
        .catch(() => null);
      const posting = appData?.posting ?? appData?.job ?? null;
      const html = (posting?.descriptionHtml || posting?.description || "") as string;
      const description = html
        .replace(/<[^>]+>/g, " ")
        .replace(/&nbsp;/g, " ")
        .replace(/\s+/g, " ")
        .trim()
        .slice(0, 6000);
      const title =
        (posting?.title as string) ||
        (await page.locator("h1, [data-qa='posting-title']").first().innerText().catch(() => "")) ||
        (await page.title()).replace(/\s*[|–-].*$/, "").trim();
      const company =
        (posting?.companyName || posting?.company || "") as string ||
        (appData?.organization?.name as string) ||
        (await page.locator("[data-qa='company-name'], .company-name").first().innerText().catch(() => "")) ||
        (() => {
          try {
            return new URL(page.url()).pathname.split("/").filter(Boolean)[0] || "";
          } catch {
            return "";
          }
        })();
      const location =
        (posting?.locationName || posting?.location || "") as string ||
        (await page
          .locator("[data-qa='posting-location'], .location")
          .first()
          .innerText()
          .catch(() => ""));
      return {
        title: title.replace(/\s+/g, " ").trim(),
        company: company.replace(/\s+/g, " ").trim(),
        location: location.replace(/\s+/g, " ").trim(),
        description,
      };
    } catch (err: any) {
      console.warn(`[Ashby] readJobContext failed: ${err?.message || err}`);
      return { title: "", company: "", location: "", description: "" };
    }
  }

  private async waitForForm(): Promise<void> {
    const page = this.getPage();
    for (let i = 0; i < 40; i++) {
      // The resume file input (#_systemfield_resume) hydrates AFTER the field
      // rows render — it must be present before fill() tries to upload, or
      // setInputFiles silently targets nothing and the resume never attaches.
      const ready = await page
        .evaluate(() => {
          const rows = Array.from(document.querySelectorAll("[data-field-path]"));
          const anyVisible = rows.some(
            (el) =>
              (el as HTMLElement).offsetParent !== null ||
              (el as HTMLElement).getBoundingClientRect().height > 0
          );
          return (
            anyVisible ||
            !!document.querySelector("#_systemfield_resume[type='file']") ||
            !!document.querySelector("button.ashby-application-form-submit-button")
          );
        })
        .catch(() => false);
      if (ready) return;
      await randomSleep(800, 1200);
    }
    throw new Error(
      "Ashby application form never appeared (no [data-field-path] elements visible)"
    );
  }

  /**
   * Land on the application form view. When the current page is a bare posting
   * (no form in the DOM), capture the job description FIRST — the JD lives on
   * the posting view and the form view may not render it — then click the Apply
   * button, which on some postings opens the form in a NEW TAB. That page is
   * adopted as the active one. Last resort: navigate to the deterministic
   * "<posting>/application" route directly.
   */
  private async ensureApplicationView(): Promise<void> {
    const page = this.getPage();
    const cur = page.url();
    try {
      if (/\/application(?:\/|$)/.test(new URL(cur).pathname)) return;
    } catch {
      // Unparseable URL; fall through to the DOM check.
    }
    if (await page.locator('[data-field-path]').first().isVisible().catch(() => false)) {
      return; // form embedded on the job page itself
    }
    // The posting view is the only reliable source for the JD — grab it now,
    // before the Apply click hands control to the form view.
    this.jobCtx = await this.readJobContext();

    await this.controls.safeAct("click the 'Apply now' button to open the application form");
    if (!(await this.controls.focusPage(/\/application(?:\/|$)/))) {
      // The tab may not carry /application (custom embeds); adopt any page
      // that now shows the form.
      let adopted = false;
      for (const p of this.stagehand.context.pages()) {
        if (await p.locator('[data-field-path]').first().isVisible().catch(() => false)) {
          this.controls.adoptPage(p);
          adopted = true;
          break;
        }
      }
      if (!adopted) {
        // Last resort: Ashby deterministically serves the form at <posting>/application.
        try {
          const applyUrl = new URL(cur);
          applyUrl.pathname = applyUrl.pathname.replace(/\/+$/, "") + "/application";
          await page.goto(applyUrl, { waitUntil: "domcontentloaded" }).catch(() => {});
        } catch {
          // Never throw here; waitForForm reports the failure.
        }
      }
    }
  }

  async fill(payload: JobPayload, rpc?: RpcHelper): Promise<void> {
    const { url, profile } = payload;
    this.profile = profile;
    console.log(`[Ashby] Navigating to ${url}...`);
    const page = this.getPage();
    await page.goto(url);
    await this.ensureApplicationView();
    await this.waitForForm();

    console.log("[Ashby] Filling deterministic profile fields...");
    const nameVal = [profile.firstName, profile.lastName].join(" ").trim();
    await this.controls.fillSystemText("_systemfield_name", nameVal);
    await this.controls.fillSystemText("_systemfield_email", profile.email);
    await this.controls.fillSystemText("_systemfield_phone", profile.phone);

    let resumeAttached = false;
    if (profile.resumePath && fs.existsSync(profile.resumePath)) {
      resumeAttached = await this.uploadResume(profile.resumePath);
    }

    if (profile.location) {
      const ok = await this.controls.fillLocation(String(profile.location));
      if (ok) console.log(`[Ashby] Committed location "${profile.location}".`);
      else console.warn(`[Ashby] Could not commit profile location "${profile.location}".`);
    }

    if (rpc) {
      // Job context first so open-ended answers are personalized to the role.
      // Prefer the JD captured from the posting view; on an /application link
      // fall back to reading __appData/DOM from the form page itself.
      const jobCtx = this.jobCtx ?? (await this.readJobContext());
      await rpc("job_context", jobCtx);
      console.log(
        `[Ashby] Job context: ${jobCtx.title || "?"} @ ${jobCtx.company || "?"}` +
          (jobCtx.location ? ` (${jobCtx.location})` : "")
      );

      const screener = new Screener(this.controls, "AshbyAdapter", profile, rpc);
      const filled: string[] = [];
      const blanked: Array<{ label: string; reason: string }> = [];
      const processedKeys = new Set<string>();
      const userSkippedKeys = new Set<string>();

      for (let pass = 0; pass < 30; pass++) {
        const fields = await this.collectQuestions();
        const fresh = fields.filter((f) => !processedKeys.has(fieldKey(f)));
        console.log(`[Ashby] Walk pass ${pass + 1}: ${fresh.length} new question(s) (total ${fields.length}).`);
        if (fresh.length === 0) {
          console.log(`[Ashby] Walk converged after ${pass + 1} pass(es).`);
          break;
        }
        for (const f of fresh) {
          processedKeys.add(fieldKey(f));
          await screener.process(f, filled, blanked, userSkippedKeys);
        }
        await this.controls.closeMenu().catch(() => {});
        await randomSleep(900, 1400);
      }

      const finalDom = await this.collectQuestions();
      const requiredBlanks = await auditBlanks({
        fields: finalDom,
        readValue: (f) => this.controls.readFieldValue(f),
        transcript: blanked,
      });
      console.log(`[Ashby] Walk complete: filled ${filled.length}, blank ${blanked.length}.`);
      for (const b of blanked) console.warn(`[Ashby]   blank: ${escapePromptValue(b.label)} (${b.reason})`);
      for (const rb of requiredBlanks) console.warn(`[Ashby]   REQUIRED blank: ${escapePromptValue(rb.label)} (${rb.reason})`);

      const sweepFilled: string[] = [];
      const sweepBlanks: Array<{ label: string; reason: string }> = [];
      for (let pass = 0; pass < 3; pass++) {
        const swept = await this.collectQuestions();
        let touched = 0;
        for (const f of swept) {
          if (PRE_FILLED_LABELS.has(norm(f.label))) continue;
          if (userSkippedKeys.has(fieldKey(f))) continue;
          if (await this.hasValue(f)) continue;
          touched += 1;
          await screener.process(f, sweepFilled, sweepBlanks, userSkippedKeys);
        }
        if (touched === 0) break;
      }
      if (sweepFilled.length) {
        console.log(`[Ashby] Final sweep filled ${sweepFilled.length} field(s):`);
        for (const l of sweepFilled) console.log(`[Ashby]   filled: ${escapePromptValue(l)}`);
      }

      await finalReverify({
        tag: "AshbyAdapter",
        collect: () => this.collectQuestions(),
        isEmpty: async (f) => !(await this.hasValue(f)),
        skippedKeys: userSkippedKeys,
        reasons: [...blanked, ...sweepBlanks],
      });
      // Surface how many required fields are still blank so the runner can
      // gate auto-submit on an incomplete form.
      setBlankedRequiredCount(requiredBlanks.length);

      const resumeBase = (profile.resumePath ?? "").split(/[\\/]/).pop() || "";
      if (profile.resumePath && !resumeAttached && !(await this.controls.isResumeAttached(resumeBase))) {
        console.warn("[Ashby] REVERIFY: resume is NOT attached after the final pass.");
      } else if (profile.resumePath) {
        console.log("[Ashby] REVERIFY: resume is attached.");
      }
    }

    console.log("[Ashby] Form filling completed.");
  }

  async submit(): Promise<SubmitOutcome> {
    console.log("[Ashby] Submitting application form...");
    const page = this.getPage();
    const submitBtn = page
      .locator("button.ashby-application-form-submit-button")
      .first();
    if (await submitBtn.isVisible().catch(() => false)) {
      await submitBtn.click();
    } else {
      await this.stagehand.act("Click the Submit Application button");
    }
    await randomSleep(500, 1000);

    return verifySubmitOutcome(page, {
      tag: "Ashby",
      submitButtonSelector: "button.ashby-application-form-submit-button",
    });
  }

  /**
   * Recheck and re-fill any required field that is still blank, then report
   * the number that remain blank. Called by the runner after a retryable
   * submit failure (validation blocked by unfilled fields) so a second submit
   * attempt starts from a complete form.
   */
  async recheckMissingFields(rpc?: RpcHelper): Promise<number> {
    console.log("[Ashby] Rechecking missing required fields...");
    const page = this.getPage();
    const stillBlank: string[] = [];
    const fields = await this.collectQuestions();
    for (const f of fields) {
      if (!f.required) continue;
      if (await this.hasValue(f)) continue;
      if (PRE_FILLED_LABELS.has(norm(f.label))) continue;
      // Try to resolve it via the screener machinery (may itself fail to
      // commit — that is recorded and re-audited below).
      const screener = new Screener(this.controls, "AshbyAdapter", this.profile, rpc ?? (async () => ({ answer: "" })));
      const filled: string[] = [];
      const blanked: { label: string; reason: string }[] = [];
      const skipped = new Set<string>();
      await screener.process(f, filled, blanked, skipped);
      if (filled.length === 0) {
        stillBlank.push(f.label);
      }
    }
    const remaining = stillBlank.length;
    setBlankedRequiredCount(remaining);
    console.log(
      `[Ashby] Recheck complete: ${remaining} required field(s) still blank.`
    );
    for (const l of stillBlank) console.warn(`[Ashby]   still blank: ${escapePromptValue(l)}`);
    return remaining;
  }

  private async uploadResume(resumePath: string): Promise<boolean> {
    const page = this.getPage();
    const baseName = resumePath.split(/[\\/]/).pop() || "";
    console.log(`[Ashby] Uploading resume from ${resumePath}...`);
    for (let attempt = 0; attempt < 3; attempt++) {
      // The resume input hydrates AFTER the field rows render. Wait for it so
      // setInputFiles targets a real element — a stale/absent locator silently
      // no-ops and the resume is never attached.
      const input = page.locator('#_systemfield_resume[type="file"]').first();
      for (let i = 0; i < 12; i++) {
        if ((await input.count()) > 0) break;
        await randomSleep(500, 800);
      }
      if ((await input.count()) === 0) {
        // No file input in the DOM. It may have been consumed by a previous
        // attempt — only count that as attached when a rendered attachment
        // chip actually shows the resume file name.
        if (await this.controls.isResumeAttached(baseName)) {
          console.log("[Ashby] Resume already registered (attachment visible).");
          return true;
        }
        console.warn(
          `[Ashby] Resume input not present (attempt ${attempt + 1}); retrying...`
        );
        continue;
      }
      try {
        await input.setInputFiles(resumePath);
      } catch (err: any) {
        console.warn(
          `[Ashby] Resume setInputFiles threw (attempt ${attempt + 1}): ${err?.message || err}`
        );
      }
      await randomSleep(1500, 2200);
      if (await this.controls.isResumeAttached(baseName)) {
        console.log("[Ashby] Resume uploaded and registered.");
        return true;
      }
      console.warn(`[Ashby] Resume upload not confirmed (attempt ${attempt + 1}); retrying...`);
    }
    return false;
  }

  private async hasValue(f: FormField): Promise<boolean> {
    return !!(await this.controls.readFieldValue(f));
  }

  // --------------------------------------------------------------------------
  // DOM inventory
  // --------------------------------------------------------------------------

  private async collectQuestions(): Promise<FormField[]> {
    const page = this.getPage();
    try {
      const rows = await page.evaluate((skipNames: string[]) => {
        const out: Array<{
          label: string;
          id: string;
          kind: string;
          required: boolean;
          options: string[];
          targets: Array<{ text: string; name: string; value: string; id?: string; button?: boolean }>;
        }> = [];
        // WARNING: only anonymous arrows may be defined inside this evaluate.
        // tsx's keepNames wraps any arrow with an inferred name in __name(),
        // and the identifier then throws when the function is stringified into
        // the page. Destructure helpers into an array so none gains a name.
        const [norm, collect] = [
          (t: string) =>
            (t || "").replace(/\s+/g, " ").trim().replace(/^\*+|\*+$/g, ""),
          (row: Element | null): string => {
            const t = row ? norm((row as HTMLElement).textContent || "") : "";
            return t;
          },
        ];
        const entries = Array.from(document.querySelectorAll("div[data-field-path]"));
        for (const entryEl of entries) {
          const el = entryEl as HTMLElement;
          if (el.closest(".ashby-survey-form-container, [class*='survey-form']")) continue;
          const id = (el.getAttribute("data-field-path") || "").trim();
          if (!id) continue;
          if (skipNames.includes(id)) continue;
          if (el.querySelector('input[type="file"]')) continue;

          const labelEl = el.querySelector('label[class*="question"], label');
          const label = norm(
            labelEl?.textContent || el.getAttribute("aria-label") || ""
          );
          if (!label) continue;

          const textInput = el.querySelector(
            'input[type="text"], input[type="email"], input[type="tel"], input[type="url"], input:not([type]), input[type="number"]'
          );
          const textarea = el.querySelector("textarea");
          const combobox = el.querySelector(
            'input[role="combobox"], input[aria-autocomplete], input[type="text"][aria-autocomplete]'
          );
          const radios = Array.from(el.querySelectorAll('input[type="radio"]'));
          const checks = Array.from(el.querySelectorAll('input[type="checkbox"]'));

          const required =
            !!el.querySelector('input[required], textarea[required]') ||
            /required/.test(labelEl?.className || "") ||
            !!el.querySelector('[aria-required="true"]');

          let kind = "";
          if (combobox && !radios.length && !checks.length) kind = "combobox";
          else if (textInput || textarea) {
            // A react-datepicker (custom calendar) is a date field, not free
            // text: free-form answers like "immediately" must be translated to
            // a real date before filling.
            kind = el.querySelector(".react-datepicker-wrapper, .react-datepicker")
              ? "date"
              : "text";
          }
          else if (radios.length) kind = "radio";
          else if (checks.length) kind = "checkbox";
          if (!kind) continue;

          if (kind === "combobox" || kind === "text" || kind === "date") {
            out.push({ label, id, kind, required, options: [], targets: [] });
            continue;
          }

          // Option groups (radio/single multi-checkbox/multi-select). The
          // uncovered checkbox/radio may carry no visible text (yes-no rows
          // render as buttons); read the option text from a wrapping or
          // sibling label, an option row, an li, or an aria-label.
          const options: string[] = [];
          const targets: Array<{ text: string; name: string; value: string; id?: string; button?: boolean }> = [];
          for (const inEl of [...radios, ...checks]) {
            const input = inEl as HTMLInputElement;
            const row = input.closest("label") || input.closest("[class*='option']") || input.closest("li");
            const labFor = input.id
              ? (document.querySelector(`label[for="${input.id}"]`)?.textContent || "").trim()
              : "";
            const text = collect(row) || labFor || input.getAttribute("aria-label") || "";
            if (text && !targets.some((t) => t.text === text)) {
              targets.push({ text, name: input.name || "", value: input.value || "", id: input.id || "" });
            }
          }
          // Fall back to option rows when the input has no textual label. The
          // yes/no toggle rows (a hidden checkbox rendered as Yes/No BUTTONS)
          // have no input text — record each button as a clickable target.
          if (targets.length === 0) {
            for (const row of Array.from(
              el.querySelectorAll(
                "[class*='option'] label, [class*='option'] span, li, button[class*='option']"
              )
            )) {
              const t = collect(row as HTMLElement);
              if (!t || options.includes(t)) continue;
              options.push(t);
              const isBtn = (row as HTMLElement).tagName === "BUTTON";
              if (isBtn) {
                targets.push({ text: t, name: "", value: "", id: "", button: true });
              }
            }
          }
          if ((kind === "radio" || kind === "checkbox") && !targets.length && !options.length) continue;
          for (const t of targets) {
            if (!options.includes(t.text)) options.push(t.text);
          }

          out.push({ label, id, kind, required, options, targets });
        }
        const seen = new Set<string>();
        const uniq: typeof out = [];
        for (const r of out) {
          const key = norm(r.label).toLowerCase() + "|" + r.kind;
          if (seen.has(key)) continue;
          seen.add(key);
          uniq.push(r);
        }
        return uniq;
      }, [...SYSTEM_SKIP]);

      return (rows ?? []).map(
        (r: any): FormField => ({
          label: r.label,
          id: r.id,
          kind: r.kind as FormField["kind"],
          required: !!r.required,
          options: r.options ?? [],
          optionTargets: (r.targets ?? []).map((t: any) => ({ text: t.text, name: t.name, value: t.value, id: t.id ?? "", button: !!t.button })),
          name: r.id,
        })
      );
    } catch (err: any) {
      console.warn(`[Ashby] collectQuestions failed: ${err?.message || err}`);
      return [];
    }
  }
}

function norm(label: string): string {
  return label.replace(/\s+/g, " ").trim().toLowerCase();
}

// ---------------------------------------------------------------------------
// Ashby interaction layer — subclasses shared FormControls so the shared
// Screener / audit machinery runs unchanged.
// ---------------------------------------------------------------------------

export class AshbyControlStack extends FormControls {
  constructor(stagehand: Stagehand, tag: string) {
    super(stagehand, { tagName: tag });
  }

  private scope(f: FormField): string {
    return `div[data-field-path="${f.id}"]`;
  }

  /** Click the option row whose normalized text exactly matches the answer.
   *  The shared base handles label[for] clicks, visually-hidden inputs,
   *  force-checks, and generic scope-based button/label/option clicks. The
   *  only Ashby-specific addition is the lone-checkbox consent-gate case. */
  override async clickGroupOption(
    field: FormField,
    answer: string,
    formSelector = "#application-form"
  ): Promise<boolean> {
    if (await super.clickGroupOption(field, answer, formSelector)) return true;
    // Fallback: a lone underlying checkbox (consent gates) with no issue UI
    // row. Only safe when there is exactly ONE — never with a multi-select.
    const page = this.getPage();
    try {
      return page
        .evaluate((fid: string) => {
          const scope = document.querySelector(`div[data-field-path="${fid}"]`);
          if (!scope) return false;
          const lone = scope.querySelectorAll('input[type="checkbox"]');
          if (lone.length !== 1) return false;
          const box = lone[0] as HTMLInputElement;
          if (box.checked) return true;
          box.click();
          return box.checked;
        }, field.id)
        .catch(() => false);
    } catch {
      return false;
    }
  }

  /** Multi-select: click each of the comma-separated picks. */
  async clickGroupMulti(field: FormField, answer: string): Promise<boolean> {
    let clicked = 0;
    for (const pick of answer.split(",").map((p) => p.trim()).filter(Boolean)) {
      if (await this.clickGroupOption({ ...field }, pick)) clicked++;
    }
    return clicked > 0;
  }

  /** The committed, human-readable value of a question (verification + audit). */
  override async readFieldValue(field: FormField): Promise<string> {
    if (field.kind === "combobox") {
      const page = this.getPage();
      try {
        const v = await page.evaluate((fid: string) => {
          const scope = document.querySelector(`div[data-field-path="${fid}"]`);
          if (!scope) return "";
          const combo = scope.querySelector('input[role="combobox"], input[aria-autocomplete]') as HTMLInputElement | null;
          return combo ? (combo.value || "").trim() : "";
        }, field.id);
        return (v as string) || "";
      } catch {
        return "";
      }
    }
    if (field.kind === "radio" || field.kind === "checkbox") {
      // The shared scope reader handles checked inputs AND active toggle
      // buttons — board-agnostic, no Ashby-specific selectors needed.
      return this.readScopedGroupValue(field);
    }
    return super.readFieldValue(field);
  }

  /** Kind-aware fill for this board. Ashby question dropdowns are comboboxes
   * answered by picking a suggestion (never raw typed text); delegate the rest
   * to the shared machinery unchanged.
   */
  override async fillByKind(
    field: FormField,
    answer: string,
    optionTexts?: string[]
  ): Promise<boolean> {
    if (field.kind === "combobox") {
      return this.fillCombobox(field, answer, optionTexts ?? []);
    }
    return super.fillByKind(field, answer, optionTexts);
  }

  /** Whether the field currently has a committed value. */
  async hasCommittedValue(f: FormField): Promise<boolean> {
    return !!(await this.readFieldValue(f));
  }

  /**
   * Fill an Ashby question combobox from its typed suggestions. Typing is only
   * a trigger to reveal options; the committed value MUST be a picked
   * suggestion — raw typed text is never accepted as an answer. Options for
   * resolution come from the field's own DOM (read via open menu) when the
   * walker did not collect them.
   */
  async fillCombobox(field: FormField, answer: string, optionTexts: string[] = []): Promise<boolean> {
    const page = this.getPage();
    const comboLocator = `${this.scope(field)} input[role="combobox"], ${this.scope(field)} input[aria-autocomplete]`;
    try {
      const input = page.locator(comboLocator).first();
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
      } else {
        await input.click();
      }
      if (!opts.length) {
        const short = answer.split(/[\s,]+/).filter((t) => t && t.length > 1)[0];
        if (short && short !== answer.trim()) {
          await input.fill(short);
          for (let i = 0; i < 6 && opts.length === 0; i++) {
            await randomSleep(900, 1200);
            opts = await this.readVisibleOptionTexts();
          }
        }
      }
      if (opts.length) {
        const picked = chooseOption(selectCandidates(answer), opts) ?? pickLocationOption(answer, opts);
        if (picked && (await this.clickVisibleOption(picked))) {
          await this.closeMenu();
          await randomSleep(300, 500);
          if (await this.readFieldValue(field)) return true;
          return false;
        }
      }
      await this.closeMenu();
      console.warn(`[${this.tagName}] No selectable suggestion for "${answer}" (${this.scope(field)}).`);
      return false;
    } catch (err: any) {
      console.warn(`[${this.tagName}] fillCombobox failed: ${err?.message || err}`);
      return false;
    }
  }

  /** Resume flag: true only when the file input carries files or a rendered
   *  attachment chip shows the resume name. A missing input does NOT prove
   *  attachment — it may simply not have hydrated yet (that was the Matic bug:
   *  an early "input consumed" returned true for an empty form). */
  async isResumeAttached(fileName = ""): Promise<boolean> {
    const page = this.getPage();
    return page
      .evaluate((name: string) => {
        const input = document.querySelector(
          '#_systemfield_resume[type="file"]'
        ) as HTMLInputElement | null;
        if (input) return !!(input.files && input.files.length > 0);
        // No file input in the DOM: attached ONLY when a rendered file chip
        // (upload/attachment area) shows the resume name — never the generic
        // "Upload your resume" hint, which would false-positive an empty form.
        if (!name) return false;
        const areas = Array.from(
          document.querySelectorAll(
            '[class*="file"], [class*="upload"], [class*="attachment"], [id^="upload-"], [class*="resume"]'
          )
        );
        for (const a of areas) {
          const t = (a.textContent || "").replace(/\s+/g, " ").trim();
          // The resume name in a chip wins even when the chip also carries
          // the upload hint ("resume.pdf · Upload a different file").
          if (t.includes(name)) return true;
          const stem = name.length > 16 ? name.slice(0, 20) : "";
          if (stem && t.includes(stem)) return true;
          if (/upload|drag|drop/i.test(t)) continue; // the upload hint, not a chip
        }
        return false;
      }, fileName)
      .catch(() => false);
  }

  /** Fill an identity/short text input by its data-field name. */
  async fillSystemText(name: string, value: string | null | undefined): Promise<void> {
    if (!value) return;
    const input = this.getPage().locator(`input[name="${name}"]`).first();
    if (await input.isVisible().catch(() => false)) {
      await input.fill(value);
      await randomSleep(150, 300);
    }
  }

  /**
   * Fill the async location combobox. Typing loads suggestions; the committed
   * value MUST be a picked suggestion — raw typed text is never an answer.
   */
  async fillLocation(value: string): Promise<boolean> {
    const page = this.getPage();
    const locator = 'div[data-field-path="_systemfield_location"] input[role="combobox"]';
    try {
      const input = page.locator(locator).first();
      if (!(await input.isVisible().catch(() => false))) return false;
      await this.closeMenu();
      await randomSleep(150, 300);
      await input.click();
      await randomSleep(200, 350);
      await input.fill(value);
      let opts: string[] = [];
      for (let i = 0; i < 8 && opts.length === 0; i++) {
        await randomSleep(900, 1200);
        opts = await this.readVisibleOptionTexts();
      }
      if (!opts.length) {
        const short = value.split(/[\s,]+/).filter((t) => t && t.length > 1)[0];
        if (short && short !== value.trim()) {
          await input.fill(short);
          for (let i = 0; i < 6 && opts.length === 0; i++) {
            await randomSleep(900, 1200);
            opts = await this.readVisibleOptionTexts();
          }
        }
      }
      if (opts.length) {
        const picked = pickLocationOption(value, opts);
        if (picked && (await this.clickVisibleOption(picked))) {
          await this.closeMenu();
          await randomSleep(300, 500);
          const committed = await page
            .evaluate(() => {
              const c = document.querySelector(locator) as HTMLInputElement | null;
              return c ? c.value.trim() : "";
            })
            .catch(() => "");
          if (committed) return true;
        }
      }
      await this.closeMenu();
      console.warn(`[${this.tagName}] No selectable location suggestion for "${value}".`);
      return false;
    } catch (err: any) {
      console.warn(`[${this.tagName}] fillLocation failed: ${err?.message || err}`);
      return false;
    }
  }
}