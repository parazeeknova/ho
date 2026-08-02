import { Stagehand } from "@browserbasehq/stagehand";
import * as fs from "fs";
import { ATSAdapter, RpcHelper } from "./base.js";
import { JobPayload, Profile } from "../types.js";
import { randomSleep } from "../utils/evasion.js";
import { auditBlanks, finalReverify } from "./shared/audit.js";
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
import { Screener } from "./shared/screener.js";

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

  constructor(stagehand: Stagehand) {
    super(stagehand);
    this.controls = new AshbyControlStack(stagehand, "AshbyAdapter");
  }

  protected getPage(): any {
    return this.controls.getPage();
  }

  private async waitForForm(): Promise<void> {
    const page = this.getPage();
    for (let i = 0; i < 40; i++) {
      const ready = await page
        .locator('[data-field-path], button.ashby-application-form-submit-button')
        .first()
        .isVisible()
        .catch(() => false);
      if (ready) return;
      await randomSleep(800, 1200);
    }
  }

  private async ensureApplicationView(): Promise<void> {
    const page = this.getPage();
    let path = "";
    try {
      path = new URL(page.url()).pathname;
    } catch {}
    if (/\/application(?:\/|$)/.test(path)) return;
    const hasForm = await page.locator('[data-field-path]').first().isVisible().catch(() => false);
    if (!hasForm) {
      await this.controls.safeAct("click the 'Apply now' button to open the application form");
      await randomSleep(1200, 2000);
    }
  }

  async fill(payload: JobPayload, rpc?: RpcHelper): Promise<void> {
    const { url, profile } = payload;
    console.log(`[Ashby] Navigating to ${url}...`);
    const page = this.getPage();
    await page.goto(url);
    await this.waitForForm();
    await this.ensureApplicationView();

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

      if (profile.resumePath && !resumeAttached && !(await this.controls.isResumeAttached())) {
        console.warn("[Ashby] REVERIFY: resume is NOT attached after the final pass.");
      } else if (profile.resumePath) {
        console.log("[Ashby] REVERIFY: resume is attached.");
      }
    }

    console.log("[Ashby] Form filling completed.");
  }

  async submit(): Promise<void> {
    console.log("[Ashby] Submitting application form...");
    const submitBtn = this.getPage()
      .locator("button.ashby-application-form-submit-button")
      .first();
    if (await submitBtn.isVisible().catch(() => false)) {
      await submitBtn.click();
    } else {
      await this.stagehand.act("Click the Submit Application button");
    }
    await randomSleep(500, 1000);
  }

  private async uploadResume(resumePath: string): Promise<boolean> {
    const page = this.getPage();
    const input = page.locator('#_systemfield_resume[type="file"]').first();
    console.log(`[Ashby] Uploading resume from ${resumePath}...`);
    for (let attempt = 0; attempt < 3; attempt++) {
      if ((await input.count()) === 0) {
        console.log("[Ashby] Resume already registered (input consumed).");
        return true;
      }
      try {
        await input.setInputFiles(resumePath);
      } catch (err: any) {
        console.warn(`[Ashby] Resume setInputFiles threw (attempt ${attempt + 1}): ${err?.message || err}`);
      }
      await randomSleep(1500, 2200);
      if (await this.controls.isResumeAttached()) {
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
          targets: Array<{ text: string; name: string; value: string }>;
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
            'input[type="text"], input[type="email"], input[type="tel"], input:not([type]), input[type="number"]'
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
          else if (radios.length) kind = "radio";
          else if (checks.length) kind = "checkbox";
          else if (textInput || textarea) kind = "text";
          if (!kind) continue;

          if (kind === "combobox" || kind === "text") {
            out.push({ label, id, kind, required, options: [], targets: [] });
            continue;
          }

          // Option groups (radio/single multi-checkbox/multi-select). The
          // uncovered checkbox/radio may carry no visible text (yes-no rows
          // render as buttons); read the option text from a wrapping or
          // sibling label, an option row, an li, or an aria-label.
          const options: string[] = [];
          const targets: Array<{ text: string; name: string; value: string }> = [];
          for (const inEl of [...radios, ...checks]) {
            const input = inEl as HTMLInputElement;
            const row = input.closest("label") || input.closest("[class*='option']") || input.closest("li");
            const labFor = input.id
              ? (document.querySelector(`label[for="${input.id}"]`)?.textContent || "").trim()
              : "";
            const text = collect(row) || labFor || input.getAttribute("aria-label") || "";
            if (text && !targets.some((t) => t.text === text)) {
              targets.push({ text, name: input.name || "", value: input.value || "" });
            }
          }
          // Fall back to option rows when the input has no textual label.
          if (targets.length === 0) {
            for (const row of Array.from(
              el.querySelectorAll("[class*='option'] label, [class*='option'] span, li")
            )) {
              const t = collect(row as HTMLElement);
              if (t && !options.includes(t)) options.push(t);
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
          optionTargets: (r.targets ?? []).map((t: any) => ({ text: t.text, name: t.name, value: t.value })),
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

  /** Click the option row whose normalized text exactly matches the answer. */
  override async clickGroupOption(
    field: FormField,
    answer: string,
    formSelector = "#application-form"
  ): Promise<boolean> {
    const page = this.getPage();
    const want = norm(answer);
    try {
      const hit = await page.evaluate((fid: string, d: string) => {
        const scope = document.querySelector(`div[data-field-path="${fid}"]`);
        if (!scope) return false;
        const rows = Array.from(
          scope.querySelectorAll(
            "[class*='option'] label, [class*='option'] span, button[class*='option'], li, label[for]"
          )
        );
        for (const row of rows) {
          const t = (row as HTMLElement).textContent || "";
          if (t.replace(/\s+/g, " ").trim().toLowerCase() === d) {
            (row as HTMLElement).click();
            return true;
          }
        }
        // Fallback: a lone underlying checkbox (consent gates) with no issue UI
        // row. Only safe when there is exactly ONE — never with a multi-select.
        const lone = scope.querySelectorAll('input[type="checkbox"]');
        if (lone.length === 1) {
          (lone[0] as HTMLInputElement).click();
          return true;
        }
        return false;
      }, field.id, want);
      if (!hit) return false;
      await randomSleep(250, 450);
      return await this.optionChecked(field);
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

  private async optionChecked(f: FormField): Promise<boolean> {
    const page = this.getPage();
    return page
      .evaluate((fid: string) => {
        const scope = document.querySelector(`div[data-field-path="${fid}"]`);
        if (!scope) return false;
        return Array.from(
          scope.querySelectorAll('input[type="radio"], input[type="checkbox"]')
        ).some((i) => (i as HTMLInputElement).checked);
      }, f.id)
      .catch(() => false);
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
      const page = this.getPage();
      try {
        return (await page.evaluate((fid: string) => {
          const scope = document.querySelector(`div[data-field-path="${fid}"]`);
          if (!scope) return "";
          const checked = scope.querySelector(
            'input[type="radio"]:checked, input[type="checkbox"]:checked'
          ) as HTMLInputElement | null;
          if (!checked) return "";
          const row = checked.closest("label") || checked.closest("[class*='option']");
          const rowText = row ? (row.textContent || "").replace(/\s+/g, " ").trim() : "";
          const lab = checked.id ? (document.querySelector(`label[for="${checked.id}"]`)?.textContent || "") : "";
          const active = scope.querySelector('button[class*="active"]')?.textContent || "";
          return (rowText || lab.trim() || active.replace(/\s+/g, " ").trim() || checked.value || "").trim();
        }, field.id)) as string;
      } catch {
        return "";
      }
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

  /** Resume flag: true only if the file input is consumed or has files. */
  async isResumeAttached(): Promise<boolean> {
    const page = this.getPage();
    return page
      .evaluate(() => {
        const input = document.querySelector(
          '#_systemfield_resume[type="file"]'
        ) as HTMLInputElement | null;
        if (!input) return true;
        return !!(input.files && input.files.length > 0);
      })
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