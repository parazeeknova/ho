import * as fs from "fs";

import { Stagehand } from "@browserbasehq/stagehand";

import { type JobPayload, type Profile } from "../types";
import { randomSleep } from "../utils/evasion";
import { ATSAdapter, type RpcHelper } from "./base";
import {
  auditBlanks,
  finalReverify,
  type SubmitOutcome,
  verifySubmitOutcome,
} from "./shared/audit";
import { FormControls } from "./shared/controls";
import { escapePromptValue, normalizeOptionText } from "./shared/matching";
import {
  fieldKey,
  type FormField,
  type JsonFieldSource,
  mergeFormInventory,
  PRE_FILLED_LABELS,
  unprocessedFields,
} from "./shared/model";
import { Screener, setBlankedRequiredCount } from "./shared/screener";

// Re-exports to keep greenhouse.test.ts import-compatible with the module as
// it existed before the shared-machinery extraction.
export {
  checkboxAction,
  fieldKey,
  isLocationAutocomplete,
  mergeFormInventory,
  unprocessedFields,
} from "./shared/model";
export {
  chooseOption,
  editDistance,
  normalizeOptionText,
  selectCandidates,
  translateToDate,
  xpathStringLiteral,
} from "./shared/matching";
export type { FormField, JsonFieldSource } from "./shared/model";

/**
 * Parse the job page HTML's embedded `window.__remixContext` question model
 * (Remix boards only). Returns the authoritative question list (field names,
 * types, required flags, option values) or null when the page has no such
 * model (legacy boards) or it cannot be parsed.
 */
export function parseRemixQuestionsModel(html: string): JsonFieldSource[] | null {
  try {
    const m = html.match(/window\.__remixContext = (\{.*?\});/);
    if (!m) return null;
    const ctx = JSON.parse(m[1]);
    const loader = Object.values(ctx?.state?.loaderData ?? {}).find(
      (v: any) => v && typeof v === "object" && v.jobPost,
    ) as any;
    const jobPost = loader?.jobPost;
    if (!jobPost) return null;
    const out: JsonFieldSource[] = [];
    const addQuestion = (q: any): void => {
      const label = (q?.label || "").replace(/\s+/g, " ").trim();
      for (const f of q?.fields ?? []) {
        out.push({
          name: f?.name || "",
          label,
          kind: String(f?.type || "input_text"),
          required: !!q?.required,
          options: (f?.values ?? []).map((v: any) => (v?.label || "").trim()).filter(Boolean),
        });
      }
    };
    for (const q of jobPost.questions ?? []) addQuestion(q);
    for (const sec of jobPost.eeoc_sections ?? []) {
      for (const q of sec.questions ?? []) addQuestion(q);
    }
    return out;
  } catch {
    return null;
  }
}

/**
 * Parse the job posting header (title, company, location) from the embedded
 * `window.__CONFIG` JSON (Remix boards only). Location (`job_post_location`)
 * is board-agnostic — no CSS-class dependence and no hydration race.
 */
export function parseRemixJobContext(
  html: string,
): { title: string; company: string; location: string } | null {
  try {
    const m = html.match(/window\.__remixContext = (\{.*?\});/);
    if (!m) return null;
    const ctx = JSON.parse(m[1]);
    const loader = Object.values(ctx?.state?.loaderData ?? {}).find(
      (v: any) => v && typeof v === "object" && v.jobPost,
    ) as any;
    const jp = loader?.jobPost;
    if (!jp) return null;
    return {
      title: String(jp?.title || "")
        .replace(/\s+/g, " ")
        .trim(),
      company: String(jp?.company_name || "")
        .replace(/\s+/g, " ")
        .trim(),
      location: String(jp?.job_post_location || "")
        .replace(/\s+/g, " ")
        .trim(),
    };
  } catch {
    return null;
  }
}

export class GreenhouseAdapter extends ATSAdapter {
  protected controls: FormControls;
  protected profile!: Profile;

  constructor(stagehand: Stagehand) {
    super(stagehand);
    this.controls = new FormControls(stagehand, { tagName: "GreenhouseAdapter" });
  }

  protected getPage(): any {
    return this.controls.getPage();
  }

  private warn(msg: string): void {
    console.warn(`[GreenhouseAdapter] ${msg}`);
  }

  private async ensureApplicationForm(): Promise<void> {
    const formVisible = async (): Promise<boolean> => {
      // The form may render in THIS page or in a NEW TAB opened by the Apply
      // click (job-boards.eu.greenhouse.io does this); adopt whichever page
      // shows the form so the rest of the fill drives the right document.
      for (const p of this.stagehand.context.pages()) {
        try {
          if (
            await p
              .locator("#first_name, #application-form")
              .first()
              .isVisible()
              .catch(() => false)
          ) {
            if (p !== this.controls.getPage()) {
              this.controls.adoptPage(p);
            }
            return true;
          }
        } catch {
          // Page may have been closed mid-poll; skip it.
        }
      }
      return false;
    };
    // Give the page a moment to hydrate, then check for the form.
    await randomSleep(1200, 2000);
    if (await formVisible()) {
      return;
    }
    // Some Greenhouse postings only reveal the application form after clicking
    // Apply. The click may open the form in a new tab OR keep the JD page —
    // poll for the form so fill() never walks an empty JD page (the ABBYY bug:
    // submit clicked nothing because the form was never reached).
    console.log("[GreenhouseAdapter] Application form not visible; clicking Apply...");
    await this.controls.safeAct("click the 'Apply for this job' or 'Apply now' button");
    for (let i = 0; i < 24; i++) {
      await randomSleep(900, 1200);
      if (await formVisible()) {
        console.log("[GreenhouseAdapter] Application form is now visible.");
        return;
      }
    }
    throw new Error(
      "Greenhouse application form never appeared after clicking Apply " +
        "(no #first_name / #application-form visible in any tab)",
    );
  }

  /**
   * Extract the job posting context (title, company, location, description)
   * from the live page. Used to personalize open-ended answers and to scope
   * country-dependent questions (work authorization, visa) to the JD's country.
   */
  private async readJobContext(): Promise<{
    title: string;
    company: string;
    location: string;
    description: string;
  }> {
    const page = this.getPage();
    try {
      // Prefer the authoritative Remix JSON, fall back to DOM selectors.
      const fetched = await fetch(page.url(), {
        headers: { "user-agent": "Mozilla/5.0" },
      });
      const html = await fetched.text();
      const remix = parseRemixJobContext(html);

      const read = async (selector: string): Promise<string> => {
        try {
          return (await page.locator(selector).first().innerText()).replace(/\s+/g, " ").trim();
        } catch {
          return "";
        }
      };

      // Title: JSON first, else DOM (.app-title / .job__title).
      let title = remix?.title ?? "";
      if (!title) {
        const titleBlock =
          (await page
            .locator(".board_title, .app-title, .job-post h1, .job-post-title h1, h1")
            .first()
            .innerText()
            .catch(() => "")) || (await page.title()).replace(/\s*[|–-].*$/, "").trim();
        title =
          titleBlock
            .split("\n")
            .map((line: string) => line.trim())
            .find(Boolean)
            ?.replace(/\s+/g, " ") ?? "";
      }

      const company =
        remix?.company ||
        (await read(".company-name, .job-post .company, [data-company]")) ||
        this.companyFromUrl();

      // Location: JSON first, else DOM below the heading.
      const location =
        remix?.location ||
        (await read(".job__location, .location, .job-location, .job-post .metadata"));

      const description = (
        await read(
          "#job-description, .job__description, .job-description, #content .job-post, #content",
        )
      ).slice(0, 6000);
      return { title, company, location, description };
    } catch (err: any) {
      this.warn(`readJobContext failed: ${err?.message || err}`);
      return { title: "", company: "", location: "", description: "" };
    }
  }

  /** Best-effort company name from the board URL token. */
  private mergeInventory(
    jsonFields: JsonFieldSource[] | null,
    domFields: FormField[],
  ): FormField[] {
    return mergeFormInventory(jsonFields, domFields);
  }

  private companyFromUrl(): string {
    try {
      const token = new URL(this.controls.getPage().url()).pathname.split("/").find(Boolean);
      return token ? token.replace(/[-_]+/g, " ") : "";
    } catch {
      return "";
    }
  }

  /**
   * Fetch the job page HTML and parse the embedded `window.__remixContext`
   * question model (Remix boards only). Returns null on legacy boards.
   */
  private async fetchQuestionsModel(): Promise<JsonFieldSource[] | null> {
    try {
      const page = this.getPage();
      const res = await fetch(page.url(), {
        headers: { "user-agent": "Mozilla/5.0" },
      });
      const html = await res.text();
      return parseRemixQuestionsModel(html);
    } catch (err: any) {
      this.warn(`fetchQuestionsModel failed: ${err?.message || err}`);
      return null;
    }
  }

  /**
   * Build a deterministic map of open questions -> {dom selector, kind} by
   * reading the Greenhouse form's labels and their associated inputs. Covers:
   *   - label[for] inputs/selects (legacy boards),
   *   - radio/checkbox groups inside fieldset / role=group / eeoc wrappers,
   *   - bare radio/checkbox groups grouped by input name.
   */
  private async collectQuestions(): Promise<FormField[]> {
    const page = this.getPage();
    try {
      const rows = await page.evaluate(() => {
        const out: Array<{
          label: string;
          id: string;
          kind: string;
          options: string[];
          targets: Array<{
            text: string;
            name: string;
            value: string;
            id?: string;
            button?: boolean;
          }>;
        }> = [];
        // WARNING: only anonymous arrows may be used here. tsx's keepNames
        // wraps ANY arrow/function with an inferred name in __name(), and
        // page.evaluate stringifies the whole function — the __name identifier
        // then throws inside the page. Only a destructured array avoids it.
        const [norm, hidden, push, addGroup] = [
          (t: string) =>
            (t || "")
              .replace(/\s+/g, " ")
              .trim()
              .replace(/^\*+|\*+$/g, ""),
          (el: Element): boolean => {
            let n: Element | null = el;
            while (n && n !== document.body) {
              const cls = (n as HTMLElement).className || "";
              if (/(^|\s)(visually-hidden|hidden)(\s|$)/.test(cls)) return true;
              if (n.getAttribute && n.getAttribute("hidden") != null) return true;
              n = n.parentElement;
            }
            return false;
          },
          (
            label: string,
            id: string,
            kind: string,
            options: string[] = [],
            targets: Array<{ text: string; name: string; value: string }> = [],
          ): void => {
            if (!label || !id) return;
            out.push({ label, id, kind, options, targets });
          },
          (inputs: HTMLInputElement[], labelText: string, anchor: string): void => {
            if (!inputs.length || !labelText) return;
            if (inputs.some((i) => hidden(i))) return;
            const single = inputs.every((i) => i.type === "radio");
            const name = inputs[0].name || "";
            if (!name) return;
            const options: string[] = [];
            const targets: Array<{
              text: string;
              name: string;
              value: string;
              id?: string;
              button?: boolean;
            }> = [];
            for (const input of inputs) {
              const wrapLabel = input.closest("label");
              const forLabel = input.id ? document.querySelector(`label[for="${input.id}"]`) : null;
              const text = norm(
                wrapLabel
                  ? wrapLabel.textContent || ""
                  : forLabel
                    ? forLabel.textContent || ""
                    : input.getAttribute("aria-label") || "",
              );
              if (!text) continue;
              options.push(text);
              targets.push({
                text,
                name: input.name || name,
                value: input.value || "",
                id: input.id || "",
              });
            }
            if (!options.length) return;
            push(labelText, anchor, single ? "radio" : "checkbox", options, targets);
          },
        ];

        // 1) label[for] -> text/select/multi inputs.
        document
          .querySelectorAll(
            "#application-form label, .application--questions label, label.select__label",
          )
          .forEach((lbl) => {
            const text = norm(lbl.textContent || "");
            const forId = (lbl as HTMLLabelElement).getAttribute("for");
            if (!text || !forId) return;
            const input = document.getElementById(forId);
            if (!input) return;
            if (hidden(input)) return;
            const type = ((input as HTMLInputElement).type || input.tagName).toLowerCase();
            if (type === "file") return;
            if (type === "radio" || type === "checkbox") return; // group collection
            const shell = input.closest("[class*='select-shell']");
            const isSelect =
              !!shell ||
              !!input.closest("[class*='select'], select") ||
              input.getAttribute("role") === "combobox";
            if (isSelect) {
              const isMulti =
                input.getAttribute("aria-multiselectable") === "true" ||
                (shell
                  ? /(^|\s)multi(\s|$)|select__multi|--is-multi/.test(shell.className as string)
                  : false);
              push(text, forId, isMulti ? "multi" : "select");
            } else {
              push(text, forId, "text");
            }
          });

        // 2) Radio/checkbox groups with an identifying container.
        const seenContainers = new Set<string>();
        document
          .querySelectorAll(
            "#application-form fieldset, #application-form [role='group'], " +
              "#application-form .eeoc__question__wrapper, " +
              ".application--questions fieldset, .application--questions [role='group']",
          )
          .forEach((container) => {
            let labelText = "";
            const legend = container.querySelector("legend");
            const labelledBy = (container as HTMLElement).getAttribute("aria-labelledby");
            if (legend) labelText = norm(legend.textContent || "");
            else if (labelledBy) {
              const l = document.getElementById(labelledBy);
              if (l) labelText = norm(l.textContent || "");
            }
            if (!labelText) {
              const wrap = container.closest(".field-wrapper, .eeoc__question__wrapper");
              const wl = wrap ? wrap.querySelector("label.select__label, label") : null;
              if (wl) labelText = norm(wl.textContent || "");
            }
            const inputs = Array.from(
              container.querySelectorAll("input[type='radio'], input[type='checkbox']"),
            ) as HTMLInputElement[];
            if (!inputs.length) return;
            const name = inputs[0].name || "";
            if (seenContainers.has(name || labelText)) return;
            seenContainers.add(name || labelText);
            addGroup(inputs, labelText, (container as HTMLElement).id || name || labelText);
          });

        // 3) Bare radio/checkbox groups grouped by input name (no container found).
        const bareSeen = new Set<string>();
        document
          .querySelectorAll(
            "#application-form input[type='radio'], #application-form input[type='checkbox'], " +
              ".application--questions input[type='radio'], .application--questions input[type='checkbox']",
          )
          .forEach((input) => {
            const i = input as HTMLInputElement;
            const name = i.name || "";
            if (!name || bareSeen.has(name)) return;
            bareSeen.add(name);
            const q = i.type === "radio" ? "radio" : "checkbox";
            const groupInputs = Array.from(
              document.querySelectorAll(`input[type="${q}"][name="${name}"]`),
            ) as HTMLInputElement[];
            if (!groupInputs.length) return;
            let labelText = "";
            const wrap = input.closest(
              ".field-wrapper, .eeoc__question__wrapper, [role='group'], fieldset",
            );
            if (wrap) {
              const legend = wrap.querySelector("legend");
              const labelledBy = (wrap as HTMLElement).getAttribute("aria-labelledby");
              if (legend) labelText = norm(legend.textContent || "");
              else if (labelledBy) {
                const l = document.getElementById(labelledBy);
                if (l) labelText = norm(l.textContent || "");
              }
              if (!labelText) {
                const wl = wrap.querySelector("label.select__label, label");
                // Must be a distinct question label, not one of this group's options.
                if (wl && !groupInputs.some((gi) => gi.closest("label") === wl)) {
                  labelText = norm(wl.textContent || "");
                }
              }
            }
            addGroup(groupInputs, labelText, name);
          });

        return out;
      });

      // De-duplicate by normalized label + kind, keep the first (deterministic
      // container-based capture wins over the bare-name pass).
      const seen = new Set<string>();
      const uniq: FormField[] = [];
      for (const r of rows ?? []) {
        const key = `${normalizeOptionText(r.label)}|${r.kind}`;
        if (seen.has(key)) continue;
        seen.add(key);
        uniq.push({
          label: r.label,
          id: r.id,
          kind: r.kind as FormField["kind"],
          required: false,
          options: r.options,
          optionTargets: r.targets,
          name: r.id.replace(/-/g, "_"),
        });
      }
      return uniq;
    } catch (err: any) {
      this.warn(`collectQuestions failed: ${err?.message || err}`);
      return [];
    }
  }

  /**
   * Upload the resume file to the resume field and verify it actually attached.
   */
  private async uploadResume(page: any, resumePath: string): Promise<boolean> {
    const baseName = resumePath.split(/[\\/]/).pop() || "";
    console.log(`[GreenhouseAdapter] Uploading resume from ${resumePath}...`);
    for (let attempt = 0; attempt < 3; attempt++) {
      // The resume input may hydrate after the rest of the form (or after the
      // Apply click opened the form in a new tab) — wait for it so
      // setInputFiles targets a real element, never an absent locator.
      const resumeInput = page.locator('input#resume[type="file"]').first();
      for (let i = 0; i < 12; i++) {
        if ((await resumeInput.count()) > 0) break;
        await randomSleep(500, 800);
      }
      if ((await resumeInput.count()) === 0) {
        // No file input in the DOM. Only count it as attached when a rendered
        // attachment shows the resume name — an absent input is NOT proof (it
        // may simply not have hydrated yet).
        if (await this.isResumeAttached(page, baseName)) {
          console.log("[GreenhouseAdapter] Resume already registered (attachment visible).");
          return true;
        }
        console.warn(
          `[GreenhouseAdapter] Resume input not present (attempt ${attempt + 1}); retrying...`,
        );
        continue;
      }
      try {
        await resumeInput.setInputFiles(resumePath);
      } catch (err: any) {
        console.warn(
          `[GreenhouseAdapter] Resume setInputFiles threw (attempt ${attempt + 1}): ${err?.message || err}`,
        );
      }
      await randomSleep(1500, 2200);
      const registered = await this.isResumeAttached(page, baseName);
      if (registered) {
        console.log("[GreenhouseAdapter] Resume uploaded and registered.");
        return true;
      }
      console.warn(
        `[GreenhouseAdapter] Resume upload not confirmed (attempt ${attempt + 1}); retrying...`,
      );
    }
    return false;
  }

  /** Whether the resume is attached (input has files, or the name is shown in
   *  a rendered file-upload chip). A missing input is NOT proof of attachment
   *  — it may simply not have hydrated yet. */
  private async isResumeAttached(page: any, fileName = ""): Promise<boolean> {
    return page
      .evaluate((name: string) => {
        const input = document.querySelector(
          'input#resume[type="file"]',
        ) as HTMLInputElement | null;
        if (input && input.files && input.files.length > 0) return true;
        if (!name) return false;
        const area = document.querySelector(
          ".file-upload, [class*='file-upload'], [id^='upload-label-']",
        );
        const text = area ? area.textContent || "" : "";
        return !!name && (text.includes(name) || text.replace(/\s+/g, "").includes(name));
      }, fileName)
      .catch(() => false);
  }

  /**
   * Locate the cover letter field across board flavors (visible textarea or a
   * file upload with an "Enter manually" toggle).
   */
  private async findCoverLetterField(): Promise<any> {
    const page = this.getPage();
    const textareaSel =
      "textarea#cover_letter_text, textarea[name*='cover_letter'], textarea[aria-label*='cover letter' i], #job_application_cover_letter";
    let ta = page.locator(textareaSel).first();
    if (await ta.isVisible().catch(() => false)) return ta;
    const manual = page.locator('button[data-testid="cover_letter-text"]').first();
    if (await manual.isVisible().catch(() => false)) {
      await manual.click();
      for (let i = 0; i < 8; i++) {
        await randomSleep(400, 700);
        ta = page.locator(textareaSel).first();
        if (await ta.isVisible().catch(() => false)) return ta;
      }
      return null;
    }
    return null;
  }

  async fill(payload: JobPayload, rpc?: RpcHelper): Promise<void> {
    const { url, profile } = payload;
    this.profile = profile;

    console.log(`[GreenhouseAdapter] Navigating to ${url}...`);
    const page = this.getPage();
    await page.goto(url);
    await randomSleep(300, 600);

    await this.ensureApplicationForm();

    const controls = this.controls;
    console.log("[GreenhouseAdapter] Filling deterministic profile fields...");
    await controls.fillField(
      "#first_name",
      profile.firstName,
      "Type %firstName% into the First Name input field",
      "firstName",
    );
    await controls.fillField(
      "#last_name",
      profile.lastName,
      "Type %lastName% into the Last Name input field",
      "lastName",
    );
    await controls.fillField(
      "#email",
      profile.email,
      "Type %email% into the Email input field",
      "email",
    );
    await controls.fillField(
      "#phone",
      profile.phone,
      "Type %phone% into the Phone input field",
      "phone",
    );

    let resumeAttached = false;
    if (profile.resumePath && fs.existsSync(profile.resumePath)) {
      resumeAttached = await this.uploadResume(page, profile.resumePath);
    }

    if (rpc) {
      // Job context first so open-ended answers are personalized to the role.
      const jobCtx = await this.readJobContext();
      await rpc("job_context", jobCtx);
      console.log(
        `[GreenhouseAdapter] Job context: ${jobCtx.title || "?"} @ ${jobCtx.company || "?"}` +
          (jobCtx.location ? ` (${jobCtx.location})` : ""),
      );

      const jsonModel = await this.fetchQuestionsModel();

      const filled: string[] = [];
      const blanked: Array<{ label: string; reason: string }> = [];
      const processedKeys = new Set<string>();
      const userSkippedKeys = new Set<string>();
      const screener = new Screener(controls, "GreenhouseAdapter", profile, rpc);

      // Iterative re-scan: conditional questions (e.g. Race, which renders only
      // after "Are you Hispanic/Latino?" is answered) appear in the DOM only
      // after an earlier interaction. Rescan each pass, process only new fields,
      // and converge when nothing new appears.
      for (let pass = 0; pass < 30; pass++) {
        const domFields = await this.collectQuestions();
        const inventory = this.mergeInventory(jsonModel, domFields);
        const newFields = unprocessedFields(inventory, processedKeys);
        if (pass === 0) {
          console.log(
            `[GreenhouseAdapter] Question inventory: ${inventory.length} ` +
              `(json: ${jsonModel?.length ?? 0}, dom: ${domFields.length})`,
          );
        }
        if (newFields.length === 0) {
          if (pass > 0) {
            console.log(`[GreenhouseAdapter] Walk converged after ${pass + 1} pass(es).`);
          }
          break;
        }
        console.log(
          `[GreenhouseAdapter] Walk pass ${pass + 1}: ${newFields.length} new question(s).`,
        );

        for (const field of newFields) {
          // Mark BEFORE filling so a re-scan can never re-ask or re-fill.
          processedKeys.add(fieldKey(field));
          await screener.process(field, filled, blanked, userSkippedKeys);
        }

        await randomSleep(900, 1400);
      }

      // Zero-blank audit after the walk (reports required blanks; records reasons
      // for optional empties into blanked).
      const finalDom = await this.collectQuestions();
      const inventory = this.mergeInventory(jsonModel, finalDom);
      const requiredBlanks = await auditBlanks({
        fields: inventory,
        readValue: (f) => controls.readFieldValue(f),
        transcript: blanked,
      });

      console.log(
        `[GreenhouseAdapter] Screener walk complete. Filled: ${filled.length}, blank (declined/unknown): ${blanked.length}.`,
      );
      for (const b of blanked) {
        console.warn(`[GreenhouseAdapter]   blank: ${escapePromptValue(b.label)} (${b.reason})`);
      }
      if (requiredBlanks.length > 0) {
        const warn = (msg: string) => console.warn(`[GreenhouseAdapter] ${msg}`);
        warn(`${requiredBlanks.length} REQUIRED field(s) left blank after the walk:`);
        for (const rb of requiredBlanks) {
          warn(`  REQUIRED blank: ${escapePromptValue(rb.label)} (${rb.reason})`);
        }
      }

      // Cover letter: LLM-generated, personalized to the job description.
      const clField = await this.findCoverLetterField();
      if (clField) {
        const coverLetterResult = await rpc("cover_letter", {});
        const coverLetter = (coverLetterResult?.answer ?? "").toString().trim();
        if (coverLetter) {
          await clField.fill(coverLetter);
          let value = await clField.inputValue().catch(() => "");
          if (!value) {
            await clField.fill(coverLetter);
            value = await clField.inputValue().catch(() => "");
          }
          if (value) {
            console.log(
              "[GreenhouseAdapter] Cover letter filled (LLM-generated, JD-personalized).",
            );
          } else {
            console.warn("[GreenhouseAdapter] Cover letter did not commit; left blank.");
          }
        } else {
          console.log("[GreenhouseAdapter] Cover letter skipped: LLM had nothing to ground it on.");
        }
      } else {
        console.log("[GreenhouseAdapter] No cover letter field on this form; skipping generation.");
      }

      // Definitive final sweep: rescan the ENTIRE form and fill ANY input still
      // empty (except identity fields + manual skips), iterating to convergence.
      const sweepPasses: string[] = [];
      const sweepBlanks: Array<{ label: string; reason: string }> = [];
      for (let pass = 0; pass < 3; pass++) {
        const sweptDom = await this.collectQuestions();
        const sweptInventory = this.mergeInventory(jsonModel, sweptDom);
        let touched = 0;
        for (const field of sweptInventory) {
          const key = normalizeOptionText(field.label);
          if (PRE_FILLED_LABELS.has(key)) continue;
          const value = await controls.readFieldValue(field);
          if (value) continue;
          if (userSkippedKeys.has(fieldKey(field))) continue;
          touched += 1;
          await screener.process(field, sweepPasses, sweepBlanks, userSkippedKeys);
        }
        if (touched === 0) break;
      }
      if (sweepPasses.length > 0) {
        console.log(
          `[GreenhouseAdapter] Final sweep filled ${sweepPasses.length} previously-empty field(s):`,
        );
        for (const label of sweepPasses) {
          console.log(`[GreenhouseAdapter]   filled: ${escapePromptValue(label)}`);
        }
      }

      // Re-run the required-blank audit over the final state.
      const finalDom2 = await this.collectQuestions();
      const finalInventory = this.mergeInventory(jsonModel, finalDom2);
      const finalRequired = await auditBlanks({
        fields: finalInventory,
        readValue: (f) => controls.readFieldValue(f),
        transcript: [...blanked, ...sweepBlanks],
      });
      if (finalRequired.length > 0) {
        console.warn(
          `[GreenhouseAdapter] ${finalRequired.length} REQUIRED field(s) still blank after the final sweep:`,
        );
        for (const rb of finalRequired) {
          console.warn(
            `[GreenhouseAdapter]   REQUIRED blank: ${escapePromptValue(rb.label)} (${rb.reason})`,
          );
        }
      } else {
        console.log("[GreenhouseAdapter] Final sweep complete: no required field is blank.");
      }
      // Surface how many required fields are still blank so the runner can
      // gate auto-submit on an incomplete form.
      setBlankedRequiredCount(finalRequired.length);

      // Definitive reverify of every still-empty field (required/optional, minus skips).
      await finalReverify({
        tag: "GreenhouseAdapter",
        collect: async () => {
          const dom = await this.collectQuestions();
          return this.mergeInventory(jsonModel, dom);
        },
        isEmpty: async (field) => !(await controls.readFieldValue(field)),
        skippedKeys: userSkippedKeys,
        reasons: [...blanked, ...sweepBlanks],
      });

      // Resume reverify: surface a failed attach instead of submitting without a CV.
      const resumeBase = (profile.resumePath ?? "").split(/[\\/]/).pop() || "";
      if (
        profile.resumePath &&
        !resumeAttached &&
        !(await this.isResumeAttached(page, resumeBase))
      ) {
        console.warn("[GreenhouseAdapter] REVERIFY: resume is NOT attached after the final pass.");
      } else if (profile.resumePath) {
        console.log("[GreenhouseAdapter] REVERIFY: resume is attached.");
      }
    }

    console.log("[GreenhouseAdapter] Form filling completed.");
  }

  async submit(): Promise<SubmitOutcome> {
    console.log("[GreenhouseAdapter] Submitting application form...");
    const page = this.getPage();
    await this.stagehand.act("Click the Submit Application button");
    await randomSleep(500, 1000);

    return verifySubmitOutcome(page, {
      tag: "Greenhouse",
      successUrlRe: /thanks|submitted|confirmation|success|applied|complete/i,
      submitButtonSelector:
        "input[type='submit'], button[type='submit'], button:has-text('Submit Application')",
    });
  }

  /**
   * Recheck and re-fill any required field that is still blank, then report
   * how many remain blank. Called by the runner after a retryable submit
   * failure (validation blocked by unfilled fields).
   */
  async recheckMissingFields(rpc?: RpcHelper): Promise<number> {
    console.log("[GreenhouseAdapter] Rechecking missing required fields...");
    const stillBlank: string[] = [];
    const controls = this.controls;
    const fields = await this.collectQuestions();
    const inventory = this.mergeInventory(null, fields);
    for (const f of inventory) {
      if (!f.required) continue;
      if (await controls.readFieldValue(f)) continue;
      if (PRE_FILLED_LABELS.has(normalizeOptionText(f.label))) continue;
      const screener = new Screener(
        controls,
        "GreenhouseAdapter",
        this.profile,
        rpc ?? (async () => ({ answer: "" })),
      );
      const filled: string[] = [];
      const blanked: { label: string; reason: string }[] = [];
      const skipped = new Set<string>();
      await screener.process(f, filled, blanked, skipped);
      if (filled.length === 0) stillBlank.push(f.label);
    }
    const remaining = stillBlank.length;
    setBlankedRequiredCount(remaining);
    console.log(
      `[GreenhouseAdapter] Recheck complete: ${remaining} required field(s) still blank.`,
    );
    for (const l of stillBlank) {
      console.warn(`[GreenhouseAdapter]   still blank: ${escapePromptValue(l)}`);
    }
    return remaining;
  }
}
