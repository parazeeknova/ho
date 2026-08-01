import { Stagehand } from "@browserbasehq/stagehand";
import * as fs from "fs";
import { ATSAdapter, RpcHelper } from "./base.js";
import { JobPayload, Profile } from "../types.js";
import { randomSleep } from "../utils/evasion.js";

function escapePromptValue(val: string): string {
  return val.replace(/\\/g, '\\\\').replace(/"/g, '\\"');
}

/**
 * Ordered candidate values to try when matching a free-form answer to a
 * dropdown option: the raw answer, its first clause (split on commas,
 * periods, semicolons), and a leading Yes/No token. Deduplicated.
 */
export function selectCandidates(answer: string): string[] {
  const candidates: string[] = [];
  const seen = new Set<string>();
  const push = (value: string): void => {
    const t = value.trim();
    if (t && !seen.has(t.toLowerCase())) {
      seen.add(t.toLowerCase());
      candidates.push(t);
    }
  };
  const raw = answer.trim();
  if (!raw) return candidates;
  push(raw);
  const clause = raw.split(/[.,;]\s+|,/, 1)[0].trim();
  if (clause && clause.length < raw.length) push(clause);
  const firstToken = raw.split(/\s+/, 1)[0].trim();
  if (/^(yes|no)$/i.test(firstToken)) push(firstToken);
  return candidates;
}

/** Case-insensitive XPath equality predicate for a literal text value. */
function optionExactXPath(text: string): string {
  const safe = escapePromptValue(text).replace(/'/g, "\\'");
  return (
    'translate(normalize-space(.), "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz") = "' +
    safe.toLowerCase() +
    '"'
  );
}

export function normalizeOptionText(text: string): string {
  return text.replace(/\s+/g, " ").trim().toLowerCase();
}

/**
 * "I don't wish to answer" style options are the user-decline choices of an
 * EEOC-style survey. They are never valid targets for a definite answer, so
 * substring matching must not resolve to them. Covers Greenhouse's
 * "I do not want to answer" phrasing too.
 */
export function isDeclineOption(text: string): boolean {
  return /(don'?t wish|do not wish|prefer not|choose not|rather not|not wish|do not want to answer|not want to answer)/i.test(text);
}

/**
 * Pick the option text that best matches an answer, from a candidate list.
 * Exact matches are preferred; a substring match is only accepted when it is
 * unambiguous within the option list (and never against a decline option).
 * Returns null when nothing matches confidently — callers must leave the
 * field blank rather than guess.
 */
export function chooseOption(candidates: string[], optionTexts: string[]): string | null {
  const eligible = optionTexts.filter((t) => !isDeclineOption(t));
  for (const candidate of candidates) {
    const nc = normalizeOptionText(candidate);
    const exact = eligible.filter((t) => normalizeOptionText(t) === nc);
    if (exact.length === 1) return exact[0];
    if (exact.length === 0) {
      const subs = eligible.filter((t) => normalizeOptionText(t).includes(nc));
      if (subs.length === 1) return subs[0];
    }
  }
  return null;
}

/** CSS attribute-value escaping (values are alphanumeric in practice). */
function cssEscape(text: string): string {
  return (text || "").replace(/\\/g, "\\\\").replace(/"/g, '\\"');
}

export interface GroupOption {
  text: string;
  name: string;
  value: string;
}

/** A single form question captured by the generic DOM walker. */
export interface FormField {
  label: string;
  /** Primary element id (text/select) or a stable group anchor (radio/checkbox). */
  id: string;
  kind: "text" | "select" | "multi" | "radio" | "checkbox";
  required: boolean;
  options: string[];
  /** Radio/checkbox option click targets (text + input name/value). */
  optionTargets: GroupOption[];
  /** Canonical field name (snake_case) when known from the board's JSON model. */
  name?: string;
}

/** A question as described by the board's embedded JSON model. */
interface JsonFieldSource {
  name: string;
  label: string;
  kind: string;
  required: boolean;
  options: string[];
}

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
      (v: any) => v && typeof v === "object" && v.jobPost
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
          options: (f?.values ?? [])
            .map((v: any) => (v?.label || "").trim())
            .filter(Boolean),
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
 * Merge the board's JSON question model (authoritative) with the DOM
 * enumeration (ground truth for what is actually rendered).
 *
 * The DOM is the BASE inventory: only rendered questions are walked and filled.
 * The JSON model ENRICHES the DOM fields with canonical names, exact option
 * values and required flags (matched by field name, then by normalized label).
 * JSON-only fields are dropped — boards often list questions (e.g. the EEOC
 * Race question) that they never render, and walking phantom fields fails
 * verification loudly for no reason.
 */
export function mergeFormInventory(
  jsonFields: JsonFieldSource[] | null,
  domFields: FormField[]
): FormField[] {
  const out: FormField[] = [];
  const seen = new Set<string>();
  const add = (f: FormField) => {
    const key = `${normalizeOptionText(f.label)}|${f.kind}`;
    if (seen.has(key)) return;
    seen.add(key);
    out.push(f);
  };

  const jsonByName = new Map<string, JsonFieldSource>();
  const jsonByLabel = new Map<string, JsonFieldSource>();
  for (const jf of jsonFields ?? []) {
    if (jf.kind === "input_file") continue; // uploads handled by dedicated paths
    if (/^resume(_text)?$|^cover_letter(_text)?$/.test(jf.name)) continue;
    jsonByName.set(jf.name, jf);
    jsonByLabel.set(normalizeOptionText(jf.label), jf);
  }

  for (const df of domFields) {
    const jf =
      (df.name ? jsonByName.get(df.name) : undefined) ||
      jsonByLabel.get(normalizeOptionText(df.label));
    if (jf) {
      add({
        ...df,
        required: df.required || jf.required,
        options: jf.options.length ? jf.options : df.options,
        name: jf.name,
      });
    } else {
      add(df);
    }
  }
  return out;
}

/** Deterministic profile-driven fills keyed by normalized question label. */
const PROFILE_FILLS: Record<string, keyof Profile> = {
  "preferred first name": "preferredName",
  linkedin: "linkedin",
  "linkedin profile": "linkedin",
  github: "github",
  website: "website",
};

/** Questions answered by the fixed deterministic identity fills before the walk. */
const PRE_FILLED_LABELS = new Set(["first name", "last name", "email", "phone"]);

/** Consent/privacy toggles are handled separately, never walked as questions. */
const CONSENT_RE = /agree|consent|retain|privacy|policy|gdpr|data protection/i;

/** Stable identity for a form field across rescans (label + kind + id). */
export function fieldKey(f: FormField): string {
  return `${normalizeOptionText(f.label)}|${f.kind}|${f.id}`;
}

/**
 * The subset of ``fields`` that has not been processed yet, as a pure
 * diff over the processed-key set. Used by the iterative re-scan walk so
 * fields revealed only after an interaction (conditional questions, e.g.
 * Race after answering "Are you Hispanic/Latino?") are picked up in a later
 * pass while already-processed fields are never re-asked or re-filled.
 */
export function unprocessedFields(
  fields: FormField[],
  processedKeys: ReadonlySet<string>
): FormField[] {
  return fields.filter((f) => !processedKeys.has(fieldKey(f)));
}

export class GreenhouseAdapter extends ATSAdapter {
  /**
   * Run an AI-powered act() call but never let a single failure abort the whole
   * form fill (e.g. a field that doesn't exist on this job's form). Values are
   * passed through Stagehand's `variables` so they are never parsed as prompt text.
   */
  private async safeAct(
    instruction: string,
    variables?: Record<string, string>
  ): Promise<void> {
    try {
      await this.stagehand.act(instruction, variables ? { variables } : {});
    } catch (err: any) {
      console.warn(
        `[GreenhouseAdapter] act() failed (continuing): ${err?.message || err}`
      );
    }
  }

  /**
   * Prefer a deterministic locator fill; fall back to an AI-driven act() using
   * %variables% so arbitrary profile text (quotes, newlines) is handled safely.
   */
  private async fillField(
    selector: string,
    value: string | undefined | null,
    actPrompt: string,
    variableName: string
  ): Promise<void> {
    if (!value) return;
    const locator = this.getPage().locator(selector).first();
    if (await locator.isVisible().catch(() => false)) {
      await locator.fill(value);
      await randomSleep(100, 300);
    } else {
      await this.safeAct(actPrompt, { [variableName]: value });
      await randomSleep(200, 500);
    }
  }

  private getPage() {
    return this.stagehand.context.pages()[0];
  }

  /** Close an open react-select menu without depending on keyboard typing. */
  private async closeMenu(): Promise<void> {
    try {
      await (this.getPage() as any).keyboard?.press("Escape");
    } catch {
      // Menu may already be closed; harmless.
    }
  }

  private async ensureApplicationForm(): Promise<void> {
    const page = this.getPage();
    const form = page.locator('#first_name, #application-form').first();
    // Give the page a moment to hydrate, then check for the form.
    await randomSleep(1200, 2000);
    if (await form.isVisible().catch(() => false)) {
      return;
    }
    // Some Greenhouse postings only reveal the application form after clicking Apply.
    console.log("[GreenhouseAdapter] Application form not visible; clicking Apply...");
    await this.safeAct("click the 'Apply for this job' or 'Apply now' button");
    await randomSleep(800, 1500);
  }

  /**
   * Extract the job posting context (title, company, location, description)
   * from the live page. Used to personalize open-ended answers and to scope
   * country-dependent questions (work authorization, visa) to the JD's country.
   *
   * NOTE: must not read the DOM via page.evaluate(fn) with named inner helper
   * functions — tsx transpiles with keepNames, which wraps them in __name(),
   * and the stringified function then throws ReferenceError inside the page.
   * Locator/read APIs (Node-side) avoid that entirely.
   */
  private async readJobContext(): Promise<{
    title: string;
    company: string;
    location: string;
    description: string;
  }> {
    const page = this.getPage();
    try {
      const read = async (selector: string): Promise<string> => {
        try {
          return (await page.locator(selector).first().innerText())
            .replace(/\s+/g, " ")
            .trim();
        } catch {
          return "";
        }
      };
      // Both legacy (.app-title, #job-description) and the newer Remix board
      // (.job__title, .job__location, .job__description) are covered. The
      // Remix title block stacks the location under the role, so split before
      // whitespace is collapsed and take the first line as the role title.
      const titleBlock =
        (await page
          .locator(".job__title, .app-title, .job-post h1, .job-post-title h1, h1")
          .first()
          .innerText()
          .catch(() => "")) ||
        (await page.title()).replace(/\s*[|–-].*$/, "").trim();
      const title =
        titleBlock
          .split("\n")
          .map((line) => line.trim())
          .filter(Boolean)[0]
          ?.replace(/\s+/g, " ") ?? "";
      const company =
        (await read(".company-name, .job-post .company, [data-company]")) ||
        this.companyFromUrl();
      const location = await read(
        ".job__location, .location, .job-location, .job-post .metadata"
      );
      const description = (
        await read(
          "#job-description, .job__description, .job-description, #content .job-post, #content"
        )
      ).slice(0, 6000);
      return { title, company, location, description };
    } catch (err: any) {
      console.warn("[GreenhouseAdapter] readJobContext failed:", err?.message || err);
      return { title: "", company: "", location: "", description: "" };
    }
  }

  /** Best-effort company name from the board URL token (job-boards.greenhouse.io/<token>/...). */
  private companyFromUrl(): string {
    try {
      const token = new URL(this.stagehand.context.pages()[0].url()).pathname
        .split("/")
        .filter(Boolean)[0];
      return token ? token.replace(/[-_]+/g, " ") : "";
    } catch {
      return "";
    }
  }

  /**
   * Fetch the job page HTML and parse the embedded `window.__remixContext`
   * question model (Remix boards only). This is the authoritative question
   * inventory: exact field names, types, required flags and option values —
   * no menu-opening needed just to read options. Returns null on legacy boards
   * (no __remixContext) or any fetch/parse failure; the caller falls back to
   * pure DOM enumeration.
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
      console.warn(
        `[GreenhouseAdapter] fetchQuestionsModel failed: ${err?.message || err}`
      );
      return null;
    }
  }

  /**
   * Build a deterministic map of open questions -> {dom selector, kind} by
   * reading the Greenhouse form's labels and their associated inputs. Covers:
   *   - label[for] inputs/selects (legacy boards),
   *   - radio/checkbox groups inside fieldset / role=group / eeoc wrappers,
   *   - bare radio/checkbox groups grouped by input name (catches questions
   *     whose group has no container with a legend/label).
   * This avoids relying on the LLM to map a question's text back to an element.
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
          targets: Array<{ text: string; name: string; value: string }>;
        }> = [];
        // NOTE: only anonymous arrows may be used here. tsx's keepNames wraps
        // ANY arrow/function with an inferred name (const assignment, object
        // property) in __name(), and page.evaluate stringifies the whole
        // function — the __name identifier then throws inside the page. Arrows
        // held in a destructured array have no inferred name, so they survive.
        const [norm, hidden, push, addGroup] = [
          (t: string) =>
            (t || "")
              .replace(/\s+/g, " ")
              .trim()
              .replace(/^\*+|\*+$/g, ""),
          // Skip elements inside visually-hidden / hidden containers (a11y-only
          // labels for toggles like the cover letter textarea).
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
            targets: Array<{ text: string; name: string; value: string }> = []
          ): void => {
            if (!label || !id) return;
            out.push({ label, id, kind, options, targets });
          },
          (
            inputs: HTMLInputElement[],
            labelText: string,
            anchor: string
          ): void => {
            if (!inputs.length || !labelText) return;
            // Module-level consts are out of scope inside evaluate, so the
            // consent regex is inlined here.
            if (/(agree|consent|retain|privacy|policy|gdpr|data protection)/i.test(labelText)) return;
            if (inputs.some((i) => hidden(i))) return;
            const single = inputs.every((i) => i.type === "radio");
            const name = inputs[0].name || "";
            if (!name) return;
            const options: string[] = [];
            const targets: Array<{ text: string; name: string; value: string }> = [];
            for (const input of inputs) {
              const wrapLabel = input.closest("label");
              const text = norm(
                wrapLabel
                  ? wrapLabel.textContent || ""
                  : input.getAttribute("aria-label") || ""
              );
              if (!text || /(agree|consent|retain|privacy|policy|gdpr|data protection)/i.test(text)) continue;
              options.push(text);
              targets.push({ text, name: input.name || name, value: input.value || "" });
            }
            if (!options.length) return;
            push(labelText, anchor, single ? "radio" : "checkbox", options, targets);
          },
        ];

        // 1) label[for] -> text/select/multi inputs (existing behaviour).
        document
          .querySelectorAll(
            "#application-form label, .application--questions label, label.select__label"
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
                  ? /(^|\s)multi(\s|$)|select__multi|--is-multi/.test(shell.className)
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
              ".application--questions fieldset, .application--questions [role='group']"
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
              const wl = wrap
                ? wrap.querySelector("label.select__label, label")
                : null;
              if (wl) labelText = norm(wl.textContent || "");
            }
            const inputs = Array.from(
              container.querySelectorAll("input[type='radio'], input[type='checkbox']")
            ) as HTMLInputElement[];
            if (!inputs.length) return;
            const name = inputs[0].name || "";
            if (seenContainers.has(name || labelText)) return;
            seenContainers.add(name || labelText);
            addGroup(
              inputs,
              labelText,
              (container as HTMLElement).id || name || labelText
            );
          });

        // 3) Bare radio/checkbox groups grouped by input name (no container found).
        const bareSeen = new Set<string>();
        document
          .querySelectorAll(
            "#application-form input[type='radio'], #application-form input[type='checkbox'], " +
              ".application--questions input[type='radio'], .application--questions input[type='checkbox']"
          )
          .forEach((input) => {
            const i = input as HTMLInputElement;
            const name = i.name || "";
            if (!name || bareSeen.has(name)) return;
            bareSeen.add(name);
            const q = i.type === "radio" ? "radio" : "checkbox";
            const groupInputs = Array.from(
              document.querySelectorAll(
                `input[type="${q}"][name="${name}"]`
              )
            ) as HTMLInputElement[];
            if (!groupInputs.length) return;
            let labelText = "";
            const wrap = input.closest(".field-wrapper, .eeoc__question__wrapper, [role='group'], fieldset");
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
      console.warn(`[GreenhouseAdapter] collectQuestions failed:`, err?.message || err);
      return [];
    }
  }

  /**
   * Merge the board's JSON question model (authoritative) with the DOM
   * enumeration (ground truth for what is actually rendered). JSON supplies
   * canonical names, options, and required flags; the DOM supplies the
   * selector/kind and catches rendered questions absent from the JSON model.
   */
  private mergeInventory(
    jsonFields: JsonFieldSource[] | null,
    domFields: FormField[]
  ): FormField[] {
    return mergeFormInventory(jsonFields, domFields);
  }

  private normalise(text: string): string {
    return text.replace(/\s+/g, " ").trim().replace(/^\*+|\*+$/g, "").toLowerCase();
  }

  private async fillQuestionText(id: string, answer: string): Promise<void> {
    const page = this.getPage();
    const loc = page.locator(`#${id}`).first();
    if (await loc.isVisible().catch(() => false)) {
      const type = await page
        .evaluate((inputId) => {
          const el = document.getElementById(inputId) as HTMLInputElement | null;
          return el?.type ?? null;
        }, id)
        .catch(() => null);
      if (type === "file") return; // file inputs crash fill(); resume handled elsewhere
      await loc.fill(answer);
      await randomSleep(200, 500);
    }
  }

  /**
   * Ensure the react-select menu for a field is open. Escape first so a stale
   * menu from a previous field can't make the control click *toggle* the menu
   * shut; then click the control and verify with Playwright's strict
   * visibility (hidden stale options must not count as "open").
   */
  private async ensureMenuOpen(id: string): Promise<boolean> {
    const page = this.getPage();
    const shellXPath = `//div[contains(@class,"select-shell")][.//input[@id="${id}"]]`;
    const control = page
      .locator(`${shellXPath}//div[contains(@class,"select__control")]`)
      .first();
    if (!(await control.isVisible().catch(() => false))) {
      return false;
    }
    await this.closeMenu();
    await randomSleep(150, 300);
    for (let attempt = 0; attempt < 3; attempt++) {
      await control.click();
      await randomSleep(300, 500);
      if (await this.hasVisibleOption()) return true;
    }
    return false;
  }

  /** Whether any option element is currently visible (the open menu's). */
  private async hasVisibleOption(): Promise<boolean> {
    const page = this.getPage();
    const options = page.locator('div[role="option"]');
    const count = await options.count().catch(() => 0);
    for (let i = 0; i < count; i++) {
      if (await options.nth(i).isVisible().catch(() => false)) {
        return true;
      }
    }
    return false;
  }

  /** Click the visible option whose text exactly matches ``picked``. */
  private async clickVisibleOption(picked: string): Promise<boolean> {
    const page = this.getPage();
    try {
      // Walk matches in DOM order and click the first visible one: closed
      // portal menus can keep hidden options in the DOM, so we must never
      // blindly take `.first()`.
      const options = page.locator(
        `//div[contains(@role,"option")][${optionExactXPath(picked)}]`
      );
      const count = await options.count().catch(() => 0);
      for (let i = 0; i < count; i++) {
        const option = options.nth(i);
        if (!(await option.isVisible().catch(() => false))) {
          continue;
        }
        await option.click();
        return true;
      }
      return false;
    } catch {
      return false;
    }
  }

  /**
   * Fill a react-select question dropdown. The option menu renders portaled
   * outside the field's own select-shell, so options are located page-wide and
   * filtered to visible ones (only one menu can be open at a time, so this is
   * unambiguous). ``optionTexts`` is the option list read by
   * ``readSelectOptions``; the picked option must come from that list so the
   * click is an exact-text match, never a substring guess.
   */
  private async fillQuestionSelect(
    id: string,
    question: string,
    answer: string,
    optionTexts: string[]
  ): Promise<boolean> {
    const page = this.getPage();
    try {
      if (!(await this.ensureMenuOpen(id))) {
        return false;
      }

      const picked = chooseOption(selectCandidates(answer), optionTexts);
      if (!picked) {
        console.warn(
          `[GreenhouseAdapter] No matching option for #${id} ` +
            `(answer "${escapePromptValue(answer)}"); leaving blank.`
        );
        await this.closeMenu();
        return false;
      }

      let clicked = await this.clickVisibleOption(picked);
      if (!clicked) {
        // A stale open menu can satisfy ensureMenuOpen while the real menu
        // is closed; close and reopen deterministically, then retry once.
        console.warn(
          `[GreenhouseAdapter] Option "${picked}" not visible on first try for #${id}; reopening...`
        );
        await this.closeMenu();
        await randomSleep(150, 300);
        if (await this.ensureMenuOpen(id)) {
          clicked = await this.clickVisibleOption(picked);
        }
      }
      if (!clicked) {
        console.warn(
          `[GreenhouseAdapter] Picked option "${picked}" not visible for #${id} ` +
            `(answer "${escapePromptValue(answer)}")`
        );
        await this.closeMenu();
        return false;
      }
      await randomSleep(300, 600);
      console.log(`[GreenhouseAdapter] Selected option "${picked}" for #${id}`);
      return true;
    } catch (err: any) {
      console.warn(`[GreenhouseAdapter] fillQuestionSelect failed for #${id}:`, err?.message || err);
      return false;
    }
  }

  /**
   * Fill a react-select multi-select: one exact match per comma-separated
   * pick, clicking each while the menu stays open.
   */
  private async fillQuestionMulti(
    id: string,
    question: string,
    answer: string,
    optionTexts: string[]
  ): Promise<boolean> {
    const page = this.getPage();
    try {
      if (!(await this.ensureMenuOpen(id))) {
        return false;
      }

      const picks = answer
        .split(",")
        .map((p) => p.trim())
        .filter(Boolean);
      let clicked = 0;
      for (const pick of picks) {
        const picked = chooseOption(selectCandidates(pick), optionTexts);
        if (!picked) {
          console.warn(
            `[GreenhouseAdapter] No matching multi option for #${id} ` +
              `(answer "${escapePromptValue(pick)}")`
          );
          continue;
        }
        if (await this.clickVisibleOption(picked)) {
          clicked += 1;
          await randomSleep(200, 400);
        } else {
          console.warn(`[GreenhouseAdapter] Multi option "${picked}" not visible for #${id}`);
        }
        // Multi-select menus normally stay open; if a pick closed it, reopen.
        await this.ensureMenuOpen(id);
      }
      await this.closeMenu();
      console.log(`[GreenhouseAdapter] Selected ${clicked} option(s) for #${id}`);
      return clicked > 0;
    } catch (err: any) {
      console.warn(`[GreenhouseAdapter] fillQuestionMulti failed for #${id}:`, err?.message || err);
      return false;
    }
  }

  /**
   * Read the real option texts of a dropdown by opening its menu. The menu is
   * portaled outside the field's select-shell, so options are collected
   * document-wide and filtered to visible ones — only the open menu is
   * visible at click time, which makes this unambiguous.
   */
  private async readSelectOptions(id: string): Promise<string[]> {
    const page = this.getPage();
    try {
      const shellXPath = `//div[contains(@class,"select-shell")][.//input[@id="${id}"]]`;
      const control = page
        .locator(`${shellXPath}//div[contains(@class,"select__control")]`)
        .first();
      if (!(await control.isVisible().catch(() => false))) {
        return [];
      }
      await this.closeMenu();
      await randomSleep(150, 300);
      await control.click();
      await randomSleep(300, 600);
      return await this.readVisibleOptionTexts();
    } catch (err: any) {
      console.warn(`[GreenhouseAdapter] readSelectOptions failed for #${id}:`, err?.message || err);
      return [];
    }
  }

  /** Option texts of whatever menu is currently open (visible only). */
  private async readVisibleOptionTexts(): Promise<string[]> {
    const page = this.getPage();
    try {
      return await page.evaluate(() => {
        const out: string[] = [];
        const seen = new Set<string>();
        for (const el of Array.from(document.querySelectorAll('div[role="option"]'))) {
          const node = el as HTMLElement;
          if (node.offsetParent === null && node.getBoundingClientRect().height === 0) {
            continue; // hidden or detached; only the open menu is visible
          }
          const text = (node.textContent || "").replace(/\s+/g, " ").trim();
          if (text && !seen.has(text)) {
            seen.add(text);
            out.push(text);
          }
        }
        return out;
      });
    } catch {
      return [];
    }
  }

  /** Read the currently displayed value of a react-select field. */
  private async readSelectValue(id: string): Promise<string> {
    const page = this.getPage();
    try {
      return await page.evaluate((inputId) => {
        const input = document.getElementById(inputId) as HTMLInputElement | null;
        if (!input) return "";
        const shell = input.closest('[class*="select-shell"]');
        if (shell) {
          const multi = Array.from(shell.querySelectorAll('div[class*="select__multi-value"]'));
          if (multi.length > 0) {
            return multi
              .map((m) => (m.textContent || "").replace(/\s+/g, " ").trim())
              .filter(Boolean)
              .join(", ");
          }
          const sv = shell.querySelector('div[class*="select__single-value"]');
          if (sv?.textContent) {
            return (sv.textContent || "").replace(/\s+/g, " ").trim();
          }
        }
        return (input.value || "").trim();
      }, id);
    } catch {
      return "";
    }
  }

  /** Read the current value of a plain text input. */
  private async readInputValue(id: string): Promise<string> {
    const page = this.getPage();
    try {
      return await page.evaluate((inputId) => {
        const input = document.getElementById(inputId) as HTMLInputElement | null;
        return input ? (input.value || "").trim() : "";
      }, id);
    } catch {
      return "";
    }
  }

  /** Read the checked option text of a radio/checkbox group. */
  private async readGroupValue(name: string): Promise<string> {
    const page = this.getPage();
    try {
      return await page.evaluate((groupName) => {
        const checked = document.querySelector(
          `input[type="radio"][name="${groupName}"]:checked, input[type="checkbox"][name="${groupName}"]:checked`
        ) as HTMLInputElement | null;
        if (!checked) return "";
        const label = checked.closest("label");
        return (label
          ? label.textContent || ""
          : checked.getAttribute("aria-label") || checked.value || ""
        )
          .replace(/\s+/g, " ")
          .trim();
      }, name);
    } catch {
      return "";
    }
  }

  /** Committed value of a field after filling, for verification/audit. */
  private async readFieldValue(field: FormField): Promise<string> {
    if (field.kind === "radio" || field.kind === "checkbox") {
      return this.readGroupValue(field.optionTargets[0]?.name || field.name || field.id);
    }
    if (field.kind === "select" || field.kind === "multi") {
      // Location autocompletes may commit as free text in the input rather
      // than a selected option, so fall back to the raw input value.
      return (
        (await this.readSelectValue(field.id)) ||
        (await this.readInputValue(field.id))
      );
    }
    return this.readInputValue(field.id);
  }

  /**
   * Click the radio/checkbox option matching ``answer``. Exact matches on the
   * collected option texts (via input name+value) are preferred; a text-based
   * label match inside the application form is the last resort.
   */
  private async clickGroupOption(field: FormField, answer: string): Promise<boolean> {
    const page = this.getPage();
    try {
      const picks = selectCandidates(answer);
      const targets = field.optionTargets;
      for (const cand of picks) {
        const nc = normalizeOptionText(cand);
        const target =
          targets.find((t) => normalizeOptionText(t.text) === nc) ||
          targets.find(
            (t) =>
              normalizeOptionText(t.text).includes(nc) ||
              nc.includes(normalizeOptionText(t.text))
          );
        if (!target) continue;
        const type = field.kind === "checkbox" ? "checkbox" : "radio";
        const nameAttr = cssEscape(target.name || field.name || field.id);
        const valueAttr = target.value ? `[value="${cssEscape(target.value)}"]` : "";
        const base = `input[type="${type}"]${valueAttr}[name="${nameAttr}"]`;
        const label = page.locator(`label:has(${base})`).first();
        if (await label.isVisible().catch(() => false)) {
          await label.click();
          return true;
        }
        const input = page.locator(base).first();
        if (await input.isVisible().catch(() => false)) {
          await (input as any).check({ force: true });
          return true;
        }
      }
      // Fallback: text-based label matching within the form.
      const formLabel = page
        .locator(
          `#application-form label:has(input[type='radio'], input[type='checkbox']):has-text("${cssEscape(answer)}")`
        )
        .first();
      if (await formLabel.isVisible().catch(() => false)) {
        await formLabel.click();
        return true;
      }
      return false;
    } catch {
      return false;
    }
  }

  /** Click every checkbox option matching a comma-separated answer. */
  private async clickGroupMulti(field: FormField, answer: string): Promise<boolean> {
    let clicked = 0;
    for (const pick of answer.split(",").map((p) => p.trim()).filter(Boolean)) {
      if (await this.clickGroupOption({ ...field, kind: "checkbox" }, pick)) {
        clicked += 1;
      }
    }
    return clicked > 0;
  }

  /**
   * Fill a typed field using the kind-aware dispatcher. Select/multi go through
   * the react-select machinery; when that fails and the field has radio/checkbox
   * targets (mis-detected kind), fall back to group clicking.
   */
  private async fillByKind(
    field: FormField,
    answer: string,
    optionTexts?: string[]
  ): Promise<boolean> {
    if (field.kind === "select") {
      const ok = await this.fillQuestionSelect(field.id, field.label, answer, optionTexts ?? []);
      if (ok) return true;
      if (field.optionTargets.length) {
        return this.clickGroupOption({ ...field, kind: "radio" }, answer);
      }
      // Native <select> fallback.
      const page = this.getPage();
      const sel = page.locator(`#${field.id} select, select#${field.id}`).first();
      if (await sel.isVisible().catch(() => false)) {
        const picked = chooseOption(selectCandidates(answer), optionTexts ?? []);
        if (picked) {
          await (sel as any).selectOption({ label: picked });
          return true;
        }
      }
      return false;
    }
    if (field.kind === "multi") {
      const ok = await this.fillQuestionMulti(field.id, field.label, answer, optionTexts ?? []);
      if (ok) return true;
      if (field.optionTargets.length) {
        return this.clickGroupMulti({ ...field, kind: "checkbox" }, answer);
      }
      return false;
    }
    if (field.kind === "radio") {
      return this.clickGroupOption(field, answer);
    }
    if (field.kind === "checkbox") {
      return this.clickGroupMulti(field, answer);
    }
    await this.fillQuestionText(field.id, answer);
    return !!(await this.readInputValue(field.id));
  }

  /**
   * Pick the best location suggestion for a free-form answer. Exact matches
   * win; otherwise the first ranked suggestion that shares the answer's
   * leading token(s) is chosen (e.g. "Bhopal, India" -> "Bhopal, Madhya
   * Pradesh, India").
   */
  private pickLocationOption(answer: string, opts: string[]): string | null {
    if (!opts.length) return null;
    const exact = opts.find((o) => normalizeOptionText(o) === normalizeOptionText(answer));
    if (exact) return exact;
    const tokens = answer
      .toLowerCase()
      .split(/[\s,]+/)
      .filter((t) => t.length > 2);
    for (const tok of tokens) {
      const start = opts.find((o) => normalizeOptionText(o).startsWith(tok));
      if (start) return start;
      const contains = opts.find((o) => normalizeOptionText(o).includes(tok));
      if (contains) return contains;
    }
    return opts[0];
  }

  /**
   * Fill a location autocomplete (react-select backed by an async geocoder):
   * open the menu, type the answer, poll for suggestions, click the best match.
   * When the full answer yields no suggestions (geocoders often need a city
   * token), retry with a shortened query. Falls back to the typed free text
   * (never cleared) when nothing can be selected.
   */
  private async fillLocation(id: string, answer: string): Promise<boolean> {
    const page = this.getPage();
    try {
      const input = page.locator(`#${id}`).first();
      if (!(await input.isVisible().catch(() => false))) return false;
      await this.closeMenu();
      await randomSleep(150, 300);
      await input.click();
      await randomSleep(200, 400);
      await input.fill(answer);

      const poll = async (): Promise<string[]> => {
        let opts: string[] = [];
        for (let i = 0; i < 6; i++) {
          await randomSleep(900, 1200);
          opts = await this.readVisibleOptionTexts();
          if (opts.length) break;
        }
        return opts;
      };

      let opts = await poll();
      if (!opts.length) {
        // Geocoders often return nothing for "City, Country"; retry with the
        // leading city token ("Bhopal, India" -> "Bhopal").
        const shortQuery = answer
          .split(/[\s,]+/)
          .filter((t) => t && t.length > 1)[0];
        if (shortQuery && shortQuery !== answer.trim()) {
          await input.fill(shortQuery);
          opts = await poll();
        }
      }
      if (opts.length) {
        const picked = this.pickLocationOption(answer, opts);
        if (picked && (await this.clickVisibleOption(picked))) {
          await this.closeMenu();
          await randomSleep(300, 500);
          console.log(`[GreenhouseAdapter] Picked location suggestion "${picked}" for #${id}`);
          return true;
        }
      }
      // Free-text fallback: the typed value stays committed in the input (never
      // blur/escape, which would clear it).
      const committed = (await this.readSelectValue(id)) || (await this.readInputValue(id));
      console.log(
        `[GreenhouseAdapter] Location #${id} free-text committed: "${committed}"`
      );
      return !!committed;
    } catch (err: any) {
      console.warn(`[GreenhouseAdapter] fillLocation failed for #${id}:`, err?.message || err);
      return false;
    }
  }

  private async consentToPrivacy(): Promise<void> {
    const page = this.getPage();
    try {
      // Deterministic: click Greenhouse consent checkboxes by label text.
      for (const hint of ["agree to allow", "retain my data", "consent"]) {
        const box = page
          .locator(
            `//div[contains(@class,"field-wrapper")][contains(., "${hint}")]//input[@type="checkbox"]`
          )
          .first();
        if (await box.isVisible().catch(() => false)) {
          if (!(await box.isChecked().catch(() => false))) {
            await box.click();
          }
          await randomSleep(200, 400);
        }
      }
    } catch (err: any) {
      console.warn("[GreenhouseAdapter] Consent checkbox deterministic fill failed:", err?.message || err);
      // Best-effort LLM fallback.
      await this.safeAct("check all the 'I agree to allow' consent and data retention checkboxes");
      await randomSleep(300, 600);
    }
  }

  /**
   * Locate the cover letter field across board flavors. Legacy boards expose a
   * visible textarea; the Remix board renders a file upload whose "Enter
   * manually" button (exact data-testid "cover_letter-text") reveals the
   * textarea. Returns null when the form has no cover letter field.
   */
  private async findCoverLetterField(): Promise<any> {
    const page = this.getPage();
    const textareaSel =
      "textarea#cover_letter_text, textarea[name*='cover_letter'], textarea[aria-label*='cover letter' i], #job_application_cover_letter";
    let ta = page.locator(textareaSel).first();
    if (await ta.isVisible().catch(() => false)) return ta;
    // Exact testid, no :has-text, no comma-list: Stagehand's locator→XPath
    // conversion can't handle those, which silently made isVisible() return
    // false before and skipped the whole field.
    const manual = page.locator('button[data-testid="cover_letter-text"]').first();
    if (await manual.isVisible().catch(() => false)) {
      await manual.click();
      // The textarea is rendered by React only after the toggle; poll for it.
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

    console.log(`[GreenhouseAdapter] Navigating to ${url}...`);
    const page = this.getPage();
    await page.goto(url);
    await randomSleep(300, 600);

    await this.ensureApplicationForm();

    console.log("[GreenhouseAdapter] Filling deterministic profile fields...");

    await this.fillField(
      '#first_name',
      profile.firstName,
      "Type %firstName% into the First Name input field",
      "firstName"
    );
    await this.fillField(
      '#last_name',
      profile.lastName,
      "Type %lastName% into the Last Name input field",
      "lastName"
    );
    await this.fillField(
      '#email',
      profile.email,
      "Type %email% into the Email input field",
      "email"
    );
    await this.fillField(
      '#phone',
      profile.phone,
      "Type %phone% into the Phone input field",
      "phone"
    );

    // Resume Upload Handling
    if (profile.resumePath && fs.existsSync(profile.resumePath)) {
      const baseName = profile.resumePath.split(/[\\/]/).pop() || "";
      console.log(`[GreenhouseAdapter] Uploading resume from ${profile.resumePath}...`);
      const fileInput = page.locator('input#resume[type="file"], input[type="file"]').first();
      if (await fileInput.count() > 0) {
        for (let attempt = 0; attempt < 2; attempt++) {
          await fileInput.setInputFiles(profile.resumePath);
          await randomSleep(1200, 2000);
          // The board consumes the file input on upload (it disappears and the
          // filename is shown in the upload area) — verify that, retry once.
          const registered = await page.evaluate((fileName) => {
            const stillThere = !!document.querySelector('input#resume[type="file"]');
            const area = document.querySelector(
              ".file-upload, [class*='file-upload'], [id^='upload-label-']"
            );
            const text = area ? (area.textContent || "") : "";
            return !stillThere || text.includes(fileName);
          }, baseName).catch(() => false);
          if (registered) {
            console.log("[GreenhouseAdapter] Resume uploaded and registered.");
            break;
          }
          console.warn(`[GreenhouseAdapter] Resume upload not confirmed (attempt ${attempt + 1}); retrying...`);
        }
      }
    }

    // Per-field screener walk driven by the merged inventory (board JSON model
    // ∪ DOM enumeration). Every rendered question is resolved one at a time via
    // the answer_question RPC (profile/KB first, Telegram with options for
    // unknowns, learned into the KB on every answer). Identity fields filled
    // deterministically above (or via PROFILE_FILLS) are not re-asked.
    if (rpc) {
      // Send the extracted job context first so open-ended answers are
      // personalized to the role and country-scoped questions resolve against
      // the JD's country.
      const jobCtx = await this.readJobContext();
      await rpc("job_context", jobCtx);
      console.log(
        `[GreenhouseAdapter] Job context: ${jobCtx.title || "?"} @ ${jobCtx.company || "?"}` +
          (jobCtx.location ? ` (${jobCtx.location})` : "")
      );

      const jsonModel = await this.fetchQuestionsModel();

      const filled: string[] = [];
      const blanked: Array<{ label: string; reason: string }> = [];
      // Iterative re-scan: conditional questions (e.g. Race, which renders
      // only after "Are you Hispanic/Latino?" is answered) appear in the DOM
      // only after an earlier interaction. Rescan the form each pass, process
      // only fields not seen before, and converge when nothing new appears.
      const processedKeys = new Set<string>();
      const MAX_WALK_PASSES = 30;

      for (let pass = 0; pass < MAX_WALK_PASSES; pass++) {
        const domFields = await this.collectQuestions();
        const inventory = this.mergeInventory(jsonModel, domFields);
        const newFields = unprocessedFields(inventory, processedKeys);
        if (pass === 0) {
          console.log(
            `[GreenhouseAdapter] Question inventory: ${inventory.length} ` +
              `(json: ${jsonModel?.length ?? 0}, dom: ${domFields.length})`
          );
        }
        if (newFields.length === 0) {
          if (pass > 0) {
            console.log(`[GreenhouseAdapter] Walk converged after ${pass + 1} pass(es).`);
          }
          break;
        }
        console.log(
          `[GreenhouseAdapter] Walk pass ${pass + 1}: ${newFields.length} new question(s).`
        );

        for (const field of newFields) {
          // Mark BEFORE filling so a re-scan can never re-ask or re-fill.
          processedKeys.add(fieldKey(field));

          const key = normalizeOptionText(field.label);
          if (PRE_FILLED_LABELS.has(key)) continue;

        // Location autocompletes resolve from the profile's current location
        // (persona) and fill through the async autocomplete path.
        const isLocation = /location|city|address/.test(key);
        if (isLocation && (profile as any)?.location) {
          const ans = String((profile as any).location);
          let ok = await this.fillLocation(field.id, ans);
          if (!ok) {
            await this.fillLocation(field.id, ans);
            ok = await this.readSelectValue(field.id).then((v) => !!v);
          }
          if (!ok) {
            throw new Error(
              `VERIFICATION_FAILED: field "${escapePromptValue(field.label)}" ` +
                `(#${field.id}) did not commit location "${escapePromptValue(ans)}"`
            );
          }
          console.log(
            `[GreenhouseAdapter] Location filled for "${escapePromptValue(field.label)}": "${escapePromptValue(ans)}"`
          );
          filled.push(field.label);
          await randomSleep(150, 300);
          continue;
        }

        // Deterministic profile-driven fills (preferred name, linkedin, github, website).
        const profileKey = PROFILE_FILLS[key];
        if (profileKey) {
          const pv = (profile as any)?.[profileKey];
          if (pv) {
            const ok = await this.fillByKind(field, String(pv));
            if (!ok) {
              throw new Error(
                `VERIFICATION_FAILED: field "${escapePromptValue(field.label)}" ` +
                  `did not commit profile value "${escapePromptValue(String(pv))}"`
              );
            }
            filled.push(field.label);
            await randomSleep(150, 300);
            continue;
          }
          // No profile value: fall through to ask.
        }

        let optionTexts = field.options.slice();
        if ((field.kind === "select" || field.kind === "multi") && optionTexts.length === 0) {
          optionTexts = await this.readSelectOptions(field.id);
          if (optionTexts.length === 0) {
            console.warn(
              `[GreenhouseAdapter] Could not read options for #${field.id} ` +
                `("${escapePromptValue(field.label)}"); resolving without them.`
            );
          }
        }

        const rpcKind = isLocation
          ? "text"
          : field.kind === "radio"
            ? "select"
            : field.kind === "checkbox"
              ? "multi"
              : field.kind;

        let result: any;
        try {
          result = await rpc("answer_question", {
            question: field.label,
            kind: rpcKind,
            options: optionTexts,
          });
        } catch (rpcErr: any) {
          // RPC failures (Telegram unconfigured, overnight deferral) must
          // abort loudly — filling a form around an unanswered personal
          // question is worse than no fill.
          console.error(
            `[GreenhouseAdapter] RPC answer_question failed for "${escapePromptValue(field.label)}":`,
            rpcErr?.message || rpcErr
          );
          throw rpcErr;
        }

        const answer: string = (result?.answer ?? "").toString().trim();
        if (!answer) {
          // User dismissed and no decline option existed; record for the audit.
          blanked.push({ label: field.label, reason: `declined (source ${result?.source ?? "unknown"})` });
          continue;
        }

        let ok: boolean;
        if (isLocation) {
          ok = await this.fillLocation(field.id, answer);
          if (!ok) {
            await this.fillLocation(field.id, answer);
            ok = await this.readSelectValue(field.id).then((v) => !!v);
          }
        } else {
          ok = await this.fillByKind(field, answer, optionTexts);
          if (!ok) {
            // One retry, then verify the committed value.
            ok = await this.fillByKind(field, answer, optionTexts);
          }
          const committed = await this.readFieldValue(field);
          ok = ok && !!committed;
        }

        if (!ok) {
          throw new Error(
            `VERIFICATION_FAILED: field "${escapePromptValue(field.label)}" ` +
              `(#${field.id}) did not commit answer "${escapePromptValue(answer)}"`
          );
        }
        filled.push(field.label);
        await randomSleep(150, 300);
        }

        // Let async conditionals (revealed questions, geocoder results) settle
        // before the next re-scan.
        await randomSleep(900, 1400);
      }

      // Final inventory for the zero-blank audit.
      const finalDom = await this.collectQuestions();
      const inventory = this.mergeInventory(jsonModel, finalDom);

      // Zero-blank audit: no required question may be left empty, and every
      // blank optional question must have a recorded reason.
      for (const field of inventory) {
        const key = normalizeOptionText(field.label);
        if (PRE_FILLED_LABELS.has(key)) continue;
        const value = await this.readFieldValue(field);
        if (value) continue;
        const reason = blanked.find(
          (b) => normalizeOptionText(b.label) === key
        )?.reason;
        if (field.required) {
          throw new Error(
            `VERIFICATION_FAILED: required field "${escapePromptValue(field.label)}" ` +
              `is blank after the walk${reason ? ` (${reason})` : ""}`
          );
        }
        if (!reason) {
          blanked.push({ label: field.label, reason: "blank after walk (no answer committed)" });
        }
      }

      console.log(
        `[GreenhouseAdapter] Screener walk complete. Filled: ${filled.length}, blank (declined/unknown): ${blanked.length}.`
      );
      for (const b of blanked) {
        console.warn(`[GreenhouseAdapter]   blank: ${escapePromptValue(b.label)} (${b.reason})`);
      }

      // Cover letter: LLM-generated, personalized to the job description.
      // Only generated when the form actually has the field (upload toggle
      // revealed), so an LLM call is never wasted on forms without one.
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
            console.log("[GreenhouseAdapter] Cover letter filled (LLM-generated, JD-personalized).");
          } else {
            console.warn("[GreenhouseAdapter] Cover letter did not commit; left blank.");
          }
        } else {
          console.log("[GreenhouseAdapter] Cover letter skipped: LLM had nothing to ground it on.");
        }
      } else {
        console.log("[GreenhouseAdapter] No cover letter field on this form; skipping generation.");
      }
    }

    // Consent checkboxes
    await this.consentToPrivacy();

    console.log("[GreenhouseAdapter] Form filling completed.");
  }

  async submit(): Promise<void> {
    console.log("[GreenhouseAdapter] Submitting application form...");
    await this.stagehand.act("Click the Submit Application button");
    await randomSleep(500, 1000);
  }
}
