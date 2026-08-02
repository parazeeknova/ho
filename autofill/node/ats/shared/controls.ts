import { Stagehand } from "@browserbasehq/stagehand";
import { randomSleep } from "../../utils/evasion.js";
import { FormField } from "./model.js";
import {
  chooseOption,
  cssEscape,
  escapePromptValue,
  normalizeOptionText,
  optionExactXPath,
  selectCandidates,
  pickLocationOption,
} from "./matching.js";

/**
 * Browser-interaction primitives shared by all ATS adapters: deterministic
 * locator fills, react-select/rale menu handling, radio/checkbox group
 * clicking, async location autocompletes, and committed-value reads. Adapters
 * compose these rather than re-implement the Playwright/Stagehand plumbing.
 */
export class FormControls {
  protected shellClass: string;
  readonly tagName: string;

  constructor(
    protected stagehand: Stagehand,
    options?: {
      /** Log tag prefix used in advisory messages (e.g. the adapter name). */
      tagName?: string;
      /** CSS class token used to find react-select shells (default "select-shell"). */
      shellClass?: string;
    }
  ) {
    this.tagName = options?.tagName ?? "FormControls";
    this.shellClass = options?.shellClass ?? "select-shell";
  }

  /** The active page (Stagehand may be configured with multiple tabs). */
  getPage(): any {
    return this.stagehand.context.pages()[0];
  }

  /**
   * Run an AI-powered act() call but never let a single failure abort the
   * whole form fill (e.g. a field that doesn't exist on this job's form).
   * Values are passed through Stagehand's `variables` so arbitrary profile
   * text is never parsed as a prompt.
   */
  async safeAct(instruction: string, variables?: Record<string, string>): Promise<void> {
    try {
      await this.stagehand.act(instruction, variables ? { variables } : {});
    } catch (err: any) {
      console.warn(`[${this.tagName}] act() failed (continuing): ${err?.message || err}`);
    }
  }

  /**
   * Prefer a deterministic locator fill; fall back to an AI-driven act() using
   * %variables% so arbitrary profile text (quotes, newlines) is handled safely.
   */
  async fillField(
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

  /** Close an open react-select menu without depending on keyboard typing. */
  async closeMenu(): Promise<void> {
    try {
      await this.getPage().keyboard?.press("Escape");
    } catch {
      // Menu may already be closed; harmless.
    }
  }

  /** XPath locating a react-select shell that contains the input with ``id``. */
  shellXPathFor(id: string): string {
    return `//div[contains(@class,"${this.shellClass}")][.//input[@id="${id}"]]`;
  }

  /** XPath of the clickable control inside a field's select shell. */
  controlXPathFor(id: string): string {
    return `${this.shellXPathFor(id)}//div[contains(@class,"select__control")]`;
  }

  /**
   * Ensure the react-select menu for a field is open. Escape first so a stale
   * menu from a previous field can't make the control click *toggle* the menu
   * shut; then click the control and verify with Playwright's strict
   * visibility (hidden stale options must not count as "open").
   */
  async ensureMenuOpen(id: string): Promise<boolean> {
    const page = this.getPage();
    const control = page.locator(this.controlXPathFor(id)).first();
    if (!(await control.isVisible().catch(() => false))) return false;
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
  async hasVisibleOption(): Promise<boolean> {
    const page = this.getPage();
    const options = page.locator('div[role="option"]');
    const count = await options.count().catch(() => 0);
    for (let i = 0; i < count; i++) {
      if (await options.nth(i).isVisible().catch(() => false)) return true;
    }
    return false;
  }

  /** Click the visible option whose text exactly matches ``picked``. */
  async clickVisibleOption(picked: string): Promise<boolean> {
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
        if (!(await option.isVisible().catch(() => false))) continue;
        await option.click();
        return true;
      }
      return false;
    } catch {
      return false;
    }
  }

  /** Any ``[role=option]`` text in the DOM, regardless of menu visibility. */
  async readAnyOptionTexts(): Promise<string[]> {
    const page = this.getPage();
    try {
      return await page.evaluate(() => {
        const out: string[] = [];
        const seen = new Set<string>();
        for (const el of Array.from(document.querySelectorAll('div[role="option"]'))) {
          const text = ((el as HTMLElement).textContent || "")
            .replace(/\s+/g, " ")
            .trim();
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

  /** Option texts of whatever menu is currently open (visible only). */
  async readVisibleOptionTexts(): Promise<string[]> {
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

  /**
   * Read the real option texts of a dropdown by opening its menu. Options are
   * collected document-wide and filtered to visible ones — only the open menu
   * is visible at click time, which makes this unambiguous.
   *
   * Async react-selects (geocoders, country pickers) load options only after a
   * keystroke, so when the initial menu is empty we type "a" to trigger the
   * loader and re-read, retrying up to 3 times. As a last resort, any
   * ``[role=option]`` present in the DOM is collected even if the menu looks
   * closed.
   */
  async readSelectOptions(id: string): Promise<string[]> {
    const page = this.getPage();
    try {
      const control = page.locator(this.controlXPathFor(id)).first();
      const input = page.locator(`#${id}`).first();
      if (!(await control.isVisible().catch(() => false))) return [];
      await this.closeMenu();
      await randomSleep(150, 300);
      await control.click();
      await randomSleep(300, 600);
      let opts = await this.readVisibleOptionTexts();
      if (opts.length === 0 && (await input.isVisible().catch(() => false))) {
        // Trigger async loaders by typing a probe character.
        for (let attempt = 0; attempt < 3 && opts.length === 0; attempt++) {
          await input.fill("a");
          await randomSleep(1200, 1800);
          opts = await this.readVisibleOptionTexts();
        }
        if (opts.length === 0) {
          await input.fill("");
        }
      }
      if (opts.length === 0) {
        // Some boards render options without a visible menu (or the menu
        // portal hasn't focused); fall back to any option in the DOM.
        opts = await this.readAnyOptionTexts();
      }
      await this.closeMenu();
      return opts;
    } catch (err: any) {
      console.warn(`[${this.tagName}] readSelectOptions failed for #${id}:`, err?.message || err);
      return [];
    }
  }

  /** Read the currently displayed value of a react-select field. */
  async readSelectValue(id: string): Promise<string> {
    const page = this.getPage();
    try {
      return await page.evaluate((inputId: string) => {
        const input = document.getElementById(inputId) as HTMLInputElement | null;
        if (!input) return "";
        const shell =
          input.closest('[class*="select-shell"]') ||
          input.closest('[class*="select__"]');
        if (shell) {
          const multi = Array.from(
            shell.querySelectorAll('div[class*="select__multi-value"]')
          );
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
  async readInputValue(id: string): Promise<string> {
    const page = this.getPage();
    try {
      return await page.evaluate((inputId: string) => {
        const input = document.getElementById(inputId) as HTMLInputElement | null;
        return input ? (input.value || "").trim() : "";
      }, id);
    } catch {
      return "";
    }
  }

  /** Read the checked radio/checkbox option text in a group. */
  async readGroupValue(name: string): Promise<string> {
    const page = this.getPage();
    try {
      return await page.evaluate((groupName: string) => {
        const checked = document.querySelector(
          `input[type="radio"][name="${groupName}"]:checked, input[type="checkbox"][name="${groupName}"]:checked`
        ) as HTMLInputElement | null;
        if (!checked) return "";
        const wrapLabel = checked.closest("label");
        const forLabel = checked.id
          ? document.querySelector(`label[for="${checked.id}"]`)
          : null;
        const label = wrapLabel || forLabel;
        return (label
          ? (label.textContent || "")
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
  async readFieldValue(field: FormField): Promise<string> {
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
   * label match inside the application form is the last resort. Handles all
   * three label renderings: wrapping ``label:has(input)``, a sibling
   * ``label[for=<input-id>]``, and a bare visible input. The checked state is
   * verified after every attempt.
   */
  async clickGroupOption(
    field: FormField,
    answer: string,
    formSelector = "#application-form"
  ): Promise<boolean> {
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
        const input = page.locator(base).first();

        // 1) Wrapping label (label:has(input)).
        const wrapLabel = page.locator(`label:has(${base})`).first();
        if (await wrapLabel.isVisible().catch(() => false)) {
          await wrapLabel.click();
          if (await input.isChecked().catch(() => false)) return true;
        }
        // 2) Sibling label[for=<input id>] — label is NOT an ancestor.
        if (await input.isVisible().catch(() => false)) {
          const inputId = await page
            .evaluate((sel: string) => document.querySelector(sel)?.id || "", base)
            .catch(() => "");
          if (inputId) {
            const forLabel = page.locator(`label[for="${cssEscape(inputId)}"]`).first();
            if (await forLabel.isVisible().catch(() => false)) {
              await forLabel.click();
              if (await input.isChecked().catch(() => false)) return true;
            }
          }
          // 3) Bare input: force-check it (never rely on a wrapping label).
          await (input as any).check({ force: true }).catch(() => {});
          if (await input.isChecked().catch(() => false)) return true;
        }
      }
      // Fallback: text-based label matching within the form.
      const formLabel = page
        .locator(
          `${formSelector} label:has(input[type='radio'], input[type='checkbox']):has-text("${cssEscape(answer)}")`
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
  async clickGroupMulti(field: FormField, answer: string): Promise<boolean> {
    let clicked = 0;
    for (const pick of answer.split(",").map((p) => p.trim()).filter(Boolean)) {
      if (await this.clickGroupOption({ ...field, kind: "checkbox" }, pick)) {
        clicked += 1;
      }
    }
    return clicked > 0;
  }

  /**
   * Fill a plain text field. Refuses to type into a combobox/select-shell
   * disguised as a text field, and silently skips file inputs (resume handled
   * by the adapter's dedicated upload path).
   */
  async fillTextById(id: string, answer: string): Promise<void> {
    const page = this.getPage();
    const loc = page.locator(`#${id}`).first();
    if (!(await loc.isVisible().catch(() => false))) return;
    const type = await page
      .evaluate((inputId: string) => {
        const el = document.getElementById(inputId) as HTMLInputElement | null;
        return el?.type ?? null;
      }, id)
      .catch(() => null);
    if (type === "file") return; // file inputs crash fill(); resume handled elsewhere
    // Never type into a combobox/select-shell disguised as a text field: a
    // dropdown is answered by selecting an option, not by typing into it.
    const isCombobox = await page
      .evaluate((inputId: string) => {
        const el = document.getElementById(inputId) as HTMLInputElement | null;
        return !!(
          el &&
          (el.getAttribute("role") === "combobox" ||
            el.closest('[class*="select-shell"]'))
        );
      }, id)
      .catch(() => false);
    if (isCombobox) {
      console.warn(
        `[${this.tagName}] Refusing to type into combobox #${id}; selecting an option instead.`
      );
      return;
    }
    await loc.fill(answer);
    await randomSleep(200, 500);
  }

  /**
   * Fill a react-select question dropdown. The option menu renders portaled
   * outside the field's own shell, so options are located page-wide and
   * filtered to visible ones (only one menu can be open at a time, so this is
   * unambiguous). ``optionTexts`` is the option list read by
   * ``readSelectOptions``; the picked option must come from that list so the
   * click is an exact-text match, never a substring guess.
   */
  async fillSelect(id: string, answer: string, optionTexts: string[]): Promise<boolean> {
    const page = this.getPage();
    try {
      if (!(await this.ensureMenuOpen(id))) return false;

      const picked = chooseOption(selectCandidates(answer), optionTexts);
      if (!picked) {
        console.warn(
          `[${this.tagName}] No matching option for #${id} ` +
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
          `[${this.tagName}] Option "${picked}" not visible on first try for #${id}; reopening...`
        );
        await this.closeMenu();
        await randomSleep(150, 300);
        if (await this.ensureMenuOpen(id)) {
          clicked = await this.clickVisibleOption(picked);
        }
      }
      if (!clicked) {
        console.warn(
          `[${this.tagName}] Picked option "${picked}" not visible for #${id} ` +
            `(answer "${escapePromptValue(answer)}")`
        );
        await this.closeMenu();
        return false;
      }
      await randomSleep(300, 600);
      console.log(`[${this.tagName}] Selected option "${picked}" for #${id}`);
      return true;
    } catch (err: any) {
      console.warn(`[${this.tagName}] fillSelect failed for #${id}:`, err?.message || err);
      return false;
    }
  }

  /**
   * Fill a react-select multi-select: one exact match per comma-separated
   * pick, clicking each while the menu stays open.
   */
  async fillMulti(id: string, answer: string, optionTexts: string[]): Promise<boolean> {
    const page = this.getPage();
    try {
      if (!(await this.ensureMenuOpen(id))) return false;

      const picks = answer
        .split(",")
        .map((p) => p.trim())
        .filter(Boolean);
      let clicked = 0;
      for (const pick of picks) {
        const picked = chooseOption(selectCandidates(pick), optionTexts);
        if (!picked) {
          console.warn(
            `[${this.tagName}] No matching multi option for #${id} ` +
              `(answer "${escapePromptValue(pick)}")`
          );
          continue;
        }
        if (await this.clickVisibleOption(picked)) {
          clicked += 1;
          await randomSleep(200, 400);
        } else {
          console.warn(`[${this.tagName}] Multi option "${picked}" not visible for #${id}`);
        }
        // Multi-select menus normally stay open; if a pick closed it, reopen.
        await this.ensureMenuOpen(id);
      }
      await this.closeMenu();
      console.log(`[${this.tagName}] Selected ${clicked} option(s) for #${id}`);
      return clicked > 0;
    } catch (err: any) {
      console.warn(`[${this.tagName}] fillMulti failed for #${id}:`, err?.message || err);
      return false;
    }
  }

  /**
   * Fill an async location autocomplete (react-select backed by a geocoder).
   * Typing here is only a trigger to reveal suggestions; the committed value
   * MUST be a selected option — raw typed text is never accepted as an answer
   * (a dropdown is never filled by typing into it). When no suggestion can be
   * selected, returns false so the walker blanks the field with a reason.
   */
  async fillAsyncAutocomplete(id: string, answer: string): Promise<boolean> {
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
        const picked = pickLocationOption(answer, opts);
        if (picked && (await this.clickVisibleOption(picked))) {
          await this.closeMenu();
          await randomSleep(300, 500);
          // Verify a real option was selected — never the raw typed text.
          const committed = await this.readSelectValue(id);
          if (committed) {
            console.log(
              `[${this.tagName}] Picked location suggestion "${picked}" for #${id}`
            );
            return true;
          }
        }
      }
      // No selectable suggestion: blank with a reason, never commit free text.
      await this.closeMenu();
      console.warn(
        `[${this.tagName}] No selectable location suggestion for #${id} ` +
          `(answer "${escapePromptValue(answer)}"); leaving blank.`
      );
      return false;
    } catch (err: any) {
      console.warn(`[${this.tagName}] fillAsyncAutocomplete failed for #${id}:`, err?.message || err);
      return false;
    }
  }

  /**
   * Fill a typed field using the kind-aware dispatcher. Select/multi go
   * through the react-select machinery; when that fails and the field has
   * radio/checkbox targets (mis-detected kind), fall back to group clicking.
   */
  async fillByKind(
    field: FormField,
    answer: string,
    optionTexts?: string[]
  ): Promise<boolean> {
    if (field.kind === "select") {
      const ok = await this.fillSelect(field.id, answer, optionTexts ?? []);
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
      const ok = await this.fillMulti(field.id, answer, optionTexts ?? []);
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
    await this.fillTextById(field.id, String(answer ?? ""));
    return !!(await this.readInputValue(field.id));
  }
}