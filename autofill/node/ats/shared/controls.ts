import { Stagehand, type Action } from "@browserbasehq/stagehand";
import { randomSleep } from "../../utils/evasion.js";
import { FormField } from "./model.js";
import {
  chooseOption,
  cssEscape,
  cssIdLocator,
  escapePromptValue,
  normalizeOptionText,
  optionExactXPath,
  selectCandidates,
  pickLocationOption,
  translateToDate,
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
  /** CSS selector for a dropdown's open option elements (default: react-select
   *  ``div[role="option"]``). Boards like Workday render options as
   *  ``li[role="option"]`` inside a ``ul`` — the subclass passes its own. */
  protected optionSelector: string;
  /** Element tag used by the XPath in ``clickVisibleOption`` (default "div").
   *  Set to "*" when options can be several tags (e.g. Workday li/div). */
  protected optionTag: string;

  constructor(
    protected stagehand: Stagehand,
    options?: {
      /** Log tag prefix used in advisory messages (e.g. the adapter name). */
      tagName?: string;
      /** CSS class token used to find react-select shells (default "select-shell"). */
      shellClass?: string;
      /** CSS selector for dropdown options (default "div[role=\"option\"]"). */
      optionSelector?: string;
      /** XPath tag for dropdown options (default "div"). */
      optionTag?: string;
    }
  ) {
    this.tagName = options?.tagName ?? "FormControls";
    this.shellClass = options?.shellClass ?? "select-shell";
    this.optionSelector = options?.optionSelector ?? 'div[role="option"]';
    this.optionTag = options?.optionTag ?? "div";
  }

  /** The active page (Stagehand may be configured with multiple tabs). */
  protected activePage: any = null;

  getPage(): any {
    return this.activePage ?? this.stagehand.context.pages()[0];
  }

  /** Adopt a specific page (e.g. a form opened in a new tab) as the active one. */
  adoptPage(page: any): void {
    this.activePage = page;
  }

  /** Poll the browser context for a page whose URL matches ``match`` and adopt
   *  it. Returns false on timeout — callers fall back to a DOM-based scan. */
  async focusPage(match: RegExp, timeoutMs = 20000): Promise<boolean> {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
      for (const p of this.stagehand.context.pages()) {
        try {
          if (match.test(p.url())) {
            this.activePage = p;
            return true;
          }
        } catch {
          // Page may have been closed mid-poll; skip it.
        }
      }
      await randomSleep(500, 900);
    }
    return false;
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
    const options = page.locator(this.optionSelector);
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
        `//${this.optionTag}[contains(@role,"option")][${optionExactXPath(picked)}]`
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
      return await page.evaluate((selector: string) => {
        const out: string[] = [];
        const seen = new Set<string>();
        for (const el of Array.from(document.querySelectorAll(selector))) {
          const text = ((el as HTMLElement).textContent || "")
            .replace(/\s+/g, " ")
            .trim();
          if (text && !seen.has(text)) {
            seen.add(text);
            out.push(text);
          }
        }
        return out;
      }, this.optionSelector);
    } catch {
      return [];
    }
  }

  /** Option texts of whatever menu is currently open (visible only). */
  async readVisibleOptionTexts(): Promise<string[]> {
    const page = this.getPage();
    try {
      return await page.evaluate((selector: string) => {
        const out: string[] = [];
        const seen = new Set<string>();
        for (const el of Array.from(document.querySelectorAll(selector))) {
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
      }, this.optionSelector);
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
      const input = page.locator(cssIdLocator(id)).first();
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

  /** Read the current value of a plain text input. Falls back to the control
   *  linked via `label[for="<id>"]` when the input itself has no id (Ashby
   *  react-datepicker shape). */
  async readInputValue(id: string): Promise<string> {
    const page = this.getPage();
    try {
      return await page.evaluate((inputId: string) => {
        const byId = document.getElementById(inputId) as HTMLInputElement | null;
        if (byId) return (byId.value || "").trim();
        const label = document.querySelector(`label[for="${inputId}"]`);
        const control = label
          ? (label as HTMLLabelElement).control
          : null;
        if (control) return ((control as HTMLInputElement).value || "").trim();
        // Last resort: the field's own container (label's `for` may dangle
        // when the input has no id — react-datepicker). Read its first control.
        // The scope itself may BE the control (generic adapter tags the input).
        const scope = document.querySelector(`[data-field-path="${inputId}"]`);
        const scoped = scope?.matches(
          'input[type="text"], input[type="date"], input[type="email"], input[type="tel"], input[type="url"], input:not([type]), textarea'
        )
          ? scope
          : scope?.querySelector(
              'input[type="text"], input[type="date"], input[type="email"], input[type="tel"], input[type="url"], input:not([type]), textarea'
            );
        return scoped ? ((scoped as HTMLInputElement).value || "").trim() : "";
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

  /**
   * Read the committed value of a radio/checkbox/button-toggle field by
   * inspecting the field's OWN scope — board-agnostic. Returns:
   *  - the checked input's label/aria text, or
   *  - the text of an active/checked toggle button (yes/no rows), or
   *  - the underlying input's value as a last resort.
   * Empty string when nothing is committed.
   */
  async readScopedGroupValue(field: FormField): Promise<string> {
    const page = this.getPage();
    try {
      return await page.evaluate((fid: string) => {
        const scope = document.querySelector(`[data-field-path="${fid}"]`);
        if (!scope) return "";
        const checked = scope.querySelector(
          'input[type="radio"]:checked, input[type="checkbox"]:checked'
        ) as HTMLInputElement | null;
        if (checked) {
          const wrapLabel = checked.closest("label");
          const forLabel = checked.id
            ? document.querySelector(`label[for="${checked.id}"]`)
            : null;
          const label = wrapLabel || forLabel;
          const txt = (label
            ? (label.textContent || "")
            : checked.getAttribute("aria-label") || ""
          )
            .replace(/\s+/g, " ")
            .trim();
          if (txt) return txt;
          return checked.value || "";
        }
        // Toggle-button rows: an active/checked button (aria-pressed or a
        // checked/active class) is the committed value.
        const activeBtn = Array.from(
          scope.querySelectorAll("button")
        ).find((b) => {
          const cls = (b.className || "").toString();
          return (
            b.getAttribute("aria-pressed") === "true" ||
            /(checked|active|selected|_checked_)/.test(cls)
          );
        });
        if (activeBtn) {
          return (activeBtn.textContent || "").replace(/\s+/g, " ").trim();
        }
        return "";
      }, field.id);
    } catch {
      return "";
    }
  }

  /** Committed value of a field after filling, for verification/audit. */
  async readFieldValue(field: FormField): Promise<string> {
    if (field.kind === "radio" || field.kind === "checkbox") {
      // Board-agnostic scope read first; fall back to the name-based read for
      // boards whose radios share a group name (Greenhouse/Lever).
      return (
        (await this.readScopedGroupValue(field)) ||
        this.readGroupValue(field.optionTargets[0]?.name || field.name || field.id)
      );
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
        // NOTE: do NOT add `[value="..."]` here. For an input without a value
        // attribute, `input.value` (the property, read by the walker) is the
        // browser default "on", but there is NO `value` attribute in the DOM —
        // so `[value="on"]` matches nothing. The name alone identifies the
        // radio/checkbox group member.
        //
        // Prefer the input's captured id: it uniquely identifies the option
        // (radio groups share one `name`, so a name-only selector always
        // resolves to the FIRST option and clicking it selects the WRONG one).
        //
        // When the input has no id (common for Greenhouse/Lever), disambiguate
        // the group by matching the option's OWN label text: walk every input
        // in the group, find the one whose wrapping/sibling label matches the
        // candidate, and use THAT input's id. This clicks the correct option
        // (e.g. "No" in a Yes/No group) instead of always the first.
        const nameSel = cssEscape(target.name || field.name || field.id);
        let optionId = target.id;
        if (!optionId) {
          optionId = await page
            .evaluate(
              (args: { name: string; type: string; want: string }) => {
                const inputs = Array.from(
                  document.querySelectorAll(
                    `input[type="${args.type}"][name="${args.name}"]`
                  )
                ) as HTMLInputElement[];
                const norm = (s: string) =>
                  (s || "").replace(/\s+/g, " ").trim().toLowerCase();
                for (const inp of inputs) {
                  const wrapLabel = inp.closest("label");
                  const forLabel = inp.id
                    ? document.querySelector(`label[for="${inp.id}"]`)
                    : null;
                  const label = wrapLabel || forLabel;
                  const txt = norm(
                    label
                      ? label.textContent || ""
                      : inp.getAttribute("aria-label") || ""
                  );
                  if (txt === args.want) {
                    return inp.id || "";
                  }
                }
                return "";
              },
              { name: nameSel, type, want: nc }
            )
            .catch(() => "");
        }
        const base = optionId
          ? `input[type="${type}"][id="${cssEscape(optionId)}"]`
          : `input[type="${type}"][name="${nameSel}"]`;
        const input = page.locator(base).first();

        // 1) Wrapping label (label:has(input)).
        const wrapLabel = page.locator(`label:has(${base})`).first();
        if (await wrapLabel.isVisible().catch(() => false)) {
          await wrapLabel.click();
          if (await input.isChecked().catch(() => false)) return true;
        }
        // 2) Sibling label[for=<input id>] — label is NOT an ancestor. The
        //    native radio/checkbox is often visually hidden (opacity:0 or a
        //    0px overlay), so `input.isVisible()` may be false — the LABEL is
        //    the clickable element and must be used regardless. Uses the
        //    matched option's OWN id so the precise option is clicked.
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
        // 3) Bare input: force-check it (never rely on a wrapping label or
        //    visibility — the input is often the styled/hidden native control).
        await (input as any).check({ force: true }).catch(() => {});
        if (await input.isChecked().catch(() => false)) return true;
      }
      // Fallback: click a matching visible element within the field's OWN
      // scope. Board-agnostic — handles toggle-button rows, pill options, and
      // option rows whose input has no id/name captured. Closes over the
      // answer and verifies the field actually flipped afterward.
      const scopeClicked = await page
        .evaluate(
          (fid: string, want: string) => {
            const scope = document.querySelector(`[data-field-path="${fid}"]`);
            if (!scope) return false;
            const norm = (s: string) =>
              (s || "").replace(/\s+/g, " ").trim().toLowerCase();
            const candidates = Array.from(
              scope.querySelectorAll(
                "button, label[for], label:has(input), [role='option'], " +
                  "[class*='option'], li, span, div[class*='option']"
              )
            ).filter((el) => {
              const txt = norm((el as HTMLElement).textContent || "");
              // Skip empty and the question label itself (a label whose text
              // equals the field's own heading is not an option).
              return txt === want || (txt && want && txt.includes(want));
            });
            for (const el of candidates) {
              const tag = el.tagName;
              // Prefer leaf nodes: a button, or an element with no option
              // descendants (avoids double-clicks on containers).
              if (tag === "BUTTON") {
                (el as HTMLElement).click();
                return true;
              }
              if (!el.querySelector("button, [class*='option']")) {
                (el as HTMLElement).click();
                return true;
              }
            }
            return false;
          },
          field.id,
          normalizeOptionText(answer)
        )
        .catch(() => false);
      if (scopeClicked) {
        await randomSleep(250, 450);
        return !!(await this.readScopedGroupValue(field));
      }
      // Fallback: text-based label matching within the form. Walks EVERY radio
      // in the group — not just the first — so an id-less group where the
      // answer is the second (or later) option still clicks the right one.
      const formLabel = page
        .locator(
          `${formSelector} label:has(input[type='radio'], input[type='checkbox']):has-text("${cssEscape(answer)}")`
        )
        .first();
      if (await formLabel.isVisible().catch(() => false)) {
        await formLabel.click();
        // Verify the chosen option (not the group's first) actually committed.
        const committed = await this.readFieldValue(field);
        return (
          !!committed && normalizeOptionText(committed) === normalizeOptionText(answer)
        );
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
   * Resolve the real control for a field, handling the Ashby DOM shapes:
   * (a) the input itself carries id == field id, (b) the input has NO id and
   * the field id lives on the label's `for` attribute, or (c) neither — the
   * label's `for` points at a nonexistent id (react-datepicker with id="")
   * and the control must be found inside the field's own `[data-field-path]`
   * scope. Returns `{ byId }`, `{ byLabel }`, or `{ byScope }`.
   */
  private async resolveTextControl(field: FormField): Promise<any | null> {
    const page = this.getPage();
    const id = field.id;
    return page
      .evaluate((fid: string) => {
        const byId = document.getElementById(fid);
        if (
          byId &&
          (byId instanceof HTMLInputElement ||
            byId instanceof HTMLTextAreaElement ||
            byId instanceof HTMLSelectElement)
        ) {
          return { byId: true };
        }
        const label = document.querySelector(`label[for="${fid}"]`);
        const control = label ? (label as HTMLLabelElement).control : null;
        if (
          control &&
          (control instanceof HTMLInputElement ||
            control instanceof HTMLTextAreaElement ||
            control instanceof HTMLSelectElement)
        ) {
          return { byLabel: true, type: (control as HTMLInputElement).type || control.tagName.toLowerCase() };
        }
        // Last resort: the field's own container. The input may carry no id
        // at all (and the label's `for` dangles) — grab the first text-like
        // control inside the `[data-field-path="<fid>"]` scope. The scope
        // itself may BE the control (generic adapter tags the input).
        const scope = document.querySelector(`[data-field-path="${fid}"]`);
        if (scope) {
          const first = scope.matches(
            'input[type="text"], input[type="date"], input[type="email"], input[type="tel"], input[type="url"], input:not([type]), textarea'
          )
            ? scope
            : scope.querySelector(
                'input[type="text"], input[type="date"], input[type="email"], input[type="tel"], input[type="url"], input:not([type]), textarea'
              );
          if (
            first instanceof HTMLInputElement ||
            first instanceof HTMLTextAreaElement
          ) {
            return { byScope: true, type: (first as HTMLInputElement).type || first.tagName.toLowerCase() };
          }
        }
        return null;
      }, id)
      .catch(() => null);
  }

  /**
   * Fill a plain text field. Refuses to type into a combobox/select-shell
   * disguised as a text field, and silently skips file inputs (resume handled
   * by the adapter's dedicated upload path). Handles the react-datepicker
   * shape (input without an id, linked via label[for]) by filling through the
   * label-associated control.
   */
  async fillTextById(id: string, answer: string): Promise<void> {
    const page = this.getPage();
    const field: FormField = { label: id, id, kind: "text", required: false, options: [], optionTargets: [] };
    const target = await this.resolveTextControl(field);
    if (process.env.DEBUG_FILL) {
      console.log(`[DEBUG_FILL] fillTextById #${id} resolveTextControl=${JSON.stringify(target)} answer="${answer}"`);
    }
    if (!target) return;
    if (target.byLabel || target.byScope) {
      // The control has no id of its own; drive it through the label's
      // associated control (or the scope's first control) with a native value
      // setter + input event so the framework (react-datepicker) observes it.
      await page
        .evaluate(
          (fid: string, value: string) => {
            const label = document.querySelector(`label[for="${fid}"]`);
            let control = label
              ? (label as HTMLLabelElement).control
              : null;
            const isControl =
              control instanceof HTMLInputElement ||
              control instanceof HTMLTextAreaElement;
            if (!isControl) {
              const scope = document.querySelector(`[data-field-path="${fid}"]`);
              const inScope = scope?.matches(
                'input[type="text"], input[type="date"], input[type="email"], input[type="tel"], input[type="url"], input:not([type]), textarea'
              )
                ? scope
                : (scope?.querySelector(
                    'input[type="text"], input[type="date"], input[type="email"], input[type="tel"], input[type="url"], input:not([type]), textarea'
                  ) as HTMLElement | null) ?? null;
              control = (inScope as HTMLElement | null) ?? null;
            }
            if (!control) return false;
            if ((control as HTMLInputElement).type === "file") return false;
            const setter = Object.getOwnPropertyDescriptor(
              (control as HTMLInputElement).constructor.prototype,
              "value"
            )?.set;
            if (setter) setter.call(control, value);
            else (control as HTMLInputElement).value = value;
            control.dispatchEvent(new Event("input", { bubbles: true }));
            control.dispatchEvent(new Event("change", { bubbles: true }));
            return true;
          },
          id,
          String(answer ?? "")
        )
        .catch(() => {});
      await randomSleep(200, 500);
      return;
    }
    const loc = page.locator(cssIdLocator(id)).first();
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
      const input = page.locator(cssIdLocator(id)).first();
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
      const sel = page
        .locator(`${cssIdLocator(field.id)} select, select${cssIdLocator(field.id)}`)
        .first();
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
    if (field.kind === "date") {
      return this.fillDate(field.id, String(answer ?? ""));
    }
    await this.fillTextById(field.id, String(answer ?? ""));
    return !!(await this.readInputValue(field.id));
  }

  /**
   * Fill a date-picker field (react-datepicker). The answer is usually a
   * free-text availability ("immediately", "in 2 weeks", "3 months") which is
   * translated to a concrete MM/DD/YYYY value that react-datepicker parses.
   * Returns true when a date value commits.
   */
  async fillDate(id: string, answer: string): Promise<boolean> {
    const date = translateToDate(answer);
    if (!date) return false;
    const page = this.getPage();
    const target = await this.resolveTextControl({ label: id, id, kind: "date", required: false, options: [], optionTargets: [] } as any);
    if (!target) return false;
    const value = `${date.getMonth() + 1}/${date.getDate()}/${date.getFullYear()}`;
    const ok = await page
      .evaluate(
        (fid: string, v: string) => {
          const byId = document.getElementById(fid);
          let control = byId;
          if (!control) {
            const label = document.querySelector(`label[for="${fid}"]`);
            control = label ? (label as HTMLLabelElement).control : null;
          }
          if (!control) {
            const scope = document.querySelector(`[data-field-path="${fid}"]`);
            control = (
              (scope?.matches('input[type="text"], input:not([type])')
                ? scope
                : scope?.querySelector('input[type="text"], input:not([type])')) as HTMLElement | null
            ) || null;
          }
          if (!control) return false;
          const setter = Object.getOwnPropertyDescriptor(
            HTMLInputElement.prototype,
            "value"
          )?.set;
          if (setter) setter.call(control, v);
          else (control as HTMLInputElement).value = v;
          control.dispatchEvent(new Event("input", { bubbles: true }));
          control.dispatchEvent(new Event("change", { bubbles: true }));
          control.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
          return true;
        },
        id,
        value
      )
      .catch(() => false);
    await randomSleep(300, 600);
    if (!ok) return false;
    const committed = await this.readInputValue(id);
    if (!committed) {
      // react-datepicker may keep the calendar open; commit by typing again.
      const input = page.locator(this.fieldScopeInputSelector(id)).first();
      if (await input.isVisible().catch(() => false)) {
        await input.fill(value);
        await randomSleep(300, 600);
      }
    }
    console.log(`[${this.tagName}] Filled date #${id} with "${value}" (from "${answer}")`);
    return !!(await this.readInputValue(id));
  }

  /** CSS selector for the first text/date input inside a field's scope. */
  private fieldScopeInputSelector(id: string): string {
    return `[data-field-path="${id}"] input[type="text"], ` +
      `[data-field-path="${id}"] input:not([type])`;
  }

  /**
   * Deterministic fill through a Stagehand observe() Action. The Action
   * carries a selector that act({ ...action, method: "fill", arguments:
   * [value] }) resolves with NO extra LLM inference — the generic adapter uses
   * this as its Tier-2 fallback for exotic form renderers the DOM walker could
   * not enumerate. Returns true once the action is issued; committed-value
   * verification is the caller's job (readObservedValue).
   */
  async fillObserved(
    action: { selector: string; description: string },
    answer: string
  ): Promise<boolean> {
    try {
      await this.stagehand.act(
        {
          selector: action.selector,
          description: action.description || "fill the field",
          method: "fill",
          arguments: [String(answer ?? "")],
        },
        { page: this.getPage() }
      );
      await randomSleep(300, 600);
      return true;
    } catch (err: any) {
      console.warn(
        `[${this.tagName}] fillObserved failed for ${action.selector}: ${err?.message || err}`
      );
      return false;
    }
  }

  /**
   * Read the committed value of an element addressed by an observe() selector
   * so a filled field can be verified without another LLM call. Resolves
   * XPath-style selectors ("xpath=…", "/…") and plain CSS selectors.
   */
  async readObservedValue(selector: string): Promise<string> {
    const page = this.getPage();
    try {
      // WARNING: only anonymous arrows may be defined inside this evaluate
      // (tsx keepNames stringifies the callback into the page). Destructure
      // helpers into an array so none gains a __name() wrapper.
      return (await page.evaluate((sel: string) => {
        const [resolve, readVal] = [
          (): Element | null => {
            const s = String(sel || "").trim();
            if (!s) return null;
            if (s.startsWith("xpath=") || s.startsWith("/")) {
              try {
                const xp = s.replace(/^xpath=/i, "");
                const node = document.evaluate(
                  xp,
                  document,
                  null,
                  XPathResult.FIRST_ORDERED_NODE_TYPE,
                  null
                ).singleNodeValue;
                return node instanceof Element ? node : null;
              } catch {
                return null;
              }
            }
            try {
              return document.querySelector(s);
            } catch {
              return null;
            }
          },
          (el: Element): string => {
            if (el instanceof HTMLSelectElement) {
              const opt = el.selectedOptions?.[0];
              return opt ? (opt.textContent || opt.value || "").trim() : "";
            }
            if (el instanceof HTMLInputElement || el instanceof HTMLTextAreaElement) {
              return (el.value || "").trim();
            }
            return (el.textContent || "").replace(/\s+/g, " ").trim();
          },
        ];
        const el = resolve();
        if (!el) return "";
        return readVal(el);
      }, selector)) as string;
    } catch {
      return "";
    }
  }
}