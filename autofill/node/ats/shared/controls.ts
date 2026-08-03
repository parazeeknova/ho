import { Stagehand } from "@browserbasehq/stagehand";
import {
  humanTypingEnabled,
  humanTypingMaxLength,
  randomSleep,
  thinkPause,
  typingDelayMs,
} from "../../utils/evasion";
import { FormField } from "./model";
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
  valuesConsistent,
} from "./matching";

/**
 * Coerce an answer for a type=number input to a bare numeric string, or ""
 * when nothing numeric survives. Stagehand's fill rejects non-numeric values
 * with an invalid-number-value error that aborts the whole fill; KB/LLM
 * answers are often ranges or labels ("0-4 Years", "Immediately").
 *
 * Quantity suffixes are expanded ("80K" -> "80000", "1.5M" -> "1500000") so a
 * salary answer like "80K INR/month" never degrades to a bare "80", and
 * separators/whitespace are stripped ("1,200" -> "1200").
 */
export function sanitizeNumberAnswer(answer: string): string {
  const original = String(answer ?? "").trim();
  if (!original) return "";
  // Quantity suffix (K/M/B): only when the magnitude is a STANDALONE token —
  // digit-run + suffix, bounded by non-letters on both sides. This expands
  // "80K INR/month" -> "80000" while never exploding ordinary words like
  // "3 BHK", "8 MB", "5 M&A deals", or the "2B" inside "10 B2B clients".
  const mul = original.match(
    /(?<![a-zA-Z0-9])([\d.]+)([kKmMbB])(?=\s|[^a-zA-Z0-9]|$)/
  );
  if (mul) {
    const n = parseFloat(mul[1]);
    const mult: Record<string, number> = { k: 1e3, m: 1e6, b: 1e9 };
    if (Number.isFinite(n)) return String(Math.round(n * mult[mul[2].toLowerCase()]));
  }
  const m = original.replace(/[,\s]/g, "").match(/-?\d+(?:\.\d+)?/);
  return m ? m[0] : "";
}

/**
 * Normalize a committed form value so a framework placeholder (the literal
 * strings "undefined", "null", "NaN") never counts as a filled answer or gets
 * committed to a field. Returns the trimmed value, or "" for placeholders.
 */
export function cleanPlaceholderValue(value: string): string {
  const v = (value ?? "").trim();
  return /^(undefined|null|nan)$/i.test(v) ? "" : v;
}

/**
 * Extract the first URL from a free-text answer. Answers to "Public links" /
 * "Portfolio" style fields are often a comma/space-separated list of several
 * URLs; a single-line URL control can only hold one, and concatenating the
 * list fails URL validation. Returns null when the answer has no URL.
 */
export function firstUrl(answer: string): string | null {
  const urls = String(answer ?? "").match(/https?:\/\/[^\s,;"']+/g);
  if (!urls || !urls.length) return null;
  return urls[0].replace(/[.,;)\]>]+$/, "").trim() || null;
}

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
      await this.fillLikeHuman(locator, value);
      await randomSleep(100, 300);
    } else {
      await this.safeAct(actPrompt, { [variableName]: value });
      await randomSleep(200, 500);
    }
  }

  /**
   * Fill a visible locator. With human typing enabled and a short-enough
   * value, type per-keystroke (real keydown/keyup events + human cadence)
   * instead of committing via the native setter with zero key events.
   */
  private async fillLikeHuman(locator: any, value: string): Promise<void> {
    if (humanTypingEnabled() && value.length <= humanTypingMaxLength()) {
      await locator.type(value, { delay: typingDelayMs() });
    } else {
      await locator.fill(value);
    }
  }

  /** Center (viewport coords) of the first VISIBLE element matching a CSS
   *  selector or XPath, or null when none is on screen. Used to drive a human
   *  mouse arc before clicking without depending on Locator internals. */
  private async elementCenter(
    page: any,
    selector: string
  ): Promise<{ x: number; y: number } | null> {
    return page
      .evaluate((sel: string) => {
        let els: Element[] = [];
        if (sel.startsWith("//") || sel.startsWith("(")) {
          const snap = document.evaluate(
            sel,
            document,
            null,
            XPathResult.ORDERED_NODE_SNAPSHOT_TYPE,
            null
          );
          for (let i = 0; i < snap.snapshotLength; i++) {
            const el = snap.snapshotItem(i);
            if (el instanceof Element) els.push(el);
          }
        } else {
          els = Array.from(document.querySelectorAll(sel));
        }
        for (const el of els) {
          const r = (el as HTMLElement).getBoundingClientRect();
          if (r.width < 1 || r.height < 1) continue;
          return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
        }
        return null;
      }, selector)
      .catch(() => null);
  }

  /** Move the cursor to a point with a natural multi-step arc and a dwell,
   *  via the page's CDP session. Falls back to a no-op when unavailable. */
  private async mouseArc(
    page: any,
    x: number,
    y: number
  ): Promise<void> {
    const session = (page as any).mainSession ?? (page as any).session;
    if (!session?.send) return;
    try {
      const steps = 5 + Math.floor(Math.random() * 4);
      const sx = x - (30 + Math.random() * 50);
      const sy = y - (15 + Math.random() * 40);
      await session.send("Input.dispatchMouseEvent", {
        type: "mouseMoved",
        x: sx,
        y: sy,
        button: "none",
      });
      for (let i = 1; i <= steps; i++) {
        const t = i / (steps + 1);
        const e = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
        await session.send("Input.dispatchMouseEvent", {
          type: "mouseMoved",
          x: sx + (x - sx) * e + (Math.random() - 0.5) * 4,
          y: sy + (y - sy) * e + (Math.random() - 0.5) * 4,
          button: "none",
        });
        await new Promise((r) => setTimeout(r, 15 + Math.random() * 35));
      }
      await session.send("Input.dispatchMouseEvent", {
        type: "mouseMoved",
        x,
        y,
        button: "none",
      });
      await randomSleep(150, 350); // dwell before the click
    } catch {
      // Best-effort; a failed arc still lets the locator click proceed.
    }
  }

  /** Click a locator like a human: move the cursor in an arc to the element
   *  (or hover + dwell), pause, then click. ``selector`` (CSS or XPath) is the
   *  geometry source for the arc; when omitted the arc degrades to a hover. */
  async humanClick(locator: any, selector?: string): Promise<void> {
    const page = this.getPage();
    if (selector) {
      const center = await this.elementCenter(page, selector);
      if (center) {
        await this.mouseArc(page, center.x, center.y);
        await locator.click();
        await thinkPause();
        return;
      }
    }
    await locator.hover().catch(() => {});
    await randomSleep(200, 450);
    await locator.click();
    await thinkPause();
  }

  /** Close an open react-select menu without depending on keyboard typing. */
  async closeMenu(): Promise<void> {
    try {
      await this.getPage().keyPress("Escape");
    } catch {
      // Menu may already be closed; harmless.
    }
  }

  /**
   * Simulate a human "reading over" the form before submitting: move the cursor
   * with a real arc to a few benign, non-input elements (headings, section
   * titles, the form header) and click them. These trusted pointer interactions
   * in the form region mirror what a human does while reviewing, without ever
   * changing a field value. Number of clicks is randomized 1-2 so it is never
   * a fixed pattern.
   */
  async humanFormInteractions(page: any, count = 2): Promise<void> {
    try {
      const targets = await page
        .evaluate(() => {
          const nodes = Array.from(
            document.querySelectorAll(
              "h1, h2, h3, h4, .ashby-application-form-header, " +
                "section[data-qa='form-section'], [class*='form-header']"
            )
          ) as HTMLElement[];
          const pts: { x: number; y: number }[] = [];
          for (const el of nodes) {
            if (el.querySelector("input, textarea, select, button")) continue;
            const r = el.getBoundingClientRect();
            if (r.width < 4 || r.height < 4) continue;
            if (r.bottom < 0 || r.top > window.innerHeight) continue;
            pts.push({ x: r.left + r.width / 2, y: r.top + r.height / 2 });
          }
          return pts;
        })
        .catch(() => [] as { x: number; y: number }[]);
      if (!targets.length) return;
      // Pick 1..2 distinct benign targets, shuffled.
      const pick = (() => {
        const n = Math.min(count, targets.length);
        const shuffled = [...targets].sort(() => Math.random() - 0.5);
        return shuffled.slice(0, n);
      })();
      for (const t of pick) {
        await this.mouseArc(page, t.x, t.y);
        const session = (page as any).mainSession ?? (page as any).session;
        if (session?.send) {
          await session
            .send("Input.dispatchMouseEvent", {
              type: "mousePressed",
              x: t.x,
              y: t.y,
              button: "left",
              clickCount: 1,
            })
            .catch(() => {});
          await randomSleep(60, 150);
          await session
            .send("Input.dispatchMouseEvent", {
              type: "mouseReleased",
              x: t.x,
              y: t.y,
              button: "left",
              clickCount: 1,
            })
            .catch(() => {});
        }
        await randomSleep(350, 800);
      }
      console.log(`[${this.tagName}] Simulated ${pick.length} human form interaction(s).`);
    } catch (err: any) {
      console.warn(`[${this.tagName}] humanFormInteractions failed (continuing):`, err?.message || err);
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
      const xpath = `//${this.optionTag}[contains(@role,"option")][${optionExactXPath(picked)}]`;
      const options = page.locator(xpath);
      const count = await options.count().catch(() => 0);
      for (let i = 0; i < count; i++) {
        const option = options.nth(i);
        if (!(await option.isVisible().catch(() => false))) continue;
        await this.humanClick(option, xpath);
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
      const raw = await page.evaluate((inputId: string) => {
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
      return cleanPlaceholderValue(raw);
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
      const raw = await page.evaluate((inputId: string) => {
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
      return cleanPlaceholderValue(raw);
    } catch {
      return "";
    }
  }

  /** Read the checked radio/checkbox option text in a group. */
  async readGroupValue(name: string): Promise<string> {
    const page = this.getPage();
    try {
      const raw = await page.evaluate((groupName: string) => {
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
      return cleanPlaceholderValue(raw);
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
      const raw = await page.evaluate((fid: string) => {
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
      return cleanPlaceholderValue(raw);
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
   * verified after every attempt, and the wrong option is never committed:
   * a click that does not flip the intended option (or that flips a different
   * one) is treated as a failure and the caller leaves the field blank.
   */
  async clickGroupOption(
    field: FormField,
    answer: string,
    formSelector = "#application-form"
  ): Promise<boolean> {
    const page = this.getPage();
    try {
      const type = field.kind === "checkbox" ? "checkbox" : "radio";
      const picks = selectCandidates(answer);
      const targets = field.optionTargets;
      for (const cand of picks) {
        const nc = normalizeOptionText(cand);
        // Never match a target whose text is empty: ``nc.includes("")`` is
        // always true, which silently selects the FIRST option in the group
        // (e.g. "Yes") no matter what the answer was.
        const nonEmptyTargets = targets.filter((t) => normalizeOptionText(t.text).length > 0);
        const target =
          nonEmptyTargets.find((t) => normalizeOptionText(t.text) === nc) ||
          nonEmptyTargets.find(
            (t) =>
              normalizeOptionText(t.text).includes(nc) ||
              nc.includes(normalizeOptionText(t.text))
          );
        if (!target) continue;
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
        // candidate, and use THAT input's id. The match is substring-aware so
        // a leading-token/clause candidate ("No", "No, I am not based in
        // Europe/UK") still resolves to the full option ("No, I am not based
        // in Europe/UK and would require visa support.") instead of being
        // skipped and falling back to a first-match click on the wrong option.
        const nameSel = cssEscape(target.name || field.name || field.id);
        let optionId = target.id;
        if (!optionId) {
          optionId = await page
            .evaluate(
              (args: { name: string; type: string; want: string }) => {
                // WARNING: only anonymous arrows here (tsx keepNames wraps
                // inferred-name arrows in __name(), which throws in page
                // context). Destructure the helper so it never gains a name.
                const [norm] = [
                  (s: string) =>
                    (s || "").replace(/\s+/g, " ").trim().toLowerCase(),
                ];
                const inputs = Array.from(
                  document.querySelectorAll(
                    `input[type="${args.type}"][name="${args.name}"]`
                  )
                ) as HTMLInputElement[];
                let best = "";
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
                  if (!txt) continue;
                  if (txt === args.want) {
                    if (inp.id) return inp.id;
                    // Exact match with no id: keep it as best rather than
                    // throwing away a substring-matched sibling.
                    best = inp.id || best;
                    continue;
                  }
                  // Substring either way: the candidate may be a leading token
                  // of a longer option, or the option a fragment of the answer.
                  if (txt.includes(args.want) || args.want.includes(txt)) {
                    best = inp.id || best;
                  }
                }
                return best;
              },
              { name: nameSel, type, want: nc }
            )
            .catch(() => "");
        }
        const base = optionId
          ? `input[type="${type}"][id="${cssEscape(optionId)}"]`
          : `input[type="${type}"][name="${nameSel}"]`;
        const input = page.locator(base).first();
        // A name-only selector resolves to the group's FIRST option (all
        // members share the name). Only an id-precise click is trusted
        // immediately; an id-less click must be verified against the intended
        // answer so a "No" answer can never commit a "Yes" first option.
        const verified = async (): Promise<boolean> => {
          if (optionId) return true;
          const committed = await this.readFieldValue(field);
          return !!committed && valuesConsistent(answer, committed);
        };

        // 1) Wrapping label (label:has(input)).
        const wrapLabel = page.locator(`label:has(${base})`).first();
        if (await wrapLabel.isVisible().catch(() => false)) {
          await wrapLabel.click();
          if (await input.isChecked().catch(() => false)) {
            if (await verified()) return true;
          }
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
            if (await input.isChecked().catch(() => false)) {
              if (await verified()) return true;
            }
          }
        }
        // 3) Bare input: force-check it (never rely on a wrapping label or
        //    visibility — the input is often the styled/hidden native control).
        await (input as any).check({ force: true }).catch(() => {});
        if (await input.isChecked().catch(() => false)) {
          if (await verified()) return true;
        }
      }

      // Fallback: click a matching visible element within the field's OWN
      // scope. Board-agnostic — handles toggle-button rows, pill options, and
      // option rows whose input has no id/name captured. Only leaf option rows
      // are clicked (a wrapper whose text spans the whole group can toggle the
      // FIRST option and commit the WRONG value), and an element whose text
      // also contains another option of the group is never a target.
      const optionTexts = field.options
        .map((o) => normalizeOptionText(o))
        .filter(Boolean);
      const scopeClicked = await page
        .evaluate(
          (fid: string, want: string, opts: string[]) => {
            const scope = document.querySelector(`[data-field-path="${fid}"]`);
            if (!scope) return false;
            // WARNING: only anonymous arrows here (tsx keepNames wraps
            // inferred-name arrows in __name(), which throws in page context).
            const [norm] = [
              (s: string) =>
                (s || "").replace(/\s+/g, " ").trim().toLowerCase(),
            ];
            const candidates = Array.from(
              scope.querySelectorAll(
                "button, label[for], label:has(input), [role='option'], " +
                  "[class*='option'], li, span, div[class*='option']"
              )
            );
            const scored: Array<{ el: Element; score: number }> = [];
            for (const el of candidates) {
              const tag = el.tagName;
              const txt = norm((el as HTMLElement).textContent || "");
              if (!txt) continue;
              // A wrapper whose text also contains ANOTHER option of this
              // group (e.g. a "Yes No" row when we want "No") must never be
              // clicked: clicking it can toggle the group's first option.
              if (opts.some((o) => o && o !== want && txt.includes(o))) continue;
              const clickableTag = tag === "BUTTON" || tag === "LABEL";
              const leaf =
                clickableTag || !el.querySelector("button, [class*='option'], input");
              const exact = txt === want;
              const near =
                leaf && txt.split(/\s+/).length <= 4 && txt.includes(want);
              if (!exact && !near) continue;
              // Exact option rows beat short containment matches; buttons are
              // the most precise target (toggle rows), then labels, then any.
              const score = exact ? (tag === "BUTTON" ? 0 : 1) : 2;
              scored.push({ el, score });
            }
            scored.sort((a, b) => a.score - b.score);
            for (const s of scored) {
              (s.el as HTMLElement).click();
              return true;
            }
            return false;
          },
          field.id,
          normalizeOptionText(answer),
          optionTexts
        )
        .catch(() => false);
      if (scopeClicked) {
        await randomSleep(250, 450);
        // Verify the committed value matches the intended answer — a click
        // on the wrong option (e.g. a group wrapper) can check something,
        // so presence alone must not count as success.
        const committed = await this.readFieldValue(field);
        return !!committed && valuesConsistent(answer, committed);
      }

      // Page-wide fallback for groups NOT inside a data-field-path scope
      // (Greenhouse/Lever radios/checkboxes). Resolve the exact option input
      // by its label anywhere in the document and click that input's own
      // label — never a first-match guess — then verify the specific input
      // checked AND the committed value is consistent with the answer.
      const groupName = targets[0]?.name || field.name || field.id;
      if (groupName) {
        const pageWideId = await page
          .evaluate(
            (args: { name: string; type: string; want: string }) => {
              const [norm] = [
                (s: string) =>
                  (s || "").replace(/\s+/g, " ").trim().toLowerCase(),
              ];
              const inputs = Array.from(
                document.querySelectorAll(
                  `input[type="${args.type}"][name="${args.name}"]`
                )
              ) as HTMLInputElement[];
              let bestId = "";
              let bestScore = 0;
              for (const inp of inputs) {
                const wrap = inp.closest("label");
                const forLabel = inp.id
                  ? document.querySelector(`label[for="${inp.id}"]`)
                  : null;
                const label = wrap || forLabel;
                const txt = norm(
                  label
                    ? label.textContent || ""
                    : inp.getAttribute("aria-label") || ""
                );
                if (!txt) continue;
                let score = 0;
                if (txt === args.want) score = 3;
                else if (txt.includes(args.want) || args.want.includes(txt)) score = 2;
                if (score > bestScore) {
                  bestScore = score;
                  bestId = inp.id || "";
                }
              }
              if (!bestId) return "";
              const target = document.getElementById(bestId);
              // Scan label[for] manually — interpolating bestId into a CSS
              // selector breaks on ids containing quotes.
              const forLabels = Array.from(document.querySelectorAll("label[for]")).filter(
                (l) => l.getAttribute("for") === bestId
              );
              const targetLabel =
                (target?.closest("label") as HTMLElement | null) ||
                (forLabels[0] as HTMLElement | null) ||
                target;
              if (targetLabel) (targetLabel as HTMLElement).click();
              return bestId;
            },
            { name: cssEscape(groupName), type, want: normalizeOptionText(answer) }
          )
          .catch(() => "");
        if (pageWideId) {
          await randomSleep(250, 450);
          const clickedInput = page
            .locator(`input[type="${type}"][id="${cssEscape(pageWideId)}"]`)
            .first();
          const checked = await clickedInput.isChecked().catch(() => false);
          if (checked) {
            const committed = await this.readFieldValue(field);
            if (committed && valuesConsistent(answer, committed)) return true;
          }
        }
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
        // Verify the chosen option (not the group's first) actually committed
        // AND is consistent with the intended answer.
        const committed = await this.readFieldValue(field);
        return !!committed && valuesConsistent(answer, committed);
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
        // WARNING: only anonymous arrows may be defined here (tsx keepNames
        // wraps inferred-name arrows in __name(), which throws in page
        // context). Destructure helpers into an array so none gains a name.
        const [describe] = [
          (el: Element): any => {
            const tag = el.tagName.toLowerCase();
            const type =
              el instanceof HTMLInputElement
                ? el.type || tag
                : tag;
            return { type, tag };
          },
        ];
        const byId = document.getElementById(fid);
        if (
          byId &&
          (byId instanceof HTMLInputElement ||
            byId instanceof HTMLTextAreaElement ||
            byId instanceof HTMLSelectElement)
        ) {
          return { byId: true, ...describe(byId) };
        }
        const label = document.querySelector(`label[for="${fid}"]`);
        const control = label ? (label as HTMLLabelElement).control : null;
        if (
          control &&
          (control instanceof HTMLInputElement ||
            control instanceof HTMLTextAreaElement ||
            control instanceof HTMLSelectElement)
        ) {
          return { byLabel: true, ...describe(control) };
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
            return { byScope: true, ...describe(first) };
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

    // Never commit a framework placeholder (the literal "undefined"/"null"/
    // "NaN" a broken datepicker can render) — leave the field blank instead.
    const cleaned = cleanPlaceholderValue(String(answer ?? ""));
    if (!cleaned) {
      console.warn(
        `[${this.tagName}] #${id} got a blank/placeholder answer ` +
          `("${escapePromptValue(String(answer ?? ""))}"); leaving blank.`
      );
      return;
    }

    // A react-datepicker control reached through the plain-text path (the
    // walker missed the date classification) must be filled as a real date —
    // typing free text like "Immediately" leaves the picker broken.
    if (target.type === "date") {
      await this.fillDate(id, cleaned);
      return;
    }

    // A single-line URL/link control holds ONE url. Answers to "Public links"
    // style fields are often a comma-separated list of several URLs; filling
    // the concatenated list fails URL validation, so fill only the first.
    // Only truncate for an actual URL control, or when the ENTIRE answer is a
    // list of URLs — a free-text answer that merely mentions a link ("Built
    // https://github.com/x, a CI tool") must never be destroyed.
    let valueToFill = cleaned;
    if (target.tag === "input") {
      const tokens = cleaned
        .split(/[\s,;]+/)
        .map((t) => t.trim())
        .filter(Boolean);
      const allUrls = tokens.length > 0 && tokens.every((t) => /^https?:\/\//i.test(t));
      if (target.type === "url" || allUrls) {
        const only = firstUrl(cleaned);
        if (only) valueToFill = only;
      }
    }

    // A type=number control resolved via label/scope still needs numeric
    // coercion — writing "0-4 Years" through the native setter silently
    // becomes "" (the browser drops non-numeric input).
    if (target.type === "number") {
      const numeric = sanitizeNumberAnswer(valueToFill);
      if (!numeric) {
        console.warn(
          `[${this.tagName}] Number field #${id} has non-numeric answer ` +
            `"${escapePromptValue(valueToFill)}"; leaving blank.`
        );
        return;
      }
      valueToFill = numeric;
    }

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
          valueToFill
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
    // A type=number input rejects non-numeric values with Stagehand's
    // invalid-number-value error, aborting the whole fill. The KB/LLM answer
    // is often a range or label ("0-4 Years"); coerce it to a bare number
    // (first integer/float token) and skip the field when nothing numeric is
    // left, instead of crashing the run.
    if (type === "number") {
      const numeric = sanitizeNumberAnswer(valueToFill);
      if (!numeric) {
        console.warn(
          `[${this.tagName}] Number field #${id} has non-numeric answer ` +
            `"${escapePromptValue(valueToFill)}"; leaving blank.`
        );
        return;
      }
      valueToFill = numeric;
    }
    await this.fillLikeHuman(loc, valueToFill);
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
      // Native <select> fallback. Stagehand's locator wrapper serializes a
      // Playwright {label}/{value} object to "[object Object]", which never
      // matches an option and silently commits nothing — pass the option TEXT
      // as a plain string (Stagehand matches by text OR value). Verify the
      // commit actually stuck before claiming success.
      const page = this.getPage();
      const sel = page
        .locator(`${cssIdLocator(field.id)} select, select${cssIdLocator(field.id)}`)
        .first();
      if (await sel.isVisible().catch(() => false)) {
        const picked = chooseOption(selectCandidates(answer), optionTexts ?? []);
        if (picked) {
          await (sel as any).selectOption(picked);
          await randomSleep(200, 400);
          const committed = !!(await this.readInputValue(field.id));
          if (committed) return true;
          console.warn(
            `[${this.tagName}] Native select #${field.id} did not commit "${picked}"`
          );
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
    const clean = cleanPlaceholderValue(String(answer ?? ""));
    const date = translateToDate(clean);
    // A translated Date can still be invalid (NaN components) — never commit it.
    if (!date || !Number.isFinite(date.getTime())) return false;
    const page = this.getPage();
    const target = await this.resolveTextControl({ label: id, id, kind: "date", required: false, options: [], optionTargets: [] } as any);
    if (!target) return false;
    // Native <input type="date"> requires YYYY-MM-DD; react-datepicker text
    // inputs parse MM/DD/YYYY. Route by the resolved control's type.
    const value =
      target.type === "date"
        ? `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(
            date.getDate()
          ).padStart(2, "0")}`
        : `${date.getMonth() + 1}/${date.getDate()}/${date.getFullYear()}`;
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
    let committed = ok ? cleanPlaceholderValue(await this.readInputValue(id)) : "";
    // A commit that left a placeholder (e.g. react-datepicker rejected the
    // native setter and re-rendered "undefined") must be re-driven by typing.
    if (!committed || /^[a-z\s]+$/i.test(committed)) {
      const input = page.locator(this.fieldScopeInputSelector(id)).first();
      if (await input.isVisible().catch(() => false)) {
        await input.click().catch(() => {});
        await input.fill(value);
        await randomSleep(300, 600);
        await page.keyPress("Enter").catch(() => {});
        await randomSleep(300, 600);
      }
      committed = cleanPlaceholderValue(await this.readInputValue(id));
    }
    if (committed && /^[a-z\s]+$/i.test(committed)) committed = "";
    console.log(`[${this.tagName}] Filled date #${id} with "${value}" (from "${answer}")`);
    return !!committed;
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