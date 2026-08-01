import { Stagehand } from "@browserbasehq/stagehand";
import * as fs from "fs";
import { ATSAdapter, RpcHelper } from "./base.js";
import { JobPayload } from "../types.js";
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
 * substring matching must not resolve to them.
 */
export function isDeclineOption(text: string): boolean {
  return /(don'?t wish|do not wish|prefer not|choose not|rather not|not wish)/i.test(text);
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
   * Build a deterministic map of open questions -> {dom selector, kind} by reading
   * the Greenhouse form's labels and their associated inputs. This avoids relying on
   * the LLM to map a question's text back to an element (which was returning null).
   */
  private async collectQuestions(): Promise<
    Array<{ label: string; id: string; kind: "text" | "select" | "multi" }>
  > {
    const page = this.getPage();
    try {
      const rows = await page.evaluate(() => {
        const out: Array<{ label: string; id: string; kind: "text" | "select" | "multi" }> = [];
        const labels = document.querySelectorAll(
          "#application-form .field-wrapper label, .application--questions label, label.select__label"
        );
        for (const lbl of Array.from(labels)) {
          const text = (lbl.textContent || "").replace(/\s+/g, " ").trim().replace(/^\*+|\*+$/g, "");
          if (!text) continue;
          const forId = lbl.getAttribute("for");
          if (!forId) continue;
          const input = document.getElementById(forId);
          if (!input) continue;
          // File uploads are handled by the resume uploader; they are not
          // text questions and cannot be answered via KB/Telegram.
          if ((input as HTMLInputElement).type === "file") continue;
          const shell = input.closest("[class*='select-shell']");
          const isSelect =
            !!shell ||
            !!input.closest("[class*='select'], select") ||
            input.getAttribute("role") === "combobox";
          if (!isSelect) {
            out.push({ label: text, id: forId, kind: "text" });
            continue;
          }
          // react-select multi: aria-multiselectable combobox, or a shell
          // whose class marks it multi-select.
          const isMulti =
            input.getAttribute("aria-multiselectable") === "true" ||
            (shell ? /(^|\s)multi(\s|$)|select__multi|--is-multi/.test(shell.className) : false);
          out.push({ label: text, id: forId, kind: isMulti ? "multi" : "select" });
        }
        return out;
      });

      // De-duplicate by label, keep later (custom screener section takes priority).
      const seen = new Set<string>();
      const uniq: Array<{ label: string; id: string; kind: "text" | "select" | "multi" }> = [];
      for (const r of rows ?? []) {
        if (seen.has(r.label)) continue;
        seen.add(r.label);
        uniq.push(r);
      }
      return uniq;
    } catch (err: any) {
      console.warn("[GreenhouseAdapter] collectQuestions failed:", err?.message || err);
      return [];
    }
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

      const optionTexts = await page.evaluate(() => {
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
      await this.closeMenu();
      await randomSleep(100, 250);
      return optionTexts;
    } catch (err: any) {
      console.warn(`[GreenhouseAdapter] readSelectOptions failed for #${id}:`, err?.message || err);
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
    if (profile.linkedin) {
      await this.fillField(
        'input[aria-label*="LinkedIn"], input[name*="linkedin"]',
        profile.linkedin,
        "Type %linkedin% into the LinkedIn Profile URL field",
        "linkedin"
      );
    }
    if (profile.github) {
      await this.fillField(
        'input[aria-label*="GitHub"], input[name*="github"]',
        profile.github,
        "Type %github% into the GitHub Profile URL field",
        "github"
      );
    }

    // Resume Upload Handling
    if (profile.resumePath && fs.existsSync(profile.resumePath)) {
      console.log(`[GreenhouseAdapter] Uploading resume from ${profile.resumePath}...`);
      const fileInput = page.locator('input#resume[type="file"], input[type="file"]').first();
      if (await fileInput.count() > 0) {
        await fileInput.setInputFiles(profile.resumePath);
        await randomSleep(300, 600);
        console.log("[GreenhouseAdapter] Resume uploaded successfully.");
      }
    }

    // Per-field screener walk: every labelled question is resolved one at a
    // time via the answer_question RPC (KB first, Telegram with options for
    // unknowns, learned into the KB on every answer). Basic identity fields
    // are skipped — they were filled deterministically above.
    if (rpc) {
      console.log("[GreenhouseAdapter] Walking custom screener questions one by one...");
      const basicFields = new Set([
        "first name",
        "last name",
        "email",
        "phone",
        "linkedin",
        "github",
        "website",
        "resume",
        "cover letter",
        "preferred first name",
      ]);
      const formFields = await this.collectQuestions();
      const filled: string[] = [];
      const blanked: string[] = [];

      for (const field of formFields) {
        const fieldNorm = this.normalise(field.label);
        if (basicFields.has(fieldNorm)) continue;

        let optionTexts: string[] = [];
        if (field.kind === "select" || field.kind === "multi") {
          optionTexts = await this.readSelectOptions(field.id);
          if (optionTexts.length === 0) {
            console.warn(
              `[GreenhouseAdapter] Could not read options for #${field.id} ` +
                `("${escapePromptValue(field.label)}"); leaving blank.`
            );
            blanked.push(field.label);
            continue;
          }
        }

        let result: any;
        try {
          result = await rpc("answer_question", {
            question: field.label,
            kind: field.kind,
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
          console.log(
            `[GreenhouseAdapter] Leaving "${escapePromptValue(field.label)}" blank ` +
              `(source: ${result?.source ?? "decline"})`
          );
          blanked.push(field.label);
          continue;
        }

        let ok: boolean;
        if (field.kind === "select") {
          ok = await this.fillQuestionSelect(field.id, field.label, answer, optionTexts);
          let value = ok ? await this.readSelectValue(field.id) : "";
          if (ok && !value) {
            // Committed value read-back: one re-fill attempt, then fail loudly.
            await this.fillQuestionSelect(field.id, field.label, answer, optionTexts);
            value = await this.readSelectValue(field.id);
          }
          ok = !!value;
        } else if (field.kind === "multi") {
          ok = await this.fillQuestionMulti(field.id, field.label, answer, optionTexts);
          let value = ok ? await this.readSelectValue(field.id) : "";
          if (ok && !value) {
            await this.fillQuestionMulti(field.id, field.label, answer, optionTexts);
            value = await this.readSelectValue(field.id);
          }
          ok = !!value;
        } else {
          await this.fillQuestionText(field.id, answer);
          let value = await this.readInputValue(field.id);
          if (!value) {
            await this.fillQuestionText(field.id, answer);
            value = await this.readInputValue(field.id);
          }
          ok = !!value;
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

      console.log(
        `[GreenhouseAdapter] Screener walk complete. Filled: ${filled.length}, blank (declined/unknown): ${blanked.length}.`
      );
      for (const b of blanked) {
        console.warn(`[GreenhouseAdapter]   blank: ${escapePromptValue(b)}`);
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
