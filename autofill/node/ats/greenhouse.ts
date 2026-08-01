import { Stagehand } from "@browserbasehq/stagehand";
import { z } from "zod";
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

/** Case-insensitive XPath `contains` predicate for a literal text value. */
function optionContainsXPath(text: string): string {
  const safe = escapePromptValue(text).replace(/'/g, "\\'");
  return (
    'contains(translate(normalize-space(.), "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz"), "' +
    safe.toLowerCase() +
    '")'
  );
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
   * Select an option from a Greenhouse react-select dropdown. Tries an AI-driven
   * selection first, then falls back to clicking the combobox and its option.
   */
  private async fillDropdown(questionText: string, answerText: string): Promise<void> {
    try {
      await this.stagehand.act(`select "${answerText}" for the dropdown asking "${questionText}"`);
      await randomSleep(400, 800);
      return;
    } catch (err: any) {
      console.warn("[GreenhouseAdapter] Dropdown AI select failed, trying manual fallback:", err?.message || err);
    }

    try {
      const page = this.getPage();
      // XPath: find the field wrapper whose label contains the question text, then its react-select control.
      const q = escapePromptValue(questionText).replace(/'/g, "\\'");
      const control = page
        .locator(
          `//div[contains(@class,"field-wrapper")][.//label[contains(., "${q}")]]//div[contains(@class,"select__control")]`
        )
        .first();
      await control.click();
      await randomSleep(300, 600);
      // The menu opens with selectable options; click the one matching the answer.
      for (const candidate of selectCandidates(answerText)) {
        const menuOption = page
          .locator(`//div[@id^="react-select"]//div[contains(@role,"option")][${optionContainsXPath(candidate)}]`)
          .first();
        if (await menuOption.isVisible().catch(() => false)) {
          await menuOption.click();
          await randomSleep(400, 800);
          return;
        }
      }
      console.warn("[GreenhouseAdapter] No manual option match for dropdown; leaving as-is.");
    } catch (err: any) {
      console.warn("[GreenhouseAdapter] Dropdown manual fallback failed:", err?.message || err);
    }
  }

  /**
   * Build a deterministic map of open questions -> {dom selector, kind} by reading
   * the Greenhouse form's labels and their associated inputs. This avoids relying on
   * the LLM to map a question's text back to an element (which was returning null).
   */
  private async collectQuestions(): Promise<Array<{ label: string; id: string; kind: "text" | "select" }>> {
    const page = this.getPage();
    try {
      const rows = await page.evaluate(() => {
        const out: Array<{ label: string; id: string; kind: "text" | "select" }> = [];
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
          const isSelect =
            !!input.closest("[class*='select'], select") ||
            input.getAttribute("role") === "combobox";
          out.push({ label: text, id: forId, kind: isSelect ? "select" : "text" });
        }
        return out;
      });

      // De-duplicate by label, keep later (custom screener section takes priority).
      const seen = new Set<string>();
      const uniq: Array<{ label: string; id: string; kind: "text" | "select" }> = [];
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
      await loc.fill(answer);
      await randomSleep(200, 500);
    }
  }

  /**
   * Fill a react-select question dropdown. Tries candidate distillations of a
   * free-form answer against the rendered options deterministically; when no
   * option matches, asks the LLM to pick the closest option using the real
   * options visible on the page. Leaves the field blank only if both fail.
   */
  private async fillQuestionSelect(id: string, question: string, answer: string): Promise<void> {
    const page = this.getPage();
    try {
      const control = page
        .locator(
          `//div[contains(@class,"select-shell")][.//input[@id="${id}"]]//div[contains(@class,"select__control")]`
        )
        .first();
      await control.click();
      await randomSleep(300, 600);

      for (const candidate of selectCandidates(answer)) {
        const option = page
          .locator(`//div[@id^="react-select"]//div[contains(@role,"option")][${optionContainsXPath(candidate)}]`)
          .first();
        if (await option.isVisible().catch(() => false)) {
          await option.click();
          await randomSleep(300, 600);
          console.log(`[GreenhouseAdapter] Selected option "${candidate}" for #${id}`);
          return;
        }
      }

      console.log(`[GreenhouseAdapter] No deterministic option match for #${id}; trying AI selection...`);
      await this.stagehand.act(
        `Select the best matching option for the question "${escapePromptValue(question)}". ` +
          `The candidate's answer is: "${escapePromptValue(answer)}". Click the matching option.`
      );
      await randomSleep(400, 800);
    } catch (err: any) {
      console.warn(`[GreenhouseAdapter] fillQuestionSelect failed for #${id}:`, err?.message || err);
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

    // Fill explicit customAnswers from profile
    for (const [questionKeyword, answer] of Object.entries(profile.customAnswers)) {
      console.log(`[GreenhouseAdapter] Answering explicit custom question matching "${questionKeyword}"...`);
      await this.safeAct(
        "Type or select %answer% into the field asking about the question matching %keyword%",
        { answer, keyword: questionKeyword }
      );
      await randomSleep(200, 500);
    }

    // RAG Dynamic Screener Extraction & Phase 3 RPC Fill
    if (rpc) {
      console.log("[GreenhouseAdapter] Extracting unanswered custom screener questions...");
      try {
        const extractionResult: any = await (this.stagehand as any).extract(
          "Extract all open-ended questions or dropdown questions that are currently unanswered on this page. Do not include basic fields like Name, Email, Phone, LinkedIn, GitHub, or Resume, and do not include the preferred first name field.",
          z.object({
            unansweredQuestions: z.array(z.string())
          })
        );

        const questions = extractionResult?.unansweredQuestions || [];
        if (questions.length === 0) {
          console.log("[GreenhouseAdapter] No unanswered custom questions found.");
        } else {
          console.log(`[GreenhouseAdapter] Found ${questions.length} unanswered custom questions. Requesting RAG answers from Python...`, questions);
          const ragAnswers: Record<string, string> = await rpc("answer_questions", { questions });

          // Deterministic DOM map of the form's labelled inputs.
          const formFields = await this.collectQuestions();
          const byNorm = new Map<string, { id: string; kind: "text" | "select" }>();
          for (const f of formFields) {
            const n = this.normalise(f.label);
            if (!byNorm.has(n)) byNorm.set(n, f);
          }

          for (const [questionText, answerText] of Object.entries(ragAnswers)) {
            // ASK_USER sentinel: Python could not determine an answer without the
            // user's input. Leave the field blank rather than fabricate.
            if (!answerText || answerText === "N/A" || answerText === "__ASK_USER__") {
              console.log(`[GreenhouseAdapter] Skipping unanswered question "${escapePromptValue(questionText)}" (no known answer)`);
              continue;
            }

            const origNorm = this.normalise(questionText);
            const target = byNorm.get(origNorm);
            if (target) {
              console.log(`[GreenhouseAdapter] Deterministic fill for "${escapePromptValue(questionText)}" (${target.kind})`);
              if (target.kind === "select") {
                await this.fillQuestionSelect(target.id, questionText, answerText);
              } else {
                await this.fillQuestionText(target.id, answerText);
              }
              continue;
            }

            // Fallback: let the LLM locate the field by its question text.
            const looksLikeDropdown = /(select|choose|which|how soon|ready to|comfortable|dropdown)/i.test(questionText);
            console.log(`[GreenhouseAdapter] No DOM match, using AI fill for "${escapePromptValue(questionText)}"`);
            if (looksLikeDropdown) {
              await this.fillDropdown(questionText, answerText);
            } else {
              await this.safeAct(
                "Type or select %answer% for the question asking %question%",
                { answer: escapePromptValue(answerText), question: questionText }
              );
            }
            await randomSleep(200, 500);
          }
        }
      } catch (extractErr) {
        console.warn("[GreenhouseAdapter] Dynamic screener extraction warning:", extractErr);
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
