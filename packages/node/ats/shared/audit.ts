import { randomSleep } from "../../utils/evasion.js";
import { normalizeOptionText, escapePromptValue } from "./matching.js";
import { type FormField, PRE_FILLED_LABELS, fieldKey, isCoverLetterField } from "./model.js";
import { type BlankEntry } from "./screener.js";

/**
 * Adaptive-form audit machinery shared by all adapters. The key insight that
 * made Greenhouse reliable is that conditional questions appear only after an
 * interaction, so the walk must re-scan the DOM each pass and converge when
 * nothing new appears. The same guard-rails and auditing are reused for every
 * form.
 */

/**
 * Find the recorded reason a field was left blank, from one or more
 * resolution/sweep transcripts. Matches on the normalized label.
 */
export function blankReason(
  label: string,
  reasons: ReadonlyArray<BlankEntry> | undefined,
): string | undefined {
  const key = normalizeOptionText(label);
  return reasons?.find((b) => normalizeOptionText(b.label) === key)?.reason;
}

/**
 * Zero-blank audit. Computes which required fields are still empty and — for
 * optional empties — records a generic reason into ``transcript`` when the
 * resolver left no reason. This is pure computation; callers own the logging.
 * Returns the required-blank report.
 */
export async function auditBlanks<T extends { label: string; required: boolean }>(params: {
  fields: T[];
  readValue: (field: T) => Promise<string>;
  transcript: BlankEntry[];
}): Promise<BlankEntry[]> {
  const transcript = params.transcript;
  const required: BlankEntry[] = [];
  for (const field of params.fields) {
    // Cover-letter prompts are handled by the adapter's dedicated path (PDF
    // upload with text fallback), which runs after the walk — never count a
    // still-empty cover-letter textarea as a required blank here.
    if (isCoverLetterField(field)) continue;
    if (await params.readValue(field)) continue;
    const reason = blankReason(field.label, transcript);
    if (field.required) {
      required.push({
        label: field.label,
        reason: reason ?? "blank after walk (no answer committed)",
      });
      continue;
    }
    if (!reason) {
      transcript.push({ label: field.label, reason: "blank after walk (no answer committed)" });
    }
  }
  return required;
}

/**
 * Reverify the ENTIRE form one last time and report every field still empty —
 * required or optional — except identity fields and manual skips. This is the
 * pre-completion checkpoint: it catches anything the walk and sweep missed
 * (including a resume that failed to attach) so the review hold shows the true
 * state. Logs via the tag and returns the still-blank list (empty means full).
 */
export async function finalReverify(params: {
  tag: string;
  collect: () => Promise<FormField[]>;
  isEmpty: (field: FormField) => Promise<boolean>;
  skippedKeys?: ReadonlySet<string>;
  reasons?: ReadonlyArray<BlankEntry>;
}): Promise<BlankEntry[]> {
  const { tag } = params;
  const fields = await params.collect();
  const stillBlank: BlankEntry[] = [];
  for (const field of fields) {
    if (PRE_FILLED_LABELS.has(normalizeOptionText(field.label))) continue;
    if (isCoverLetterField(field)) continue; // dedicated cover-letter path owns it
    if (params.skippedKeys?.has(fieldKey(field))) continue;
    if (!(await params.isEmpty(field))) continue;
    stillBlank.push({
      label: field.label,
      reason:
        blankReason(field.label, params.reasons) ??
        "empty after final reverify (no answer committed)",
    });
  }
  if (stillBlank.length > 0) {
    console.warn(
      `[${tag}] REVERIFY: ${stillBlank.length} field(s) still unfilled after completion:`,
    );
    for (const sb of stillBlank) {
      console.warn(`[${tag}]   unfilled: ${escapeLog(sb.label)} (${sb.reason})`);
    }
  } else {
    console.log(`[${tag}] REVERIFY: every field is filled (only manual skips excluded).`);
  }
  return stillBlank;
}

function escapeLog(val: string): string {
  return val.replace(/"/g, '\\"');
}

/**
 * The result of a submit attempt. `confirmed` is ONLY true when the ATS
 * actually reached a confirmed-submitted state: a success-page URL redirect,
 * or the form leaving the submission state (submit button gone) while a
 * success phrase is rendered. Bare inline body text is NOT confirmation —
 * a static "thank you" string on the still-unsubmitted form page never counts.
 * `retryable` is true when the failure looks like a client-side validation
 * error (an error banner, or the submit button still visible) that a
 * field-recheck + resubmit could fix; it is false when no outcome could be
 * detected at all (the form may have navigated somewhere unexpected, or the
 * page is in an unknown state).
 */
export interface SubmitOutcome {
  confirmed: boolean;
  /** Human-readable reason for a non-confirmed result (banner text or a
   *  description of what was detected). */
  error?: string;
  /** Whether a field-recheck + resubmit is worth attempting. */
  retryable: boolean;
}

/** Success-URL tokens shared across ATS confirmation pages. */
const SUCCESS_URL_RE = /thanks|submitted|confirmation|success|applied|complete|received/i;

/** Inline success phrases. Only trusted as confirmation when the form has
 *  structurally left the submission state (submit button gone) — see
 *  ``verifySubmitOutcome``. */
const CONFIRM_TEXT_RE =
  /application (has been )?(successfully )?submitted|your application was successfully submitted|thank (you|u) for applying|your application has been received|we (have )?received your application|application complete|we have received your application|application received|your application has been submitted|you'?re all set|we'?ll (be )?in touch/i;

/**
 * Verify a submit by polling for a success-page redirect, a structurally-gone
 * submit form with a success phrase, or a visible error banner. This is the
 * ONLY path that declares a submission confirmed — a bare submit click + sleep
 * is never a confirmation.
 *
 * @param page active Playwright/Stagehand page
 * @param opts
 *   - `tag`: log prefix (e.g. "Ashby").
 *   - `successUrlRe`: extra URL tokens for this board (default: shared set).
 *   - `submitButtonSelector`: when provided, a still-visible submit button
 *     after the click is a retryable failure signal (the form never left the
 *     page / validation kept it); when it has GONE and a success phrase is
 *     present, the submit is confirmed (SPA-style inline success).
 *   - `errorSelectors`: selectors for error banners (default shared set).
 *   - `polls`: number of poll iterations (default 10).
 */
export async function verifySubmitOutcome(
  page: any,
  opts: {
    tag: string;
    successUrlRe?: RegExp;
    submitButtonSelector?: string;
    errorSelectors?: string[];
    polls?: number;
    /** Listen for the submit network request and require a 2xx response
     *  before declaring the submission confirmed. When provided, `confirmed`
     *  is only true if BOTH a 2xx submit response AND the page-level success
     *  signal (redirect / success text) are observed. */
    submitResponse?: () => Promise<{ ok: boolean; status?: number } | undefined>;
  },
): Promise<SubmitOutcome> {
  const { tag } = opts;
  const urlRe = opts.successUrlRe ?? SUCCESS_URL_RE;
  const errorSelectors = opts.errorSelectors ?? [
    '[role="alert"]',
    '.error, .error-message, [class*="error"]',
  ];
  const polls = opts.polls ?? 10;

  for (let i = 0; i < polls; i++) {
    const url = page.url();
    if (urlRe.test(url)) {
      // A success-page redirect is only a confirmation when the submit request
      // itself succeeded (2xx). If we cannot observe the response (no hook),
      // the redirect alone stands (backwards compatible).
      if (opts.submitResponse) {
        const resp = await opts.submitResponse().catch(() => undefined);
        if (!resp || !resp.ok) {
          const code = resp?.status ?? "no-response";
          console.error(`[${tag}] Success URL reached but submit response was not 2xx (${code}).`);
          return {
            confirmed: false,
            error: `submit response not 2xx (${code}) despite success-page redirect`,
            retryable: true,
          };
        }
      }
      console.log(`[${tag}] Submitted: success-page redirect confirmed (${url}).`);
      return { confirmed: true, retryable: false };
    }

    // Error banner / validation feedback — retryable (a recheck may fix it).
    for (const sel of errorSelectors) {
      const err = await page
        .locator(sel)
        .first()
        .innerText()
        .catch(() => "");
      const clean = (err || "").trim();
      // Ignore benign upload-size errors that are really an upload hint, not a
      // submission blocker.
      if (clean && !/exceeds? the maximum upload size|too large|100MB/i.test(clean)) {
        console.error(`[${tag}] Submit error banner: ${escapePromptValue(clean)}`);
        return { confirmed: false, error: clean, retryable: true };
      }
    }

    if (opts.submitButtonSelector) {
      const stillVisible = await page
        .locator(opts.submitButtonSelector)
        .first()
        .isVisible()
        .catch(() => false);
      // Structural confirmation: after the first poll the form is gone (submit
      // button no longer present) AND a success phrase is rendered. Stricter
      // than raw inline text — the button must actually be gone, so a static
      // "thank you" string on the un-submitted form page can never confirm.
      // Gated on i >= 1 so a slow re-render right after the click isn't misread.
      if (!stillVisible && i >= 1) {
        const bodyText = await page
          .evaluate(() => document.body?.innerText?.slice(0, 4000) ?? "")
          .catch(() => "");
        if (CONFIRM_TEXT_RE.test(bodyText)) {
          console.log(`[${tag}] Submitted: form gone + inline confirmation text detected.`);
          return { confirmed: true, retryable: false };
        }
      }
      if (stillVisible && i >= 1) {
        // A button that is still on the page after the click usually means
        // validation refused the submit.
        console.warn(`[${tag}] Submit button still visible after click; validation likely failed.`);
        return {
          confirmed: false,
          error: "submit button still visible after submit (validation blocked it)",
          retryable: true,
        };
      }
    }

    await randomSleep(1500, 2000);
  }
  console.error(`[${tag}] Submit outcome not detected at ${page.url()}.`);
  const lastUrl = page.url();
  // Keep the error diagnostic short (and avoid dumping the applicant's own
  // form answers — name/email/work history — into the persisted job error).
  const bodySnip = (
    await page.evaluate(() => document.body?.innerText?.slice(0, 120) ?? "").catch(() => "")
  )
    .replace(/\s+/g, " ")
    .trim();
  return {
    confirmed: false,
    error: `no success-page redirect or error outcome detected after clicking submit (final url: ${lastUrl}; body: ${escapePromptValue(bodySnip)})`,
    retryable: false,
  };
}
