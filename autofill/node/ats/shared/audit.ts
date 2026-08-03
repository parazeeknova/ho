import { FormField, PRE_FILLED_LABELS, fieldKey } from "./model.js";
import { BlankEntry } from "./screener.js";
import { normalizeOptionText, escapePromptValue } from "./matching.js";
import { randomSleep } from "../../utils/evasion.js";

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
  reasons: ReadonlyArray<BlankEntry> | undefined
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
    if (await params.readValue(field)) continue;
    const reason = blankReason(field.label, transcript);
    if (field.required) {
      required.push({ label: field.label, reason: reason ?? "blank after walk (no answer committed)" });
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
    console.warn(`[${tag}] REVERIFY: ${stillBlank.length} field(s) still unfilled after completion:`);
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
 * actually reached a confirmation state (success-URL redirect or inline
 * confirmation text). `retryable` is true when the failure looks like a
 * client-side validation error (an error banner, or the submit button still
 * visible) that a field-recheck + resubmit could fix; it is false when no
 * outcome could be detected at all (the form may have navigated somewhere
 * unexpected, or the page is in an unknown state).
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
const SUCCESS_URL_RE = /thanks|submitted|confirmation|success|applied|complete/i;

/** Inline confirmation phrases shared across ATS success pages. */
const CONFIRM_TEXT_RE =
  /application (has been )?(successfully )?submitted|thank (you|u) for applying|your application has been received|we (have )?received your application|application complete|we have received your application/i;

/**
 * Verify a submit by polling for a success URL, inline confirmation text, or
 * a visible error banner. This is the ONLY path that declares a submission
 * confirmed — adapters must never report `submitted` on a bare click + sleep.
 *
 * @param page active Playwright/Stagehand page
 * @param opts
 *   - `tag`: log prefix (e.g. "Ashby").
 *   - `successUrlRe`: extra URL tokens for this board (default: shared set).
 *   - `submitButtonSelector`: when provided, a still-visible submit button
 *     after the click is a retryable failure signal (the form never left the
 *     page / validation kept it).
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
  }
): Promise<SubmitOutcome> {
  const { tag } = opts;
  const urlRe = opts.successUrlRe ?? SUCCESS_URL_RE;
  const errorSelectors =
    opts.errorSelectors ?? ['[role="alert"]', '.error, .error-message, [class*="error"]'];
  const polls = opts.polls ?? 10;

  for (let i = 0; i < polls; i++) {
    const url = page.url();
    if (urlRe.test(url)) {
      console.log(`[${tag}] Submitted: redirect confirmed (${url}).`);
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

    // Inline confirmation text (some ATS show success without a URL change).
    const bodyText = await page
      .evaluate(() => document.body?.innerText?.slice(0, 4000) ?? "")
      .catch(() => "");
    if (CONFIRM_TEXT_RE.test(bodyText)) {
      console.log(`[${tag}] Submitted: inline confirmation text detected.`);
      return { confirmed: true, retryable: false };
    }

    // Board-specific fast signal: submit button still present = not submitted.
    if (opts.submitButtonSelector) {
      const stillVisible = await page
        .locator(opts.submitButtonSelector)
        .first()
        .isVisible()
        .catch(() => false);
      if (stillVisible) {
        // A button that is still on the page after the click usually means
        // validation refused the submit. Double-check it is not a disabled
        // re-render — if we are past the first poll and the button persists,
        // treat it as a retryable validation failure.
        if (i >= 1) {
          console.warn(`[${tag}] Submit button still visible after click; validation likely failed.`);
          return {
            confirmed: false,
            error: "submit button still visible after submit (validation blocked it)",
            retryable: true,
          };
        }
      }
    }

    await randomSleep(1500, 2000);
  }
  console.error(`[${tag}] Submit outcome not detected at ${page.url()}.`);
  return {
    confirmed: false,
    error: "no success or error outcome detected after clicking submit",
    retryable: false,
  };
}