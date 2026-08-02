import { FormField, PRE_FILLED_LABELS, fieldKey } from "./model.js";
import { BlankEntry } from "./screener.js";
import { normalizeOptionText } from "./matching.js";

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