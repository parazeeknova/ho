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
 * Iterative re-scan walk. Enumerate the form fresh each pass, resolve only
 * fields not seen before, and converge when a pass yields nothing new (or the
 * cap is reached). ``processedKeys`` is the adapter's set of already-handled
 * field keys — the walker marks a field BEFORE asking the caller to resolve it
 * so a re-scan can never re-ask a field the resolver is mid-flight on.
 *
 * Returns the number of passes it took to converge.
 */
export async function walkUntilConverged<T>(
  params: {
    collect: () => Promise<T[]>;
    keyOf: (item: T) => string;
    resolve: (item: T) => Promise<void>;
    processedKeys: Set<string>;
    maxPasses?: number;
    settle?: () => Promise<void>;
    onPass?: (pass: number, freshCount: number, total: number) => void;
  }
): Promise<number> {
  const maxPasses = params.maxPasses ?? 30;
  for (let pass = 0; pass < maxPasses; pass++) {
    const items = await params.collect();
    const fresh = items.filter((item) => !params.processedKeys.has(params.keyOf(item)));
    params.onPass?.(pass + 1, fresh.length, items.length);
    if (fresh.length === 0) {
      return pass + 1;
    }
    for (const item of fresh) {
      // Mark BEFORE resolving so a re-scan can never re-ask or re-fill a field
      // currently being resolved.
      params.processedKeys.add(params.keyOf(item));
      await params.resolve(item);
    }
    if (params.settle) await params.settle();
  }
  return maxPasses;
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
 * Definitive final sweep: re-enumerate the ENTIRE form and resolve any field
 * that is still empty, except identity fields (audited separately) and fields
 * the user deliberately skipped. Iterates until no new field fills, so it
 * converges like the walk. Returns the labels it filled.
 */
export async function sweepUntilStable(params: {
  collect: () => Promise<FormField[]>;
  isEmpty: (field: FormField) => Promise<boolean>;
  resolve: (field: FormField) => Promise<void>;
  userSkippedKeys?: ReadonlySet<string>;
}): Promise<string[]> {
  const filled: string[] = [];
  for (let pass = 0; pass < 3; pass++) {
    const fields = await params.collect();
    let touched = 0;
    for (const field of fields) {
      if (PRE_FILLED_LABELS.has(normalizeOptionText(field.label))) continue;
      if (params.userSkippedKeys?.has(fieldKey(field))) continue;
      if (await params.isEmpty(field)) continue;
      touched += 1;
      filled.push(field.label);
      await params.resolve(field);
    }
    if (touched === 0) break;
  }
  return filled;
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
    if (await params.isEmpty(field)) continue;
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