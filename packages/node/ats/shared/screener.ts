import type { Profile } from "../../types.js";
import { randomSleep } from "../../utils/evasion.js";
import type { RpcHelper } from "../base.js";
import { FormControls } from "./controls.js";
import { escapePromptValue, valuesConsistent } from "./matching.js";
import {
  checkboxAction,
  fieldKey,
  type FormField,
  IDENTITY_FILLS,
  isCoverLetterField,
  isLocationAutocomplete,
  PROFILE_FILLS,
} from "./model.js";

/** A question left blank during resolution, with the reason it was skipped. */
export interface BlankEntry {
  label: string;
  reason: string;
}

// Deferral count shared across every adapter run in this process. A question
// deferred overnight leaves its field blank, so submission must never happen
// for that job — the runner reads this after fill() to decide whether to
// abort instead of submitting an incomplete application.
let deferredFieldCount = 0;

export function resetDeferredFieldCount(): void {
  deferredFieldCount = 0;
}

export function getDeferredFieldCount(): number {
  return deferredFieldCount;
}

// Count of REQUIRED fields left blank after the fill/sweep/reverify pass. A
// required field that failed to commit (or a skipped radio, unchecked consent
// box, etc.) must never be submitted — the runner reads this alongside the
// deferred count to gate auto-submit. Set by each adapter's fill() from its
// final required-blank audit.
let blankedRequiredCount = 0;

export function resetBlankedRequiredCount(): void {
  blankedRequiredCount = 0;
}

export function setBlankedRequiredCount(n: number): void {
  blankedRequiredCount = Math.max(0, Math.floor(n));
}

export function getBlankedRequiredCount(): number {
  return blankedRequiredCount;
}

/**
 * Resolve and fill one question field via the shared resolution chain:
 * identity/profile value → checkbox structural rules → async location
 * autocomplete → RPC answer (KB/Telegram) → kind-aware fill → committed-value
 * verification. Records results in ``filled``/``blanked``; a user-declined
 * skip is remembered in ``userSkippedKeys`` so the final sweep never re-asks
 * a manually-skipped field.
 */
export class Screener {
  protected controls: FormControls;
  protected tagName: string;
  protected profile: Profile;
  protected rpc: RpcHelper;
  /** Boards with a dedicated cover-letter path skip cover-letter fields in
   *  the walk — those are generated once via the "cover_letter" RPC. */
  protected skipCoverLetterFields: boolean;
  /** Answers pre-resolved in one batch RPC, keyed by normalized label. */
  protected batchCache: Map<string, { answer: string; source: string }> = new Map();

  constructor(
    controls: FormControls,
    tagName: string,
    profile: Profile,
    rpc: RpcHelper,
    skipCoverLetterFields = false,
  ) {
    this.controls = controls;
    this.tagName = tagName;
    this.profile = profile;
    this.rpc = rpc;
    this.skipCoverLetterFields = skipCoverLetterFields;
  }

  /**
   * Batch-resolve a set of fields' answers in one RPC round-trip (one LLM call
   * for the whole form instead of one per field). Fields that answer from the
   * profile, structural checkbox rules, or async location are excluded — only
   * fields that would otherwise fall through to the per-field answer_question
   * RPC are sent. Results are cached by normalized label; process() consults
   * the cache before issuing an individual RPC.
   */
  async preResolveBatch(fields: FormField[]): Promise<void> {
    const specs: Array<{ question: string; kind: string; options: string[]; required: boolean }> =
      [];
    for (const field of fields) {
      if (this.skipCoverLetterFields && isCoverLetterField(field)) continue;
      const key = normalizeQuestionLabel(field.label);
      // Already covered deterministically — skip.
      const profileKey = PROFILE_FILLS[key] ?? IDENTITY_FILLS[key];
      if (profileKey && (this.profile as any)?.[profileKey]) continue;
      if (field.kind === "checkbox" && checkboxAction(field) !== "ask") continue;
      if (isLocationAutocomplete(field) && (this.profile as any)?.location) continue;
      let optionTexts = field.options.slice();
      if ((field.kind === "select" || field.kind === "multi") && optionTexts.length === 0) {
        // Read options now so the batch gets the same ground truth a per-field
        // walk would. Best-effort: if reading fails, resolve without them.
        optionTexts = await this.controls.readSelectOptions(field.id).catch(() => []);
      }
      specs.push({
        question: field.label,
        kind: field.kind === "radio" ? "select" : field.kind === "checkbox" ? "multi" : field.kind,
        options: optionTexts,
        required: !!field.required,
      });
    }
    if (specs.length === 0) return;
    try {
      const result: any = await this.rpc("answer_questions_batch", { questions: specs });
      const answers: Record<string, string> = result?.answers ?? {};
      for (const s of specs) {
        const ans = (answers[s.question] ?? "").toString().trim();
        if (ans && ans !== "__ASK_USER__") {
          this.batchCache.set(normalizeQuestionLabel(s.question), { answer: ans, source: "kb" });
        }
      }
      console.log(
        `[${this.tagName}] Batch-resolved ${specs.length} question(s) in one RPC ` +
          `(${this.batchCache.size} answered from cache).`,
      );
    } catch (err: any) {
      // Fall back to per-field resolution on any batch failure.
      console.warn(
        `[${this.tagName}] Batch pre-resolve failed; using per-field resolution: ` +
          `${err?.message || err}`,
      );
    }
  }

  async process(
    field: FormField,
    filled: string[],
    blanked: BlankEntry[],
    userSkippedKeys: Set<string>,
  ): Promise<void> {
    // Cover-letter prompts are handled by the adapter's dedicated path (PDF
    // upload with text fallback) — never resolved here. Silently skipped: no
    // blanked entry, since the dedicated path fills it after the walk.
    if (this.skipCoverLetterFields && isCoverLetterField(field)) {
      console.log(
        `[${this.controls.tagName}] Cover-letter field "${escapePromptValue(field.label)}" ` +
          "left for the dedicated cover-letter path.",
      );
      return;
    }

    const key = normalizeQuestionLabel(field.label);

    // Identity + profile-driven fields are filled deterministically from the
    // profile — never resolved via RPC or asked.
    const profileKey = PROFILE_FILLS[key] ?? IDENTITY_FILLS[key];
    if (profileKey) {
      const pv = (this.profile as any)?.[profileKey];
      if (pv) {
        const ok = await this.controls.fillByKind(field, String(pv));
        if (!ok) {
          const reason = `profile value "${escapePromptValue(String(pv))}" could not be committed`;
          console.warn(
            `[${this.controls.tagName}] Leaving "${escapePromptValue(field.label)}" blank (${reason})`,
          );
          blanked.push({ label: field.label, reason });
          return;
        }
        filled.push(field.label);
        await randomSleep(150, 300);
        return;
      }
    }

    // Structural checkbox semantics — label-agnostic, so any phrasing works:
    //  - required single-option checkbox = an acceptance/consent gate (privacy
    //    policy, code of conduct, "I agree…"): it gates submission, so agree.
    //  - optional single-option checkbox = an opt-in preference (marketing,
    //    newsletters): never presume consent, leave unchecked.
    //  - multi-option checkbox = a real multi-select question: resolve normally.
    if (field.kind === "checkbox") {
      const action = checkboxAction(field);
      if (action === "leave") {
        console.log(
          `[${this.controls.tagName}] Optional opt-in checkbox "${escapePromptValue(field.label)}" left unchecked.`,
        );
        return;
      }
      if (action === "accept") {
        const option = field.optionTargets[0]?.text ?? "yes";
        const ok = await this.controls.clickGroupOption(field, option);
        const committed = await this.controls.readFieldValue(field);
        if (ok && committed) {
          console.log(
            `[${this.controls.tagName}] Accepted required checkbox "${escapePromptValue(field.label)}".`,
          );
          filled.push(field.label);
          await randomSleep(150, 300);
          return;
        }
        const reason = "required acceptance checkbox could not be checked";
        console.warn(
          `[${this.controls.tagName}] Leaving "${escapePromptValue(field.label)}" blank (${reason})`,
        );
        blanked.push({ label: field.label, reason });
        return;
      }
      // multi-option checkbox: fall through to normal resolution.
    }

    // The ONLY place typing is allowed is a genuine async location autocomplete
    // (no static options + anchored label). Everything else is answered by
    // selecting an option, never by typing into the field.
    const isAsyncLocation = isLocationAutocomplete(field);
    if (isAsyncLocation && (this.profile as any)?.location) {
      const ans = String((this.profile as any).location);
      let ok = await this.controls.fillAsyncAutocomplete(field.id, ans);
      if (!ok) {
        ok = await this.controls.fillAsyncAutocomplete(field.id, ans);
      }
      if (!ok) {
        const reason = `location "${escapePromptValue(ans)}" had no selectable suggestion`;
        console.warn(
          `[${this.controls.tagName}] Leaving "${escapePromptValue(field.label)}" blank (${reason})`,
        );
        blanked.push({ label: field.label, reason });
        return;
      }
      console.log(
        `[${this.controls.tagName}] Location filled for "${escapePromptValue(field.label)}": "${escapePromptValue(ans)}"`,
      );
      filled.push(field.label);
      await randomSleep(150, 300);
      return;
    }

    let optionTexts = field.options.slice();
    if ((field.kind === "select" || field.kind === "multi") && optionTexts.length === 0) {
      optionTexts = await this.controls.readSelectOptions(field.id);
      if (optionTexts.length === 0) {
        console.warn(
          `[${this.controls.tagName}] Could not read options for #${field.id} ` +
            `("${escapePromptValue(field.label)}"); resolving without them.`,
        );
      }
    }

    const rpcKind =
      field.kind === "radio" ? "select" : field.kind === "checkbox" ? "multi" : field.kind;

    // Batch cache hit: an answer pre-resolved in one form-wide RPC. Use it
    // directly (fill + verify) without a second per-field RPC round-trip.
    const cached = this.batchCache.get(normalizeQuestionLabel(field.label));
    if (cached && cached.answer) {
      let ok: boolean;
      if (isAsyncLocation) {
        ok = await this.controls.fillAsyncAutocomplete(field.id, cached.answer);
        if (!ok) {
          ok = await this.controls.fillAsyncAutocomplete(field.id, cached.answer);
        }
        const committed = await this.controls.readSelectValue(field.id);
        ok = ok && !!committed;
      } else {
        ok = await this.controls.fillByKind(field, cached.answer, optionTexts);
        if (!ok) {
          ok = await this.controls.fillByKind(field, cached.answer, optionTexts);
        }
      }
      if (ok) {
        filled.push(field.label);
        await randomSleep(150, 300);
      } else {
        blanked.push({
          label: field.label,
          reason: `batch answer "${cached.answer}" not committable`,
        });
      }
      return;
    }

    let result: any;
    try {
      result = await this.rpc("answer_question", {
        question: field.label,
        kind: rpcKind,
        options: optionTexts,
        // Whether the form marks this field required (its label has the
        // asterisk). Overnight, an UNRESOLVED optional question is skipped
        // rather than deferred; the Python side decides with this flag.
        required: !!field.required,
      });
    } catch (rpcErr: any) {
      // A deferred question (overnight, no human) must NOT abort the whole
      // run: the Python side has already recorded it for the morning digest
      // and prompted the user. Skip this field, keep it out of the final
      // sweep, and continue filling the rest of the form.
      if (/(AUTOFILL_DEFER|DEFER)/.test(rpcErr?.message || String(rpcErr))) {
        deferredFieldCount += 1;
        console.warn(
          `[${this.controls.tagName}] Question "${escapePromptValue(field.label)}" deferred for user input; ` +
            "leaving blank and continuing.",
        );
        userSkippedKeys.add(fieldKey(field));
        blanked.push({
          label: field.label,
          reason: "deferred for user input (resume via the CLI)",
        });
        return;
      }
      // Real RPC failures (Telegram unconfigured, transport) must abort loudly —
      // filling a form around an unanswered personal question is worse than no fill.
      console.error(
        `[${this.controls.tagName}] RPC answer_question failed for "${escapePromptValue(field.label)}":`,
        rpcErr?.message || rpcErr,
      );
      throw rpcErr;
    }

    const answer: string = (result?.answer ?? "").toString().trim();
    if (!answer) {
      // User dismissed (source "decline"): honor the manual skip — record it
      // so the final sweep never re-asks or re-fills this field.
      userSkippedKeys.add(fieldKey(field));
      blanked.push({
        label: field.label,
        reason: `declined (source ${result?.source ?? "unknown"})`,
      });
      return;
    }

    let ok: boolean;
    if (isAsyncLocation) {
      ok = await this.controls.fillAsyncAutocomplete(field.id, answer);
      if (!ok) {
        ok = await this.controls.fillAsyncAutocomplete(field.id, answer);
      }
      const committed = await this.controls.readSelectValue(field.id);
      ok = ok && !!committed;
    } else {
      ok = await this.controls.fillByKind(field, answer, optionTexts);
      if (!ok) {
        // One retry, then verify the committed value.
        ok = await this.controls.fillByKind(field, answer, optionTexts);
      }
      const committed = await this.controls.readFieldValue(field);
      // Presence alone is not enough: an option group click can commit a
      // DIFFERENT option than the answer resolved to (e.g. the first radio in
      // a Yes/No group when the "No" target was missed). Only count the field
      // as filled when the committed value is consistent with the answer.
      const isOptionKind =
        field.kind === "radio" ||
        field.kind === "checkbox" ||
        field.kind === "select" ||
        field.kind === "multi";
      ok = ok && !!committed && (!isOptionKind || valuesConsistent(answer, committed));
    }

    if (!ok) {
      const reason = `answer "${escapePromptValue(answer)}" could not be committed`;
      console.warn(
        `[${this.controls.tagName}] Leaving "${escapePromptValue(field.label)}" blank (${reason})`,
      );
      blanked.push({ label: field.label, reason });
      await this.controls.closeMenu();
      return;
    }
    filled.push(field.label);
    await randomSleep(150, 300);
  }
}

function normalizeQuestionLabel(label: string): string {
  return label.replace(/\s+/g, " ").trim().toLowerCase();
}
