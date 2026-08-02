import { randomSleep } from "../../utils/evasion.js";
import type { RpcHelper } from "../base.js";
import type { Profile } from "../../types.js";
import { FormControls } from "./controls.js";
import {
  checkboxAction,
  fieldKey,
  FormField,
  IDENTITY_FILLS,
  isLocationAutocomplete,
  PROFILE_FILLS,
} from "./model.js";
import { escapePromptValue } from "./matching.js";

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

/**
 * Resolve and fill one question field via the shared resolution chain:
 * identity/profile value → checkbox structural rules → async location
 * autocomplete → RPC answer (KB/Telegram) → kind-aware fill → committed-value
 * verification. Records results in ``filled``/``blanked``; a user-declined
 * skip is remembered in ``userSkippedKeys`` so the final sweep never re-asks
 * a manually-skipped field.
 */
export class Screener {
  constructor(
    protected controls: FormControls,
    protected tagName: string,
    protected profile: Profile,
    protected rpc: RpcHelper
  ) {}

  async process(
    field: FormField,
    filled: string[],
    blanked: BlankEntry[],
    userSkippedKeys: Set<string>
  ): Promise<void> {
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
            `[${this.controls.tagName}] Leaving "${escapePromptValue(field.label)}" blank (${reason})`
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
          `[${this.controls.tagName}] Optional opt-in checkbox "${escapePromptValue(field.label)}" left unchecked.`
        );
        return;
      }
      if (action === "accept") {
        const option = field.optionTargets[0]?.text ?? "yes";
        const ok = await this.controls.clickGroupOption(field, option);
        const committed = await this.controls.readFieldValue(field);
        if (ok && committed) {
          console.log(
            `[${this.controls.tagName}] Accepted required checkbox "${escapePromptValue(field.label)}".`
          );
          filled.push(field.label);
          await randomSleep(150, 300);
          return;
        }
        const reason = "required acceptance checkbox could not be checked";
        console.warn(
          `[${this.controls.tagName}] Leaving "${escapePromptValue(field.label)}" blank (${reason})`
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
          `[${this.controls.tagName}] Leaving "${escapePromptValue(field.label)}" blank (${reason})`
        );
        blanked.push({ label: field.label, reason });
        return;
      }
      console.log(
        `[${this.controls.tagName}] Location filled for "${escapePromptValue(field.label)}": "${escapePromptValue(ans)}"`
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
            `("${escapePromptValue(field.label)}"); resolving without them.`
        );
      }
    }

    const rpcKind =
      field.kind === "radio"
        ? "select"
        : field.kind === "checkbox"
          ? "multi"
          : field.kind;

    let result: any;
    try {
      result = await this.rpc("answer_question", {
        question: field.label,
        kind: rpcKind,
        options: optionTexts,
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
            "leaving blank and continuing."
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
        rpcErr?.message || rpcErr
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
      ok = ok && !!committed;
    }

    if (!ok) {
      const reason = `answer "${escapePromptValue(answer)}" could not be committed`;
      console.warn(
        `[${this.controls.tagName}] Leaving "${escapePromptValue(field.label)}" blank (${reason})`
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