import type { Profile } from "../../types.js";
import { normalizeOptionText } from "./matching.js";

/** A single rendered radio/checkbox option and its click target. */
export interface GroupOption {
  text: string;
  name: string;
  value: string;
  /** The underlying input's id (used to click its label[for] precisely). */
  id?: string;
  /** True when the option is a toggle BUTTON (Ashby yes/no rows), not an input. */
  button?: boolean;
}

/** A single form question captured by an adapter's DOM walker. */
export interface FormField {
  label: string;
  /** Primary element id (text/select) or a stable group anchor (radio/checkbox). */
  id: string;
  kind:
    | "text"
    | "select"
    | "multi"
    | "radio"
    | "checkbox"
    | "combobox"
    /** A date picker (react-datepicker): value must be a real date, and
     *  free-text answers like "immediately" are translated to a date. */
    | "date";
  required: boolean;
  options: string[];
  /** Radio/checkbox option click targets (text + input name/value). */
  optionTargets: GroupOption[];
  /** Canonical field name (snake_case) when known from an authoritative model. */
  name?: string;
}

/** A question as described by an ATS's embedded JSON question model. */
export interface JsonFieldSource {
  name: string;
  label: string;
  kind: string;
  required: boolean;
  options: string[];
}

/**
 * Structural semantics for a checkbox field — label-agnostic so ANY phrasing
 * is handled correctly without a phrase list:
 *  - "accept": a REQUIRED checkbox with a single option is an acceptance/consent
 *    gate (privacy policy, code of conduct, "I agree…"). It gates submission,
 *    so agreeing is the only way to apply.
 *  - "leave": an OPTIONAL checkbox with a single option is an opt-in preference
 *    (marketing, newsletters). Never presume consent; leave it unchecked.
 *  - "ask": a multi-option checkbox is a real multi-select question and must be
 *    resolved normally (profile/KB/Telegram), never auto-accepted.
 */
export type CheckboxAction = "accept" | "leave" | "ask";

export function checkboxAction(field: FormField): CheckboxAction {
  if (field.kind !== "checkbox") return "ask";
  const single = field.optionTargets.length <= 1;
  if (!single) return "ask"; // multi-option = real multi-select question
  return field.required ? "accept" : "leave";
}

/**
 * True only for a genuine async location autocomplete — a react-select that
 * has NO static options (a geocoder loads them after typing) and whose label is
 * about the candidate's current city (Location (City), Candidate Location, What
 * is your current location?). A pick-list that merely mentions location (e.g.
 * "…willing to relocate to the job's location?") is NOT one — it has options
 * and must be answered by selecting, never by typing.
 */
export function isLocationAutocomplete(field: FormField): boolean {
  if (field.kind === "radio" || field.kind === "checkbox") return false;
  if (field.kind !== "select" && field.kind !== "multi") return false;
  if (field.options.length > 0) return false; // static pick-list
  if (field.optionTargets.length > 0) return false;
  const label = normalizeOptionText(field.label);
  const isLocationSubject =
    /\bcurrent location\b/.test(label) ||
    /\bcandidate location\b/.test(label) ||
    /\b(?:city|location)\b/.test(label);
  if (!isLocationSubject) return false;
  // A willingness/relocation question ("…willing to relocate to the job's
  // location?", "Are you currently living in…") is NOT a city autocomplete —
  // it is a yes/no pick-list.
  if (/relocat|willing|are you currently|are you open|job's location|job’s location/.test(label)) {
    return false;
  }
  return true;
}

/** Stable identity for a form field across rescans (label + kind + id). */
export function fieldKey(f: FormField): string {
  return `${normalizeOptionText(f.label)}|${f.kind}|${f.id}`;
}

/**
 * The subset of ``fields`` that has not been processed yet, as a pure
 * diff over the processed-key set. Used by the iterative re-scan walk so
 * fields revealed only after an interaction (conditional questions) are
 * picked up in a later pass while already-processed fields are never
 * re-asked or re-filled.
 */
export function unprocessedFields(
  fields: FormField[],
  processedKeys: ReadonlySet<string>,
): FormField[] {
  return fields.filter((f) => !processedKeys.has(fieldKey(f)));
}

/**
 * Merge an authoritative JSON question model with the DOM enumeration.
 *
 * The DOM is the BASE inventory: only rendered questions are walked and filled.
 * The JSON model ENRICHES the DOM fields with canonical names, exact option
 * values and required flags (matched by field name, then by normalized label).
 * JSON-only fields are dropped — boards often list questions (e.g. EEOC Race)
 * that they never render, and walking phantom fields fails verification loudly
 * for no reason.
 */
export function mergeFormInventory(
  jsonFields: JsonFieldSource[] | null,
  domFields: FormField[],
): FormField[] {
  const out: FormField[] = [];
  const seen = new Set<string>();
  const add = (f: FormField) => {
    const key = `${normalizeOptionText(f.label)}|${f.kind}`;
    if (seen.has(key)) return;
    seen.add(key);
    out.push(f);
  };

  const jsonByName = new Map<string, JsonFieldSource>();
  const jsonByLabel = new Map<string, JsonFieldSource>();
  for (const jf of jsonFields ?? []) {
    if (jf.kind === "input_file") continue; // uploads handled by dedicated paths
    if (/^resume(_text)?$|^cover_letter(_text)?$/.test(jf.name)) continue;
    jsonByName.set(jf.name, jf);
    jsonByLabel.set(normalizeLabel(jf.label), jf);
  }

  for (const df of domFields) {
    const jf =
      (df.name ? jsonByName.get(df.name) : undefined) || jsonByLabel.get(normalizeLabel(df.label));
    if (jf) {
      add({
        ...df,
        required: df.required || jf.required,
        options: jf.options.length ? jf.options : df.options,
        name: jf.name,
      });
    } else {
      add(df);
    }
  }
  return out;
}

function normalizeLabel(label: string): string {
  return normalizeOptionText(label);
}

/** Deterministic profile-driven fills keyed by normalized question label. */
export const PROFILE_FILLS: Record<string, keyof Profile> = {
  "preferred first name": "preferredName",
  linkedin: "linkedin",
  "linkedin profile": "linkedin",
  "linkedin url": "linkedin",
  github: "github",
  "github profile": "github",
  "github url": "github",
  website: "website",
  portfolio: "website",
  "portfolio url": "website",
  "your website": "website",
};

/** Identity fields filled deterministically from the profile (never asked). */
export const IDENTITY_FILLS: Record<string, keyof Profile> = {
  "first name": "firstName",
  "legal first name": "firstName",
  firstname: "firstName",
  "given name": "firstName",
  "given name(s)": "firstName",
  "given names": "firstName",
  "last name": "lastName",
  "legal last name": "lastName",
  lastname: "lastName",
  surname: "lastName",
  "family name": "lastName",
  "family name(s)": "lastName",
  "family names": "lastName",
  "legal name": "lastName",
  "local given name(s)": "firstName",
  "local given name": "firstName",
  "local family name(s)": "lastName",
  "local family name": "lastName",
  email: "email",
  "email address": "email",
  "email address (username)": "email",
  "e-mail address": "email",
  "email id": "email",
  "your email": "email",
  "what is your email address?": "email",
  phone: "phone",
  "phone number": "phone",
  "mobile phone": "phone",
  "mobile phone number": "phone",
  "cell phone": "phone",
  "contact number": "phone",
  "primary phone number": "phone",
  "phone (e.g. +91 99999 99999)": "phone",
};

/** Questions answered by the fixed deterministic identity fills before the walk. */
export const PRE_FILLED_LABELS = new Set(["first name", "last name", "email", "phone"]);

/**
 * True when a field is driven deterministically from the profile (identity /
 * profile fills) rather than resolved by the walk. These are typically
 * pre-filled before the walk begins, so the walk must never re-process them.
 */
export function isProfileDrivenField(f: FormField): boolean {
  const key = normalizeBlankLabel(f.label);
  return key in IDENTITY_FILLS || key in PROFILE_FILLS;
}

/**
 * True for a free-text cover-letter prompt (a textarea asking for a cover
 * letter or an open "anything you would like us to know" blurb). Boards with a
 * dedicated cover-letter path (greenhouse/generic/workday) generate the letter
 * ONCE via the "cover_letter" RPC (PDF upload with text fallback) — so the walk
 * must never resolve these via "answer_question", which would both burn a
 * second LLM generation and pre-fill the textarea before the PDF path runs.
 */
export function isCoverLetterField(f: { label: string; kind?: string }): boolean {
  const kind = f.kind ?? "text";
  if (kind === "select" || kind === "multi" || kind === "radio" || kind === "checkbox") {
    return false;
  }
  const label = normalizeBlankLabel(f.label);
  return (
    /cover letter|cover_letter/.test(label) ||
    /^additional information$/.test(label) ||
    /anything else you|more about you|tell us about yourself|anything you would like/.test(label)
  );
}

function normalizeBlankLabel(label: string): string {
  return (label || "").replace(/\s+/g, " ").trim().toLowerCase();
}
