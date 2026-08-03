/**
 * Pure text/option matching helpers shared by all ATS adapters. No page or
 * Stagehand dependencies: each function operates on plain strings/lists so
 * they are unit-testable in isolation.
 */

/** Escape a value so it can be embedded in a prompt string safely. */
export function escapePromptValue(val: string): string {
  return val.replace(/\\/g, "\\\\").replace(/"/g, '\\"');
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
export function optionExactXPath(text: string): string {
  const needle = text.toLowerCase();
  return (
    'translate(normalize-space(.), "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz") = ' +
    xpathStringLiteral(needle)
  );
}

/**
 * Build an XPath 1.0 string literal for arbitrary text. XPath has no backslash
 * escapes: single quotes are fine inside a double-quoted literal, and text
 * containing double quotes switches to single quotes (or ``concat`` when it
 * contains both). Never append a backslash — it becomes a literal character.
 */
export function xpathStringLiteral(text: string): string {
  if (!text.includes('"')) {
    return `"${text}"`;
  }
  if (!text.includes("'")) {
    return `'${text}'`;
  }
  const parts = text.split('"').map((p) => `'${p}'`);
  return `concat(${parts.join(', \'"\', ')})`;
}

export function normalizeOptionText(text: string): string {
  return text.replace(/\s+/g, " ").trim().toLowerCase();
}

/**
 * "I don't wish to answer" style options are the user-decline choices of an
 * EEOC-style survey. They are never valid targets for a definite answer, so
 * substring matching must not resolve to them. Covers Greenhouse's
 * "I do not want to answer" phrasing too.
 */
export function isDeclineOption(text: string): boolean {
  return /(don'?t wish|do not wish|prefer not|choose not|rather not|not wish|do not want to answer|not want to answer)/i.test(text);
}

/**
 * Levenshtein edit distance (iterative, unicode-safe). Used to forgive a small
 * typo when matching a typed answer to an option label — never enough to jump
 * between distinct options.
 */
export function editDistance(a: string, b: string): number {
  const x = a ?? "";
  const y = b ?? "";
  if (x === y) return 0;
  if (!x.length) return y.length;
  if (!y.length) return x.length;
  const prev = Array.from({ length: y.length + 1 }, (_, j) => j);
  for (let i = 1; i <= x.length; i++) {
    const curr = [i];
    for (let j = 1; j <= y.length; j++) {
      const cost = x[i - 1] === y[j - 1] ? 0 : 1;
      curr.push(Math.min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost));
    }
    prev.splice(0, prev.length, ...curr);
  }
  return prev[y.length];
}

/**
 * Pick the option text that best matches an answer, from a candidate list.
 * Exact matches are preferred; a substring match is only accepted when it is
 * unambiguous within the option list (and never against a decline option);
 * finally a conservative edit-distance match forgives a small typo when
 * exactly one option is within tolerance. Returns null when nothing matches
 * confidently — callers must leave the field blank rather than guess.
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
    const threshold = Math.max(1, Math.min(2, Math.floor(nc.length / 4)));
    const near = eligible.filter((t) => {
      const tokens = normalizeOptionText(t).split(/\s+/).filter(Boolean);
      return tokens.some((tok) => tok && editDistance(nc, tok) <= threshold);
    });
    if (near.length === 1) return near[0];
  }
  return null;
}

/**
 * True when a committed form value is consistent with the intended answer.
 * Mirrors the matching rules used to pick options (exact, then substring,
 * then a forgiving edit distance) so a value that the option resolver would
 * have chosen counts as committed, while a different option (e.g. "No"
 * committed where the answer was "Yes") is flagged. Empty strings never match.
 */
export function valuesConsistent(answer: string, committed: string): boolean {
  const a = normalizeOptionText(answer);
  const c = normalizeOptionText(committed);
  if (!a || !c) return false;
  if (a === c) return true;
  // The answer is a leading token/phrase of the committed option text
  // ("No" vs "No, I will require immediate visa sponsorship").
  if (c.includes(a)) return true;
  if (a.includes(c)) return true;
  const threshold = Math.max(1, Math.min(2, Math.floor(a.length / 4)));
  const tokens = a.split(/\s+/).filter(Boolean);
  return tokens.some((tok) => tok && editDistance(tok, c) <= threshold);
}

/** CSS attribute-value escaping (values are alphanumeric in practice). */
export function cssEscape(text: string): string {
  return (text || "").replace(/\\/g, "\\\\").replace(/"/g, '\\"');
}

/**
 * Translate a free-text availability answer into a concrete Date. Handles
 * "immediately"/"asap"/"now" (today), "in N days/weeks/months",
 * "within N ...", and "N day(s)/week(s)/month(s)". Returns null when the
 * answer is not date-like (callers leave the field blank rather than guess).
 */
export function translateToDate(answer: string): Date | null {
  const a = (answer || "").trim();
  if (!a) return null;
  const low = a.toLowerCase();
  if (/^(immediately|immeditely|immediate|asap|now|right away|today)$/.test(low)) {
    return new Date();
  }
  const m = low.match(
    /^(?:in\s+|within\s+)?(\d+|one|two|three|a)\s+(day|week|month|days|weeks|months|wk|wks|mo)\b/
  );
  if (m) {
    const n = { one: 1, two: 2, three: 3, a: 1 }[m[1]] ?? parseInt(m[1], 10);
    if (Number.isFinite(n) && n > 0) {
      const unit = m[2].toLowerCase();
      const d = new Date();
      if (/day/.test(unit)) d.setDate(d.getDate() + n);
      else if (/week|wk/.test(unit)) d.setDate(d.getDate() + n * 7);
      else if (/month|mo/.test(unit)) d.setMonth(d.getMonth() + n);
      return d;
    }
  }
  // A value that is already an ISO or MM/DD/YYYY date.
  const iso = a.match(/^(\d{4})-(\d{1,2})-(\d{1,2})/);
  if (iso) return new Date(+iso[1], +iso[2] - 1, +iso[3]);
  const md = a.match(/^(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})/);
  if (md) {
    const yy = md[3].length === 2 ? 2000 + +md[3] : +md[3];
    return new Date(yy, +md[1] - 1, +md[2]);
  }
  // A bare year (e.g. "2027") — common for "Expected graduation year"
  // fields. Default to December 31 of that year.
  const yr = a.match(/^(19|20)\d{2}$/);
  if (yr) return new Date(+yr[0], 11, 31);
  return null;
}

/**
 * CSS selector targeting an element by id. The bare `#id` form is invalid CSS
 * when the id starts with a digit (very common for ATS question ids like
 * `3e05737b-…`), which makes `querySelector('#3e…')` throw and locators
 * silently match nothing. The attribute form `[id="…"]` is valid for ANY id,
 * so it is always used instead.
 */
export function cssIdLocator(id: string): string {
  return `[id="${cssEscape(id)}"]`;
}

/**
 * Pick the best location suggestion for a free-form answer. Exact matches
 * win; otherwise the first ranked suggestion that shares the answer's
 * leading token(s) is chosen (e.g. "Bhopal, India" -> "Bhopal, Madhya
 * Pradesh, India").
 */
export function pickLocationOption(answer: string, opts: string[]): string | null {
  if (!opts.length) return null;
  const exact = opts.find((o) => normalizeOptionText(o) === normalizeOptionText(answer));
  if (exact) return exact;
  const tokens = answer
    .toLowerCase()
    .split(/[\s,]+/)
    .filter((t) => t.length > 2);
  for (const tok of tokens) {
    const start = opts.find((o) => normalizeOptionText(o).startsWith(tok));
    if (start) return start;
    const contains = opts.find((o) => normalizeOptionText(o).includes(tok));
    if (contains) return contains;
  }
  return opts[0];
}