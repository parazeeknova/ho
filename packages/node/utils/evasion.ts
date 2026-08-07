export async function randomSleep(minMs: number = 200, maxMs: number = 600): Promise<void> {
  const duration = Math.floor(Math.random() * (maxMs - minMs + 1)) + minMs;
  // AUTOFILL_PACING scales every sleep so machine-speed fills can be slowed
  // to human-like timing (e.g. 4 => 4x slower). Must be >= 1; ignored otherwise.
  const pacing = parseFloat(process.env.AUTOFILL_PACING || "1");
  const scale = Number.isFinite(pacing) && pacing >= 1 ? pacing : 1;
  const scaled = Math.round(duration * scale);
  await new Promise((resolve) => setTimeout(resolve, scaled));
}

/**
 * Per-key "human typing" for text fields. ON BY DEFAULT: a locator.type() with
 * a randomized per-keystroke delay replaces the instant native fill so an ATS
 * heuristics scan sees real keyboard events (keydown/keyup) instead of a form
 * committed with zero key events — the strongest automation tell there is.
 * Set AUTOFILL_TYPING=0 to disable. Long values (cover letters, essays) are
 * capped by AUTOFILL_TYPING_MAX so a run is never slowed to a crawl by one
 * textarea (a human pastes those anyway — the native insert is paste-like).
 */
export function humanTypingEnabled(): boolean {
  return process.env.AUTOFILL_TYPING !== "0";
}

export function humanTypingMaxLength(): number {
  const v = parseInt(process.env.AUTOFILL_TYPING_MAX || "600", 10);
  return Number.isFinite(v) && v > 0 ? v : 600;
}

/**
 * Human per-keystroke delay: mostly 40-140ms with an occasional longer
 * "thinking" pause between bursts, so the inter-key cadence is never uniform.
 * AUTOFILL_TYPING_SPEED scales the cadence down (>1 = faster typing while
 * still emitting real per-keystroke events); 1 = default, ignored below 1.
 */
export function typingDelayMs(): number {
  const speed = parseFloat(process.env.AUTOFILL_TYPING_SPEED || "1");
  const divisor = Number.isFinite(speed) && speed >= 1 ? speed : 1;
  const base = (40 + Math.floor(Math.random() * 100)) / divisor;
  // ~7% of keys: a longer pause (hand on keyboard, re-reading).
  return Math.random() < 0.07 ? base + (250 + Math.floor(Math.random() * 350)) / divisor : base;
}

/**
 * A human pause between discrete actions (field-to-field, before a click).
 * Wider than randomSleep and scaled by AUTOFILL_PACING.
 */
export async function thinkPause(): Promise<void> {
  const min = 900 + Math.floor(Math.random() * 1200);
  const jitter = Math.floor(Math.random() * 1500);
  const pacing = parseFloat(process.env.AUTOFILL_PACING || "1");
  const scale = Number.isFinite(pacing) && pacing >= 1 ? pacing : 1;
  await new Promise((resolve) => setTimeout(resolve, Math.round((min + jitter) * scale)));
}
