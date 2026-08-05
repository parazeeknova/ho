import { ImapFlow } from "imapflow";
import { simpleParser } from "mailparser";

/**
 * Sender substrings accepted for Greenhouse verification emails.
 *
 * Modern Greenhouse OTP mail is sent from SendGrid-style subdomains such as
 * no-reply@us.greenhouse-mail.io (us./eu./api. variants), so the primary
 * match is "greenhouse-mail.io" while "greenhouse.io" is kept for legacy
 * direct sending (no-reply@greenhouse.io, noreply@greenhouse.io, ...). The
 * JS-side filter uses substring `.includes()`, which handles both host
 * styles regardless of which router host Gmail fronted the message with.
 */
export const GREENHOUSE_VERIFICATION_SENDERS = [
  "greenhouse-mail.io",
  "greenhouse.io",
];

/** Whether Gmail IMAP credentials are present in the environment. */
export function gmailConfigured(): boolean {
  return !!process.env.GMAIL_EMAIL && !!process.env.GMAIL_APP_PASSWORD;
}

/**
 * Extract a verification code from email text.
 *
 * Greenhouse sends two code shapes:
 *  - modern: 8-char mixed-case alphanumeric codes ("gfMvrZ38") rendered as an
 *    <h1> in an HTML-only (multipart/related) email;
 *  - legacy: plain 6-digit numeric codes ("482913 is your code").
 *
 * Strategy: scan for 6-12 char tokens containing both a letter and a digit
 * (English words and bare numbers can never qualify), keep those within a
 * bounded window of a code keyword ("code", "verification", "otp", "pin",
 * "resubmit", ...), and pick the one closest to a keyword with the length
 * closest to 8. Fall back to the anchored 6-digit numeric rules, then a bare
 * standalone 6-digit number. Returns null when nothing plausible exists.
 */
export function extractVerificationCode(text: string): string | null {
  const src = String(text ?? "")
    .replace(/\s+/g, " ")
    .trim();
  if (!src) return null;
  const WINDOW = 80;
  const KEYWORD = /\b(?:code|verification|one[ -]?time|otp|pin|resubmit|security)\b/gi;
  let best: { token: string; penalty: number } | null = null;
  for (const m of src.matchAll(/\b([A-Za-z0-9]{6,12})\b/g)) {
    const token = m[1];
    // Codes are 6-12 alnum chars with mixed case and/or digits ("gfMvrZ38",
    // "vCipmku6", "lJwcuQgh"). Common English words fail every arm:
    //  - title-case words ("Security", "Street"): only the initial cap;
    //  - lowercase words ("application", "resubmit"): no interior caps/digits;
    //  - all-caps words: no lowercase, and (without digits) rejected below.
    // Interior uppercase letters are the tell of a generated code.
    // Pure-digit tokens are skipped here and handled by the numeric rules
    // below, which correctly reject 7+-digit reference numbers.
    if (!/[A-Za-z]/.test(token)) continue;
    const hasDigit = /\d/.test(token);
    const hasLower = /[a-z]/.test(token);
    const hasUpper = /[A-Z]/.test(token);
    const interiorUpper = hasUpper && !/^[A-Z][a-z]+$/.test(token);
    if (!(hasDigit || interiorUpper)) continue;
    const start = m.index ?? 0;
    const end = start + token.length;
    let dist = Infinity;
    const preStart = Math.max(0, start - WINDOW);
    for (const k of src.slice(preStart, start).matchAll(KEYWORD)) {
      dist = Math.min(dist, start - (preStart + (k.index ?? 0)));
    }
    for (const k of src.slice(end, end + WINDOW).matchAll(KEYWORD)) {
      dist = Math.min(dist, (k.index ?? 0) + 1);
    }
    if (dist > WINDOW - 20) continue;
    const penalty = Math.abs(token.length - 8) * 2 + dist;
    if (!best || penalty < best.penalty) best = { token, penalty };
  }
  if (best) return best.token;
  // Legacy numeric shape: 6 digits anchored to a code/verification keyword
  // (bounded window so a phone or reference number elsewhere never wins).
  const anchored =
    src.match(
      /(?:\bcode\b|verification|security code|confirmation code|one[- ]?time|pin)[^\d]{0,40}?(\b\d{6}\b)/i
    ) ||
    // "482913 is your code" / "482913 is your verification code" phrasing.
    src.match(/(\b\d{6}\b)[^\d]{0,40}?\bcode\b/i);
  if (anchored) return anchored[1];
  const bare = src.match(/\b\d{6}\b/);
  return bare ? bare[0] : null;
}

function stripHtml(html: string): string {
  return html
    .replace(/<style[\s\S]*?<\/style>/gi, " ")
    .replace(/<script[\s\S]*?<\/script>/gi, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/&#\d+;/g, " ");
}

/**
 * Parse a raw RFC 2822 message (Buffer or string) and extract a verification
 * code from its subject + text/HTML bodies. HTML-only and multipart/related
 * emails (the modern Greenhouse format) carry the code in the `html` part, so
 * that is included here too. Returns null when the message cannot be parsed or
 * contains no code.
 */
export async function extractCodeFromEmail(
  source: Buffer | string
): Promise<string | null> {
  try {
    const parsed = await simpleParser(source);
    const parts: string[] = [
      parsed.subject || "",
      parsed.text || "",
      stripHtml(parsed.html || ""),
    ];
    if (parsed.textAsHtml) parts.push(stripHtml(parsed.textAsHtml));
    return extractVerificationCode(parts.join("\n"));
  } catch {
    return null;
  }
}

export interface GmailWaitOptions {
  /** How long to keep polling the INBOX before failing (default 60000ms,
   *  overridable via AUTOFILL_GMAIL_TIMEOUT_MS). */
  timeoutMs?: number;
  /** Delay between IMAP poll cycles (default 4000ms). */
  pollMs?: number;
  /** Only accept emails received within this window (default 5 min — codes
   *  arrive seconds after submit, so 5 min is generous for delivery while
   *  excluding stale unread codes from a previous attempt). */
  maxAgeMs?: number;
  /** Sender substrings to accept (default: Greenhouse senders). */
  senders?: string[];
  /** Log sink (default: console.log with a [Gmail] tag). */
  log?: (message: string) => void;
}

/**
 * Poll the Gmail INBOX (via IMAP) for an unread Greenhouse verification email
 * and return its 6-digit code. Marks the matched email as read so later runs
 * never re-parse it. Throws a clean error on timeout or IMAP failure — the
 * caller decides whether to skip the job, never hangs the worker.
 */
export async function waitForGreenhouseCode(
  options: GmailWaitOptions = {}
): Promise<string> {
  const timeoutMs =
    options.timeoutMs ??
    parseInt(process.env.AUTOFILL_GMAIL_TIMEOUT_MS || "60000", 10);
  const pollMs = options.pollMs ?? 4000;
  const maxAgeMs = options.maxAgeMs ?? 5 * 60 * 1000;
  const senders = options.senders ?? GREENHOUSE_VERIFICATION_SENDERS;
  const log =
    options.log ?? ((message: string) => console.log(`[Gmail] ${message}`));

  const email = process.env.GMAIL_EMAIL;
  const appPassword = process.env.GMAIL_APP_PASSWORD;
  if (!email || !appPassword) {
    throw new Error(
      "GMAIL_EMAIL and GMAIL_APP_PASSWORD are not configured; cannot fetch the verification code"
    );
  }

  const client = new ImapFlow({
    host: "imap.gmail.com",
    port: 993,
    secure: true,
    auth: { user: email, pass: appPassword },
    logger: false,
  });

  const deadline = Date.now() + timeoutMs;
  // Gmail's IMAP SEARCH is backed by its message index, and `from`/`since`
  // criteria can miss freshly delivered messages for minutes (observed
  // repeatedly). The \Seen flag search is metadata-based and dependable, so
  // search flags only and filter sender/age/code in JS. Only the newest
  // POLL_TOP unread messages are inspected per cycle — the INBOX holds tens
  // of thousands of unread messages and the code email is always fresh.
  const POLL_TOP = 20;
  try {
    await client.connect();
    const lock = await client.getMailboxLock("INBOX");
    try {
      while (Date.now() < deadline) {
        const uids =
          (await client.search({ seen: false }, { uid: true }).catch(() => [] as number[])) ||
          [];
        // Gmail returns UIDs ascending, so the tail is the newest.
        for (const uid of uids.slice(-POLL_TOP).reverse()) {
          const meta = await client
            .fetchOne(uid, { envelope: true, internalDate: true }, { uid: true })
            .catch(() => null);
          if (!meta) continue;
          const from = (meta.envelope?.from?.[0]?.address ?? "").toLowerCase();
          if (!senders.some((s) => from.includes(s.toLowerCase()))) continue;
          const received = meta.internalDate ? new Date(meta.internalDate) : null;
          if (received && Date.now() - received.getTime() > maxAgeMs) continue;
          const msg = await client
            .fetchOne(uid, { source: true }, { uid: true })
            .catch(() => null);
          if (!msg || !msg.source) continue;
          const code = await extractCodeFromEmail(msg.source);
          if (code) {
            // Mark read so a later run never re-parses the same email.
            await client.messageFlagsAdd(uid, ["\\Seen"], { uid: true }).catch(() => false);
            log(`Verification code found in email from ${from} (${maskCode(code)}).`);
            return code;
          }
        }
        const left = Math.max(0, Math.round((deadline - Date.now()) / 1000));
        log(`No Greenhouse verification email yet (${left}s left).`);
        await new Promise((resolve) => setTimeout(resolve, pollMs));
      }
      throw new Error(
        `Timed out after ${timeoutMs}ms waiting for a Greenhouse verification email from ${senders.join(", ")}`
      );
    } finally {
      lock.release();
    }
  } catch (err: any) {
    // imapflow reports failures as a bare "Command failed" with the server
    // response text on `response` — surface it so auth/search failures are
    // diagnosable in the job log instead of a one-line dead end.
    const detail = [err?.response, err?.text, err?.cmd]
      .filter((v) => typeof v === "string" && v.trim())
      .join(" / ");
    throw new Error(
      `Failed to fetch Greenhouse verification code from Gmail: ${err?.message || err}` +
        (detail ? ` (${detail.slice(0, 200)})` : "")
    );
  } finally {
    await client.logout().catch(() => {});
  }
}

function maskCode(code: string): string {
  if (code.length <= 2) return code;
  return `${code[0]}${"*".repeat(Math.max(0, code.length - 2))}${code[code.length - 1]}`;
}
