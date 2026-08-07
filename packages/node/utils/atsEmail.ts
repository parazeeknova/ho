import { ImapFlow } from "imapflow";
import { simpleParser } from "mailparser";

import { gmailConfigured } from "./gmail.js";

/**
 * Post-submit email feedback loop.
 *
 * After the ATS confirmation page is reached, briefly poll the Gmail INBOX for
 * the email the ATS sends back and classify it — confirmation received,
 * rejection, screening request, or a verification/OTP gate. This is a SOFT
 * secondary check: it never blocks a confirmed submission, it just records the
 * email evidence so the worker can surface "we got a rejection" or "they asked
 * us to schedule an interview" instead of assuming every 2xx is still open.
 *
 * Reuses the same IMAP machinery as the Greenhouse OTP waiter (gmail.ts):
 * imapflow + mailparser, unread-only search, sender + age filtering, and marks
 * the matched message \Seen so a later run never re-classifies it.
 */

/** ATS senders whose application-status mail we read back. */
export const ATS_SENDERS = [
  "greenhouse-mail.io",
  "greenhouse.io",
  "ashbyhq.com",
  "ashbyhq",
  "lever.co",
  "lever",
  "workable.com",
  "workable",
  "smartrecruiters.com",
  "smartrecruiters",
  "recruitee.com",
  "bamboohr.com",
  "teamtailor.com",
  "myworkdayjobs.com",
  "workday.com",
  "jazzhr.com",
  "rippling.com",
  "hello@",
  "no-reply@",
  "noreply@",
  "noreply",
  "jobs@",
];

/** Email classifications we can produce. */
export type AtsEmailKind = "confirmation" | "rejection" | "screening" | "otp" | "other";

export interface AtsEmailResult {
  kind: AtsEmailKind;
  from: string;
  subject: string;
  /** Short text snippet (bounded) for the status event / logs. */
  snippet: string;
}

/** Subject markers → confirmation ("we received your application"). */
const CONFIRM_RE =
  /\b(received|confirm|submitted|complete|successful|thanks?|thank you for applying|applied)\b/i;
/** Subject/body markers → rejection ("not moving forward", "other candidates"). */
const REJECT_RE =
  /\b(unfortunately|regret|not (moving|c|selected)|other candidates|position has been filled|no longer under consideration|won'?t be moving|we will not be moving|decided to move forward with other candidates|not successful)\b/i;
/** Screening/next-step markers → they want an interview / assessment. */
const SCREENING_RE =
  /\b(interview|schedule|phone screen|assessment|take[ -]?home|coding challenge|code test|next step|book a|call (with|us)|video (call|interview)|we'?d like to|excited to (speak|meet|discuss))\b/i;
/** Verification/OTP markers → a code gate, not application status. */
const OTP_RE =
  /\b(verification code|one[ -]?time|security code|otp|confirm your (email|address)|verify your (email|identity))\b/i;

/** A confirmation email whose subject is NOT a rejection/screening. */
export function classifyAtsEmail(subject: string, body: string): AtsEmailKind {
  const text = `${subject}\n${body}`;
  // OTP first: a code mail is a gate, never a status.
  if (OTP_RE.test(text)) return "otp";
  // A rejection subject wins over a generic confirmation ("thank you" +
  // "unfortunately" is a rejection).
  if (REJECT_RE.test(subject) || REJECT_RE.test(body.slice(0, 400))) return "rejection";
  if (SCREENING_RE.test(subject) || SCREENING_RE.test(body.slice(0, 400))) return "screening";
  if (CONFIRM_RE.test(subject)) return "confirmation";
  return "other";
}

function stripHtml(html: string): string {
  return html
    .replace(/<style[\s\S]*?<\/style>/gi, " ")
    .replace(/<script[\s\S]*?<\/script>/gi, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/&#\d+;/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

export async function parseAtsEmail(source: Buffer | string): Promise<{
  subject: string;
  text: string;
} | null> {
  try {
    const parsed = await simpleParser(source);
    const subject = parsed.subject || "";
    const text = `${parsed.text || ""} ${stripHtml(parsed.html || "")}`.trim();
    // simpleParser never throws on garbage — it returns an empty message.
    if (!subject && !text) return null;
    return { subject, text };
  } catch {
    return null;
  }
}

export interface AtsEmailWaitOptions {
  /** How long to keep polling (default 60s, overridable via env). */
  timeoutMs?: number;
  /** Delay between IMAP poll cycles (default 4000ms). */
  pollMs?: number;
  /** Only accept emails received within this window (default 10 min). */
  maxAgeMs?: number;
  /** Sender substrings to accept (default: ATS_SENDERS). */
  senders?: string[];
  /** The role/company to match in the subject (loose) — helps pick the right
   *  application's mail when many are in flight. */
  context?: string;
  log?: (message: string) => void;
}

/**
 * Poll the Gmail INBOX for a recent unread ATS email and classify it. Returns
 * the first match, or null on timeout / no match. Never throws a fatal error:
 * a missing config just returns null (the submission stands on its own page
 * confirmation). Marks the matched message \Seen.
 */
export async function waitForAtsEmail(
  options: AtsEmailWaitOptions = {},
): Promise<AtsEmailResult | null> {
  if (!gmailConfigured()) return null;
  const timeoutMs =
    options.timeoutMs ?? parseInt(process.env.AUTOFILL_GMAIL_TIMEOUT_MS || "60000", 10);
  const pollMs = options.pollMs ?? 4000;
  const maxAgeMs = options.maxAgeMs ?? 10 * 60 * 1000;
  const senders = options.senders ?? ATS_SENDERS;
  const log = options.log ?? ((message: string) => console.log(`[AtsEmail] ${message}`));
  const contextNorm = (options.context ?? "").toLowerCase();

  const email = process.env.GMAIL_EMAIL;
  const appPassword = process.env.GMAIL_APP_PASSWORD;
  if (!email || !appPassword) return null;

  const client = new ImapFlow({
    host: "imap.gmail.com",
    port: 993,
    secure: true,
    auth: { user: email, pass: appPassword },
    logger: false,
  });

  const deadline = Date.now() + timeoutMs;
  const POLL_TOP = 30;
  try {
    await client.connect();
    const lock = await client.getMailboxLock("INBOX");
    try {
      while (Date.now() < deadline) {
        const uids =
          (await client.search({ seen: false }, { uid: true }).catch(() => [] as number[])) || [];
        for (const uid of uids.slice(-POLL_TOP).toReversed()) {
          const meta = await client
            .fetchOne(uid, { envelope: true, internalDate: true }, { uid: true })
            .catch(() => null);
          if (!meta) continue;
          const from = (meta.envelope?.from?.[0]?.address ?? "").toLowerCase();
          if (!senders.some((s) => from.includes(s.toLowerCase()))) continue;
          const received = meta.internalDate ? new Date(meta.internalDate) : null;
          if (received && Date.now() - received.getTime() > maxAgeMs) continue;
          const msg = await client.fetchOne(uid, { source: true }, { uid: true }).catch(() => null);
          if (!msg || !msg.source) continue;
          const parsed = await parseAtsEmail(msg.source);
          if (!parsed) continue;
          // Optional loose context match (role/company in subject) so we grab
          // THIS application's mail, not an unrelated ATS email.
          if (contextNorm && !parsed.subject.toLowerCase().includes(contextNorm)) {
            continue;
          }
          const kind = classifyAtsEmail(parsed.subject, parsed.text);
          // Mark read so a later run never re-parses it.
          await client.messageFlagsAdd(uid, ["\\Seen"], { uid: true }).catch(() => false);
          const snippet = `${parsed.subject} — ${parsed.text.slice(0, 140).trim()}`;
          log(`Classified ${kind} email from ${from}: ${snippet.slice(0, 160)}`);
          return { kind, from, subject: parsed.subject, snippet };
        }
        const left = Math.max(0, Math.round((deadline - Date.now()) / 1000));
        log(`No ATS status email yet (${left}s left).`);
        await new Promise((resolve) => setTimeout(resolve, pollMs));
      }
      return null;
    } finally {
      lock.release();
    }
  } catch (err: any) {
    const detail = [err?.response, err?.text, err?.cmd]
      .filter((v) => typeof v === "string" && v.trim())
      .join(" / ");
    log(`IMAP poll failed: ${err?.message || err}${detail ? ` (${detail.slice(0, 160)})` : ""}`);
    return null;
  } finally {
    await client.logout().catch(() => {});
  }
}
