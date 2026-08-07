import { z } from "zod";

export const ProfileSchema = z.object({
  firstName: z.string().default("John"),
  lastName: z.string().default("Doe"),
  email: z.string().email().default("john.doe@example.com"),
  phone: z.string().default("+1234567890"),
  linkedin: z.string().url().nullable().optional(),
  github: z.string().url().nullable().optional(),
  website: z.string().url().nullable().optional(),
  twitter: z.string().url().nullable().optional(),
  preferredName: z.string().nullable().optional(),
  location: z.string().nullable().optional(),
  resumePath: z.string().nullable().optional(),
  customAnswers: z.record(z.string(), z.string()).default({}),
});

export const JobPayloadSchema = z.preprocess(
  (data: any) => {
    if (data && typeof data === "object") {
      return {
        jobId: data.jobId,
        url: data.url || data.applyLink,
        mode: data.mode || data.applyMode || "review",
        profile: data.profile,
        submitAllowed: data.submitAllowed,
        siteKnowledge: data.siteKnowledge,
      };
    }
    return data;
  },
  z.object({
    jobId: z.string(),
    url: z.string().url(),
    profile: ProfileSchema,
    mode: z.enum(["auto", "review"]).default("review"),
    // No-apply phase: when false the fill completes with a screenshot but the
    // application is never submitted, regardless of mode or review choice.
    submitAllowed: z.boolean().default(true),
    // Learned per-site selectors/flow (procedural memory) passed from the
    // worker so the generic adapter can consult known-good selectors before
    // re-probing the DOM.
    siteKnowledge: z.record(z.string(), z.any()).default({}),
  }),
);

export const StatusEventSchema = z.object({
  jobId: z.string(),
  status: z.enum(["in_progress", "awaiting_review", "submitted", "failed", "skipped", "expired"]),
  screenshotPath: z.string().optional(),
  filledFields: z.record(z.string(), z.string()).optional(),
  error: z.string().optional(),
  message: z.string().optional(),
  // Post-submit email feedback: {kind, from, subject, snippet} from the soft
  // ATS-email poll, surfaced by the worker (confirmation/rejection/screening).
  emailStatus: z
    .object({
      kind: z.enum(["confirmation", "rejection", "screening", "otp", "other"]),
      from: z.string(),
      subject: z.string(),
      snippet: z.string(),
    })
    .optional(),
});

export const ActionCallbackSchema = z.object({
  action: z.enum(["submit", "skip", "correct"]),
  // label -> corrected value, sent when action is "correct".
  corrections: z.record(z.string(), z.string()).optional(),
});

export type Profile = z.infer<typeof ProfileSchema>;
export type JobPayload = z.infer<typeof JobPayloadSchema>;
export type StatusEvent = z.infer<typeof StatusEventSchema>;
export type ActionCallback = z.infer<typeof ActionCallbackSchema>;
