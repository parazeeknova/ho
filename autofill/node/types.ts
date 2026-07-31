import { z } from "zod";

export const ProfileSchema = z.object({
  firstName: z.string().default("John"),
  lastName: z.string().default("Doe"),
  email: z.string().email().default("john.doe@example.com"),
  phone: z.string().default("+1234567890"),
  linkedin: z.string().url().nullable().optional(),
  github: z.string().url().nullable().optional(),
  website: z.string().url().nullable().optional(),
  resumePath: z.string().nullable().optional(),
  customAnswers: z.record(z.string(), z.string()).default({})
});

export const JobPayloadSchema = z.object({
  jobId: z.string(),
  url: z.string().url(),
  profile: ProfileSchema,
  mode: z.enum(["auto", "review"]).default("review")
});

export const StatusEventSchema = z.object({
  jobId: z.string(),
  status: z.enum(["in_progress", "awaiting_review", "submitted", "failed", "skipped"]),
  screenshotPath: z.string().optional(),
  error: z.string().optional(),
  message: z.string().optional()
});

export const ActionCallbackSchema = z.object({
  action: z.enum(["submit", "skip"])
});

export type Profile = z.infer<typeof ProfileSchema>;
export type JobPayload = z.infer<typeof JobPayloadSchema>;
export type StatusEvent = z.infer<typeof StatusEventSchema>;
export type ActionCallback = z.infer<typeof ActionCallbackSchema>;
