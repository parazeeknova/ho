import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { JobPayloadSchema } from "./types";

describe("JobPayloadSchema", () => {
  it("accepts a payload with learned site knowledge", () => {
    const r = JobPayloadSchema.safeParse({
      jobId: "job-1",
      url: "https://boards.greenhouse.io/neo4j/jobs/123",
      mode: "auto",
      submitAllowed: true,
      profile: {
        firstName: "Harsh",
        lastName: "Sahu",
        email: "harsh@example.com",
        phone: "+917000000000",
      },
      siteKnowledge: {
        host: "boards.greenhouse.io",
        form_signature: "greenhouse:boards.greenhouse.io",
        platform: "greenhouse",
        selectors: { location: 'input[role="combobox"]' },
        flow: "wizard",
      },
    });
    assert.equal(r.success, true, JSON.stringify(r.error));
    const data = r.success ? r.data : null;
    assert.ok(data);
    assert.equal(data?.siteKnowledge.platform, "greenhouse");
  });

  it("defaults site knowledge to empty when absent", () => {
    const r = JobPayloadSchema.safeParse({
      jobId: "job-2",
      url: "https://jobs.ashbyhq.com/replit/abc",
      mode: "auto",
      submitAllowed: true,
      profile: { firstName: "A", lastName: "B", email: "a@b.com", phone: "+1" },
    });
    assert.equal(r.success, true, JSON.stringify(r.error));
    assert.deepEqual(r.success ? r.data?.siteKnowledge : null, {});
  });
});

describe("JobPayloadSchema URL normalization (user-input prevention)", () => {
  it("accepts scheme-less linkedin/github/website by prepending https://", () => {
    const r = JobPayloadSchema.safeParse({
      jobId: "job-3",
      url: "https://jobs.lever.co/acme/1",
      mode: "auto",
      profile: {
        firstName: "A",
        lastName: "B",
        email: "a@b.com",
        phone: "+1",
        linkedin: "linkedin.com/in/foo",
        github: "github.com/bar",
        website: "example.com",
      },
    });
    assert.equal(r.success, true, JSON.stringify(r.error));
    const d = r.success ? r.data : null;
    assert.equal(d?.profile.linkedin, "https://linkedin.com/in/foo");
    assert.equal(d?.profile.github, "https://github.com/bar");
    assert.equal(d?.profile.website, "https://example.com");
  });

  it("keeps already-absolute URLs untouched", () => {
    const r = JobPayloadSchema.safeParse({
      jobId: "job-4",
      url: "https://boards.greenhouse.io/x/jobs/1",
      mode: "auto",
      profile: {
        firstName: "A",
        lastName: "B",
        email: "a@b.com",
        phone: "+1",
        linkedin: "https://linkedin.com/in/foo",
      },
    });
    assert.equal(r.success, true, JSON.stringify(r.error));
    assert.equal(r.success ? r.data?.profile.linkedin : null, "https://linkedin.com/in/foo");
  });

  it("accepts null/empty profile URLs without prepending", () => {
    const r = JobPayloadSchema.safeParse({
      jobId: "job-5",
      url: "https://boards.greenhouse.io/x/jobs/1",
      mode: "auto",
      profile: {
        firstName: "A",
        lastName: "B",
        email: "a@b.com",
        phone: "+1",
        linkedin: null,
        github: "",
        website: "N/A",
      },
    });
    assert.equal(r.success, true, JSON.stringify(r.error));
  });
});
