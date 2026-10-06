import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { JobPayloadSchema } from "./types";

describe("JobPayloadSchema", () => {
  it("accepts a payload with learned site knowledge", () => {
    const r = JobPayloadSchema.safeParse({
      jobId: "job-1",
      mode: "auto",
      profile: {
        email: "harsh@example.com",
        firstName: "Harsh",
        lastName: "Sahu",
        phone: "+917000000000",
      },
      siteKnowledge: {
        flow: "wizard",
        form_signature: "greenhouse:boards.greenhouse.io",
        host: "boards.greenhouse.io",
        platform: "greenhouse",
        selectors: { location: 'input[role="combobox"]' },
      },
      submitAllowed: true,
      url: "https://boards.greenhouse.io/neo4j/jobs/123",
    });
    assert.equal(r.success, true, JSON.stringify(r.error));
    const data = r.success ? r.data : null;
    assert.ok(data);
    assert.equal(data?.siteKnowledge.platform, "greenhouse");
  });

  it("defaults site knowledge to empty when absent", () => {
    const r = JobPayloadSchema.safeParse({
      jobId: "job-2",
      mode: "auto",
      profile: { email: "a@b.com", firstName: "A", lastName: "B", phone: "+1" },
      submitAllowed: true,
      url: "https://jobs.ashbyhq.com/replit/abc",
    });
    assert.equal(r.success, true, JSON.stringify(r.error));
    assert.deepEqual(r.success ? r.data?.siteKnowledge : null, {});
  });
});

describe("JobPayloadSchema URL normalization (user-input prevention)", () => {
  it("accepts scheme-less linkedin/github/website by prepending https://", () => {
    const r = JobPayloadSchema.safeParse({
      jobId: "job-3",
      mode: "auto",
      profile: {
        email: "a@b.com",
        firstName: "A",
        github: "github.com/bar",
        lastName: "B",
        linkedin: "linkedin.com/in/foo",
        phone: "+1",
        website: "example.com",
      },
      url: "https://jobs.lever.co/acme/1",
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
      mode: "auto",
      profile: {
        email: "a@b.com",
        firstName: "A",
        lastName: "B",
        linkedin: "https://linkedin.com/in/foo",
        phone: "+1",
      },
      url: "https://boards.greenhouse.io/x/jobs/1",
    });
    assert.equal(r.success, true, JSON.stringify(r.error));
    assert.equal(
      r.success ? r.data?.profile.linkedin : null,
      "https://linkedin.com/in/foo"
    );
  });

  it("accepts null/empty profile URLs without prepending", () => {
    const r = JobPayloadSchema.safeParse({
      jobId: "job-5",
      mode: "auto",
      profile: {
        email: "a@b.com",
        firstName: "A",
        github: "",
        lastName: "B",
        linkedin: null,
        phone: "+1",
        website: "N/A",
      },
      url: "https://boards.greenhouse.io/x/jobs/1",
    });
    assert.equal(r.success, true, JSON.stringify(r.error));
  });
});
