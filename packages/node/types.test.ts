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
