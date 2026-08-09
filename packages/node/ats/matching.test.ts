import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { pickLocationOption } from "./shared/matching.js";

describe("pickLocationOption", () => {
  it("exact match wins", () => {
    const opts = ["Bhopal, Madhya Pradesh, India", "Delhi, India"];
    assert.equal(pickLocationOption("Bhopal, Madhya Pradesh, India", opts), opts[0]);
  });

  it("bare India never matches Indianapolis", () => {
    // Regression: "India" token-matched "Indianapolis, Indiana, United States"
    // via startsWith("india") and picked a US city for an India-based candidate.
    const opts = [
      "Indianapolis, Indiana, United States",
      "India",
      "New Delhi, India",
      "Bengaluru, Karnataka, India",
    ];
    const picked = pickLocationOption("India", opts);
    assert.notEqual(picked, opts[0]);
    assert.ok(picked.includes("India"));
  });

  it("city token matches within the same country", () => {
    const opts = [
      "Indianapolis, Indiana, United States",
      "Bhopal, Madhya Pradesh, India",
      "Delhi, India",
    ];
    const picked = pickLocationOption("Bhopal, India", opts);
    assert.equal(picked, "Bhopal, Madhya Pradesh, India");
  });

  it("no country in answer prefers first suggestion", () => {
    const opts = ["Bangalore, Karnataka, India", "Remote"];
    assert.equal(pickLocationOption("Bangalore", opts), opts[0]);
  });

  it("full specific location matches exact city", () => {
    const opts = ["Bhopal, Madhya Pradesh, India", "Indianapolis, Indiana, United States"];
    const picked = pickLocationOption("Bhopal, Madhya Pradesh, India", opts);
    assert.equal(picked, "Bhopal, Madhya Pradesh, India");
  });
});
