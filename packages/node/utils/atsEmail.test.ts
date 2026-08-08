import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { classifyAtsEmail, parseAtsEmail } from "./atsEmail.js";

describe("classifyAtsEmail", () => {
  it("confirmation: we received your application", () => {
    const kind = classifyAtsEmail(
      "Application received — Software Engineer at Stripe",
      "Thank you for applying. We have received your application and will review it.",
    );
    assert.equal(kind, "confirmation");
  });

  it("rejection: unfortunately / not moving forward", () => {
    const kind = classifyAtsEmail(
      "Update on your application",
      "Unfortunately, we will not be moving forward with your application at this time.",
    );
    assert.equal(kind, "rejection");
  });

  it("rejection wins over thank-you subject", () => {
    const kind = classifyAtsEmail(
      "Thank you for your application",
      "Thank you for applying, but the position has been filled with other candidates.",
    );
    assert.equal(kind, "rejection");
  });

  it("screening: interview / next step request", () => {
    const kind = classifyAtsEmail(
      "Let's schedule a time to talk",
      "We'd like to schedule a phone screen with you for the Backend Engineer role.",
    );
    assert.equal(kind, "screening");
  });

  it("otp: verification code email is not application status", () => {
    const kind = classifyAtsEmail(
      "Your verification code",
      "Your one-time verification code is 482913.",
    );
    assert.equal(kind, "otp");
  });

  it("other: unrelated mail", () => {
    const kind = classifyAtsEmail("Your invoice", "Here is your monthly invoice for services.");
    assert.equal(kind, "other");
  });
});

describe("parseAtsEmail", () => {
  it("parses a plain-text RFC822 message", async () => {
    const source = [
      "From: no-reply@greenhouse.io",
      "Subject: Application received",
      "",
      "Thank you for applying to join our team.",
      "",
    ].join("\r\n");
    const parsed = await parseAtsEmail(source);
    assert.ok(parsed);
    assert.match(parsed.subject, /Application received/);
    assert.match(parsed.text, /Thank you for applying/);
  });

  it("returns null for garbage", async () => {
    const parsed = await parseAtsEmail(Buffer.from("not a real email"));
    assert.equal(parsed, null);
  });
});
