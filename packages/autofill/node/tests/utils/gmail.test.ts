import { describe, it } from "node:test";
import assert from "node:assert/strict";
import {
  extractCodeFromEmail,
  extractVerificationCode,
  gmailConfigured,
} from "../../utils/gmail.js";

describe("extractVerificationCode", () => {
  it("extracts a keyword-anchored 6-digit code", () => {
    assert.equal(extractVerificationCode("Your verification code is 482913."), "482913");
    assert.equal(extractVerificationCode("Security code: 482913"), "482913");
    assert.equal(extractVerificationCode("Enter the code 482913 below"), "482913");
    assert.equal(extractVerificationCode("Your one-time pin is 482913"), "482913");
  });

  it("extracts a code in the '482913 is your code' phrasing", () => {
    assert.equal(extractVerificationCode("482913 is your code"), "482913");
    assert.equal(extractVerificationCode("482913 is your verification code."), "482913");
  });

  it("falls back to any standalone 6-digit number", () => {
    assert.equal(extractVerificationCode("Here is the number: 123456"), "123456");
  });

  it("prefers the code anchored to a keyword over other 6-digit numbers", () => {
    const text = "Reference number 654321. Your verification code is 482913.";
    assert.equal(extractVerificationCode(text), "482913");
  });

  it("returns null when no 6-digit number exists", () => {
    assert.equal(extractVerificationCode("No code here, just 123 and 4567."), null);
    assert.equal(extractVerificationCode(""), null);
    assert.equal(extractVerificationCode("   "), null);
  });

  it("ignores 7-digit numbers", () => {
    assert.equal(extractVerificationCode("Your code is 1234567"), null);
  });

  it("normalizes whitespace across the body", () => {
    assert.equal(
      extractVerificationCode("Your\nverification\ncode\nis\n482913"),
      "482913"
    );
  });

  it("extracts the modern alphanumeric code shape (greenhouse-mail.io)", () => {
    const text =
      "Copy and paste this code into the security code field on your application: " +
      "gfMvrZ38 After you enter the code, resubmit your application.";
    assert.equal(extractVerificationCode(text), "gfMvrZ38");
  });

  it("extracts a digits-less mixed-case code over title-case words", () => {
    const text =
      "Security code for your application to Anduril Industries. " +
      "Copy and paste this code into the security code field on your application: " +
      "lJwcuQgh After you enter the code, resubmit your application.";
    assert.equal(extractVerificationCode(text), "lJwcuQgh");
  });

  it("picks the alphanumeric code over English words near 'code'", () => {
    const text =
      "Copy and paste this code into the security code field on your application: " +
      "gfMvrZ38 After you enter the code, resubmit your application";
    assert.equal(extractVerificationCode(text), "gfMvrZ38");
  });

  it("does not mistake words or bare numbers for alphanumeric codes", () => {
    const text =
      "Security code field on your application:  application  After you enter the code";
    assert.equal(extractVerificationCode(text), null);
  });

  it("falls back to numeric rules when the alphanumeric shape is absent", () => {
    assert.equal(extractVerificationCode("Your code is 482913"), "482913");
  });
});

describe("extractCodeFromEmail", () => {
  const plain = [
    "From: no-reply@greenhouse.io",
    "To: user@gmail.com",
    "Subject: Your application verification code",
    "MIME-Version: 1.0",
    "Content-Type: text/plain; charset=utf-8",
    "",
    "Hi John,",
    "",
    "Enter the following code to verify your email: 482913",
    "",
    "Thanks!",
  ].join("\r\n");

  it("parses a plain-text verification email", async () => {
    assert.equal(await extractCodeFromEmail(plain), "482913");
  });

  it("parses an HTML verification email", async () => {
    const raw = [
      "From: no-reply@greenhouse.io",
      "To: user@gmail.com",
      "Subject: Verify your email address",
      "MIME-Version: 1.0",
      "Content-Type: text/html; charset=utf-8",
      "",
      "<html><body><p>Your code is</p><div style='font-size:32px'>735190</div></body></html>",
    ].join("\r\n");
    assert.equal(await extractCodeFromEmail(raw), "735190");
  });

  it("reads the code from the subject when the body is empty", async () => {
    const raw = [
      "From: no-reply@greenhouse.io",
      "To: user@gmail.com",
      "Subject: Your verification code is 903214",
      "MIME-Version: 1.0",
      "Content-Type: text/plain; charset=utf-8",
      "",
      "",
    ].join("\r\n");
    assert.equal(await extractCodeFromEmail(raw), "903214");
  });

  it("returns null for emails without a code", async () => {
    const raw = [
      "From: no-reply@greenhouse.io",
      "To: user@gmail.com",
      "Subject: Welcome",
      "MIME-Version: 1.0",
      "Content-Type: text/plain; charset=utf-8",
      "",
      "Thanks for applying!",
    ].join("\r\n");
    assert.equal(await extractCodeFromEmail(raw), null);
  });

  it("parses the real greenhouse-mail.io HTML-only message", async () => {
    const raw = [
      "From: no-reply@us.greenhouse-mail.io",
      "To: user@gmail.com",
      "Subject: Security code for your application to Anduril Industries",
      "MIME-Version: 1.0",
      'Content-Type: multipart/related; boundary="b1"',
      "",
      "--b1",
      'Content-Type: text/html; charset="utf-8"',
      "",
      "<html><body>",
      "  <p>Hi Aman,<br/><br/>",
      "    Copy and paste this code into the security code field on your application:",
      "  </p>",
      "  <h1>gfMvrZ38</h1>",
      "  <p>After you enter the code, resubmit your application.</p>",
      "</body></html>",
      "--b1",
      'Content-Type: image/png; name="6a71891e7838e_6d4d688901b@prod-jben-web.mail"',
      "Content-Transfer-Encoding: base64",
      "",
      "iVBORw0KGgoAAAANSUhEUg==",
      "--b1--",
    ].join("\r\n");
    assert.equal(await extractCodeFromEmail(raw), "gfMvrZ38");
  });
});

describe("gmailConfigured", () => {
  it("reflects GMAIL_EMAIL / GMAIL_APP_PASSWORD presence", () => {
    const prevEmail = process.env.GMAIL_EMAIL;
    const prevPass = process.env.GMAIL_APP_PASSWORD;
    delete process.env.GMAIL_EMAIL;
    delete process.env.GMAIL_APP_PASSWORD;
    try {
      assert.equal(gmailConfigured(), false);
      process.env.GMAIL_EMAIL = "someone@gmail.com";
      assert.equal(gmailConfigured(), false);
      process.env.GMAIL_APP_PASSWORD = "abcd efgh ijkl mnop";
      assert.equal(gmailConfigured(), true);
    } finally {
      if (prevEmail === undefined) delete process.env.GMAIL_EMAIL;
      else process.env.GMAIL_EMAIL = prevEmail;
      if (prevPass === undefined) delete process.env.GMAIL_APP_PASSWORD;
      else process.env.GMAIL_APP_PASSWORD = prevPass;
    }
  });
});
