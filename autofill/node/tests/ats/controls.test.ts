import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { sanitizeNumberAnswer, cleanPlaceholderValue, firstUrl } from "../../ats/shared/controls.js";

describe("sanitizeNumberAnswer", () => {
  it("passes a plain integer through", () => {
    assert.equal(sanitizeNumberAnswer("5"), "5");
    assert.equal(sanitizeNumberAnswer("0"), "0");
  });

  it("extracts the leading number from a range/label", () => {
    assert.equal(sanitizeNumberAnswer("0-4 Years"), "0");
    assert.equal(sanitizeNumberAnswer("5+ years of experience"), "5");
    assert.equal(sanitizeNumberAnswer("around 3.5 years"), "3.5");
  });

  it("handles negative values", () => {
    assert.equal(sanitizeNumberAnswer("-2"), "-2");
  });

  it("returns empty string when nothing numeric exists", () => {
    assert.equal(sanitizeNumberAnswer("Immediately"), "");
    assert.equal(sanitizeNumberAnswer(""), "");
    assert.equal(sanitizeNumberAnswer("N/A"), "");
  });

  it("expands K/M/B salary suffixes and strips separators", () => {
    assert.equal(sanitizeNumberAnswer("80K INR/month"), "80000");
    assert.equal(sanitizeNumberAnswer("80k"), "80000");
    assert.equal(sanitizeNumberAnswer("1.5M"), "1500000");
    assert.equal(sanitizeNumberAnswer("1,200"), "1200");
    assert.equal(sanitizeNumberAnswer("€80k"), "80000");
  });

  it("never expands ordinary words with a k/m/b letter", () => {
    assert.equal(sanitizeNumberAnswer("3 BHK"), "3");
    assert.equal(sanitizeNumberAnswer("8 MB"), "8");
    assert.equal(sanitizeNumberAnswer("5 M&A deals"), "5");
    assert.equal(sanitizeNumberAnswer("10 B2B clients"), "10");
    assert.equal(sanitizeNumberAnswer("2+ KB of memory"), "2");
  });
});

describe("cleanPlaceholderValue", () => {
  it("rejects framework placeholders", () => {
    assert.equal(cleanPlaceholderValue("undefined"), "");
    assert.equal(cleanPlaceholderValue("null"), "");
    assert.equal(cleanPlaceholderValue("NaN"), "");
  });

  it("passes real values through", () => {
    assert.equal(cleanPlaceholderValue("0"), "0");
    assert.equal(cleanPlaceholderValue("Immediately"), "Immediately");
    assert.equal(cleanPlaceholderValue("  hello "), "hello");
    assert.equal(cleanPlaceholderValue(""), "");
  });
});

describe("firstUrl", () => {
  it("extracts the first URL from a comma-separated list", () => {
    assert.equal(
      firstUrl("https://linkedin.com/in/me, https://github.com/me, https://me.dev"),
      "https://linkedin.com/in/me"
    );
  });

  it("passes a single URL through", () => {
    assert.equal(firstUrl("https://github.com/me"), "https://github.com/me");
  });

  it("strips trailing sentence punctuation from a URL", () => {
    assert.equal(firstUrl("See https://github.com/me."), "https://github.com/me");
  });

  it("returns null when there is no URL", () => {
    assert.equal(firstUrl("just text"), null);
    assert.equal(firstUrl(""), null);
  });
});
