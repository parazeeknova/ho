import { describe, it } from "node:test";
import assert from "node:assert/strict";
import {
  FormControls,
  sanitizeNumberAnswer,
  cleanPlaceholderValue,
  firstUrl,
} from "../../ats/shared/controls.js";

/**
 * A FormControls whose page serves a single controllable locator. Records the
 * fill/type calls so tests can assert exactly how a value was committed.
 */
function makeControls(initialValue: string) {
  const calls: Array<{ op: string; value: string }> = [];
  let value = initialValue;
  const locator = {
    isVisible: async () => true,
    inputValue: async () => value,
    fill: async (v: string) => {
      calls.push({ op: "fill", value: v });
      value = v;
    },
    type: async (v: string) => {
      calls.push({ op: "type", value: v });
      value = v;
    },
  };
  const page = { locator: () => ({ first: () => locator }) };
  const stagehand = { context: { pages: () => [page] } };
  const controls = new FormControls(stagehand as any);
  return { controls, calls, readValue: () => value };
}

describe("fillField / fillLikeHuman (no-append typing)", () => {
  it("skips a field that already holds the target value (identity re-fill)", async () => {
    const { controls, calls } = makeControls("Aman");
    await controls.fillField("#first_name", "Aman", "", "firstName");
    assert.deepEqual(calls, []);
  });

  it("clears existing content before typing so a re-fill never appends", async () => {
    const { controls, calls, readValue } = makeControls("Ama");
    await controls.fillField("#first_name", "Aman", "", "firstName");
    // The wrong/partial value must be cleared first, then typed fresh.
    assert.deepEqual(
      calls.map((c) => `${c.op}:${c.value}`),
      ["fill:", "type:Aman"]
    );
    assert.equal(readValue(), "Aman");
  });

  it("fills an empty field by typing (human typing enabled)", async () => {
    const { controls, calls, readValue } = makeControls("");
    await controls.fillField("#email", "aman@example.com", "", "email");
    assert.deepEqual(
      calls.map((c) => `${c.op}:${c.value}`),
      ["type:aman@example.com"]
    );
    assert.equal(readValue(), "aman@example.com");
  });

  it("uses the native setter (fill) for long values over the typing cap", async () => {
    const { controls, calls, readValue } = makeControls("");
    const long = "x".repeat(700);
    await controls.fillField("#desc", long, "", "desc");
    assert.deepEqual(
      calls.map((c) => `${c.op}:${c.value.length}`),
      ["fill:700"]
    );
    assert.equal(readValue(), long);
  });
});

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
