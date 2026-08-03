import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { verifySubmitOutcome } from "./audit.js";

// A minimal fake page that models the outcomes verifySubmitOutcome polls for:
// URL transitions, error banners, inline confirmation text, and a submit
// button that either disappears (success) or persists (validation failure).
class FakePage {
  private _url = "https://example.com/apply";
  private _bodyText = "";
  private _errors: string[] = [];
  private _submitVisible = false;
  polls = 0;

  get urlValue() {
    return this._url;
  }
  set urlValue(u: string) {
    this._url = u;
  }

  url() {
    return this._url;
  }

  locator(sel: string) {
    return {
      first: () => ({
        innerText: async () => {
          if (sel.includes("alert") || sel.includes("error")) {
            return this._errors[0] ?? "";
          }
          return "";
        },
        isVisible: async () => this._submitVisible,
      }),
    };
  }

  async evaluate(fn: any) {
    return fn();
  }

  set bodyText(t: string) {
    this._bodyText = t;
  }
  set errors(e: string[]) {
    this._errors = e;
  }
  set submitVisible(v: boolean) {
    this._submitVisible = v;
  }
  // The helper calls document.body?.innerText inside evaluate; emulate by
  // returning a minimal DOM shim when requested.
  get bodyShim() {
    const self = this;
    return {
      document: {
        body: {
          innerText: self._bodyText,
        },
      },
    };
  }
}

// Wait: the real helper uses page.evaluate(() => document.body?.innerText...).
// The FakePage above returns fn() without a document shim. Patch evaluate to
// expose a fake `document` global so the inline-text branch can be exercised.
function makeEvaluator(page: FakePage) {
  return async (fn: any) => {
    const prev = (globalThis as any).document;
    (globalThis as any).document = page.bodyShim.document;
    try {
      return await fn();
    } finally {
      (globalThis as any).document = prev;
    }
  };
}

describe("verifySubmitOutcome", () => {
  it("confirms on a success-URL redirect", async () => {
    const page = new FakePage();
    page.evaluate = makeEvaluator(page) as any;
    page.urlValue = "https://example.com/apply/confirmation";
    page.submitVisible = false;
    const out = await verifySubmitOutcome(page as any, { tag: "Test" });
    assert.equal(out.confirmed, true);
    assert.equal(out.retryable, false);
  });

  it("confirms on inline confirmation body text", async () => {
    const page = new FakePage();
    page.evaluate = makeEvaluator(page) as any;
    page.bodyText = "Thank you for applying. Your application has been submitted.";
    page.submitVisible = false;
    const out = await verifySubmitOutcome(page as any, { tag: "Test" });
    assert.equal(out.confirmed, true);
  });

  it("flags a visible error banner as retryable", async () => {
    const page = new FakePage();
    page.evaluate = makeEvaluator(page) as any;
    page.errors = ["Please complete the required field."];
    page.submitVisible = true;
    const out = await verifySubmitOutcome(page as any, {
      tag: "Test",
      submitButtonSelector: "button[type='submit']",
    });
    assert.equal(out.confirmed, false);
    assert.equal(out.retryable, true);
    assert.match(out.error ?? "", /required field/);
  });

  it("flags a still-visible submit button as a retryable validation failure", async () => {
    const page = new FakePage();
    page.evaluate = makeEvaluator(page) as any;
    page.submitVisible = true;
    const out = await verifySubmitOutcome(page as any, {
      tag: "Test",
      submitButtonSelector: "button[type='submit']",
      polls: 3,
    });
    assert.equal(out.confirmed, false);
    assert.equal(out.retryable, true);
  });

  it("returns non-retryable when nothing resolves", async () => {
    const page = new FakePage();
    page.evaluate = makeEvaluator(page) as any;
    page.submitVisible = false;
    page.bodyText = "Some unrelated page";
    const out = await verifySubmitOutcome(page as any, {
      tag: "Test",
      polls: 2,
    });
    assert.equal(out.confirmed, false);
    assert.equal(out.retryable, false);
  });

  it("ignores benign upload-size error text", async () => {
    const page = new FakePage();
    page.evaluate = makeEvaluator(page) as any;
    page.errors = ["File exceeds the maximum upload size of 100MB"];
    page.submitVisible = false;
    const out = await verifySubmitOutcome(page as any, {
      tag: "Test",
      submitButtonSelector: "button[type='submit']",
      polls: 2,
    });
    assert.equal(out.confirmed, false);
    assert.equal(out.retryable, false);
  });
});
