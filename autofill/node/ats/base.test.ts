import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { ATSAdapter } from "./base.js";

// Runs the real detectCaptcha page.evaluate body against a stubbed DOM so the
// exact classification logic (selectors + size + viewport + visibility) is
// tested. viewportSize defaults to 1280x800; pass {top,left,display,visibility}
// per element to simulate on-screen/hidden/offscreen states.
type RectEl = {
  w: number;
  h: number;
  top: number;
  left: number;
  display: string;
  visibility: string;
  textContent?: string;
  getBoundingClientRect(): { width: number; height: number; top: number; left: number; bottom: number; right: number };
};

function el(
  w: number,
  h: number,
  opts: { top?: number; left?: number; display?: string; visibility?: string; textContent?: string } = {}
): RectEl {
  const top = opts.top ?? 0;
  const left = opts.left ?? 0;
  return {
    w,
    h,
    top,
    left,
    display: opts.display ?? "",
    visibility: opts.visibility ?? "visible",
    textContent: opts.textContent ?? "",
    getBoundingClientRect: () => ({
      width: w,
      height: h,
      top,
      left,
      bottom: top + h,
      right: left + w,
    }),
  };
}

class FakePage {
  constructor(
    private iframes: RectEl[],
    private widgets: RectEl[],
    private bodyText: string = "",
    private pageFrames: FakeFrame[] = [],
    private viewport = { w: 1280, h: 800 },
    private puzzleEls: RectEl[] = []
  ) {}

  async evaluate(fn: () => any): Promise<any> {
    const prevDoc = (globalThis as any).document;
    const prevWin = (globalThis as any).window;
    (globalThis as any).window = {
      innerHeight: this.viewport.h,
      innerWidth: this.viewport.w,
      getComputedStyle: (n: any) => ({
        display: n.display || "",
        visibility: n.visibility || "visible",
      }),
    };
    (globalThis as any).document = {
      body: {
        textContent: this.bodyText,
        parentElement: null,
      },
      querySelectorAll: (sel: string): RectEl[] => {
        if (sel.includes("div, section, form, iframe")) return this.puzzleEls;
        if (sel.includes("iframe")) return this.iframes;
        if (sel.includes("recaptcha") || sel.includes("challenge")) return this.widgets;
        return [];
      },
    };
    // Wire the body into the parentElement chain so the visibility walk
    // terminates.
    (globalThis as any).document.body.parentElement = null;
    (globalThis as any).document.documentElement = (globalThis as any).document.body;
    try {
      return await fn();
    } finally {
      (globalThis as any).document = prevDoc;
      (globalThis as any).window = prevWin;
    }
  }

  frames() {
    return this.pageFrames;
  }

  async waitForTimeout(_ms: number) {}
}

class FakeFrame {
  constructor(private urlStr: string, private clickable: string[]) {}

  url() {
    return this.urlStr;
  }

  locator(sel: string) {
    return new FakeLocator(this.clickable.includes(sel));
  }
}

class FakeLocator {
  constructor(private isClickable: boolean) {}

  first() {
    return this;
  }

  async isVisible() {
    return this.isClickable;
  }

  async click() {}
}

class TestAdapter extends ATSAdapter {
  constructor(private page: FakePage) {
    super(null as any);
  }
  getActivePage(): any {
    return this.page;
  }
  async fill(): Promise<void> {}
  async submit(): Promise<any> {
    return { confirmed: true, retryable: false };
  }
}

describe("detectCaptcha", () => {
  it("ignores the invisible reCAPTCHA v3 badge (256x60) — the false-positive fix", async () => {
    const adapter = new TestAdapter(
      new FakePage([el(256, 60)], []) // src would contain recaptcha
    );
    assert.equal(await adapter.detectCaptcha(), null);
  });

  it("ignores auto-solving Turnstile / checkbox frames (~300x65)", async () => {
    const adapter = new TestAdapter(new FakePage([el(300, 65)], []));
    assert.equal(await adapter.detectCaptcha(), null);
  });

  it("ignores a large recaptcha frame hidden by an ancestor (display:none)", async () => {
    const adapter = new TestAdapter(new FakePage([el(304, 600, { display: "none" })], []));
    assert.equal(await adapter.detectCaptcha(), null);
  });

  it("ignores a large recaptcha frame parked off-screen below the fold", async () => {
    const adapter = new TestAdapter(
      new FakePage([el(304, 600, { top: 4000, left: 0 })], [], "", [], { w: 1280, h: 800 })
    );
    assert.equal(await adapter.detectCaptcha(), null);
  });

  it("ignores a large recaptcha frame positioned off-viewport to the right", async () => {
    const adapter = new TestAdapter(
      new FakePage([el(304, 600, { top: 0, left: 5000 })], [], "", [], { w: 1280, h: 800 })
    );
    assert.equal(await adapter.detectCaptcha(), null);
  });

  it("flags a large hCaptcha challenge iframe on screen (300x250)", async () => {
    const adapter = new TestAdapter(new FakePage([el(300, 250)], []));
    assert.equal(await adapter.detectCaptcha(), "captcha challenge iframe");
  });

  it("flags a large reCAPTCHA challenge frame on screen (304x600)", async () => {
    const adapter = new TestAdapter(new FakePage([el(304, 600)], []));
    assert.equal(await adapter.detectCaptcha(), "captcha challenge iframe");
  });

  it("flags a large visible challenge widget", async () => {
    const adapter = new TestAdapter(new FakePage([], [el(304, 600)]));
    assert.equal(await adapter.detectCaptcha(), "captcha challenge widget");
  });

  it("flags a FunCaptcha puzzle widget with its prompt text", async () => {
    const adapter = new TestAdapter(
      new FakePage([], [], "", [], { w: 1280, h: 800 }, [
        el(360, 260, {
          textContent: "Place the correct animal into the empty spot to complete the pattern",
        }),
      ])
    );
    assert.equal(await adapter.detectCaptcha(), "fun captcha challenge");
  });

  it("does NOT flag a large visible widget without puzzle text", async () => {
    const adapter = new TestAdapter(
      new FakePage([], [], "", [], { w: 1280, h: 800 }, [
        el(360, 260, { textContent: "Application form — please fill out the fields below" }),
      ])
    );
    assert.equal(await adapter.detectCaptcha(), null);
  });

  it("flags a hidden ancestor that becomes visible (no display:none anywhere)", async () => {
    const adapter = new TestAdapter(
      new FakePage([el(300, 250, { display: "", visibility: "visible" })], [])
    );
    assert.equal(await adapter.detectCaptcha(), "captcha challenge iframe");
  });

  it("flags a Cloudflare 'checking your browser' interstitial by text", async () => {
    const adapter = new TestAdapter(
      new FakePage([], [], "Just a moment... Checking your browser before accessing.")
    );
    assert.equal(await adapter.detectCaptcha(), "challenge interstitial page");
  });

  it("returns null for a clean page with no captcha elements", async () => {
    const adapter = new TestAdapter(new FakePage([], [], "Welcome to the application"));
    assert.equal(await adapter.detectCaptcha(), null);
  });
});

describe("attemptCaptcha", () => {
  it("returns false when no captcha frames exist", async () => {
    const adapter = new TestAdapter(new FakePage([], [], "", []));
    assert.equal(await adapter.attemptCaptcha(), false);
  });

  it("clicks the reCAPTCHA checkbox inside a matching frame", async () => {
    const frame = new FakeFrame(
      "https://www.google.com/recaptcha/api2/anchor",
      [".recaptcha-checkbox-border"]
    );
    const adapter = new TestAdapter(new FakePage([], [], "", [frame]));
    assert.equal(await adapter.attemptCaptcha(), true);
  });

  it("returns false when the matching frame has no clickable control", async () => {
    const frame = new FakeFrame("https://challenges.cloudflare.com/turnstile/foo", []);
    const adapter = new TestAdapter(new FakePage([], [], "", [frame]));
    assert.equal(await adapter.attemptCaptcha(), false);
  });

  it("ignores non-captcha frames", async () => {
    const frame = new FakeFrame("https://example.com/some-app", [".recaptcha-checkbox"]);
    const adapter = new TestAdapter(new FakePage([], [], "", [frame]));
    assert.equal(await adapter.attemptCaptcha(), false);
  });
});
