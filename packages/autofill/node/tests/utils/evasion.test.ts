import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { humanTypingEnabled, humanTypingMaxLength, typingDelayMs } from "../../utils/evasion";

describe("human typing knobs", () => {
  it("is enabled by default and disabled by AUTOFILL_TYPING=0", () => {
    const prev = process.env.AUTOFILL_TYPING;
    delete process.env.AUTOFILL_TYPING;
    assert.equal(humanTypingEnabled(), true);
    process.env.AUTOFILL_TYPING = "0";
    assert.equal(humanTypingEnabled(), false);
    process.env.AUTOFILL_TYPING = "1";
    assert.equal(humanTypingEnabled(), true);
    if (prev === undefined) delete process.env.AUTOFILL_TYPING;
    else process.env.AUTOFILL_TYPING = prev;
  });

  it("caps typing length with a sane default and honors the env override", () => {
    const prev = process.env.AUTOFILL_TYPING_MAX;
    delete process.env.AUTOFILL_TYPING_MAX;
    assert.equal(humanTypingMaxLength(), 600);
    process.env.AUTOFILL_TYPING_MAX = "80";
    assert.equal(humanTypingMaxLength(), 80);
    process.env.AUTOFILL_TYPING_MAX = "junk";
    assert.equal(humanTypingMaxLength(), 600);
    if (prev === undefined) delete process.env.AUTOFILL_TYPING_MAX;
    else process.env.AUTOFILL_TYPING_MAX = prev;
  });

  it("yields a positive per-keystroke delay", () => {
    for (let i = 0; i < 200; i++) {
      const d = typingDelayMs();
      assert.ok(d >= 40, `delay ${d} below human floor`);
    }
  });
});
