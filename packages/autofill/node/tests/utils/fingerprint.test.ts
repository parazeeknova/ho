import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { loadFingerprint, type BrowserFingerprint } from "../../utils/fingerprint";

describe("loadFingerprint", () => {
  it("is deterministic for a given seed", () => {
    const a = loadFingerprint("seed-abc");
    const b = loadFingerprint("seed-abc");
    assert.deepEqual(a, b);
  });

  it("differs across seeds", () => {
    const seen = new Set<string>();
    for (let i = 0; i < 8; i++) {
      const fp = loadFingerprint(`job-${i}`);
      seen.add(fp.userAgent);
      seen.add(`${fp.viewport.width}x${fp.viewport.height}`);
      seen.add(`${fp.hardwareConcurrency}-${fp.deviceMemory}`);
      seen.add(fp.platform);
    }
    // The OS platform alone will collide by chance, but the UA string across 8
    // seeds must not collapse to a single value (that is the whole point).
    assert.ok(seen.size > 4, "fingerprint variation must not collapse across seeds");
  });

  it("keeps India-consistent locale/timezone/languages", () => {
    for (let i = 0; i < 6; i++) {
      const fp = loadFingerprint(`loc-${i}`);
      assert.equal(fp.locale, "en-IN");
      assert.equal(fp.timezoneId, "Asia/Kolkata");
      assert.ok(fp.languages[0] === "en-IN");
      assert.match(fp.acceptLanguage, /^en-IN,/);
    }
  });

  it("produces a coherent UA/platform pair", () => {
    const fps: BrowserFingerprint[] = [];
    for (let i = 0; i < 20; i++) fps.push(loadFingerprint(`ua-${i}`));
    for (const fp of fps) {
      assert.match(fp.userAgent, /^Mozilla\/5\.0 \(/);
      assert.match(fp.userAgent, /Chrome\/\d+\.0\./);
      if (fp.platform === "Win32") {
        assert.match(fp.userAgent, /Windows NT 10\.0/);
      } else if (fp.platform === "MacIntel") {
        assert.match(fp.userAgent, /Macintosh; Intel Mac OS X/);
      } else {
        assert.match(fp.userAgent, /X11; Linux x86_64/);
      }
    }
  });

  it("falls back to a random seed when none is provided", () => {
    const a = loadFingerprint();
    const b = loadFingerprint();
    // Overwhelmingly likely to differ (random seed each call).
    assert.notDeepEqual(a.seed, b.seed);
  });

  it("validates viewport/dpr/cores/memory are in sane ranges", () => {
    for (let i = 0; i < 20; i++) {
      const fp = loadFingerprint(`range-${i}`);
      assert.ok(fp.viewport.width >= 1000 && fp.viewport.width <= 2000);
      assert.ok(fp.viewport.height >= 700 && fp.viewport.height <= 1200);
      assert.ok(fp.deviceScaleFactor === 1 || fp.deviceScaleFactor === 2);
      assert.ok([4, 8, 12, 16].includes(fp.hardwareConcurrency));
      assert.ok([4, 8, 16].includes(fp.deviceMemory));
      assert.ok(fp.languages.length >= 2);
    }
  });
});
