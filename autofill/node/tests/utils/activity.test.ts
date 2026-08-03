import { describe, it, beforeEach } from "node:test";
import assert from "node:assert/strict";
import { ActivityWatchdog } from "../../utils/activity.js";

describe("ActivityWatchdog", () => {
  beforeEach(() => {
    // The watchdog uses real timers; give each test enough time to settle.
  });

  it("fires onTimeout when idle exceeds the limit", async () => {
    let fired = 0;
    const wd = new ActivityWatchdog(60, () => (fired += 1), 20);
    wd.start();
    await new Promise((r) => setTimeout(r, 120));
    assert.equal(fired, 1, "watchdog should fire exactly once");
    wd.stop();
  });

  it("does not fire while activity keeps resetting the idle timer", async () => {
    let fired = 0;
    const wd = new ActivityWatchdog(80, () => (fired += 1), 20);
    wd.start();
    for (let i = 0; i < 6; i++) {
      await new Promise((r) => setTimeout(r, 40));
      wd.touch();
    }
    assert.equal(fired, 0, "activity must keep the watchdog from firing");
    wd.stop();
  });

  it("is a no-op when the timeout is zero/disabled", async () => {
    let fired = 0;
    const wd = new ActivityWatchdog(0, () => (fired += 1), 20);
    wd.start();
    await new Promise((r) => setTimeout(r, 60));
    assert.equal(fired, 0);
  });

  it("stops permanently after firing", async () => {
    let fired = 0;
    const wd = new ActivityWatchdog(40, () => (fired += 1), 10);
    wd.start();
    await new Promise((r) => setTimeout(r, 100));
    assert.equal(fired, 1, "fires once");
    await new Promise((r) => setTimeout(r, 60));
    assert.equal(fired, 1, "must not fire again after stop");
  });
});
