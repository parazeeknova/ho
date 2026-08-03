/**
 * Idle/activity watchdog for the fill runner.
 *
 * Overnight there is no human to notice a hung browser, and a Stagehand
 * `act`/`observe` that never resolves blocks the whole runner forever — the DB
 * lease would only reclaim the job after an hour. The watchdog aborts a run
 * that makes no observable progress (RPC traffic, status events, adapter logs)
 * for the configured timeout, so a stuck browser is killed quickly instead of
 * holding a worker slot all night.
 */
export class ActivityWatchdog {
  private lastTouch: number;
  private timer: NodeJS.Timeout | null = null;

  constructor(
    private readonly timeoutMs: number,
    private readonly onTimeout: () => void,
    private readonly intervalMs: number = 1000
  ) {
    this.lastTouch = Date.now();
  }

  start(): void {
    if (this.timer || this.timeoutMs <= 0) return;
    this.touch();
    this.timer = setInterval(() => {
      if (Date.now() - this.lastTouch >= this.timeoutMs) {
        this.stop();
        this.onTimeout();
      }
    }, this.intervalMs);
  }

  touch(): void {
    this.lastTouch = Date.now();
  }

  stop(): void {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
  }
}
