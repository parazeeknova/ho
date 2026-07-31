import { Stagehand } from "@browserbasehq/stagehand";
import { JobPayload } from "../types.js";

export abstract class ATSAdapter {
  protected stagehand: Stagehand;

  constructor(stagehand: Stagehand) {
    this.stagehand = stagehand;
  }

  abstract fill(payload: JobPayload): Promise<void>;
  abstract submit(): Promise<void>;
}
