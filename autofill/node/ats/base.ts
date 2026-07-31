import { Stagehand } from "@browserbasehq/stagehand";
import { JobPayload } from "../types.js";

export type RpcHelper = (method: string, args: Record<string, any>) => Promise<any>;

export abstract class ATSAdapter {
  protected stagehand: Stagehand;

  constructor(stagehand: Stagehand) {
    this.stagehand = stagehand;
  }

  abstract fill(payload: JobPayload, rpc?: RpcHelper): Promise<void>;
  abstract submit(): Promise<void>;
}
