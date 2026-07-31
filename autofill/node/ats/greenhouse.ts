import { Stagehand } from "@browserbasehq/stagehand";
import { z } from "zod";
import * as fs from "fs";
import { ATSAdapter, RpcHelper } from "./base.js";
import { JobPayload } from "../types.js";
import { randomSleep } from "../utils/evasion.js";

function escapePromptValue(val: string): string {
  return val.replace(/\\/g, '\\\\').replace(/"/g, '\\"');
}

export class GreenhouseAdapter extends ATSAdapter {
  async fill(payload: JobPayload, rpc?: RpcHelper): Promise<void> {
    const { url, profile } = payload;

    console.log(`[GreenhouseAdapter] Navigating to ${url}...`);
    const page = this.stagehand.context.pages()[0];
    await page.goto(url);
    await randomSleep(300, 600);

    console.log("[GreenhouseAdapter] Filling deterministic profile fields...");

    // First Name
    const firstNameInput = page.locator('#first_name, input[name="job_application[first_name]"]').first();
    if (await firstNameInput.isVisible().catch(() => false)) {
      await firstNameInput.fill(profile.firstName);
      await randomSleep(100, 300);
    } else {
      const safeFirstName = escapePromptValue(profile.firstName);
      await this.stagehand.act(`Type "${safeFirstName}" into the First Name input field`);
      await randomSleep(200, 500);
    }

    // Last Name
    const lastNameInput = page.locator('#last_name, input[name="job_application[last_name]"]').first();
    if (await lastNameInput.isVisible().catch(() => false)) {
      await lastNameInput.fill(profile.lastName);
      await randomSleep(100, 300);
    } else {
      const safeLastName = escapePromptValue(profile.lastName);
      await this.stagehand.act(`Type "${safeLastName}" into the Last Name input field`);
      await randomSleep(200, 500);
    }

    // Email
    const emailInput = page.locator('#email, input[name="job_application[email]"]').first();
    if (await emailInput.isVisible().catch(() => false)) {
      await emailInput.fill(profile.email);
      await randomSleep(100, 300);
    } else {
      const safeEmail = escapePromptValue(profile.email);
      await this.stagehand.act(`Type "${safeEmail}" into the Email input field`);
      await randomSleep(200, 500);
    }

    // Phone
    const phoneInput = page.locator('#phone, input[name="job_application[phone]"]').first();
    if (await phoneInput.isVisible().catch(() => false)) {
      await phoneInput.fill(profile.phone);
      await randomSleep(100, 300);
    } else {
      const safePhone = escapePromptValue(profile.phone);
      await this.stagehand.act(`Type "${safePhone}" into the Phone input field`);
      await randomSleep(200, 500);
    }

    // LinkedIn
    if (profile.linkedin) {
      const linkedinInput = page.locator('input[aria-label*="LinkedIn"], input[name*="linkedin"]').first();
      if (await linkedinInput.isVisible().catch(() => false)) {
        await linkedinInput.fill(profile.linkedin);
        await randomSleep(100, 300);
      } else {
        const safeLinkedin = escapePromptValue(profile.linkedin);
        await this.stagehand.act(`Type "${safeLinkedin}" into the LinkedIn Profile URL field`);
        await randomSleep(200, 500);
      }
    }

    // GitHub
    if (profile.github) {
      const githubInput = page.locator('input[aria-label*="GitHub"], input[name*="github"]').first();
      if (await githubInput.isVisible().catch(() => false)) {
        await githubInput.fill(profile.github);
        await randomSleep(100, 300);
      } else {
        const safeGithub = escapePromptValue(profile.github);
        await this.stagehand.act(`Type "${safeGithub}" into the GitHub Profile URL field`);
        await randomSleep(200, 500);
      }
    }

    // Resume Upload Handling
    if (profile.resumePath && fs.existsSync(profile.resumePath)) {
      console.log(`[GreenhouseAdapter] Uploading resume from ${profile.resumePath}...`);
      const fileInput = page.locator('input[type="file"]').first();
      if (await fileInput.count() > 0) {
        await fileInput.setInputFiles(profile.resumePath);
        await randomSleep(300, 600);
        console.log("[GreenhouseAdapter] Resume uploaded successfully.");
      }
    }

    // Fill explicit customAnswers from profile
    for (const [questionKeyword, answer] of Object.entries(profile.customAnswers)) {
      const safeKeyword = escapePromptValue(questionKeyword);
      const safeAnswer = escapePromptValue(answer);
      console.log(`[GreenhouseAdapter] Answering explicit custom question matching "${safeKeyword}"...`);
      await this.stagehand.act(`Type "${safeAnswer}" into the field asking about "${safeKeyword}"`);
      await randomSleep(200, 500);
    }

    // RAG Dynamic Screener Extraction & Phase 3 RPC Fill
    if (rpc) {
      console.log("[GreenhouseAdapter] Extracting remaining unanswered custom screener questions...");
      try {
        const extractionResult: any = await (this.stagehand as any).extract(
          "Extract all open-ended questions or dropdown questions that are currently unanswered on this page. Do not include basic fields like Name, Email, Phone, LinkedIn, GitHub, or Resume.",
          z.object({
            unansweredQuestions: z.array(z.string())
          })
        );

        const questions = extractionResult?.unansweredQuestions || [];
        if (questions.length > 0) {
          console.log(`[GreenhouseAdapter] Found ${questions.length} unanswered custom questions. Requesting RAG answers from Python...`, questions);
          const ragAnswers: Record<string, string> = await rpc("answer_questions", { questions });

          for (const [questionText, answerText] of Object.entries(ragAnswers)) {
            if (!answerText || answerText === "N/A") continue;
            const safeQ = escapePromptValue(questionText);
            const safeA = escapePromptValue(answerText);
            console.log(`[GreenhouseAdapter] RAG Filling question "${safeQ}"...`);
            await this.stagehand.act(`Type or select "${safeA}" for the question asking "${safeQ}"`);
            await randomSleep(200, 500);
          }
        }
      } catch (extractErr) {
        console.warn("[GreenhouseAdapter] Dynamic screener extraction warning:", extractErr);
      }
    }

    console.log("[GreenhouseAdapter] Form filling completed.");
  }

  async submit(): Promise<void> {
    console.log("[GreenhouseAdapter] Submitting application form...");
    await this.stagehand.act("Click the Submit Application button");
    await randomSleep(500, 1000);
  }
}
