import { Stagehand } from "@browserbasehq/stagehand";
import * as fs from "fs";
import { ATSAdapter, RpcHelper } from "./base.js";
import { JobPayload, Profile } from "../types.js";
import { randomSleep } from "../utils/evasion.js";
import { auditBlanks, finalReverify } from "./shared/audit.js";
import { FormControls } from "./shared/controls.js";
import {
  escapePromptValue,
  pickLocationOption,
} from "./shared/matching.js";
import {
  fieldKey,
  FormField,
  PRE_FILLED_LABELS,
} from "./shared/model.js";
import { Screener } from "./shared/screener.js";

/**
 * Lever adapter.
 *
 * Lever's apply form (`<form id="application-form">`) renders every question as
 * an `<li class="application-question">` with a `.application-label` heading and
 * the input in `.application-field`. Verified against live postings:
 *  - standard contact fields: name, email, phone, org, `urls[LinkedIn]`,
 *    `urls[Portfolio]`, `urls[GitHub]`, `urls[Other]`;
 *  - a resume file input (`input[type=file][name=resume]`) — uploading it also
 *    fires `POST /parseResume`, which autofills profile fields;
 *  - a location autocomplete: `#location-input` with a hidden
 *    `#selected-location` (name `selectedLocation`); the committed value is
 *    written only when a `.dropdown-location` suggestion is picked (the native
 *    address is a text field, not an answer);
 *  - custom questions come through as text/select/radio/checkbox rows;
 *  - voluntary EEO selects (`eeo[`...) and an optional embedded survey section
 *    (rendered inside a hidden `.application-form`) are never walked.
 * Submit is the real button `button.template-btn-submit` (type=button) — the
 * native `button[type=submit]` is hidden and clicking it is a no-op.
 */
export class LeverAdapter extends ATSAdapter {
  protected controls!: LeverControlStack;

  constructor(stagehand: Stagehand) {
    super(stagehand);
    this.controls = new LeverControlStack(stagehand, "LeverAdapter");
  }

  protected getPage(): any {
    return this.controls.getPage();
  }

  private async waitForForm(): Promise<void> {
    const page = this.getPage();
    for (let i = 0; i < 40; i++) {
      const ready = await page
        .locator("#application-form input[name='email'], #application-form input[name='name']")
        .first()
        .isVisible()
        .catch(() => false);
      if (ready) return;
      await randomSleep(800, 1200);
    }
  }

  private async uploadResume(resumePath: string): Promise<boolean> {
    const page = this.getPage();
    const input = page
      .locator('#application-form input[type="file"][name="resume"]')
      .first();
    console.log(`[Lever] Uploading resume from ${resumePath}...`);
    for (let attempt = 0; attempt < 3; attempt++) {
      if ((await input.count()) === 0) {
        console.log("[Lever] Resume already registered (input consumed).");
        return true;
      }
      try {
        // Click the label first so the board associates the upload with the
        // (clickable) resume drop zone, then set the file.
        await input.evaluate((el: any) => {
          const lbl = el.closest("label");
          if (lbl) (lbl as HTMLElement).click();
        }).catch(() => {});
        await randomSleep(300, 600);
        await input.setInputFiles(resumePath);
      } catch (err: any) {
        console.warn(
          `[Lever] Resume setInputFiles threw (attempt ${attempt + 1}): ${err?.message || err}`
        );
      }
      await randomSleep(2500, 3500);
      if (await this.controls.isResumeAttached()) {
        console.log("[Lever] Resume uploaded and registered.");
        return true;
      }
      console.warn(`[Lever] Resume upload not confirmed (attempt ${attempt + 1}); retrying...`);
    }
    return false;
  }

  private async fillLocation(value: string): Promise<boolean> {
    const ok = await this.controls.fillLeverLocation(value);
    if (ok) console.log(`[Lever] Committed location "${value}".`);
    else console.warn(`[Lever] Could not commit profile location "${value}".`);
    return ok;
  }

  private async collectQuestions(): Promise<FormField[]> {
    const page = this.getPage();
    try {
      const rows = await page.evaluate(() => {
        const out: Array<{
          label: string;
          id: string;
          kind: string;
          required: boolean;
          options: string[];
          targets: Array<{ text: string; name: string; value: string }>;
        }> = [];
        // WARNING: only anonymous arrows may be defined inside this evaluate
        // (tsx keepNames wraps inferred-name arrows in __name()). Destructure
        // helpers into an array so none gains a name.
        const [norm] = [
          (t: string) =>
            (t || "").replace(/\s+/g, " ").trim().replace(/^\*+|\*+$/g, ""),
        ];
        const isSurvey = (el: Element): boolean => {
          // All survey/EEO/geography questions live inside a hidden container.
          let n: Element | null = el;
          while (n && n !== document.body) {
            const cls = (n.className || "").toString();
            const id = (n.getAttribute && n.getAttribute("id")) || "";
            if (
              /(survey|eeo|\bonboarding)/i.test(cls + " " + id) ||
              n === document.querySelector('#survey-job-questions') ||
              n === document.querySelector(".application-form.hidden")
            ) {
              // Hidden survey sections carry their own "hidden" class; only
              // skip rows that are not visible (optional != include).
              return !(n as HTMLElement).getClientRects().length;
            }
            n = n.parentElement;
          }
          return false;
        };
        const VISIBLE = (el: Element): boolean => {
          const r = el.getBoundingClientRect();
          return r.height > 0 && r.width > 0;
        };
        const entries = Array.from(
          document.querySelectorAll("#application-form li.application-question")
        );
        let seq = 0;
        for (const li of entries) {
          if (!VISIBLE(li)) continue;
          if (isSurvey(li)) continue;
          const labelEl = li.querySelector(":scope > .application-label") || li.querySelector(".application-label");
          const label = norm(labelEl?.textContent || "");
          if (!label) continue;
          const nameAttr = li.querySelector("input, textarea, select")?.getAttribute("name") || "";
          // The EEO / survey / standard-context fields are handled elsewhere or
          // never auto-answered. `location`/`selectedLocation` is the combobox.
          if (/^(name|resume|phone|email|org|location|selectedLocation)$/.test(nameAttr)) continue;
          if (/(^|\[)(eeo|surveys?|states?)\[/.test(nameAttr) || /^eeo\[/.test(nameAttr)) continue;
          if (/^urls\[/.test(nameAttr)) continue;

          const input = li.querySelector("input[type='text'], input[type='email'], input[type='tel'], input:not([type])") as HTMLInputElement | null;
          const textarea = li.querySelector("textarea") as HTMLTextAreaElement | null;
          const select = li.querySelector("select") as HTMLSelectElement | null;
          const radios = Array.from(li.querySelectorAll('input[type="radio"]')) as HTMLInputElement[];
          const checks = Array.from(li.querySelectorAll('input[type="checkbox"]')) as HTMLInputElement[];

          const required =
            !!li.querySelector("input[required], textarea[required], select[required]") ||
            /aria-required/.test(li.outerHTML);

          let kind = "";
          if (input && !radios.length && !checks.length && !select) kind = "text";
          else if (textarea) kind = "text";
          else if (select) kind = "select";
          else if (radios.length) kind = "radio";
          else if (checks.length) kind = "checkbox";
          else continue;

          // Native inputs carry no id; assign a stable synthetic one so the
          // shared locator machinery (#<id>) finds the element deterministically.
          const id = `leverq-${seq++}`;
          if (input) input.setAttribute("id", id);
          else if (textarea) textarea.setAttribute("id", id);
          else if (select) select.setAttribute("id", id);

          let options: string[] = [];
          const targets: Array<{ text: string; name: string; value: string }> = [];
          if (select) {
            options = Array.from(select.options)
              .map((o) => norm(o.textContent || ""))
              .filter((t) => t && t !== "");
            // discard placeholder / "Select..." option (first empty value).
            if (options.length > 1 && select.options[0]?.value === "") {
              options = options.slice(1);
            }
          }
          for (const inEl of [...radios, ...checks]) {
            const wrapLabel = inEl.closest("label");
            const row = inEl.closest("[class*='option']") || inEl.closest("li");
            const text = norm(
              wrapLabel
                ? wrapLabel.textContent || ""
                : row
                  ? row.textContent || ""
                  : inEl.getAttribute("aria-label") || ""
            );
            if (!text) continue;
            targets.push({ text, name: inEl.name || "", value: inEl.value || "" });
            if (!options.includes(text)) options.push(text);
          }
          if (kind === "select" && options.length === 0) continue;
          if ((kind === "radio" || kind === "checkbox") && targets.length === 0) continue;

          out.push({ label, id, kind, required, options, targets });
        }
        const seen = new Set<string>();
        const uniq: typeof out = [];
        for (const r of out) {
          const key = `${norm(r.label).toLowerCase()}|${r.kind}`;
          if (seen.has(key)) continue;
          seen.add(key);
          uniq.push(r);
        }
        return uniq;
      });

      return (rows ?? []).map(
        (r: any): FormField => ({
          label: r.label,
          id: r.id,
          kind: r.kind as FormField["kind"],
          required: !!r.required,
          options: r.options ?? [],
          optionTargets: (r.targets ?? []).map((t: any) => ({
            text: t.text,
            name: t.name,
            value: t.value,
          })),
          name: r.id,
        })
      );
    } catch (err: any) {
      console.warn(`[Lever] collectQuestions failed: ${err?.message || err}`);
      return [];
    }
  }

  async fill(payload: JobPayload, rpc?: RpcHelper): Promise<void> {
    const { url, profile } = payload;
    console.log(`[Lever] Navigating to ${url}...`);
    const page = this.getPage();
    await page.goto(url);
    await this.waitForForm();

    console.log("[Lever] Uploading resume (parseResume autofills standard fields)...");
    let resumeAttached = false;
    if (profile.resumePath && fs.existsSync(profile.resumePath)) {
      resumeAttached = await this.uploadResume(profile.resumePath);
    }

    console.log("[Lever] Overriding deterministic profile fields (post-parse)...");
    await this.controls.fillField(
      'input[name="name"]',
      [profile.firstName, profile.lastName].join(" ").trim(),
      "Type %value% into the Full name input field",
      "value"
    );
    await this.controls.fillField(
      'input[name="email"]',
      profile.email,
      "Type %value% into the Email input field",
      "value"
    );
    await this.controls.fillField(
      'input[name="phone"]',
      profile.phone,
      "Type %value% into the Phone input field",
      "value"
    );
    await this.controls.fillField(
      'input[name="urls[LinkedIn]"]',
      profile.linkedin ?? "",
      "Type %value% into the LinkedIn URL input field",
      "value"
    );
    await this.controls.fillField(
      'input[name="urls[Portfolio]"]',
      profile.website ?? "",
      "Type %value% into the Portfolio URL input field",
      "value"
    );
    await this.controls.fillField(
      'input[name="urls[GitHub]"]',
      profile.github ?? "",
      "Type %value% into the GitHub URL input field",
      "value"
    );

    if (profile.location) {
      await this.fillLocation(String(profile.location));
    }

    if (rpc) {
      const screener = new Screener(this.controls, "LeverAdapter", profile, rpc);
      const filled: string[] = [];
      const blanked: Array<{ label: string; reason: string }> = [];
      const processedKeys = new Set<string>();
      const userSkippedKeys = new Set<string>();

      for (let pass = 0; pass < 30; pass++) {
        const fields = await this.collectQuestions();
        const fresh = fields.filter((f) => !processedKeys.has(fieldKey(f)));
        console.log(`[Lever] Walk pass ${pass + 1}: ${fresh.length} new question(s) (total ${fields.length}).`);
        if (fresh.length === 0) {
          console.log(`[Lever] Walk converged after ${pass + 1} pass(es).`);
          break;
        }
        for (const f of fresh) {
          processedKeys.add(fieldKey(f));
          await screener.process(f, filled, blanked, userSkippedKeys);
        }
        await this.controls.closeMenu().catch(() => {});
        await randomSleep(900, 1400);
      }

      const finalDom = await this.collectQuestions();
      const requiredBlanks = await auditBlanks({
        fields: finalDom,
        readValue: (f) => this.controls.readFieldValue(f),
        transcript: blanked,
      });
      console.log(`[Lever] Walk complete: filled ${filled.length}, blank ${blanked.length}.`);
      for (const b of blanked) console.warn(`[Lever]   blank: ${escapePromptValue(b.label)} (${b.reason})`);
      for (const rb of requiredBlanks) console.warn(`[Lever]   REQUIRED blank: ${escapePromptValue(rb.label)} (${rb.reason})`);

      // Final sweep: re-enumerate and fill any remaining empty fields.
      const sweepFilled: string[] = [];
      const sweepBlanks: Array<{ label: string; reason: string }> = [];
      for (let pass = 0; pass < 3; pass++) {
        const swept = await this.collectQuestions();
        let touched = 0;
        for (const f of swept) {
          if (PRE_FILLED_LABELS.has(`${f.label}`.replace(/\s+/g, " ").toLowerCase())) continue;
          if (userSkippedKeys.has(fieldKey(f))) continue;
          if (await this.hasValue(f)) continue;
          touched += 1;
          await screener.process(f, sweepFilled, sweepBlanks, userSkippedKeys);
        }
        if (touched === 0) break;
      }
      if (sweepFilled.length) {
        console.log(`[Lever] Final sweep filled ${sweepFilled.length} field(s):`);
        for (const l of sweepFilled) console.log(`[Lever]   filled: ${escapePromptValue(l)}`);
      }

      await finalReverify({
        tag: "LeverAdapter",
        collect: () => this.collectQuestions(),
        isEmpty: async (f) => !(await this.hasValue(f)),
        skippedKeys: userSkippedKeys,
        reasons: [...blanked, ...sweepBlanks],
      });

      if (profile.resumePath && !resumeAttached && !(await this.controls.isResumeAttached())) {
        console.warn("[Lever] REVERIFY: resume is NOT attached after the final pass.");
      } else if (profile.resumePath) {
        console.log("[Lever] REVERIFY: resume is attached.");
      }
    }

    console.log("[Lever] Form filling completed.");
  }

  private async hasValue(f: FormField): Promise<boolean> {
    return !!(await this.controls.readFieldValue(f));
  }

  async submit(): Promise<void> {
    const page = this.getPage();
    console.log("[Lever] Submitting application form...");
    const submitBtn = page.locator("button.template-btn-submit").first();
    await submitBtn.scrollIntoViewIfNeeded().catch(() => {});
    await submitBtn.click();
    await randomSleep(1200, 2000);

    // Wait for navigation off the apply page (success) or an error re-render.
    for (let i = 0; i < 10; i++) {
      const url = page.url();
      if (/thanks|leverappid|leverapplicationid/i.test(url)) {
        console.log("[Lever] Submitted: redirect confirmed.");
        return;
      }
      const err = await page
        .locator(".error-message:visible")
        .first()
        .innerText()
        .catch(() => "");
      if (err && !/exceeds? the maximum upload size|too large|100MB/i.test(err)) {
        console.error(`[Lever] Submit error banner: ${escapePromptValue(err)}`);
        throw new Error(`Lever submit failed (form re-rendered): ${escapePromptValue(err)}`);
      }
      await randomSleep(1500, 2000);
    }
    console.warn("[Lever] Submit outcome not detected; treating as failed.");
    throw new Error("Lever submit: no success or error outcome detected after clicking submit");
  }
}

/**
 * Lever interaction layer — subclasses shared FormControls so the shared
 * Screener / audit machinery runs unchanged. Adds the location-dropdown picker.
 */
export class LeverControlStack extends FormControls {
  constructor(stagehand: Stagehand, tag: string) {
    super(stagehand, { tagName: tag });
  }

  override async readFieldValue(field: FormField): Promise<string> {
    if (field.kind === "select") {
      const page = this.getPage();
      try {
        return (await page.evaluate((id: string) => {
          const sel = document.getElementById(id) as HTMLSelectElement | null;
          if (!sel) return "";
          const idx = sel.selectedIndex;
          if (idx < 0) return "";
          return (sel.options[idx]?.textContent || "").replace(/\s+/g, " ").trim();
        }, field.id)) as string;
      } catch {
        return "";
      }
    }
    return super.readFieldValue(field);
  }

  async isResumeAttached(): Promise<boolean> {
    const page = this.getPage();
    return page
      .evaluate(() => {
        const input = document.querySelector(
          '#application-form input[type="file"][name="resume"]'
        ) as HTMLInputElement | null;
        if (!input) return true; // consumed by the board = attached
        if (input.files && input.files.length > 0) return true;
        const zone = document.querySelector(
          ".file-upload, [class*='upload'], [class*='resume-upload']"
        );
        const text = zone ? (zone.textContent || "") : "";
        return /attached|uploaded|reading|✓|Added/i.test(text);
      })
      .catch(() => false);
  }

  /** Fill the async location autocomplete. Type to load suggestions, then pick
   *  the row that matches the profile location; only a picked suggestion is a
   *  committed value. */
  async fillLeverLocation(value: string): Promise<boolean> {
    const page = this.getPage();
    try {
      const input = page.locator("#location-input").first();
      if (!(await input.isVisible().catch(() => false))) return false;
      await this.closeMenu();
      await randomSleep(150, 300);
      await input.click().catch(() => {});
      await input.focus().catch(() => {});
      await input.pressSequentially(value, { delay: 40 }).catch(async () => {
        await input.fill(value);
      });
      let rows: string[] = [];
      for (let i = 0; i < 10 && rows.length === 0; i++) {
        await randomSleep(900, 1200);
        rows = await this.locationRows();
      }
      if (!rows.length) {
        const shortToken = value.split(/[\s,]+/).filter((t) => t && t.length > 1)[0];
        if (shortToken && shortToken !== value.trim()) {
          await input.fill(shortToken);
          for (let i = 0; i < 8 && rows.length === 0; i++) {
            await randomSleep(900, 1200);
            rows = await this.locationRows();
          }
        }
      }
      if (rows.length) {
        const picked = pickLocationOption(value, rows);
        if (picked && (await this.clickLocationOption(picked))) {
          await randomSleep(400, 700);
          const committed = await this.readSelectedLocation();
          if (committed) return true;
        }
      }
      // Never commit the raw typed text; blank with a reason.
      await this.closeMenu();
      console.warn(`[L] No location suggestion matched "${value}".`);
      return false;
    } catch (err: any) {
      console.warn(`[L] fillLeverLocation failed: ${err?.message || err}`);
      return false;
    }
  }

  private async readSelectedLocation(): Promise<string> {
    const page = this.getPage();
    try {
      const v = await page.evaluate(() => {
        const li = document.querySelector("#selected-location") as HTMLInputElement | null;
        return li ? (li.value || "").trim() : "";
      });
      return (v as string) || "";
    } catch {
      return "";
    }
  }

  private async clickLocationOption(text: string): Promise<boolean> {
    const page = this.getPage();
    return page
      .evaluate((want: string) => {
        const rows = Array.from(
          document.querySelectorAll(".dropdown-results .dropdown-location")
        );
        for (const el of rows) {
          const t = (el.textContent || "").replace(/\s+/g, " ").trim();
          if (t === want) {
            (el as HTMLElement).click();
            return true;
          }
        }
        return false;
      }, text)
      .catch(() => false);
  }

  private async locationRows(): Promise<string[]> {
    const page = this.getPage();
    try {
      return await page.evaluate(() => {
        const out: string[] = [];
        for (const el of Array.from(
          document.querySelectorAll(".dropdown-location")
        )) {
          if ((el as HTMLElement).offsetParent === null) continue;
          const text = (el.textContent || "").replace(/\s+/g, " ").trim();
          if (text) out.push(text);
        }
        return out;
      });
    } catch {
      return [];
    }
  }
}