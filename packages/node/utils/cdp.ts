/**
 * File-input upload helpers for Stagehand v3's understudy Page.
 *
 * Both Stagehand's own `setInputFiles` (which, for a `cdpUrl`-attached browser
 * like Steel, routes through JS File-object injection) and a raw CDP
 * `DOM.setFileInputFiles` can leave `input.files` EMPTY on Greenhouse's
 * React-controlled `visually-hidden` file inputs — the submit then fails with
 * "Resume/CV*". The approach that reliably registers is building a `File` from
 * the byte payload inside the page, assigning it to the input via a
 * `DataTransfer`, and dispatching bubbling `change`/`input` events exactly the
 * way a real user's file picker would.
 */

/**
 * Set a file input's value by injecting the file bytes into the page (base64)
 * and assigning them via a DataTransfer + change/input events. Returns true
 * when the input now holds the file (files.length > 0). Works whether or not
 * the browser can see the local filesystem — it never needs the path on the
 * Chrome host.
 */
export async function setFileInputViaDataTransfer(
  page: any,
  selector: string,
  filePath: string,
  fileName?: string,
): Promise<boolean> {
  try {
    const fs = await import("fs");
    if (!fs.existsSync(filePath)) return false;
    const buf = fs.readFileSync(filePath);
    const name = fileName || filePath.split(/[\\/]/).pop() || "resume.pdf";
    const mime = name.toLowerCase().endsWith(".pdf")
      ? "application/pdf"
      : "application/octet-stream";

    const result = await page.evaluate(
      async ({
        b64,
        fname,
        mtype,
        sel,
      }: {
        b64: string;
        fname: string;
        mtype: string;
        sel: string;
      }) => {
        const input = document.querySelector(sel) as HTMLInputElement | null;
        if (!input) return "NO_INPUT";
        const bin = atob(b64);
        const bytes = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
        const file = new File([bytes], fname, { type: mtype });
        const dt = new DataTransfer();
        dt.items.add(file);
        input.files = dt.files;
        input.dispatchEvent(new Event("change", { bubbles: true }));
        input.dispatchEvent(new Event("input", { bubbles: true }));
        await new Promise((r) => setTimeout(r, 1500));
        return JSON.stringify({
          files: input.files.length,
          first: input.files[0]?.name ?? null,
        });
      },
      { b64: buf.toString("base64"), fname: name, mtype: mime, sel: selector },
    );

    if (typeof result === "string" && result.startsWith("{")) {
      const parsed = JSON.parse(result);
      return parsed.files > 0;
    }
    return false;
  } catch {
    return false;
  }
}

/**
 * Check a checkbox with a REAL trusted mouse click via CDP. Setting
 * `cb.checked = true` then dispatching synthetic events does NOT work on
 * Greenhouse: React's controlled checkbox ignores the programmatic property
 * change, and a synthetic `click` TOGGLES the box back to unchecked (click's
 * default action flips `checked`), so the submit still fails with "Please
 * accept the terms to proceed". A real Input.dispatchMouseEvent press+release
 * on the checkbox's visible label/visual (scrolled into view) is processed
 * exactly like a user's click, toggles `checked`, and fires React's onChange.
 * Returns true when the checkbox is checked afterwards.
 */
export async function checkCheckboxViaCdpClick(page: any, selector: string): Promise<boolean> {
  const session = page?.mainSession ?? page?.session ?? null;
  if (!session || typeof session.send !== "function") return false;
  try {
    await session.send("DOM.enable").catch(() => {});
    await session.send("Runtime.enable").catch(() => {});

    // Scroll the checkbox's visible label/visual into view FIRST, let the
    // scroll settle, THEN measure its center (measuring immediately after
    // scrollIntoView returns stale coordinates while the page is still
    // scrolling). Greenhouse wraps the input in a custom checkbox with a
    // sibling visual; clicking the INPUT directly can be intercepted by an
    // overlay, so click the nearest clickable visual/label element instead.
    const scrolled = await page.evaluate((sel: string) => {
      const cb = document.querySelector(sel) as HTMLInputElement | null;
      if (!cb) return false;
      const parent = cb.parentElement;
      const vis = parent?.querySelector(
        '.checkbox__visual, [class*="visual"], [class*="box"], [class*="checkmark"]',
      ) as HTMLElement | null;
      const el: HTMLElement = vis || (parent as HTMLElement) || cb;
      el.scrollIntoView({ block: "center", inline: "center" });
      return true;
    }, selector);
    if (!scrolled) return false;
    await new Promise((r) => setTimeout(r, 1500));

    const target = await page.evaluate((sel: string) => {
      const cb = document.querySelector(sel) as HTMLInputElement | null;
      if (!cb) return null;
      const parent = cb.parentElement;
      const vis = parent?.querySelector(
        '.checkbox__visual, [class*="visual"], [class*="box"], [class*="checkmark"]',
      ) as HTMLElement | null;
      const el: HTMLElement = vis || (parent as HTMLElement) || cb;
      const r = el.getBoundingClientRect();
      if (r.width < 2 || r.height < 2) return null;
      return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
    }, selector);
    if (!target) return false;

    await session.send("Input.dispatchMouseEvent", {
      type: "mouseMoved",
      x: target.x,
      y: target.y,
    });
    await session.send("Input.dispatchMouseEvent", {
      type: "mousePressed",
      x: target.x,
      y: target.y,
      button: "left",
      clickCount: 1,
    });
    await session.send("Input.dispatchMouseEvent", {
      type: "mouseReleased",
      x: target.x,
      y: target.y,
      button: "left",
      clickCount: 1,
    });

    const checked = await page.evaluate((sel: string) => {
      const cb = document.querySelector(sel) as HTMLInputElement | null;
      return cb ? cb.checked : false;
    }, selector);
    return checked === true;
  } catch {
    return false;
  }
}
