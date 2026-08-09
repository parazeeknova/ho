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
