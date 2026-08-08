import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  atsApiJobContext,
  classifyFlow,
  cleanObserveLabel,
  extractBalancedObject,
  extractJsonJobContext,
  extractQuestionsFromJsonObject,
  isVoluntaryText,
  parseJsonQuestions,
} from "./generic";
import type { FlowProbe } from "./generic";

describe("classifyFlow", () => {
  const probe = (p: Partial<FlowProbe>): FlowProbe => ({
    formDetected: false,
    applyDetected: false,
    wizardDetected: false,
    gateDetected: false,
    ...p,
  });

  it("gate wins over everything else (password + account form)", () => {
    assert.equal(classifyFlow(probe({ gateDetected: true, formDetected: true })), "gate");
  });

  it("wizard wins over a live form (multi-step with Continue)", () => {
    assert.equal(classifyFlow(probe({ wizardDetected: true, formDetected: true })), "wizard");
  });

  it("a live form without navigation is just a form", () => {
    assert.equal(classifyFlow(probe({ formDetected: true })), "form");
  });

  it("a JD page (apply detected, no form) is apply", () => {
    assert.equal(classifyFlow(probe({ applyDetected: true })), "apply");
  });

  it("falls back to form when nothing is detected (walker decides)", () => {
    assert.equal(classifyFlow(probe({})), "form");
  });
});

describe("cleanObserveLabel", () => {
  it("strips leading fill/type/enter verbs, possessives, and trailing field/input nouns", () => {
    assert.equal(cleanObserveLabel("fill the First Name field"), "First Name");
    assert.equal(cleanObserveLabel("Type your email address"), "email address");
    assert.equal(cleanObserveLabel("enter the Company input"), "Company");
    assert.equal(cleanObserveLabel("Please fill in the phone number textbox"), "phone number");
  });

  it("passes already-clean labels through", () => {
    assert.equal(cleanObserveLabel("First Name"), "First Name");
    assert.equal(cleanObserveLabel(""), "");
  });
});

describe("isVoluntaryText", () => {
  it("flags voluntary / demographic / EEOC text", () => {
    assert.equal(isVoluntaryText("Voluntary Self-Identification Continue"), true);
    assert.equal(isVoluntaryText("Demographic Information (optional)"), true);
    assert.equal(isVoluntaryText("Equal Opportunity Employer Statement"), true);
  });

  it("does not flag ordinary step text", () => {
    assert.equal(isVoluntaryText("Personal Information First Name * Last Name *"), false);
    assert.equal(isVoluntaryText(""), false);
  });
});

describe("extractBalancedObject", () => {
  it("extracts a balanced JSON object after the marker", () => {
    const html = '<script>window.__remixContext = {"a":{"b":[1,2]}};</script>';
    assert.equal(extractBalancedObject(html, /window\.__remixContext\s*=\s*/), '{"a":{"b":[1,2]}}');
  });

  it("handles strings containing braces and escaped quotes", () => {
    const html = 'x = {"s":"}{ \\"quoted\\" ","n":1};';
    assert.equal(extractBalancedObject(html, /x\s*=\s*/), '{"s":"}{ \\"quoted\\" ","n":1}');
  });

  it("returns null when no marker or no object", () => {
    assert.equal(extractBalancedObject("<html></html>", /window\.__remixContext/), null);
    assert.equal(
      extractBalancedObject("window.__remixContext = nope;", /window\.__remixContext\s*=\s*/),
      null,
    );
  });
});

describe("extractQuestionsFromJsonObject", () => {
  it("extracts remix-style questions with fields and option values", () => {
    const obj = {
      state: {
        loaderData: {
          jobPost: {
            questions: [
              {
                required: true,
                label: "Preferred First Name",
                fields: [{ name: "preferred_name", type: "input_text" }],
              },
              {
                required: false,
                label: "Race",
                fields: [
                  {
                    name: "race",
                    type: "multi_value_single_select",
                    values: [
                      { label: "Asian", value: "2" },
                      { label: "White", value: "5" },
                    ],
                  },
                ],
              },
            ],
            eeoc_sections: [
              {
                questions: [
                  {
                    required: false,
                    label: "Veteran Status",
                    fields: [{ name: "veteran_status", type: "multi_value_single_select" }],
                  },
                ],
              },
            ],
          },
        },
      },
    };
    const out = extractQuestionsFromJsonObject(obj);
    assert.equal(out.length, 3);
    const pref = out.find((f) => f.name === "preferred_name");
    assert.ok(pref);
    assert.equal(pref.label, "Preferred First Name");
    assert.equal(pref.required, true);
    const race = out.find((f) => f.name === "race");
    assert.ok(race);
    assert.deepEqual(race.options, ["Asian", "White"]);
    assert.ok(out.some((f) => f.name === "veteran_status"));
  });

  it("drops file / resume / cover-letter entries", () => {
    const obj = {
      questions: [
        { required: true, label: "Resume", fields: [{ name: "resume", type: "input_file" }] },
        {
          required: false,
          label: "Cover Letter",
          fields: [{ name: "cover_letter", type: "input_file" }],
        },
        { required: false, label: "LinkedIn", fields: [{ name: "linkedin", type: "input_text" }] },
      ],
    };
    const out = extractQuestionsFromJsonObject(obj);
    assert.equal(out.length, 1);
    assert.equal(out[0].name, "linkedin");
  });
});

describe("parseJsonQuestions", () => {
  it("parses a remix blob out of full HTML", () => {
    const html = `<html><script>window.__remixContext = {"state":{"loaderData":{"r":{"jobPost":{"questions":[{"required":true,"label":"First Name","fields":[{"name":"first_name","type":"input_text"}]}]}}}}};</script></html>`;
    const out = parseJsonQuestions(html);
    assert.equal(out.length, 1);
    assert.equal(out[0].name, "first_name");
    assert.equal(out[0].label, "First Name");
  });

  it("parses a __NEXT_DATA__ blob", () => {
    const html = `<script id="__NEXT_DATA__" type="application/json">{"props":{"pageProps":{"job":{"title":"Engineer","questions":[{"label":"GitHub","name":"github","type":"input_text"}]}}}}</script>`;
    const out = parseJsonQuestions(html);
    assert.equal(out.length, 1);
    assert.equal(out[0].name, "github");
  });

  it("returns an empty list when no question model exists", () => {
    assert.deepEqual(parseJsonQuestions("<html><body>form</body></html>"), []);
  });
});

describe("extractJsonJobContext", () => {
  it("extracts title/company/location from a remix jobPost", () => {
    const html = `<script>window.__remixContext = {"state":{"loaderData":{"r":{"jobPost":{"title":"Backend Engineer","company_name":"Acme","job_post_location":"Remote / Berlin"}}}}};</script>`;
    const ctx = extractJsonJobContext(html);
    assert.ok(ctx);
    assert.equal(ctx.title, "Backend Engineer");
    assert.equal(ctx.company, "Acme");
    assert.equal(ctx.location, "Remote / Berlin");
  });

  it("extracts a posting title from __NEXT_DATA__", () => {
    const html = `<script>__NEXT_DATA__ = {"props":{"pageProps":{"posting":{"title":"ML Engineer","descriptionHtml":"<p>Build models</p>"}}}};</script>`;
    const ctx = extractJsonJobContext(html);
    assert.ok(ctx);
    assert.equal(ctx.title, "ML Engineer");
    assert.match(ctx.description, /Build models/);
  });

  it("returns null when nothing is parseable", () => {
    assert.equal(extractJsonJobContext("<html><body>hi</body></html>"), null);
  });
});

describe("atsApiJobContext", () => {
  it("parses the Ashby posting API response into job context", async () => {
    const origFetch = globalThis.fetch;
    // Mock the fetch so the test is hermetic (no network).
    (globalThis as any).fetch = async (_url: string) => ({
      ok: true,
      status: 200,
      async json() {
        return {
          jobs: [
            {
              id: "abc-123",
              title: "AI Platform Engineer",
              locationName: "Remote",
              team: "Platform",
              descriptionHtml: "<h2>About</h2><p>Build AI infra.</p>",
            },
          ],
        };
      },
    });
    try {
      const ctx = await atsApiJobContext("https://jobs.ashbyhq.com/supabase/abc-123");
      assert.ok(ctx);
      assert.equal(ctx?.title, "AI Platform Engineer");
      assert.equal(ctx?.location, "Remote");
      assert.match(ctx?.description ?? "", /Build AI infra/);
    } finally {
      globalThis.fetch = origFetch;
    }
  });

  it("returns null for non-ATS URLs", async () => {
    const ctx = await atsApiJobContext("https://example.com/careers");
    assert.equal(ctx, null);
  });
});
