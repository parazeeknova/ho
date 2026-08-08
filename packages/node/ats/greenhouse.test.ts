import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  checkboxAction,
  chooseOption,
  editDistance,
  fieldKey,
  isLocationAutocomplete,
  isCoverLetterField,
  isProfileDrivenField,
  normalizeOptionText,
  parseRemixJobContext,
  parseRemixQuestionsModel,
  mergeFormInventory,
  selectCandidates,
  translateToDate,
  unprocessedFields,
  xpathStringLiteral,
} from "./greenhouse";
import type { FormField } from "./greenhouse";

describe("normalizeOptionText", () => {
  it("normalizes case and whitespace", () => {
    assert.equal(normalizeOptionText("  I am not  a veteran\n "), "i am not a veteran");
  });
});

describe("xpathStringLiteral", () => {
  it("keeps apostrophes unescaped inside a double-quoted literal", () => {
    // Regression: "Bachelor's Degree" was previously escaped with a backslash
    // (bachelor\'s), which XPath treats as a literal backslash and never matches.
    assert.equal(xpathStringLiteral("bachelor's degree"), '"bachelor\'s degree"');
  });

  it("switches to single quotes when the text contains double quotes", () => {
    assert.equal(xpathStringLiteral('say "hi"'), "'say \"hi\"'");
  });

  it("falls back to concat when both quote types are present", () => {
    const out = xpathStringLiteral("a \"quote\" and 'apostrophe'");
    assert.ok(out.startsWith("concat("));
    assert.ok(out.includes("'\"'"));
  });
});

describe("selectCandidates", () => {
  it("returns the raw answer, clause, and leading yes/no token", () => {
    const out = selectCandidates("Yes, I have experience with AWS");
    assert.deepEqual(out, ["Yes, I have experience with AWS", "Yes"]);
  });

  it("returns nothing for empty answers", () => {
    assert.deepEqual(selectCandidates("   "), []);
  });

  it("deduplicates candidates", () => {
    assert.deepEqual(selectCandidates("No"), ["No"]);
  });
});

describe("chooseOption", () => {
  it("prefers an unambiguous exact match", () => {
    const picked = chooseOption(["No"], ["Yes", "No", "I don't wish to answer"]);
    assert.equal(picked, "No");
  });

  it("falls back to an unambiguous substring match", () => {
    const picked = chooseOption(["No"], ["Yes", "No, I don't have a disability"]);
    assert.equal(picked, "No, I don't have a disability");
  });

  it("rejects ambiguous substring matches", () => {
    const picked = chooseOption(["No"], ["No, I am not a veteran", "No, I am a veteran"]);
    assert.equal(picked, null);
  });

  it("resolves disability survey with a decline option present", () => {
    const options = [
      "Yes, I have a disability, or have had one in the past",
      "No, I don't have a disability, or have not had one in the past",
      "I don't wish to answer",
    ];
    assert.equal(chooseOption(["No"], options), options[1]);
    assert.equal(chooseOption(["Yes"], options), options[0]);
  });

  it("resolves veteran survey against the negation option", () => {
    const options = [
      "I identify as a protected veteran",
      "I am not a protected veteran",
      "I don't wish to answer",
    ];
    assert.equal(chooseOption(["No"], options), options[1]);
  });

  it("rejects when nothing matches", () => {
    assert.equal(chooseOption(["30-40"], ["Yes", "No"]), null);
  });

  it("is case-insensitive", () => {
    assert.equal(chooseOption(["yes"], ["YES", "NO"]), "YES");
  });

  it("tries candidates in order", () => {
    const options = ["Male", "Female", "Other"];
    assert.equal(chooseOption(["Male, cisgender", "Male"], options), "Male");
  });

  it("maps a No answer onto a disability negation option", () => {
    // The "I do not want to answer" decline must be excluded so the "no"
    // substring is unambiguous (it also appears inside "not").
    const options = [
      "Yes, I have a disability, or have had one in the past",
      "No, I do not have a disability and have not had one in the past",
      "I do not want to answer",
    ];
    assert.equal(chooseOption(["No"], options), options[1]);
    assert.equal(chooseOption(["Yes"], options), options[0]);
  });

  it("forgives a small typo via edit distance (unambiguous only)", () => {
    const options = ["Bachelor's Degree", "Master's Degree", "PhD"];
    assert.equal(chooseOption(["bachlors"], options), "Bachelor's Degree");
    // A typo that lands near two options must not pick one.
    assert.equal(chooseOption(["degre"], ["Bachelor's Degree", "Master's Degree"]), null);
  });

  it("editDistance is symmetric and handles unicode", () => {
    assert.equal(editDistance("kitten", "sitting"), 3);
    assert.equal(editDistance("", "abc"), 3);
    assert.equal(editDistance("abc", "abc"), 0);
    assert.equal(editDistance("mañana", "manana"), 1);
  });
});

describe("isLocationAutocomplete (dropdown-typing guard)", () => {
  const mk = (label: string, kind: string, options: string[] = []): any => ({
    label,
    id: "x",
    kind,
    required: false,
    options,
    optionTargets: [],
  });

  it("accepts a genuine current-city autocomplete (no static options, anchored label)", () => {
    assert.equal(isLocationAutocomplete(mk("Location (City)", "select")), true);
    assert.equal(isLocationAutocomplete(mk("Candidate Location", "select")), true);
    assert.equal(isLocationAutocomplete(mk("City", "select")), true);
    assert.equal(isLocationAutocomplete(mk("What is your current location?", "select")), true);
  });

  it("rejects a pick-list select that happens to mention location", () => {
    // Cloudflare's relocate question has options — it is a dropdown, never
    // typed into.
    const relocate = mk(
      "Do you currently live or are you willing to relocate to the job’s location?",
      "select",
      ["I currently live in this job's location.", "I am willing to relocate."],
    );
    assert.equal(isLocationAutocomplete(relocate), false);
    // Any select with static options is a pick-list.
    assert.equal(isLocationAutocomplete(mk("Location", "select", ["A", "B"])), false);
  });

  it("rejects non-location anchored labels and non-select kinds", () => {
    assert.equal(isLocationAutocomplete(mk("Where do you work currently?", "select")), false);
    assert.equal(isLocationAutocomplete(mk("Location", "text")), false);
    assert.equal(isLocationAutocomplete(mk("Location", "radio")), false);
  });
});

describe("checkboxAction (structural consent/opt-in semantics, label-agnostic)", () => {
  const mk = (label: string, required: boolean, nTargets: number): any => ({
    label,
    id: "x",
    kind: "checkbox",
    required,
    options: [],
    optionTargets: Array.from({ length: nTargets }, (_, i) => ({
      text: `opt ${i}`,
      name: "x[]",
      value: String(i),
    })),
  });

  it("accepts a required single-option checkbox (any phrasing)", () => {
    assert.equal(
      checkboxAction(mk("Please review and acknowledge the Privacy Policy", true, 1)),
      "accept",
    );
    assert.equal(checkboxAction(mk("I agree to the Code of Conduct", true, 1)), "accept");
    assert.equal(checkboxAction(mk("Terms and conditions", true, 1)), "accept");
  });

  it("leaves an optional single-option checkbox unchecked (opt-in)", () => {
    assert.equal(checkboxAction(mk("Send me marketing emails", false, 1)), "leave");
    assert.equal(checkboxAction(mk("Subscribe to newsletter", false, 1)), "leave");
  });

  it("asks for a multi-option checkbox (real multi-select question)", () => {
    assert.equal(checkboxAction(mk("Which teams interest you?", false, 3)), "ask");
    assert.equal(checkboxAction(mk("Which teams interest you?", true, 3)), "ask");
    // Non-checkbox fields are never auto-handled by these rules.
    assert.equal(checkboxAction({ ...mk("x", true, 1), kind: "select" }), "ask");
  });
});

describe("parseRemixQuestionsModel", () => {
  const jobPost = {
    questions: [
      {
        required: true,
        label: "Preferred First Name",
        fields: [{ name: "preferred_name", type: "input_text" }],
      },
      {
        required: true,
        label: "Candidate Location",
        fields: [{ name: "candidate_location", type: "input_text" }],
      },
      {
        required: false,
        label: "LinkedIn Profile",
        fields: [{ name: "question_65822505", type: "input_text" }],
      },
      {
        required: false,
        label: "Cover Letter",
        fields: [{ name: "cover_letter", type: "input_file" }],
      },
    ],
    eeoc_sections: [
      {
        description: "...",
        questions: [
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
                  { label: "Decline To Self Identify", value: "8" },
                ],
              },
            ],
          },
        ],
      },
    ],
  };
  const html = `<html><script>window.__remixContext = {"state":{"loaderData":{"root":{},"route":{"jobPost":${JSON.stringify(
    jobPost,
  )}}}}};</script></html>`;

  it("extracts core questions with names, types and required flags", () => {
    const model = parseRemixQuestionsModel(html);
    assert.ok(model);
    const pref = model.find((f) => f.name === "preferred_name");
    assert.ok(pref);
    assert.equal(pref.label, "Preferred First Name");
    assert.equal(pref.required, true);
    assert.equal(pref.kind, "input_text");
  });

  it("extracts EEOC questions with their exact option values", () => {
    const model = parseRemixQuestionsModel(html);
    assert.ok(model);
    const race = model.find((f) => f.name === "race");
    assert.ok(race);
    assert.deepEqual(race.options, ["Asian", "White", "Decline To Self Identify"]);
    assert.equal(race.kind, "multi_value_single_select");
  });

  it("returns null for pages without __remixContext (legacy boards)", () => {
    assert.equal(parseRemixQuestionsModel("<html><body>legacy form</body></html>"), null);
  });
});

describe("parseRemixJobContext", () => {
  const jobPost = {
    title: "Software Engineer (TypeScript/JavaScript)",
    company_name: "Databento",
    job_post_location: "Remote / Boston / Salt Lake City / San Francisco / New York",
  };
  const html = `<html><script>window.__remixContext = {"state":{"loaderData":{"root":{},"route":{"jobPost":${JSON.stringify(
    jobPost,
  )}}}}};</script></html>`;

  it("extracts title, company and location from the Remix JSON", () => {
    const ctx = parseRemixJobContext(html);
    assert.ok(ctx);
    assert.equal(ctx.title, "Software Engineer (TypeScript/JavaScript)");
    assert.equal(ctx.company, "Databento");
    assert.equal(ctx.location, "Remote / Boston / Salt Lake City / San Francisco / New York");
  });

  it("returns null for legacy pages without __remixContext", () => {
    assert.equal(parseRemixJobContext("<html><body>legacy</body></html>"), null);
  });
});

describe("mergeFormInventory", () => {
  const jsonFields = [
    {
      name: "preferred_name",
      label: "Preferred First Name",
      kind: "input_text",
      required: true,
      options: [],
    },
    {
      name: "candidate_location",
      label: "Candidate Location",
      kind: "input_text",
      required: true,
      options: [],
    },
    {
      name: "race",
      label: "Race",
      kind: "multi_value_single_select",
      required: false,
      options: ["Asian", "White", "Decline To Self Identify"],
    },
    {
      // Listed by the board JSON but never rendered — must be dropped.
      name: "social_security",
      label: "Social Security Number",
      kind: "input_text",
      required: false,
      options: [],
    },
  ];

  const domFields: FormField[] = [
    {
      label: "Preferred First Name",
      id: "preferred-name",
      kind: "text",
      required: false,
      options: [],
      optionTargets: [],
      name: "preferred_name",
    },
    {
      label: "Location (City)",
      id: "candidate-location",
      kind: "select",
      required: false,
      options: [],
      optionTargets: [],
      name: "candidate_location",
    },
    {
      label: "Country",
      id: "country",
      kind: "select",
      required: false,
      options: [],
      optionTargets: [],
      name: "country",
    },
    {
      label: "Race",
      id: "race",
      kind: "radio",
      required: false,
      options: ["Asian", "White"],
      optionTargets: [
        { text: "Asian", name: "race", value: "2" },
        { text: "White", name: "race", value: "5" },
      ],
      name: "race",
    },
  ];

  it("enriches DOM fields with JSON options and required flags", () => {
    const merged = mergeFormInventory(jsonFields, domFields);
    const location = merged.find((f) => f.name === "candidate_location");
    assert.ok(location);
    assert.equal(location.required, true);
    assert.equal(location.label, "Location (City)");
    assert.equal(location.id, "candidate-location");
  });

  it("prefers DOM group knowledge (radio) over the JSON select kind", () => {
    const merged = mergeFormInventory(jsonFields, domFields);
    const race = merged.find((f) => f.name === "race");
    assert.ok(race);
    assert.equal(race.kind, "radio");
    assert.deepEqual(race.options, ["Asian", "White", "Decline To Self Identify"]);
    assert.equal(race.optionTargets.length, 2);
  });

  it("keeps DOM-only fields (country) when they have no JSON match", () => {
    const merged = mergeFormInventory(jsonFields, domFields);
    const country = merged.find((f) => f.name === "country");
    assert.ok(country);
    assert.equal(country.label, "Country");
  });

  it("drops JSON-only questions that are not rendered in the DOM", () => {
    const merged = mergeFormInventory(jsonFields, domFields);
    const ssn = merged.find((f) => f.name === "social_security");
    assert.equal(ssn, undefined);
    assert.ok(!merged.some((f) => f.label === "Social Security Number"));
  });

  it("falls back to DOM-only inventory when there is no JSON model", () => {
    const merged = mergeFormInventory(null, domFields);
    assert.equal(merged.length, domFields.length);
  });
});

describe("isProfileDrivenField", () => {
  const mk = (label: string, kind: FormField["kind"] = "text"): FormField => ({
    label,
    id: "x",
    kind,
    required: false,
    options: [],
    optionTargets: [],
  });

  it("marks identity fields (first name, email, phone)", () => {
    assert.equal(isProfileDrivenField(mk("First Name")), true);
    assert.equal(isProfileDrivenField(mk("What is your email address?")), true);
    assert.equal(isProfileDrivenField(mk("Phone (e.g. +91 99999 99999)")), true);
  });

  it("marks profile fields (linkedin, github, website)", () => {
    assert.equal(isProfileDrivenField(mk("LinkedIn Profile")), true);
    assert.equal(isProfileDrivenField(mk("Portfolio URL")), true);
  });

  it("does not mark ordinary screener questions", () => {
    assert.equal(isProfileDrivenField(mk("Race", "radio")), false);
    assert.equal(isProfileDrivenField(mk("Are you currently based in Europe?", "select")), false);
    assert.equal(isProfileDrivenField(mk("")), false);
  });
});

describe("isCoverLetterField", () => {
  const mk = (label: string, kind: FormField["kind"] = "text"): FormField => ({
    label,
    id: "x",
    kind,
    required: false,
    options: [],
    optionTargets: [],
  });

  it("marks cover-letter prompts and open blurb textareas", () => {
    assert.equal(isCoverLetterField(mk("Cover Letter")), true);
    assert.equal(isCoverLetterField(mk("Please add your cover letter below")), true);
    // "Anything else" / "Tell us about yourself" prompts ARE cover-letter
    // holders — they ask for open prose about the candidate.
    assert.equal(isCoverLetterField(mk("Anything else you would like us to know?")), true);
    assert.equal(isCoverLetterField(mk("Tell us about yourself")), true);
    // Bare "Additional Information" is a conditional companion to a sourcing
    // select, not a cover-letter field: it must stay blank (the walk never
    // fills it with a signed letter), so it is NOT a cover-letter field.
    assert.equal(isCoverLetterField(mk("Additional Information")), false);
  });

  it("never marks structured questions (selects, radios, checkboxes)", () => {
    assert.equal(isCoverLetterField(mk("Do you need visa sponsorship?", "select")), false);
    assert.equal(isCoverLetterField(mk("Race", "radio")), false);
    assert.equal(isCoverLetterField(mk("")), false);
  });
});

describe("fieldKey / unprocessedFields (iterative re-scan walk)", () => {
  const mk = (label: string, kind: FormField["kind"], id: string): FormField => ({
    label,
    id,
    kind,
    required: false,
    options: [],
    optionTargets: [],
  });

  it("keys by label + kind + id", () => {
    const a = mk("Race", "radio", "race");
    assert.equal(fieldKey(a), "race|radio|race");
  });

  it("returns fields not yet processed and excludes processed ones", () => {
    const hispanic = mk("Are you Hispanic/Latino?", "select", "hispanic_ethnicity");
    const race = mk("Race", "radio", "race");
    const processed = new Set([fieldKey(hispanic)]);
    const out = unprocessedFields([hispanic, race], processed);
    assert.deepEqual(
      out.map((f) => f.id),
      ["race"],
    );
  });

  it("picks up a field revealed in a later pass (conditional race question)", () => {
    // Pass 1: only hispanic present.
    const pass1 = [mk("Are you Hispanic/Latino?", "select", "hispanic_ethnicity")];
    const processed = new Set<string>();
    const round1 = unprocessedFields(pass1, processed);
    round1.forEach((f) => processed.add(fieldKey(f)));
    // After answering "No", Race is revealed.
    const pass2 = [...pass1, mk("Race", "radio", "race")];
    const round2 = unprocessedFields(pass2, processed);
    assert.deepEqual(
      round2.map((f) => f.id),
      ["race"],
    );
    // And once processed it is never returned again.
    round2.forEach((f) => processed.add(fieldKey(f)));
    const round3 = unprocessedFields(pass2, processed);
    assert.deepEqual(round3, []);
  });
});

describe("translateToDate", () => {
  it("maps immediate/now to today", () => {
    const d = translateToDate("Immediately")!;
    const now = new Date();
    assert.equal(d.getFullYear(), now.getFullYear());
    assert.equal(d.getMonth(), now.getMonth());
    assert.equal(d.getDate(), now.getDate());
  });

  it("adds relative offsets", () => {
    const d = translateToDate("in 2 weeks")!;
    const expected = new Date();
    expected.setDate(expected.getDate() + 14);
    assert.equal(d.getFullYear(), expected.getFullYear());
    assert.equal(d.getMonth(), expected.getMonth());
    assert.equal(d.getDate(), expected.getDate());
  });

  it("parses explicit dates", () => {
    const d = translateToDate("2026-08-15")!;
    assert.equal(d.getFullYear(), 2026);
    assert.equal(d.getMonth(), 7);
    assert.equal(d.getDate(), 15);
  });

  it("returns null for non-dates", () => {
    assert.equal(translateToDate("flexible"), null);
    assert.equal(translateToDate(""), null);
  });

  it("parses a bare graduation year as Dec 31", () => {
    const d = translateToDate("2027")!;
    assert.equal(d.getFullYear(), 2027);
    assert.equal(d.getMonth(), 11);
    assert.equal(d.getDate(), 31);
  });
});
