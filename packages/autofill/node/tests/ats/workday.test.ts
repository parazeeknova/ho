import { describe, it } from "node:test";
import assert from "node:assert/strict";
import {
  classifyWorkdayControl,
  isVoluntaryStepText,
  workdayApplyManuallyUrl,
  workdayCompanyFromHostname,
  workdayPostingUrl,
} from "../../ats/workday";

describe("workdayCompanyFromHostname", () => {
  it("extracts the company from a *.wd<N>.myworkdayjobs.com host", () => {
    assert.equal(workdayCompanyFromHostname("intel.wd1.myworkdayjobs.com"), "intel");
    assert.equal(workdayCompanyFromHostname("salesforce.wd12.myworkdayjobs.com"), "salesforce");
  });

  it("falls back to the first label for non-standard hosts", () => {
    assert.equal(workdayCompanyFromHostname("careers.example.com"), "careers");
    assert.equal(workdayCompanyFromHostname(""), "");
  });
});

describe("workdayPostingUrl / workdayApplyManuallyUrl", () => {
  it("strips an applyManually suffix back to the posting URL", () => {
    assert.equal(
      workdayPostingUrl(
        "https://intel.wd1.myworkdayjobs.com/en-US/External/job/Senior-AI_Engineer_JR0282751/apply/applyManually",
      ),
      "https://intel.wd1.myworkdayjobs.com/en-US/External/job/Senior-AI_Engineer_JR0282751",
    );
  });

  it("strips a plain /apply suffix too", () => {
    assert.equal(
      workdayPostingUrl(
        "https://salesforce.wd12.myworkdayjobs.com/External_Career_Site/job/Intern_JR337715/apply",
      ),
      "https://salesforce.wd12.myworkdayjobs.com/External_Career_Site/job/Intern_JR337715",
    );
  });

  it("derives the deterministic applyManually URL", () => {
    assert.equal(
      workdayApplyManuallyUrl(
        "https://intel.wd1.myworkdayjobs.com/en-US/External/job/Senior-AI_Engineer_JR0282751",
      ),
      "https://intel.wd1.myworkdayjobs.com/en-US/External/job/Senior-AI_Engineer_JR0282751/apply/applyManually",
    );
  });
});

describe("isVoluntaryStepText", () => {
  it("flags voluntary disclosure / demographic / self-identification screens", () => {
    assert.equal(isVoluntaryStepText("Voluntary Self-Identification Continue"), true);
    assert.equal(isVoluntaryStepText("Demographic Information (optional)"), true);
    assert.equal(isVoluntaryStepText("EEOC Disability self-identification"), true);
  });

  it("does not flag a normal question step", () => {
    assert.equal(isVoluntaryStepText("Personal Information First Name * Last Name *"), false);
    assert.equal(isVoluntaryStepText(""), false);
  });
});

describe("classifyWorkdayControl", () => {
  it("classifies comboboxes by role or aria-autocomplete", () => {
    assert.equal(
      classifyWorkdayControl({ tag: "INPUT", type: "text", role: "combobox" }),
      "combobox",
    );
    assert.equal(
      classifyWorkdayControl({ tag: "INPUT", type: "text", ariaAutocomplete: true }),
      "combobox",
    );
  });

  it("classifies radio/checkbox/select by tag/type", () => {
    assert.equal(classifyWorkdayControl({ tag: "INPUT", type: "radio" }), "radio");
    assert.equal(classifyWorkdayControl({ tag: "INPUT", type: "checkbox" }), "checkbox");
    assert.equal(classifyWorkdayControl({ tag: "SELECT" }), "select");
  });

  it("defaults plain text inputs to text", () => {
    assert.equal(classifyWorkdayControl({ tag: "INPUT", type: "email" }), "text");
    assert.equal(classifyWorkdayControl({ tag: "TEXTAREA" }), "text");
  });
});
