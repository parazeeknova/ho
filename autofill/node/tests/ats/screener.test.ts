import { describe, it, beforeEach } from "node:test";
import assert from "node:assert/strict";
import { Screener } from "../../ats/shared/screener";
import {
  getDeferredFieldCount,
  resetDeferredFieldCount,
  getBlankedRequiredCount,
  resetBlankedRequiredCount,
  setBlankedRequiredCount,
} from "../../ats/shared/screener";
import { valuesConsistent } from "../../ats/shared/matching";
import type { FormField } from "../../ats/shared/model";

const textField = (label: string): FormField => ({
  label,
  id: "f1",
  kind: "text",
  required: true,
  options: [],
  optionTargets: [],
});

const fakeControls = () =>
  ({
    tagName: "Test",
    fillByKind: async () => true,
    readFieldValue: async () => "filled",
    readSelectOptions: async () => [],
    closeMenu: async () => {},
    readSelectValue: async () => "filled",
    fillAsyncAutocomplete: async () => true,
  }) as any;

const profile: any = {
  firstName: "Jane",
  lastName: "Doe",
  email: "jane@example.com",
  phone: "+123",
  location: "Berlin",
  resumePath: null,
};

describe("deferral counter (screener)", () => {
  beforeEach(() => resetDeferredFieldCount());

  it("increments when answer_question defers (overnight)", async () => {
    const rpc = async () => {
      throw new Error("AUTOFILL_DEFER: Question deferred");
    };
    const screener = new Screener(fakeControls(), "Test", profile, rpc);
    await screener.process(textField("Favorite color"), [], [], new Set());
    assert.equal(getDeferredFieldCount(), 1);
  });

  it("does not increment on a normal resolved question", async () => {
    const rpc = async () => ({ answer: "Blue", source: "llm" });
    const screener = new Screener(fakeControls(), "Test", profile, rpc);
    await screener.process(textField("Favorite color"), [], [], new Set());
    assert.equal(getDeferredFieldCount(), 0);
  });
});

describe("valuesConsistent", () => {
  it("accepts an exact match", () => {
    assert.equal(valuesConsistent("No", "No"), true);
    assert.equal(valuesConsistent("Yes", "Yes"), true);
  });

  it("accepts the answer as a leading token/phrase of the committed option", () => {
    assert.equal(
      valuesConsistent("No", "No, I will require immediate visa sponsorship"),
      true
    );
    assert.equal(
      valuesConsistent("Yes", "Yes, I am willing to relocate"),
      true
    );
  });

  it("rejects a DIFFERENT option committing (the Clera/Faros/Lio bug)", () => {
    assert.equal(valuesConsistent("No", "Yes"), false);
    assert.equal(valuesConsistent("Yes", "No"), false);
  });

  it("rejects empty committed values", () => {
    assert.equal(valuesConsistent("No", ""), false);
    assert.equal(valuesConsistent("", "Yes"), false);
  });
});

describe("screener commit verification", () => {
  beforeEach(() => resetDeferredFieldCount());

  const optionField = (): FormField => ({
    label: "Work Authorization",
    id: "wa",
    kind: "multi",
    required: true,
    options: ["Yes", "No"],
    optionTargets: [],
  });

  it("blanks the field when the resolved answer is not what committed", async () => {
    // The controls commit "Yes" no matter what answer was resolved: presence
    // is true, but value-consistency must reject it.
    const controls = {
      tagName: "Test",
      fillByKind: async () => true,
      readFieldValue: async () => "Yes",
      readSelectOptions: async () => ["Yes", "No"],
      closeMenu: async () => {},
      readSelectValue: async () => "Yes",
      fillAsyncAutocomplete: async () => true,
    } as any;
    const rpc = async () => ({ answer: "No", source: "kb" });
    const screener = new Screener(controls, "Test", profile, rpc);
    const filled: string[] = [];
    const blanked: { label: string; reason: string }[] = [];
    await screener.process(optionField(), filled, blanked, new Set());
    assert.equal(filled.length, 0);
    assert.equal(blanked.length, 1);
    assert.match(blanked[0].reason, /could not be committed/);
  });

  it("accepts the field when the committed value matches the answer", async () => {
    const controls = {
      tagName: "Test",
      fillByKind: async () => true,
      readFieldValue: async () => "No",
      readSelectOptions: async () => ["Yes", "No"],
      closeMenu: async () => {},
      readSelectValue: async () => "No",
      fillAsyncAutocomplete: async () => true,
    } as any;
    const rpc = async () => ({ answer: "No", source: "kb" });
    const screener = new Screener(controls, "Test", profile, rpc);
    const filled: string[] = [];
    const blanked: { label: string; reason: string }[] = [];
    await screener.process(optionField(), filled, blanked, new Set());
    assert.equal(filled.length, 1);
    assert.equal(blanked.length, 0);
  });
});

describe("blanked-required counter (screener)", () => {
  beforeEach(() => resetBlankedRequiredCount());

  it("defaults to 0", () => {
    assert.equal(getBlankedRequiredCount(), 0);
  });

  it("tracks the count set by adapters", () => {
    setBlankedRequiredCount(3);
    assert.equal(getBlankedRequiredCount(), 3);
  });

  it("clamps negatives and non-integers", () => {
    setBlankedRequiredCount(-1);
    assert.equal(getBlankedRequiredCount(), 0);
    setBlankedRequiredCount(2.9);
    assert.equal(getBlankedRequiredCount(), 2);
  });
});
