import { defineConfig } from "oxlint";
import core from "ultracite/oxlint/core";

export default defineConfig({
  extends: [core],
  ignorePatterns: [
    ...core.ignorePatterns,
    "refs/**",
    "artifacts/**",
    "crawler_out/**",
    "data/**",
    "logs/**",
    "azure_dump/**",
  ],
  rules: {
    "eslint/class-methods-use-this": "off",
    "eslint/complexity": "off",
    "eslint/func-style": "off",
    "eslint/max-classes-per-file": "off",
    "eslint/no-await-in-loop": "off",
    "eslint/no-bitwise": "off",
    "eslint/no-empty-function": "off",
    "eslint/no-inline-comments": "off",
    "eslint/no-plusplus": "off",
    "eslint/no-use-before-define": "off",
    "eslint/prefer-named-capture-group": "off",
    "eslint/require-await": "off",
    "eslint/require-unicode-regexp": "off",
    "promise/avoid-new": "off",
    "promise/param-names": "off",
    "promise/prefer-await-to-callbacks": "off",
    "promise/prefer-await-to-then": "off",
    "typescript/no-explicit-any": "off",
    "unicorn/consistent-function-scoping": "off",
    "unicorn/filename-case": "off",
    "unicorn/prefer-number-coercion": "off",
  },
});
