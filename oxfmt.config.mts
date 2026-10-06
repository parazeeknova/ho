import { defineConfig } from "oxfmt";
import ultracite from "ultracite/oxfmt";

export default defineConfig({
  ...ultracite,
  ignorePatterns: [
    ...ultracite.ignorePatterns,
    "refs/**",
    "artifacts/**",
    "crawler_out/**",
    "data/**",
    "logs/**",
    "azure_dump/**",
  ],
});
