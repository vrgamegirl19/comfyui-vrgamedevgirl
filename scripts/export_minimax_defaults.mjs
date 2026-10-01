// Exports the MiniMax H3 video settings defaults the Video Builder UI uses
// (cloneMiniMaxH3Settings({})) to minimax/h3_settings_defaults.json, so the
// Python side (agent_api, MCP, standalone app) shares one list of setting names,
// types and defaults with the browser.
//
// Usage (pack root):
//   node scripts/export_minimax_defaults.mjs          write the JSON file
//   node scripts/export_minimax_defaults.mjs --check  exit 1 if the file is stale
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const target = join(root, "minimax", "h3_settings_defaults.json");
const { cloneMiniMaxH3Settings } = await import(
  pathToFileURL(join(root, "web", "music_video_builder", "minimax_h3.mjs")).href
);

const defaults = cloneMiniMaxH3Settings({});
const text = `${JSON.stringify(defaults, null, 2)}\n`;

if (process.argv.includes("--check")) {
  let current = "";
  try { current = readFileSync(target, "utf8"); } catch { /* missing counts as stale */ }
  if (current.replace(/\r\n/g, "\n") !== text) {
    console.error("[VRGDG] minimax/h3_settings_defaults.json is out of date. Run node scripts/export_minimax_defaults.mjs");
    process.exit(1);
  }
  console.log("[VRGDG] minimax/h3_settings_defaults.json is up to date.");
} else {
  writeFileSync(target, text, "utf8");
  console.log(`[VRGDG] Wrote ${target} (${Object.keys(defaults).length} settings).`);
}
