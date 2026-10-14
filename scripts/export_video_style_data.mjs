// Exports the video style and FX presets the Video Builder UI uses
// (web/storyboard_builder/video_style.mjs) to agent_api/video_style_data.json, so the
// agent API writes the same style and FX wording into its prompts.
//
// Usage (pack root):
//   node scripts/export_video_style_data.mjs          write the JSON file
//   node scripts/export_video_style_data.mjs --check  exit 1 if the file is stale
import { readFileSync, writeFileSync } from "node:fs";
import { register } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const target = join(root, "agent_api", "video_style_data.json");
register(pathToFileURL(join(root, "tests", "stub_comfy_loader.mjs")).href);
const { MINIMAX_VIDEO_STYLE_PRESETS, STORYBOARD_FX_PRESETS } = await import(
  pathToFileURL(join(root, "web", "storyboard_builder", "video_style.mjs")).href
);

const data = {
  video_styles: Object.fromEntries(MINIMAX_VIDEO_STYLE_PRESETS.filter((p) => p.value && p.value !== "custom")
    .map((p) => [p.value, { label: p.label, guidance: p.prompt_guidance }])),
  fx: Object.fromEntries(STORYBOARD_FX_PRESETS.filter((p) => p.value && p.value !== "custom")
    .map((p) => [p.value, { label: p.label, cues: p.cues }])),
};
const text = `${JSON.stringify(data, null, 2)}\n`;

if (process.argv.includes("--check")) {
  let current = "";
  try { current = readFileSync(target, "utf8"); } catch { /* missing counts as stale */ }
  if (current.replace(/\r\n/g, "\n") !== text) {
    console.error("[VRGDG] agent_api/video_style_data.json is out of date. Run node scripts/export_video_style_data.mjs");
    process.exit(1);
  }
  console.log("[VRGDG] agent_api/video_style_data.json is up to date.");
} else {
  writeFileSync(target, text, "utf8");
  console.log(`[VRGDG] Wrote ${target} (${Object.keys(data.video_styles).length} styles, ${Object.keys(data.fx).length} FX presets).`);
}
