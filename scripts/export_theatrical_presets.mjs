// Keep the Agent API's Theatrical directions identical to the browser presets.
import fs from "node:fs";
import { fileURLToPath } from "node:url";
import { STORYBOARD_CAMERA_FLOW_PRESETS } from "../web/storyboard_builder/shot_presets.mjs";
import { storyboardPerformancePreset, storyboardFacialPerformancePreset } from "../web/storyboard_builder/performance_presets.mjs";

const path = fileURLToPath(new URL("../storyboard/theatrical_presets.json", import.meta.url));
const data = {
  camera_key: "pretty_little_malfunction",
  camera: STORYBOARD_CAMERA_FLOW_PRESETS.pretty_little_malfunction,
  performance: storyboardPerformancePreset("theatrical"),
  facial: storyboardFacialPerformancePreset("theatrical"),
};
const content = `${JSON.stringify(data, null, 2)}\n`;
if (process.argv.includes("--check")) {
  if (!fs.existsSync(path) || fs.readFileSync(path, "utf8").replace(/\r\n/g, "\n") !== content) {
    throw new Error("Theatrical presets are stale. Run node scripts/export_theatrical_presets.mjs");
  }
} else {
  fs.writeFileSync(path, content, "utf8");
}
