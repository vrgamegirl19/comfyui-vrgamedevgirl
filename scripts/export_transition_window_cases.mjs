// Keep the transition window cases in tests/transition_window_cases.json in step with the rule in web/music_video_builder/transition_model.mjs.
// The cases are generated scene layouts and blocks with the window the player's rule gives each. tests/transition_model.cjs runs them, and so does
// the H3 Studio app's Rust planner (crates/h3-media, `plan`), so the live preview and the exported video agree.
import fs from "node:fs";
import { fileURLToPath } from "node:url";
import { transitionWindow } from "../web/music_video_builder/transition_model.mjs";

// A small seeded generator, so the file only changes when the rule does.
function random(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const next = random(20261028);
const between = (low, high) => low + (high - low) * next();
const cases = [];
for (let n = 0; n < 700; n += 1) {
  const count = 1 + Math.floor(next() * 5);
  // Scene lengths that are not whole frames, so the rounding is exercised. Some are very short (but not under a frame).
  const clips = [];
  let at = 0;
  for (let i = 0; i < count; i += 1) {
    const length = next() < 0.2 ? between(0.05, 0.3) : between(0.4, 4);
    clips.push([Math.round(at * 1000) / 1000, Math.round((at + length) * 1000) / 1000]);
    at = clips[i][1];
  }
  // A block anywhere from the start to a little past the end, sometimes right at a cut.
  const total = clips[clips.length - 1][1];
  let start = between(0, total + 0.5);
  if (clips.length > 1 && next() < 0.35) start = clips[Math.floor(next() * (clips.length - 1))][1] - [0, 0.25, 0.5, 1][Math.floor(next() * 4)];
  start = Math.max(0, Math.round(start * 1000) / 1000);
  const seconds = Math.round(between(0.1, 3) * 1000) / 1000;
  const segments = clips.map(([from, to], i) => ({ id: `s${i}`, start: from, end: to }));
  const w = transitionWindow({ start, seconds }, segments);
  cases.push({ clips, start, seconds, expect: w.frames >= 2 ? { first: w.first, frames: w.frames, cut: w.cut } : null });
}

const path = fileURLToPath(new URL("../tests/transition_window_cases.json", import.meta.url));
const content = `${JSON.stringify(cases)}\n`;
if (process.argv.includes("--check")) {
  if (!fs.existsSync(path) || fs.readFileSync(path, "utf8").replace(/\r\n/g, "\n") !== content) {
    throw new Error("The transition window cases are stale. Run node scripts/export_transition_window_cases.mjs");
  }
} else {
  fs.writeFileSync(path, content, "utf8");
  console.log(`${cases.length} cases, ${cases.filter((c) => c.expect).length} with a window, ${cases.filter((c) => c.expect?.cut === null).length} inside a scene`);
}
