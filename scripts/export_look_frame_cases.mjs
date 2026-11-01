// Keep the look frame cases in tests/look_frame_cases.json in step with the rule in web/music_video_builder/look_model.mjs.
// The cases are blocks of the effect and filter lanes on a video of a given number of frames, with the frames each covers and how far it has run on some of them.
// tests/look_model.cjs runs them, and so does the H3 Studio app's Rust planner (crates/h3-media, looks.rs, `plan` and `block_state`), so the live preview and the
// exported video agree. Run with `node scripts/export_look_frame_cases.mjs` (`--check` fails when the file is stale).
import fs from "node:fs";
import { fileURLToPath } from "node:url";
import { lookFrames, lookStateAt } from "../web/music_video_builder/look_model.mjs";

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

const next = random(20261029);
const between = (low, high) => low + (high - low) * next();
const cases = [];
for (let n = 0; n < 600; n += 1) {
  const total = 24 + Math.floor(next() * 2400);
  // Times that are not whole frames, so the rounding is exercised. Some blocks start past the end and some reach past it.
  let start = between(0, total / 24 + 0.6);
  if (next() < 0.25) start = Math.round(start * 24) / 24;
  start = Math.round(start * 1000) / 1000;
  const seconds = Math.round((next() < 0.2 ? between(0.05, 0.12) : between(0.1, 6)) * 1000) / 1000;
  const entry = { start, seconds };
  const frames = lookFrames(entry, total);
  const states = [];
  if (frames) {
    const middle = Math.floor((frames.first + frames.last) / 2);
    for (const frame of [frames.first - 1, frames.first, middle, frames.last - 1, frames.last]) {
      const run = lookStateAt(entry, frame, total);
      states.push({ frame, progress: run ? run.progress : null, time: run ? run.time : null });
    }
  }
  cases.push({ start, seconds, total, frames, states });
}

const target = fileURLToPath(new URL("../tests/look_frame_cases.json", import.meta.url));
const content = `${JSON.stringify({ fps: 24, cases })}\n`;
if (process.argv.includes("--check")) {
  if (!fs.existsSync(target) || fs.readFileSync(target, "utf8").replace(/\r\n/g, "\n") !== content) {
    throw new Error("The look frame cases are stale. Run node scripts/export_look_frame_cases.mjs");
  }
} else {
  fs.writeFileSync(target, content, "utf8");
  console.log(`${cases.length} cases, ${cases.filter((c) => c.frames).length} with frames in the video`);
}
