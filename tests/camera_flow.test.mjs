import assert from "node:assert/strict";
import { test } from "node:test";

import {
  STORYBOARD_CAMERA_FLOW_PRESETS,
  storyboardCameraFlowEntry,
} from "../web/storyboard_builder/shot_presets.mjs";

test("Medium shots cycles through portraits and upper bodies with occasional full bodies", () => {
  const preset = STORYBOARD_CAMERA_FLOW_PRESETS.medium_shots;
  assert.equal(preset.label, "Medium shots");
  assert.equal(preset.framing_candidates, true);
  let previousMotion = "";
  const entries = Array.from({ length: preset.sequence.length * 3 }, (_, index) => {
    const entry = storyboardCameraFlowEntry("medium_shots", index, previousMotion);
    previousMotion = entry.camera;
    return entry;
  });
  const portraits = entries.filter(({ shot }) => /headshot/.test(shot));
  const upperBodies = entries.filter(({ shot }) => /upper-body/.test(shot));
  const fullBodies = entries.filter(({ shot }) => /full-body/.test(shot));
  assert.ok(portraits.length > 0);
  assert.ok(upperBodies.length > portraits.length);
  assert.ok(fullBodies.length > 0 && fullBodies.length < portraits.length);
  assert.equal(portraits.length + upperBodies.length + fullBodies.length, entries.length);
  for (const { shot, camera } of entries) {
    assert.doesNotMatch(`${shot} ${camera}`, /wide|establishing|extreme|macro|eyes-only/i);
    assert.match(camera, /framing|composition|subject size/);
  }
  for (const { shot } of portraits) assert.match(shot, /complete head and shoulders.*headroom/);
  for (const { shot } of fullBodies) assert.match(shot, /head to toe.*minimal|minimal.*head.to.toe/);
  assert.equal(new Set(entries.map(({ shot }) => shot)).size, preset.sequence.length);
});

test("Medium shots limits also govern cuts, camera motion, and ending framing", () => {
  const { guidance } = STORYBOARD_CAMERA_FLOW_PRESETS.medium_shots;
  assert.match(guidance, /Never use wide, medium-wide, establishing/);
  assert.match(guidance, /never crop into eyes-only, macro, or extreme close-ups/);
  assert.match(guidance, /throughout every camera move, cut, and ending composition/);
  assert.match(guidance, /push-in or pullback must stay within these limits/);
});
