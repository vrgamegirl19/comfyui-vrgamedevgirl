import assert from "node:assert/strict";
import { register } from "node:module";
import { test } from "node:test";

// gpt_payload.mjs imports ComfyUI's scripts/api.js, which only exists in the browser. The loader stubs it.
register("./stub_comfy_loader.mjs", import.meta.url);
const { storyboardGptPayload } = await import("../web/storyboard_builder/gpt_payload.mjs");

test("storyboard LLM payload carries temporal and per-shot FX writing guidance", () => {
  const payload = storyboardGptPayload({
    mode: "image_to_video_prep", projectVideoEngine: "minimax_h3",
    cutFrequency: 2.5, fxPreset: "lighting", temporalWorldEffect: "frozen_world",
    scenes: [{ id: "s1", scene_number: 1, exact_duration: 10, project_video_engine: "minimax_h3" }],
    referenceBuilder: { subjects: [], locations: [] }, storyLayer: {},
  });
  const scene = payload.scenes[0];
  assert.equal(scene.temporal_world_effect.enabled, true);
  assert.match(scene.temporal_world_effect.writing_guidance, /world frozen/);
  assert.match(scene.temporal_world_effect_instruction, /naturally inside every shot/);
  assert.equal(scene.temporal_world_effect_verbiage, undefined);
  assert.equal(scene.shot_fx.length, 4);
  assert.deepEqual(scene.shot_fx.map(effect => effect.shot), [1, 2, 3, 4]);
  assert.match(scene.shot_fx_instruction, /natural prose/);
});
