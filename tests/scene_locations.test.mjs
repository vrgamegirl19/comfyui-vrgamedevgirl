import assert from "node:assert/strict";
import { test } from "node:test";

import {
  eligibleSharedLocationRuns, hasMappedSceneLocation, mappedLocation, sharedLocationRuns,
} from "../web/music_video_builder/scene_locations.mjs";

const scenes = [
  { id: "one", start: 0, end: 5 },
  { id: "two", start: 5, end: 10 },
  { id: "three", start: 10, end: 15 },
  { id: "four", start: 15, end: 20 },
  { id: "five", start: 20, end: 25 },
  { id: "six", start: 25, end: 30 },
];
const refs = {
  locations: [{ id: "a", name: "Hall" }, { id: "b", name: "Roof" }],
  scene_map: { one: "a", two: "a", three: "a", four: "b", five: "b", six: "b" },
};

test("mapped locations require a real location and scene assignment", () => {
  assert.equal(hasMappedSceneLocation({ locations: refs.locations, scene_map: {} }, scenes), false);
  assert.equal(hasMappedSceneLocation({ locations: [], scene_map: { one: "a" } }, scenes), false);
  assert.equal(mappedLocation(refs, scenes[0]).name, "Hall");
  assert.equal(hasMappedSceneLocation(refs, scenes), true);
});

test("continuous location groups start anew after each location change", () => {
  assert.deepEqual(sharedLocationRuns(refs, scenes).map((run) => run.map((scene) => scene.id)), [
    ["one", "two", "three"],
    ["four", "five", "six"],
  ]);
});

test("unmapped scenes, different locations, and timeline gaps break runs", () => {
  const changed = { ...refs, scene_map: { ...refs.scene_map, three: "", five: "a" } };
  assert.deepEqual(sharedLocationRuns(changed, scenes).map((run) => run.map((scene) => scene.id)), [["one", "two"]]);
  const gap = scenes.map((scene) => ({ ...scene }));
  gap[1].start = 5.2;
  assert.deepEqual(sharedLocationRuns(refs, gap).map((run) => run.map((scene) => scene.id)), [
    ["two", "three"], ["four", "five", "six"],
  ]);
});

test("a group is skipped when one continuation scene uses an unsupported MiniMax mode", () => {
  const configured = scenes.map((scene) => ({ ...scene }));
  configured[2].use_scene_minimax_h3_settings = true;
  configured[2].minimax_h3_settings = { video_mode: "image_reference_to_video", render_pass: "two_pass" };
  assert.deepEqual(eligibleSharedLocationRuns(refs, configured, {
    video_mode: "text_to_video", render_pass: "single",
  }).map((run) => run.map((scene) => scene.id)), [["four", "five", "six"]]);
});
