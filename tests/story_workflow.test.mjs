import assert from "node:assert/strict";
import { test } from "node:test";

import { runStoryGenerationSequence } from "../web/storyboard_builder/story_workflow.mjs";

import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import vm from "node:vm";
const { functionSource } = createRequire(import.meta.url)("./builder_source.cjs");
const storySource = readFileSync(new URL("../web/storyboard_builder/story_layer.mjs", import.meta.url), "utf8");

for (const locations of [[], [{ id: "hall", name: "Hall", description: "A narrow hallway" }]]) {
  test(`arc and brief work with ${locations.length} unmapped text-only locations`, async () => {
    const requests = [];
    const context = {
      state: { storyLayer: {}, referenceBuilder: { locations }, scenes: [{ id: "one", lyrics: "Leaving home" }] },
      syncStoryLayerFromInputs() {}, confirmStoryStepRerun: () => true,
      normalizeStoryboardScriptImportState: () => ({}), normalizeStoryLayer: x => x,
      createStoryboardProgressWindow: () => ({ set() {}, close() {} }), promptRunnerName: () => "LLM",
      overallStoryIdeaInput: { value: "A homecoming" }, userStoryArcInput: { value: "" }, songStoryBriefInput: {},
      slimStoryboardForRequest: () => ({}), storyboardTimelineNotesForRequest: () => [],
      lyricsForStoryBrief: () => "Leaving home", slimSceneForRequest: x => x,
      storyboardSpeedValue: () => 4, normalizeStoryArcDetail: () => "compact", createToast() {},
      postJson: async (path, payload) => { requests.push({ path, payload }); return { story_arc: "At the Hall", story_brief: "Returning to the Hall" }; },
    };
    vm.createContext(context);
    for (const name of ["createStoryArcWithGemma", "createStoryBriefWithGemma"]) {
      vm.runInContext(functionSource(storySource, name), context);
      assert.ok(await context[name]());
    }
    assert.equal(requests.length, 2);
    assert.deepEqual(requests[0].payload.reference_builder.locations, locations);
    assert.equal(context.state.storyLayer.user_story_arc, "At the Hall");
    assert.equal(context.state.storyLayer.song_story_brief, "Returning to the Hall");
  });
}

test("combined story generation saves each stage before the next", async () => {
  const order = [];
  const result = await runStoryGenerationSequence({
    createArc: async () => { order.push("arc"); return "arc text"; },
    createBrief: async () => { order.push("brief"); return "brief text"; },
    createBeats: async () => { order.push("beats"); return { created: 2, failures: [] }; },
    save: async () => { order.push("save"); },
  });
  assert.deepEqual(order, ["arc", "save", "brief", "save", "beats"]);
  assert.deepEqual(result, { completed: true, created: 2 });
});

test("combined story generation stops after an empty brief or failed beats", async () => {
  const order = [];
  const steps = {
    createArc: async () => { order.push("arc"); return "arc"; },
    createBrief: async () => { order.push("brief"); return ""; },
    createBeats: async () => { order.push("beats"); return { created: 1, failures: [] }; },
    save: async () => { order.push("save"); },
  };
  assert.deepEqual(await runStoryGenerationSequence(steps), { completed: false, stoppedAt: "brief" });
  assert.deepEqual(order, ["arc", "save", "brief"]);
  steps.createBrief = async () => "brief";
  steps.createBeats = async () => ({ created: 1, failures: [{ scene: "two" }] });
  assert.deepEqual(await runStoryGenerationSequence(steps), { completed: false, stoppedAt: "beats" });
});

test("LM Extract uses a saved story with no timeline text and preserves existing images", async () => {
  const source = readFileSync(new URL("../web/music_video_builder/reference_locations.mjs", import.meta.url), "utf8");
  const location = { id: "hall", name: "Hall", description: "", image: { path: "hall.png" } };
  const requests = [];
  const context = {
    state: { builderStoryLayer: { user_story_arc: "Return to the Hall under amber lights." } },
    refs: { locations: [location] }, extractLocations: {}, autoMapLocations: {},
    createProgressWindow: () => ({ set() {}, close() {} }), allEditableSegments: () => [],
    t2iTextGemmaModelSelect: { value: "" }, i2vTextGemmaModelSelect: { value: "" },
    locationStyleTheme: { value: "" }, locationExtractionStyleTheme: async x => x,
    gemmaRunnerLine: () => "LLM", textGemmaRunnerPayload: () => ({}),
    referenceSubjectContextForLocations: () => "", normalizeGemmaContextLimit: () => 16384,
    postJson: async (path, payload) => { requests.push({ path, payload }); return { locations: [{ name: "Hall", description: "Amber lights." }, { name: "Garden", description: "Glass roof." }] }; },
    locationByName: name => context.refs.locations.find(x => x.name === name),
    createLocation: (name, description) => context.refs.locations.push({ name, description, image: {} }),
    renderAll() {}, toast() {},
  };
  vm.createContext(context);
  vm.runInContext(functionSource(source, "extractLocationsWithGemma"), context);
  await context.extractLocationsWithGemma();
  assert.equal(requests.length, 1);
  assert.equal(requests[0].payload.story_layer.user_story_arc, context.state.builderStoryLayer.user_story_arc);
  assert.equal(requests[0].payload.lyrics_text, "");
  assert.equal(location.image.path, "hall.png");
  assert.equal(location.description, "Amber lights.");
  assert.equal(context.refs.locations.length, 2);
  assert.equal(context.refs.locations[1].name, "Garden");
  assert.equal(context.extractLocations.disabled, false);
});
