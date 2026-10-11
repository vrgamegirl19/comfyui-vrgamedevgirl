const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const test = require("node:test");

function fixture(locations) {
  const context = vm.createContext({});
  for (const file of ["music_video_builder/refmod_labels.mjs", "storyboard_builder/references.mjs"]) {
    const source = fs.readFileSync(path.join(__dirname, "../web", file), "utf8")
      .replace(/^import [^;]*;\r?\n/gm, "").replace(/^export /gm, "");
    vm.runInContext(source, context, { filename: file });
  }
  const state = { referenceBuilder: { subjects: [], locations } };
  const references = context.createStoryboardReferences({ state, renderTable: () => {} });
  return { state, absorb: references.absorbSceneReferencesIntoCatalog };
}

test("scene settings reuse existing short-named locations and preserve reference data", () => {
  const description = "A spacious station with ornate pillars and reflections. ".repeat(10).trim();
  const { state, absorb } = fixture([
    { id: "station", name: "Station", description, image: { path: "station.png" }, trigger_phrase: "station" },
  ]);
  const scenes = [
    { scene_number: 1, setting: description },
    { scene_number: 2, setting: "station" },
  ];
  absorb(scenes);
  absorb(scenes);
  assert.equal(state.referenceBuilder.locations.length, 1);
  for (const scene of scenes) {
    assert.equal(scene.location_ref.id, "station");
    assert.equal(scene.location_ref.name, "Station");
    assert.equal(scene.location_ref.description, description);
    assert.equal(scene.location_ref.image.path, "station.png");
    assert.equal(scene.location_ref.trigger_phrase, "station");
  }
});

test("unmatched scene descriptions get short distinct names without repeated duplicates", () => {
  const { state, absorb } = fixture([{ id: "existing", name: "Location 2", description: "Existing setting" }]);
  const description = "An entirely new location with detailed scenery. ".repeat(10).trim();
  const scenes = [
    { scene_number: 1, setting: description },
    { scene_number: 2, setting: description },
    { scene_number: 3, setting: "A different setting" },
  ];
  absorb(scenes);
  absorb(scenes);
  assert.equal(state.referenceBuilder.locations.length, 3);
  assert.equal(scenes[0].location_ref.name, "Location 3");
  assert.equal(scenes[0].location_ref.description, description);
  assert.equal(scenes[1].location_ref.id, scenes[0].location_ref.id);
  assert.equal(scenes[2].location_ref.name, "Location 4");
  assert.notEqual(scenes[2].location_ref.id, scenes[0].location_ref.id);
});
