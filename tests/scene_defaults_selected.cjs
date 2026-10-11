const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");

function fixture(mode = "image_to_video_prep") {
  const context = vm.createContext({});
  for (const name of ["controls", "scenes", "shot_presets", "video_style", "performance_presets", "defaults_actions", "settings_panel", "references"]) {
    const source = fs.readFileSync(path.join(__dirname, "../web/storyboard_builder", `${name}.mjs`), "utf8")
      .replace(/^import [^;]*;\r?\n/gm, "").replace(/^export /gm, "");
    vm.runInContext(source, context, { filename: name });
  }
  const scenes = ["a", "b", "c"].map((id) => ({ id, shot_type: "Manual shot", camera_motion: "Manual camera",
    performance_style: "old style", facial_performance: "old face", facial_performance_custom: "old custom",
    video_style: "old video", video_prompt_type: "t2v", project_video_engine: "minimax_h3", motion_summary: "Image aesthetic: old\nKeep this note", video_prompt: "Existing prompt" }));
  const state = { scenes, selected: new Set(["b"]), mode, cameraFlow: "medium_shots", imageShotFlow: "test",
    imageAesthetic: "test", performanceStyle: "rock_punk", facialPerformance: "rock_intense",
    facialPerformanceCustom: "New custom", videoStyle: "custom", videoStyleCustom: "My exact visual style" };
  let renders = 0;
  const toasts = [];
  context.createToast = (text) => toasts.push(text);
  const renderTable = () => { renders++; };
  const settings = context.createSettingsPanel({ state, renderTable,
    imageShotFlowPresets: { test: { sequence: ["First shot", "Second shot", "Third shot"] } },
    imageAestheticPresets: [{ value: "test", description: "New aesthetic" }],
  });
  const references = context.createStoryboardReferences({ state, renderTable });
  return { context, state, settings: { ...settings, ...references }, toasts, renders: () => renders };
}

for (const [name, mode] of [["applyCameraFlow", "image_to_video_prep"], ["applyImageShotFlow", "storyboard_prompts"],
  ["applyImageAesthetic", "storyboard_prompts"], ["applyVideoStyle", "image_to_video_prep"],
  ["applyPerformanceStyle", "image_to_video_prep"], ["applyFacialPerformance", "image_to_video_prep"]]) {
  test(`${name} replaces only checked scenes and never edits prompts`, () => {
    const f = fixture(mode), before = f.state.scenes.map((scene) => JSON.stringify(scene));
    f.settings[name]({ overwrite: true, selectedOnly: true });
    assert.equal(JSON.stringify(f.state.scenes[0]), before[0]);
    assert.notEqual(JSON.stringify(f.state.scenes[1]), before[1]);
    assert.equal(JSON.stringify(f.state.scenes[2]), before[2]);
    assert.equal(f.state.scenes[1].video_prompt, "Existing prompt");
    if (name === "applyImageShotFlow") assert.equal(f.state.scenes[1].shot_type, "Second shot");
    if (name === "applyFacialPerformance") assert.equal(f.state.scenes[1].facial_performance_custom, "New custom");
    if (name === "applyVideoStyle") assert.equal(f.state.scenes[1].video_style_custom, "My exact visual style");
    assert.equal(f.renders(), 1);
  });
}

test("Empty or stale selection never falls back to applying all scenes", () => {
  const f = fixture(), before = JSON.stringify(f.state.scenes);
  for (const selected of [new Set(), new Set(["deleted-scene"])]) {
    f.state.selected = selected;
    for (const action of ["applyCameraFlow", "applyVideoStyle", "applyPerformanceStyle", "applyFacialPerformance"]) {
      f.settings[action]({ overwrite: true, selectedOnly: true });
    }
  }
  assert.equal(JSON.stringify(f.state.scenes), before);
  assert.equal(f.renders(), 0);
  assert.ok(f.toasts.every((text) => text === "Select one or more scenes first."));
});

test("Fill Missing and Replace All keep their existing all-scene scope", () => {
  const f = fixture();
  f.state.scenes[0].performance_style = "";
  f.settings.applyPerformanceStyle({ overwrite: false });
  assert.equal(f.state.scenes[0].performance_style, "rock_punk");
  assert.equal(f.state.scenes[1].performance_style, "old style");
  f.settings.applyPerformanceStyle({ overwrite: true });
  assert.ok(f.state.scenes.every((scene) => scene.performance_style === "rock_punk"));
});

test("Selected action visibility follows checking, clearing and removed scene IDs", () => {
  const f = fixture(), actions = { applyPerformanceStyle: { hidden: true } };
  const refresh = f.context.wireSelectedDefaultActions({ actions, handlers: f.settings, state: f.state });
  refresh(); assert.equal(actions.applyPerformanceStyle.hidden, false);
  actions.applyPerformanceStyle.onclick(); assert.equal(f.state.scenes[1].performance_style, "rock_punk");
  f.state.selected.clear(); refresh(); assert.equal(actions.applyPerformanceStyle.hidden, true);
  f.state.selected.add("removed"); refresh(); assert.equal(actions.applyPerformanceStyle.hidden, true);
  f.state.selected.add("c"); refresh(); assert.equal(actions.applyPerformanceStyle.hidden, false);
  assert.match(actions.applyPerformanceStyle.title, /1 selected scene only/);
});

test("Explicit scene aesthetic overrides global aesthetic and preserves scene custom wording", () => {
  const f = fixture();
  const own = f.context.storyboardSceneVideoStyle({ video_style: "custom", video_style_custom: "Scene only" }, f.state);
  assert.equal(own.value, "custom"); assert.equal(own.custom, "Scene only");
  const inherited = f.context.storyboardSceneVideoStyle({}, f.state);
  assert.equal(inherited.custom, "My exact visual style");
  const other = f.context.storyboardSceneVideoStyle({ video_style: "1990s_grunge" }, f.state);
  assert.equal(other.value, "1990s_grunge"); assert.equal(other.custom, "");
});

test("Actual scene payload keeps selected and unselected aesthetics independent of global default", () => {
  const f = fixture();
  vm.runInContext(fs.readFileSync(path.join(__dirname, "../web/music_video_builder/refmod_labels.mjs"), "utf8").replace(/^export /gm, ""), f.context);
  vm.runInContext(fs.readFileSync(path.join(__dirname, "../web/storyboard_builder/gpt_payload.mjs"), "utf8")
    .replace(/^import [^;]*;\r?\n/gm, "").replace(/^export /gm, ""), f.context);
  const state = { mode: "image_to_video_prep", storyLayer: {}, referenceBuilder: {}, cameraFlow: "medium_shots",
    videoStyle: "custom", videoStyleCustom: "Global style", scenes: [
      { scene_number: 1, video_prompt_type: "t2v", project_video_engine: "minimax_h3", video_style: "custom", video_style_custom: "Selected style" },
      { scene_number: 2, video_prompt_type: "t2v", project_video_engine: "minimax_h3", video_style: "custom", video_style_custom: "Unselected style" },
      { scene_number: 3, video_prompt_type: "t2v", project_video_engine: "minimax_h3" },
    ] };
  const scenes = f.context.storyboardScenesForGpt(state);
  assert.equal(scenes[0].video_style_verbiage, "Selected style");
  assert.equal(scenes[1].video_style_verbiage, "Unselected style");
  assert.equal(scenes[2].video_style_verbiage, "Global style");
});
