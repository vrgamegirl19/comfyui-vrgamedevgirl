const assert = require("node:assert/strict");
const { test } = require("node:test");
const vm = require("node:vm");
const fs = require("node:fs"), path = require("node:path");

function fixture({ refreshControls, onSettingApplied } = {}) {
  const context = vm.createContext({ Object, Set, JSON });
  for (const name of ["scene_defaults_state", "scenes", "shot_presets", "video_style", "defaults_presets", "defaults_auto_apply"]) {
    const source = fs.readFileSync(path.join(__dirname, "../web/storyboard_builder", name + ".mjs"), "utf8")
      .replace(/^import [^;]*;\r?\n/gm, "").replace(/^export /gm, "");
    vm.runInContext(source, context);
  }
  const { SCENE_DEFAULT_KEYS, DEFAULT_PRESET_CONTROLS } = vm.runInContext("({ SCENE_DEFAULT_KEYS, DEFAULT_PRESET_CONTROLS })", context);
  const state = { mode: "image_to_video_prep", selected: new Set(), scenes: [1, 5, 7, 10].map(number => ({
    id: "scene" + number, scene_number: number, project_video_engine: "minimax_h3", video_prompt_type: "t2v",
    performance_style: "manual", video_prompt: "Existing prompt", label: "Keep me",
  })), cameraFlow: "balanced", cameraMotionSpeed: 4, characterMotionSpeed: 4, cutFrequency: 0,
    cutStyle: "free", performanceStyle: "", facialPerformance: "", facialPerformanceCustom: "",
    videoStyle: "", videoStyleCustom: "", temporalWorldEffect: "", temporalWorldEffectCustom: "",
    temporalAllowBackgroundExtras: true, temporalBackgroundIntensity: 8, temporalEnvironmentTimePassage: true,
    temporalProtectedCharacters: "all_referenced", temporalProtectedCustom: "", fxPreset: "", fxCustomJson: "",
    globalConsistencyPhrase: "Old consistency", storyArcDetail: "standard", customCameraFlowSequence: [] };
  const controls = { defaultsScope: {}, defaultsScopeFooter: {}, defaultsResult: {}, defaultsUndo: {} };
  for (const ref of Object.values(DEFAULT_PRESET_CONTROLS)) controls[ref] = {
    value: "", addEventListener(type, fn) { this[type] = fn; },
  };
  let saves = 0;
  const controller = context.createAutoSceneDefaults({ state, controls, onSettingApplied, persist: () => saves++,
    renderTable: () => controller.syncScope(), refreshControls: () => refreshControls?.(controller, state), imageShotFlowPresets: {}, imageAestheticPresets: [] });
  controller.reset();
  const change = (key, value) => {
    controls[DEFAULT_PRESET_CONTROLS[key]]?.change?.();
    state[key] = value;
    controller.commit();
  };
  return { context, state, controls, controller, change, saves: () => saves };
}

test("all-scene edits are immediate, stored per scene and leave existing prompts alone", () => {
  const f = fixture();
  f.change("cameraMotionSpeed", 0);
  assert.equal(f.controls.defaultsScope.textContent, "Editing all scenes");
  assert.equal(f.controls.defaultsResult.textContent, "Updated all scenes");
  assert.equal(f.state.cameraMotionSpeed, 4, "camera speed is no longer a live global value");
  for (const scene of f.state.scenes) {
    assert.equal(scene.scene_defaults.camera_motion_speed, 0);
    assert.equal(scene.video_prompt, "Existing prompt");
  }
  assert.equal(f.saves(), 1);
});

test("verse/chorus selection applies only to selected scenes and reports their real numbers", () => {
  const f = fixture();
  f.state.selected = new Set(["scene5", "scene10"]);
  f.controller.syncScope();
  assert.equal(f.controls.defaultsScope.textContent, "Editing selected scenes: 5, 10");
  const untouched = JSON.stringify([f.state.scenes[0], f.state.scenes[2]]);
  f.change("performanceStyle", "metal_screaming");
  assert.equal(f.state.scenes[1].performance_style, "metal_screaming");
  assert.equal(f.state.scenes[3].performance_style, "metal_screaming");
  assert.equal(JSON.stringify([f.state.scenes[0], f.state.scenes[2]]), untouched);
  assert.equal(f.state.performanceStyle, "");
  assert.equal(f.controls.defaultsResult.textContent, "Updated scenes 5, 10");
  f.state.selected = new Set(["scene1", "scene7"]);
  f.change("performanceStyle", "rock_punk");
  assert.equal(f.state.scenes[1].performance_style, "metal_screaming");
  assert.equal(f.state.scenes[0].performance_style, "rock_punk");
});

test("choosing the same default still replaces different existing scene choices", () => {
  const f = fixture();
  f.state.scenes[0].scene_defaults = { camera_motion_speed: 10 };
  f.change("cameraMotionSpeed", 4);
  assert.equal(f.state.scenes[0].scene_defaults.camera_motion_speed, 4);
});

test("a preset is one undo transaction for camera, acting, facial and visual fields", () => {
  const f = fixture();
  const before = JSON.stringify(f.state.scenes);
  for (const [key, value] of Object.entries({ cameraMotionSpeed: 2, characterMotionSpeed: 2,
    cutFrequency: 1, performanceStyle: "rock_punk", facialPerformance: "rock_intense", videoStyle: "" })) {
    f.controls[vm.runInContext("DEFAULT_PRESET_CONTROLS", f.context)[key]]?.change?.();
    f.state[key] = value;
  }
  f.controller.commit();
  assert.equal(f.saves(), 1);
  assert.equal(f.state.scenes[0].scene_defaults.character_motion_speed, 2);
  f.state.scenes[0].label = "Later unrelated edit";
  f.controls.defaultsUndo.onclick();
  const restored = JSON.parse(JSON.stringify(f.state.scenes));
  restored[0].label = "Keep me";
  assert.equal(JSON.stringify(restored), before);
  assert.equal(f.state.scenes[0].label, "Later unrelated edit");
  assert.equal(f.saves(), 2);
});

test("Story arc detail is always project-wide and never touches scenes", () => {
  const f = fixture();
  f.state.selected.add("scene7");
  const before = JSON.stringify(f.state.scenes);
  f.change("storyArcDetail", "rich");
  assert.equal(f.state.storyArcDetail, "rich");
  assert.equal(JSON.stringify(f.state.scenes), before);
  assert.match(f.controls.defaultsResult.textContent, /project-wide/);
  f.controls.defaultsUndo.onclick();
  assert.equal(f.state.storyArcDetail, "standard");
});

test("stale selected IDs never fall back to all scenes or change project defaults", () => {
  const f = fixture();
  f.state.selected.add("deleted");
  const before = JSON.stringify(f.state.scenes);
  f.change("videoStyle", "1990s_grunge");
  assert.equal(JSON.stringify(f.state.scenes), before);
  assert.equal(f.state.videoStyle, "");
  assert.match(f.controls.defaultsResult.textContent, /No scenes updated/);
});

test("explicit zero, Off and custom text survive scene-level serialization", () => {
  const f = fixture();
  const original = { scene_defaults: { camera_motion_speed: 0, facial_performance: "",
    facial_performance_custom: "Her eyebrows lift slowly", temporal_allow_background_extras: false } };
  const restored = JSON.parse(JSON.stringify(original));
  const effective = f.context.effectiveSceneDefaultsState(f.state, restored);
  assert.equal(effective.cameraMotionSpeed, 0);
  assert.equal(effective.facialPerformance, "");
  assert.equal(effective.temporalAllowBackgroundExtras, false);
  assert.equal(effective.facialPerformanceCustom, original.scene_defaults.facial_performance_custom);
  assert.equal(f.context.normalizeSceneDefaults({ story_arc_detail: "rich" }).story_arc_detail, undefined);
});

test("Undo preserves a later unrelated scene-default edit", () => {
  const f = fixture();
  f.change("cameraMotionSpeed", 8);
  f.state.scenes[0].scene_defaults.fx_preset = "lighting";
  f.controls.defaultsUndo.onclick();
  assert.equal(f.state.scenes[0].scene_defaults.fx_preset, "lighting");
  assert.equal(f.state.scenes[0].scene_defaults.camera_motion_speed, undefined);
});

test("an actual preset auto-applies in one transaction, including an unchanged custom flow", () => {
  const f = fixture();
  f.context.Event = class { constructor(type) { this.type = type; } };
  const refs = vm.runInContext("DEFAULT_PRESET_CONTROLS", f.context);
  const fields = { cameraMotionSpeed: 2, characterMotionSpeed: 2, performanceStyle: "rock_punk", facialPerformance: "rock_intense" };
  const preset = { ...fields, cameraFlow: "custom", customCameraFlowSequence: [{ shot: "Close-up", camera: "Pan left" }] };
  for (const [key, ref] of Object.entries(refs)) {
    const control = f.controls[ref];
    control.tagName = ref.endsWith("Select") ? "SELECT" : "INPUT";
    control.options = ["", "custom", "rock_punk", "rock_intense"].map(value => ({ value }));
    control.dispatchEvent = event => {
      if (event.type === "change") control.change?.();
      f.state[key] = typeof f.state[key] === "number" ? Number(control.value) : control.value;
      if (event.type === "change") f.controller.commit();
    };
  }
  f.context.applyDefaultPreset(preset, { controls: f.controls, state: f.state,
    refreshCameraFlowInfo() {}, notifyStoryboardDefaultsChanged: f.controller.commit,
    noteDefaultKeys: keys => f.controller.note(keys) });
  assert.equal(f.saves(), 1);
  assert.equal(f.state.scenes[0].camera_motion, "Pan left");
  assert.equal(f.state.scenes[0].scene_defaults.camera_motion_speed, 2);
  assert.equal(f.state.scenes[0].performance_style, "rock_punk");
  f.controls.defaultsUndo.onclick();
  assert.equal(f.state.scenes[0].performance_style, "manual");
});

test("choosing settings with no scenes never claims every scene was updated", () => {
  const f = fixture();
  f.state.scenes = [];
  f.controller.reset();
  f.change("cameraMotionSpeed", 8);
  assert.match(f.controls.defaultsResult.textContent, /No eligible scenes/);
  assert.equal(f.state.cameraMotionSpeed, 4);
  assert.equal(f.controls.cameraSpeedInput.value, "4");
});

test("a mixed mode project updates video aesthetic only on eligible scenes", () => {
  const f = fixture();
  f.state.scenes[0].video_prompt_type = "i2v";
  f.state.scenes[0].minimax_h3_mode = "image_to_video";
  f.change("videoStyle", "1990s_grunge");
  assert.equal(f.state.scenes[0].scene_defaults, undefined);
  assert.equal(f.state.scenes[1].video_style, "1990s_grunge");
  assert.match(f.controls.defaultsResult.textContent, /eligible scenes: 5, 7, 10/);
});

test("scope refresh can re-enter through Storyboard summaries without blocking window setup", () => {
  let refreshes = 0;
  const f = fixture({ refreshControls(controller) {
    assert.ok(++refreshes < 20, "scope refresh must not recursively rebuild its own summaries");
    controller.syncScope();
  } });
  assert.equal(refreshes, 1);
  f.state.selected = new Set(["scene5"]);
  f.controller.syncScope();
  assert.equal(refreshes, 2);
  assert.equal(f.controls.defaultsScope.textContent, "Editing selected scenes: 5");
  f.change("cameraMotionSpeed", 2);
  assert.equal(f.state.scenes[1].scene_defaults.camera_motion_speed, 2);
  f.controls.defaultsUndo.onclick();
  assert.equal(f.state.scenes[1].scene_defaults, undefined);
});

test("actual scene checkbox updates the visible scope and isolates the next edit", () => {
  const f = fixture();
  class Node {
    constructor() { this.style = {}; this.children = []; this.actions = {}; this.events = {}; }
    append(...children) { this.children.push(...children); }
    appendChild(child) { this.append(child); }
    replaceChildren(...children) { this.children = children; }
    addEventListener(type, fn) { this.events[type] = fn; }
    set innerHTML(value) {
      this.html = value;
      for (const match of value.matchAll(/data-action="([^"]+)"/g)) this.actions[match[1]] = new Node();
    }
    querySelector(selector) { return this.actions[/data-action="([^"]+)"/.exec(selector)?.[1]] || null; }
  }
  f.context.document = { createElement: () => new Node() };
  Object.assign(f.context, {
    escapeHtml: value => String(value || ""), truncate: value => String(value || ""),
    storyboardReferenceImageSrc: () => "", subjectRefsHtml: () => "", settingRefHtml: () => "",
    videoPromptTypeLabel: () => "T2V",
  });
  const source = fs.readFileSync(path.join(__dirname, "../web/storyboard_builder/scene_table.mjs"), "utf8")
    .replace(/^import [^;]*;\r?\n/gm, "").replace(/^export /gm, "");
  vm.runInContext(source, f.context);
  const tableWrap = new Node();
  const visibleRows = f.state.scenes.slice(0, 3);
  const table = f.context.createSceneTable({ state: f.state, tableWrap, stats: {},
    currentRows: () => visibleRows, promptRunnerName: () => "LLM",
    refreshSetupPanelSummaries() {}, refreshActionButtons: () => f.controller.syncScope() });
  const checkbox = () => tableWrap.children[0].children.find(node => node.children.some(row => row.actions.select))
    .children[0].actions.select;
  table.renderTable();
  checkbox().events.change({ target: { checked: true } });
  assert.equal(f.state.selected.size, 1);
  assert.equal(f.controls.defaultsScopeFooter.textContent, "Editing selected scenes: 1");
  f.change("characterMotionSpeed", 2);
  assert.equal(f.state.scenes[0].scene_defaults.character_motion_speed, 2);
  assert.equal(f.state.scenes[1].scene_defaults, undefined);
  assert.equal(f.controls.defaultsResult.textContent, "Updated scenes 1");
  f.controls.defaultsUndo.onclick();
  assert.equal(f.controls.defaultsScopeFooter.textContent, "Editing selected scenes: 1");
  assert.equal(f.controls.defaultsResult.textContent, "Undid update to scenes 1");
  assert.equal(f.state.scenes[0].scene_defaults, undefined);
  checkbox().events.change({ target: { checked: false } });
  assert.equal(f.controls.defaultsScopeFooter.textContent, "Editing all scenes");
  const invert = () => tableWrap.children[0].children.find(node => node.actions["invert-selection"])
    .actions["invert-selection"].events.click();
  f.state.selected.add("scene10"); // Hidden by the current table filter.
  invert();
  assert.deepEqual([...f.state.selected].sort(), ["scene1", "scene10", "scene5", "scene7"]);
  invert();
  assert.deepEqual([...f.state.selected], ["scene10"], "hidden selection is preserved, visible inversion is reversible");
  checkbox().events.change({ target: { checked: true } });
  invert();
  assert.deepEqual([...f.state.selected].sort(), ["scene10", "scene5", "scene7"]);
});

test("Undo keeps the selected scene's restored value in its slider and hint", () => {
  let shownSpeed;
  const f = fixture({ refreshControls(_controller, state) { shownSpeed = state.characterMotionSpeed; } });
  f.state.scenes[0].scene_defaults = { character_motion_speed: 7 };
  f.state.selected = new Set(["scene1"]);
  f.controller.syncScope();
  assert.equal(f.controls.characterSpeedInput.value, "7");
  f.change("characterMotionSpeed", 2);
  f.controls.defaultsUndo.onclick();
  assert.equal(f.controls.characterSpeedInput.value, "7");
  assert.equal(shownSpeed, 7, "the hint must describe the selected scene rather than the global fallback");
});

test("individual facial dropdown confirms its option and selected scene numbers", () => {
  const confirmations = [];
  const f = fixture({ onSettingApplied: (name, result) => confirmations.push({ name, result }) });
  f.state.selected = new Set(["scene1", "scene5", "scene7"]);
  f.controller.syncScope();
  const control = f.controls.facialSelect;
  control.options = [{ value: "love_tender", textContent: "Love song / tender" }];
  control.getAttribute = () => "Facial performance";
  control.value = "love_tender";
  f.change("facialPerformance", "love_tender");
  assert.deepEqual(confirmations, [{ name: "Facial performance — Love song / tender", result: "Updated scenes 1, 5, 7" }]);
  assert.equal(f.state.scenes[3].facial_performance, undefined);
  f.controller.note(["facialPerformance"]);
  f.state.facialPerformance = "love_tender";
  f.controller.commit({ preset: true });
  assert.equal(confirmations.length, 1, "a toolbar preset produces its own single combined confirmation");
});
