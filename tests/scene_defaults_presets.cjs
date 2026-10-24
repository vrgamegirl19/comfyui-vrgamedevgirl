const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");

class Element {
  constructor(tag) { this.tagName = tag; this.children = []; this.style = { cssText: "" }; this.listeners = {}; this.value = ""; }
  append(...items) { this.children.push(...items); }
  replaceChildren(...items) { this.children = items; }
  setAttribute() {}
  addEventListener(type, fn, capture = false) { (this.listeners[type] ||= []).push({ fn, capture }); }
  dispatchEvent(event) { const listeners = this.listeners[event.type] || []; for (const item of listeners.filter(item => item.capture)) item.fn(event); this[`on${event.type}`]?.(event); for (const item of listeners.filter(item => !item.capture)) item.fn(event); }
  get options() { return this.children; }
}
function fixture() {
  const context = vm.createContext({ document: { createElement: (tag) => new Element(tag) }, Event,
    window: { confirm: () => true }, Array, Object, String });
  let eventsSource;
  for (const name of ["controls", "scene_defaults_state", "scenes", "shot_presets", "video_style", "defaults_presets", "storyboard_events"]) {
    const source = fs.readFileSync(path.join(__dirname, "../web/storyboard_builder", `${name}.mjs`), "utf8")
      .replace(/^import [^;]*;\r?\n/gm, "").replace(/^export /gm, "");
    vm.runInContext(source, context);
    if (name === "storyboard_events") eventsSource = source;
  }
  const bindings = vm.runInContext("({ DEFAULT_PRESET_CONTROLS, SPEED_DEFAULT_PRESETS, MUSIC_DEFAULT_PRESETS, PRESET_BASELINE, musicDefaultSettings })", context);
  const state = { cameraFlow: "balanced", cameraMotionSpeed: 4, characterMotionSpeed: 4, cutFrequency: 0,
    customCameraFlowSequence: [], scenes: [{ shot_type: "User's own shot", prompt: "Existing prompt" }] };
  const args = {};
  for (const name of eventsSource.match(/function wireStoryboardEvents\(\{([\s\S]*?)\}\)/)[1].split(",").map((name) => name.trim()).filter(Boolean)) args[name] = new Element("input");
  for (const ref of Object.values(bindings.DEFAULT_PRESET_CONTROLS)) args[ref] = new Element(ref.endsWith("Select") ? "select" : "input");
  const choices = {
    cameraFlowSelect: ["balanced", "medium_shots", "music_video", "energetic", "custom"],
    performanceSelect: ["", "rock_punk", "metal_screaming", "rap_hiphop", "ballad_emotional", "pop_performance"],
    facialSelect: ["", "rock_intense", "metal_rage", "rap_high_intensity", "love_tender", "pop_polished", "custom"],
    videoStyleSelect: ["", "custom"], temporalEffectSelect: ["", "custom"], fxSelect: ["", "custom"],
    imageShotSelect: ["intimate"], imageAestheticSelect: [""], imageWorldStyleSelect: ["", "custom"], storyArcDetailSelect: ["standard", "rich"],
  };
  for (const [ref, values] of Object.entries(choices)) {
    args[ref].children = values.map((value) => ({ value })); args[ref].value = values[0];
  }
  let notifications = 0, dialogs = 0;
  for (const key of Object.keys(args).filter((key) => key.startsWith("refresh"))) args[key] = () => {};
  args.notifyStoryboardDefaultsChanged = () => { if (!state.applyingDefaultsPreset) notifications++; };
  args.openCustomCameraFlowDialog = async () => { dialogs++; return null; };
  args.imageAestheticPresets = [{ value: "" }]; args.imageShotFlowPresets = { intimate: {} };
  args.state = state;
  context.wireStoryboardEvents(args);
  const apply = (settings) => context.applyDefaultPreset(settings, { controls: args, state,
    refreshCameraFlowInfo: args.refreshCameraFlowInfo, notifyStoryboardDefaultsChanged: args.notifyStoryboardDefaultsChanged });
  return { context, bindings, args, state, apply, notifications: () => notifications, dialogs: () => dialogs };
}

test("All speed presets update real handlers, reset effects, persist once and leave scenes unchanged", () => {
  const f = fixture(), original = JSON.stringify(f.state.scenes);
  for (const [key, settings] of Object.entries(f.bindings.SPEED_DEFAULT_PRESETS)) {
    f.args.videoStyleSelect.value = "custom"; f.args.temporalEffectSelect.value = "custom"; f.args.fxSelect.value = "custom";
    const before = f.notifications();
    f.apply(settings);
    for (const field of Object.keys(settings)) assert.equal(f.state[field], settings[field], `${key}: ${field}`);
    for (const field of Object.keys(f.bindings.PRESET_BASELINE)) assert.equal(f.state[field], "");
    assert.equal(f.notifications(), before + 1);
    assert.equal(JSON.stringify(f.state.scenes), original);
    assert.equal("applyingDefaultsPreset" in f.state, false);
  }
});

test("Music styles compose with speed and remain editable afterward", () => {
  const f = fixture(); f.apply(f.bindings.SPEED_DEFAULT_PRESETS.fast);
  for (const key of Object.keys(f.bindings.MUSIC_DEFAULT_PRESETS)) {
    const settings = f.bindings.musicDefaultSettings(key);
    f.apply(settings);
    for (const [field, value] of Object.entries(settings)) assert.equal(f.state[field], value);
    assert.equal(f.state.cameraMotionSpeed, 8); assert.equal(f.state.cutFrequency, 7);
  }
  f.args.cameraSpeedInput.value = "3";
  f.args.cameraSpeedInput.dispatchEvent(new Event("input"));
  assert.equal(f.state.cameraMotionSpeed, 3);
});

test("Film mappings use valid film performance/facial presets", () => {
  const f = fixture();
  const valid = vm.runInContext("({ acting: ID_LORA_PERFORMANCE_STYLE_PRESETS.map(p => p.value), face: ID_LORA_FACIAL_PERFORMANCE_PRESETS.map(p => p.value) })",
    (() => { vm.runInContext(fs.readFileSync(path.join(__dirname, "../web/storyboard_builder/performance_presets.mjs"), "utf8").replace(/^export /gm, ""), f.context); return f.context; })());
  for (const key of Object.keys(f.bindings.MUSIC_DEFAULT_PRESETS)) {
    const settings = f.bindings.musicDefaultSettings(key, true);
    assert.ok(valid.acting.includes(settings.performanceStyle)); assert.ok(valid.face.includes(settings.facialPerformance));
  }
});

test("Named preset restores custom camera sequence without import dialog and other custom text", () => {
  const f = fixture();
  f.apply({ cameraFlow: "custom", customCameraFlowSequence: [{ shot: "medium shot", camera: "pan" }],
    facialPerformance: "custom", facialPerformanceCustom: "Subtle smile", globalConsistencyPhrase: "Silver jacket", storyArcDetail: "rich" });
  assert.equal(f.state.cameraFlow, "custom"); assert.equal(f.state.customCameraFlowSequence[0].shot, "medium shot");
  assert.equal(f.dialogs(), 0); assert.equal(f.state.facialPerformanceCustom, "Subtle smile");
  assert.equal(f.state.globalConsistencyPhrase, "Silver jacket"); assert.equal(f.state.storyArcDetail, "rich");
  assert.equal(f.notifications(), 1);
});

test("Toolbar saves and loads via shared backend, with baseline and editable labels", async () => {
  const f = fixture(), toolbar = f.context.buildDefaultPresetsToolbar(), requests = [];
  const savedSettings = { cameraMotionSpeed: 2, cutFrequency: 1, globalConsistencyPhrase: "Saved phrase" };
  const postJson = async (url, payload) => {
    requests.push({ url, payload });
    return url.includes("list_") ? { presets: [{ name: "My mix" }] } : { preset: { name: "My mix", settings: savedSettings } };
  };
  f.context.wireDefaultPresets({ toolbar, controls: f.args, state: f.state, film: false, postJson,
    createToast: () => {}, refreshCameraFlowInfo: f.args.refreshCameraFlowInfo, notifyStoryboardDefaultsChanged: f.args.notifyStoryboardDefaultsChanged });
  await new Promise((resolve) => setImmediate(resolve));
  toolbar.speed.value = "fast"; toolbar.speed.onchange(); toolbar.music.value = "rap"; toolbar.music.onchange();
  assert.equal(toolbar.speed.value, "fast"); assert.equal(f.state.cameraMotionSpeed, 8);
  f.args.cameraSpeedInput.dispatchEvent(new Event("change")); assert.equal(toolbar.speed.value, "");
  toolbar.name.value = "My mix"; await toolbar.save.onclick();
  const save = requests.find((request) => request.url.includes("save_"));
  assert.equal(save.payload.name, "My mix"); assert.equal(save.payload.settings.fxPreset, "");
  assert.equal(save.payload.settings.temporalWorldEffect, ""); assert.equal(save.payload.settings.videoStyle, "");
  assert.equal("scenes" in save.payload.settings, false);
  toolbar.saved.value = "My mix"; await toolbar.load.onclick();
  assert.equal(f.state.globalConsistencyPhrase, "Saved phrase"); assert.equal(f.state.cameraMotionSpeed, 2);
  assert.equal(toolbar.name.value, "My mix"); assert.equal(toolbar.load.disabled, false);
});

test("the chosen Speed and Music style are remembered and shown again", async () => {
  const f = fixture();
  const toolbar = {};
  for (const key of ["speed", "music", "saved", "name", "save", "load"]) toolbar[key] = new Element(key === "name" ? "input" : "select");
  toolbar.speed.children = ["", "static", "slow", "medium", "fast"].map((value) => ({ value }));
  toolbar.music.children = ["", "rock", "metal", "rap", "soft", "pop"].map((value) => ({ value }));
  f.state.defaultsSpeedPreset = "fast";
  f.state.defaultsMusicPreset = "rap";
  const wired = f.context.wireDefaultPresets({ toolbar, controls: f.args, state: f.state, film: false,
    postJson: async () => ({ presets: [] }), createToast() {}, refreshCameraFlowInfo() {},
    notifyStoryboardDefaultsChanged: f.args.notifyStoryboardDefaultsChanged });
  assert.deepEqual([toolbar.speed.value, toolbar.music.value], ["fast", "rap"], "saved choices are shown on open");
  toolbar.music.value = "rock";
  toolbar.music.onchange();
  assert.equal(f.state.defaultsMusicPreset, "rock");
  assert.equal(f.state.defaultsSpeedPreset, "fast", "speed and music style combine");
  // Editing a setting the music style set clears it, and that is remembered too.
  f.args.cameraFlowSelect.value = "balanced";
  f.args.cameraFlowSelect.dispatchEvent(new Event("change"));
  assert.equal(f.state.defaultsMusicPreset, "");
  f.state.defaultsSpeedPreset = "slow";
  wired.syncFromState();
  assert.equal(toolbar.speed.value, "slow", "a reloaded storyboard's choice is shown");
});

test("choosing or replacing speed, music and saved presets confirms immediately after applying", async () => {
  const f = fixture(), toolbar = f.context.buildDefaultPresetsToolbar();
  const confirmations = [];
  f.context.wireDefaultPresets({ toolbar, controls: f.args, state: f.state, film: false,
    postJson: async url => url.includes("list_") ? { presets: [{ name: "Gentle verse" }] }
      : { preset: { name: "Gentle verse", settings: { characterMotionSpeed: 1 } } },
    createToast() {}, refreshCameraFlowInfo: f.args.refreshCameraFlowInfo,
    notifyStoryboardDefaultsChanged: f.args.notifyStoryboardDefaultsChanged,
    onPresetApplied: name => confirmations.push({ name, saves: f.notifications(), speed: f.state.characterMotionSpeed }),
  });
  await new Promise(resolve => setImmediate(resolve));
  toolbar.speed.value = "slow"; toolbar.speed.onchange();
  assert.deepEqual(confirmations[0], { name: "Slow speed", saves: 1, speed: 2 });
  toolbar.speed.value = "fast"; toolbar.speed.onchange();
  assert.deepEqual(confirmations[1], { name: "Fast speed", saves: 2, speed: 8 });
  toolbar.music.value = "metal"; toolbar.music.onchange();
  assert.equal(confirmations[2].name, "Metal music style");
  toolbar.saved.value = "Gentle verse"; await toolbar.saved.onchange();
  assert.deepEqual(confirmations[3], { name: "Gentle verse", saves: 4, speed: 1 });
});

test("real Speed and Music handlers replace presets on selected scenes without Save", async () => {
  const f = fixture();
  const source = fs.readFileSync(path.join(__dirname, "../web/storyboard_builder/defaults_auto_apply.mjs"), "utf8")
    .replace(/^import [^;]*;\r?\n/gm, "").replace(/^export /gm, "");
  vm.runInContext(source, f.context);
  f.state.mode = "image_to_video_prep";
  f.state.scenes = [1, 2, 3].map(n => ({ id: `scene${n}`, scene_number: n,
    project_video_engine: "minimax_h3", video_prompt_type: "t2v" }));
  f.state.selected = new Set(["scene2", "scene3"]);
  const toolbar = f.context.buildDefaultPresetsToolbar();
  Object.assign(f.args, { defaultsPresetsToolbar: toolbar, defaultsScope: {}, defaultsResult: {}, defaultsUndo: {} });
  let saves = 0;
  const controller = f.context.createAutoSceneDefaults({ state: f.state, controls: f.args,
    persist: () => saves++, renderTable: () => controller.syncScope(), imageShotFlowPresets: {}, imageAestheticPresets: [] });
  f.args.notifyStoryboardDefaultsChanged = options => controller.commit(options);
  // Rewire the actual individual control handlers to the auto-apply controller.
  f.context.wireStoryboardEvents(f.args);
  const confirmations = [];
  f.context.wireDefaultPresets({ toolbar, controls: f.args, state: f.state, film: false,
    postJson: async () => ({ presets: [] }), createToast() {}, refreshCameraFlowInfo() {},
    notifyStoryboardDefaultsChanged: f.args.notifyStoryboardDefaultsChanged,
    noteDefaultKeys: keys => controller.note(keys),
    onPresetApplied: (name, details) => confirmations.push({ name, details, result: f.args.defaultsResult.textContent }) });
  controller.reset();
  await new Promise(resolve => setImmediate(resolve));
  toolbar.speed.value = "static"; toolbar.speed.onchange();
  for (const scene of f.state.scenes.slice(1)) {
    assert.equal(scene.scene_defaults.character_motion_speed, 0);
    assert.equal(scene.scene_defaults.camera_motion_speed, 0);
    assert.equal(scene.scene_defaults.minimax_h3_cut_frequency, 0);
  }
  toolbar.music.value = "soft"; toolbar.music.onchange();
  for (const scene of f.state.scenes.slice(1)) {
    assert.equal(scene.scene_defaults.performance_style, "ballad_emotional");
    assert.equal(scene.scene_defaults.facial_performance, "love_tender");
    assert.equal(scene.scene_defaults.character_motion_speed, 0, "music preserves the chosen speed");
  }
  toolbar.speed.value = "fast"; toolbar.speed.onchange();
  assert.equal(f.state.scenes[1].scene_defaults.character_motion_speed, 8);
  assert.equal(f.state.scenes[1].scene_defaults.performance_style, "ballad_emotional");
  assert.equal(f.state.scenes[0].scene_defaults, undefined);
  assert.equal(saves, 3);
  assert.deepEqual(confirmations.map(item => `${item.name}: ${item.result}`), ["Static speed: Updated scenes 2, 3", "Soft music style: Updated scenes 2, 3", "Fast speed: Updated scenes 2, 3"]);
  assert.match(confirmations[0].details, /characterMotionSpeed: 0\/10/);
  assert.match(confirmations[1].details, /facialPerformance: love_tender/);
});
