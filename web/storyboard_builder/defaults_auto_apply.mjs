import { storyboardSpeedGuidance } from "./scenes.mjs";
import { SCENE_DEFAULT_KEYS, effectiveSceneDefaultsState } from "./scene_defaults_state.mjs";
import { DEFAULT_PRESET_CONTROLS, SPEED_DEFAULT_PRESETS, MUSIC_DEFAULT_PRESETS, musicDefaultSettings } from "./defaults_presets.mjs";
import { storyboardCameraFlowEntry } from "./shot_presets.mjs";
import { storyboardSceneSupportsVideoStyle } from "./video_style.mjs";

const CONTROLS = {
  ...DEFAULT_PRESET_CONTROLS, videoStyleCustom: "videoStyleCustomInput", cutStyle: "cutStyleSelect",
  temporalWorldEffect: "temporalEffectSelect", temporalWorldEffectCustom: "temporalEffectCustomInput",
  temporalAllowBackgroundExtras: "temporalExtrasInput", temporalBackgroundIntensity: "temporalIntensityInput",
  temporalEnvironmentTimePassage: "temporalEnvironmentInput", temporalProtectedCharacters: "temporalProtectedSelect",
  temporalProtectedCustom: "temporalProtectedCustomInput", fxPreset: "fxSelect", fxCustomJson: "fxCustomInput",
};
const clone = value => value === undefined ? undefined : JSON.parse(JSON.stringify(value));
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const SCENE_ONLY_FIELDS = new Set(["cameraFlow", "customCameraFlowSequence", "cameraMotionSpeed", "characterMotionSpeed",
  "cutFrequency", "cutStyle", "performanceStyle", "facialPerformance", "facialPerformanceCustom",
  "defaultsSpeedPreset", "defaultsMusicPreset"]);
const SNAPSHOT_KEYS = [...Object.keys(SCENE_DEFAULT_KEYS), "storyArcDetail", "defaultsSpeedPreset", "defaultsMusicPreset"];
const snapshot = state => Object.fromEntries(SNAPSHOT_KEYS.map(key => [key, clone(state[key])]));

export function createAutoSceneDefaults({ state, controls, persist, renderTable, refreshControls,
  imageShotFlowPresets, imageAestheticPresets, film = false }) {
  let baseline = snapshot(state), undo = null, lastScope = null, busy = false;
  const pending = new Set();
  const { defaultsScope, defaultsResult, defaultsUndo } = controls;
  const targets = () => state.scenes.map((scene, index) => ({ scene, index }))
    .filter(({ scene }) => !state.selected?.size || state.selected.has(scene.id));
  const numbers = rows => rows.map(({ scene, index }) => scene.scene_number || index + 1).join(", ");
  const refreshDisplay = source => {
    const saved = snapshot(state);
    for (const key of SNAPSHOT_KEYS) if (source[key] !== undefined) state[key] = clone(source[key]);
    try { refreshControls?.(); } finally { Object.assign(state, saved); }
    controls.defaultsCards?.refreshSummaries?.();
  };
  const showControls = source => {
    for (const [key, ref] of Object.entries(CONTROLS)) {
      const control = controls[ref];
      if (!control || source[key] === undefined) continue;
      if (control.type === "checkbox") control.checked = Boolean(source[key]);
      else control.value = String(source[key]);
    }
    refreshDisplay(source);
  };
  const syncScope = () => {
    const rows = targets(), selected = Boolean(state.selected?.size);
    defaultsScope.textContent = selected ? `Editing selected scenes: ${numbers(rows) || "none (selection unavailable)"}` : "Editing all scenes";
    if (controls.defaultsScopeFooter) controls.defaultsScopeFooter.textContent = defaultsScope.textContent;
    defaultsScope.title = "Camera, editing, performance and facial settings are stored per scene. Only fields you change are updated. Story arc detail is project-wide.";
    const signature = (selected ? "selected:" : "all:") + rows.map(row => row.scene.id).join("|")
      + JSON.stringify(rows.map(row => row.scene.scene_defaults || {}));
    if (signature !== lastScope) {
      // Summary refreshes call back into syncScope. Mark this scope before refreshing.
      lastScope = signature;
      showControls(rows.length ? effectiveSceneDefaultsState(state, rows[0].scene) : state);
      const toolbar = controls.defaultsPresetsToolbar;
      if (toolbar) {
        const values = rows.map(row => effectiveSceneDefaultsState(state, row.scene));
        const matches = settings => values.length && values.every(value => Object.entries(settings).every(([key, item]) => same(value[key], item)));
        toolbar.speed.value = Object.keys(SPEED_DEFAULT_PRESETS).find(key => matches(SPEED_DEFAULT_PRESETS[key])) || "";
        toolbar.music.value = Object.keys(MUSIC_DEFAULT_PRESETS).find(key => matches(musicDefaultSettings(key, film))) || "";
        toolbar.saved.value = "";
      }
    }
  };
  for (const [key, ref] of Object.entries(CONTROLS)) {
    controls[ref]?.addEventListener("change", () => pending.add(key), true);
  }
  function apply(scene, fields, index) {
    scene.scene_defaults = { ...scene.scene_defaults };
    for (const key of fields) {
      if (["videoStyle", "videoStyleCustom"].includes(key) && !storyboardSceneSupportsVideoStyle(scene)) continue;
      if (state[key] !== undefined) scene.scene_defaults[SCENE_DEFAULT_KEYS[key]] = clone(state[key]);
    }
    if (fields.includes("cameraMotionSpeed")) scene.scene_defaults.camera_guidance = storyboardSpeedGuidance(state.cameraMotionSpeed, "camera");
    if (fields.includes("characterMotionSpeed")) scene.scene_defaults.character_guidance = storyboardSpeedGuidance(state.characterMotionSpeed, "character");
    for (const [field, key] of [["performance_style", "performanceStyle"], ["facial_performance", "facialPerformance"],
      ["facial_performance_custom", "facialPerformanceCustom"], ["video_style", "videoStyle"], ["video_style_custom", "videoStyleCustom"]]) {
      if (fields.includes(key) && (!key.startsWith("videoStyle") || storyboardSceneSupportsVideoStyle(scene))) scene[field] = state[key] || "";
    }
    if (fields.includes("temporalWorldEffect") || fields.includes("temporalWorldEffectCustom")) {
      scene.temporal_world_effect_override = state.temporalWorldEffect || "off";
      scene.temporal_world_effect_custom = state.temporalWorldEffectCustom || "";
    }
    if (fields.includes("cameraFlow") && state.mode === "image_to_video_prep") {
      const entry = state.cameraFlow === "off" ? {} : storyboardCameraFlowEntry(state.cameraFlow, index, "", state.customCameraFlowSequence);
      scene.shot_type = entry?.shot || "";
      scene.camera_motion = entry?.camera || "";
    }
    if (fields.includes("imageShotFlow") && state.mode !== "image_to_video_prep") {
      const sequence = imageShotFlowPresets[state.imageShotFlow]?.sequence || [];
      scene.shot_type = sequence.length ? sequence[index % sequence.length] : "";
    }
    if (fields.includes("imageAesthetic") && state.mode !== "image_to_video_prep") {
      const description = imageAestheticPresets.find(item => item.value === state.imageAesthetic)?.description || "";
      scene.motion_summary = String(scene.motion_summary || "").split(/\r?\n/)
        .filter(line => !/^image aesthetic:/i.test(line.trim())).concat(description ? [`Image aesthetic: ${description}`] : []).join("\n").trim();
    }
  }
  function commit() {
    if (state.applyingDefaultsPreset || busy) return;
    const now = snapshot(state);
    const fields = Object.keys(SCENE_DEFAULT_KEYS).filter(key => pending.has(key) || !same(now[key], baseline[key]));
    const storyChanged = pending.has("storyArcDetail") || !same(now.storyArcDetail, baseline.storyArcDetail);
    pending.clear();
    if (!fields.length && !storyChanged) { persist(); return; }
    const requested = fields.length ? targets() : [], selected = Boolean(state.selected?.size);
    const rows = requested.filter(({ scene }) => fields.some(key => !["videoStyle", "videoStyleCustom"].includes(key)
      || storyboardSceneSupportsVideoStyle(scene)));
    const before = rows.map(({ scene }) => ({ id: scene.id, value: clone(scene) }));
    const globalBefore = clone(baseline);
    busy = true;
    try {
      for (const { scene, index } of rows) apply(scene, fields, index);
      for (const key of SNAPSHOT_KEYS) {
        if (SCENE_ONLY_FIELDS.has(key) || (selected && key !== "storyArcDetail")) state[key] = clone(baseline[key]);
      }
      const patches = before.map(({ id, value }) => {
        const current = state.scenes.find(scene => scene.id === id);
        const keys = [...new Set([...Object.keys(value), ...Object.keys(current)])].filter(key => !same(value[key], current[key]));
        const defaultsKeys = [...new Set([...Object.keys(value.scene_defaults || {}), ...Object.keys(current.scene_defaults || {})])]
          .filter(key => !same(value.scene_defaults?.[key], current.scene_defaults?.[key]));
        return { id, keys: keys.filter(key => key !== "scene_defaults"), defaultsKeys, value };
      });
      undo = { patches, scope: numbers(rows), storyChanged, globalBefore, globalKeys: SNAPSHOT_KEYS.filter(key => !same(globalBefore[key], state[key])) };
      baseline = snapshot(state);
      defaultsResult.textContent = fields.length
        ? (!selected ? (rows.length === state.scenes.length && rows.length ? "Updated all scenes"
          : rows.length ? `Updated all eligible scenes: ${numbers(rows)}` : "No eligible scenes to update")
          : rows.length ? `Updated scenes ${numbers(rows)}` : "No scenes updated: selected scenes are unavailable")
        : "Story arc detail updated project-wide";
      if (storyChanged && fields.length) defaultsResult.textContent += ". Story arc detail updated project-wide";
      defaultsUndo.hidden = false;
      if (!rows.length && fields.length) showControls(state);
      renderTable();
      persist();
      refreshDisplay(rows.length ? effectiveSceneDefaultsState(state, rows[0].scene) : state);
    } finally { busy = false; }
  }
  defaultsUndo.onclick = () => {
    if (!undo || busy) return;
    for (const { id, keys, defaultsKeys, value } of undo.patches) {
      const scene = state.scenes.find(item => item.id === id);
      if (!scene) continue;
      if (defaultsKeys.length) scene.scene_defaults ||= {};
      for (const key of defaultsKeys) {
        if (Object.hasOwn(value.scene_defaults || {}, key)) scene.scene_defaults[key] = clone(value.scene_defaults[key]);
        else delete scene.scene_defaults[key];
      }
      if (!Object.keys(scene.scene_defaults || {}).length && !Object.hasOwn(value, "scene_defaults")) delete scene.scene_defaults;
      for (const key of keys) {
        if (Object.hasOwn(value, key)) scene[key] = clone(value[key]);
        else delete scene[key];
      }
    }
    for (const key of undo.globalKeys) state[key] = clone(undo.globalBefore[key]);
    const undoneScope = undo.scope;
    const undoneStory = undo.storyChanged;
    undo = null; baseline = snapshot(state); pending.clear();
    defaultsUndo.hidden = true;
    defaultsResult.textContent = undoneScope ? `Undid update to scenes ${undoneScope}`
      : undoneStory ? "Undid project-wide story arc detail update" : "Scene Defaults update undone";
    lastScope = null; syncScope(); renderTable(); persist();
  };
  return { commit, syncScope, note(keys) { for (const key of keys) pending.add(key); }, reset() { baseline = snapshot(state); pending.clear(); lastScope = null; syncScope(); } };
}
