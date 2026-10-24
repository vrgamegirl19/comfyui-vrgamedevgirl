import { makeButton, makeInput, makeSelect, setButtonDisabled } from "./controls.mjs";
import { normalizeStoryboardCustomCameraFlowSequence } from "./shot_presets.mjs";

export const SPEED_DEFAULT_PRESETS = {
  static: { cameraMotionSpeed: 0, characterMotionSpeed: 0, cutFrequency: 0 },
  slow: { cameraMotionSpeed: 2, characterMotionSpeed: 2, cutFrequency: 1 },
  medium: { cameraMotionSpeed: 5, characterMotionSpeed: 5, cutFrequency: 4 },
  fast: { cameraMotionSpeed: 8, characterMotionSpeed: 8, cutFrequency: 7 },
};
export const MUSIC_DEFAULT_PRESETS = {
  rock: { cameraFlow: "music_video", performanceStyle: "rock_punk", facialPerformance: "rock_intense" },
  metal: { cameraFlow: "energetic", performanceStyle: "metal_screaming", facialPerformance: "metal_rage" },
  rap: { cameraFlow: "medium_shots", performanceStyle: "rap_hiphop", facialPerformance: "rap_high_intensity" },
  soft: { cameraFlow: "medium_shots", performanceStyle: "ballad_emotional", facialPerformance: "love_tender" },
  pop: { cameraFlow: "music_video", performanceStyle: "pop_performance", facialPerformance: "pop_polished" },
};
const FILM_MUSIC_DEFAULTS = {
  rock: ["punk_bar_attitude", "defiant_controlled"], metal: ["tense_confrontation", "defiant_controlled"],
  rap: ["noir_restraint", "defiant_controlled"], soft: ["emotional_confession", "vulnerable_confession"],
  pop: ["dialogue_naturalism", ""],
};
export const PRESET_BASELINE = { videoStyle: "", temporalWorldEffect: "", fxPreset: "" };
export const DEFAULT_PRESET_CONTROLS = {
  cameraFlow: "cameraFlowSelect", cameraMotionSpeed: "cameraSpeedInput", characterMotionSpeed: "characterSpeedInput",
  cutFrequency: "cutFrequencyInput", performanceStyle: "performanceSelect", facialPerformance: "facialSelect",
  facialPerformanceCustom: "facialCustomInput", globalConsistencyPhrase: "consistencyInput",
  storyArcDetail: "storyArcDetailSelect", imageAesthetic: "imageAestheticSelect", imageShotFlow: "imageShotSelect",
  imageWorldStyle: "imageWorldStyleSelect", imageCustomStyle: "imageCustomStyleInput",
  videoStyle: "videoStyleSelect", temporalWorldEffect: "temporalEffectSelect", fxPreset: "fxSelect",
};

export function musicDefaultSettings(key, film = false) {
  const settings = MUSIC_DEFAULT_PRESETS[key];
  if (!settings) return null;
  const [performanceStyle, facialPerformance] = film ? FILM_MUSIC_DEFAULTS[key] : [settings.performanceStyle, settings.facialPerformance];
  return { ...settings, performanceStyle, facialPerformance, facialPerformanceCustom: "" };
}

export function buildDefaultPresetsToolbar() {
  const toolbar = document.createElement("div");
  toolbar.style.cssText = "display:flex;flex-wrap:wrap;gap:8px;align-items:end;padding:10px;background:#182538;border:1px solid #334155;border-radius:8px;margin-bottom:10px;";
  const speed = makeSelect([{ value: "", label: "Custom / choose speed" }, ...Object.keys(SPEED_DEFAULT_PRESETS).map((value) => ({ value, label: value[0].toUpperCase() + value.slice(1) }))]);
  const music = makeSelect([{ value: "", label: "Choose music style" }, ...Object.keys(MUSIC_DEFAULT_PRESETS).map((value) => ({ value, label: value[0].toUpperCase() + value.slice(1) }))]);
  const saved = makeSelect([{ value: "", label: "Choose saved preset" }]);
  const name = makeInput("", "Preset name");
  name.maxLength = 60;
  const save = makeButton("Save preset", "primary"), load = makeButton("Load preset");
  for (const button of [save, load]) button.style.cssText += "padding:5px 8px;font-size:11px;font-weight:700;line-height:1.2;";
  for (const [caption, control] of [["Speed", speed], ["Music style", music], ["Saved presets", saved], ["Name", name]]) {
    const label = document.createElement("label");
    label.style.cssText = "display:flex;flex-direction:column;gap:5px;flex:1 1 155px;min-width:0;color:#cffafe;font-weight:700;";
    const text = document.createElement("span");
    text.textContent = caption;
    control.style.minWidth = "0";
    control.setAttribute("aria-label", caption);
    label.append(text, control);
    toolbar.append(label);
  }
  toolbar.append(save);
  const help = document.createElement("div");
  help.style.cssText = "flex-basis:100%;font-size:11px;color:#94a3b8;line-height:1.4;";
  help.textContent = "Presets update scene defaults; each setting remains editable. Speed and music style combine. Presets use Default video aesthetic with world FX and shot FX Off. Changes apply automatically to all scenes or only checked scenes; Story arc detail remains project-wide.";
  toolbar.append(help);
  return { toolbar, speed, music, saved, name, save, load };
}

// Use the same input/change handlers as individual edits. A saved custom camera list is
// already supplied, so it can be restored without opening the import dialog.
export function applyDefaultPreset(settings, { controls, state, refreshCameraFlowInfo, notifyStoryboardDefaultsChanged, noteDefaultKeys }) {
  const imageOnly = new Set(["imageWorldStyle", "imageCustomStyle", "imageShotFlow", "imageAesthetic"]);
  const applied = Object.fromEntries(Object.entries({ ...settings, ...PRESET_BASELINE })
    .filter(([key]) => state.mode !== "image_to_video_prep" || !imageOnly.has(key)));
  noteDefaultKeys?.(Object.keys(applied));
  state.applyingDefaultsPreset = true;
  try {
    if (Array.isArray(settings.customCameraFlowSequence)) {
      state.customCameraFlowSequence = normalizeStoryboardCustomCameraFlowSequence(settings.customCameraFlowSequence);
    }
    for (const [key, value] of Object.entries(applied)) {
      const control = controls[DEFAULT_PRESET_CONTROLS[key]];
      if (!control) continue;
      if (control.tagName.toLowerCase() === "select" && !Array.from(control.options).some((option) => option.value === String(value))) continue;
      control.value = String(value);
      if (key === "cameraFlow" && value === "custom") {
        state.cameraFlow = value;
        refreshCameraFlowInfo();
      } else {
        control.dispatchEvent(new Event("input", { bubbles: true }));
        control.dispatchEvent(new Event("change", { bubbles: true }));
      }
    }
  } finally {
    delete state.applyingDefaultsPreset;
    notifyStoryboardDefaultsChanged();
  }
}

export function wireDefaultPresets({ toolbar, controls, state, film, postJson, createToast, refreshCameraFlowInfo, notifyStoryboardDefaultsChanged, noteDefaultKeys }) {
  const context = { controls, state, refreshCameraFlowInfo, notifyStoryboardDefaultsChanged, noteDefaultKeys };
  let applying = false;
  const apply = (settings) => {
    applying = true;
    try { applyDefaultPreset(settings, context); } finally { applying = false; }
  };
  // The chosen Speed and Music style are saved with the Scene Defaults (state.defaultsSpeedPreset and
  // state.defaultsMusicPreset) and shown again when the panel opens.
  const remember = () => {
    state.defaultsSpeedPreset = toolbar.speed.value || "";
    state.defaultsMusicPreset = toolbar.music.value || "";
  };
  const syncFromState = () => {
    toolbar.speed.value = SPEED_DEFAULT_PRESETS[state.defaultsSpeedPreset] ? state.defaultsSpeedPreset : "";
    toolbar.music.value = MUSIC_DEFAULT_PRESETS[state.defaultsMusicPreset] ? state.defaultsMusicPreset : "";
  };
  syncFromState();
  toolbar.speed.onchange = () => {
    if (!SPEED_DEFAULT_PRESETS[toolbar.speed.value]) {
      remember();
      notifyStoryboardDefaultsChanged();
      return;
    }
    remember();
    apply(SPEED_DEFAULT_PRESETS[toolbar.speed.value]);
    toolbar.saved.value = "";
  };
  toolbar.music.onchange = () => {
    const settings = musicDefaultSettings(toolbar.music.value, film);
    remember();
    if (!settings) {
      notifyStoryboardDefaultsChanged();
      return;
    }
    apply(settings);
    toolbar.saved.value = "";
  };
  // Avoid showing a preset label once any of its component settings is edited.
  for (const [key, ref] of Object.entries(DEFAULT_PRESET_CONTROLS)) {
    controls[ref].addEventListener("change", () => {
      if (applying) return;
      if (key in SPEED_DEFAULT_PRESETS.medium) toolbar.speed.value = "";
      if (["cameraFlow", "performanceStyle", "facialPerformance", "facialPerformanceCustom"].includes(key)) toolbar.music.value = "";
      if (key in PRESET_BASELINE) { toolbar.speed.value = ""; toolbar.music.value = ""; }
      toolbar.saved.value = "";
      remember();
    });
  }
  const refresh = async (selected = "") => {
    const data = await postJson("/vrgdg/music_builder/list_scene_defaults_presets");
    toolbar.saved.replaceChildren();
    for (const value of ["", ...data.presets.map((preset) => preset.name)]) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value || "Choose saved preset";
      toolbar.saved.append(option);
    }
    toolbar.saved.value = selected;
  };
  const busy = async (task) => {
    for (const control of [toolbar.save, toolbar.load, toolbar.saved, toolbar.name, toolbar.speed, toolbar.music]) setButtonDisabled(control, true);
    try { await task(); } catch (error) { createToast(String(error?.message || error), true); }
    finally { for (const control of [toolbar.save, toolbar.load, toolbar.saved, toolbar.name, toolbar.speed, toolbar.music]) setButtonDisabled(control, false); }
  };
  toolbar.load.onclick = () => busy(async () => {
    if (!toolbar.saved.value) throw new Error("Choose a saved preset first.");
    const data = await postJson("/vrgdg/music_builder/load_scene_defaults_preset", { name: toolbar.saved.value });
    apply(data.preset.settings);
    toolbar.speed.value = "";
    toolbar.music.value = "";
    remember();
    toolbar.name.value = data.preset.name;
    createToast(`Loaded Scene Defaults preset: ${data.preset.name}`);
  });
  toolbar.saved.onchange = () => toolbar.saved.value ? toolbar.load.onclick() : undefined;
  toolbar.save.onclick = () => busy(async () => {
    const settings = Object.fromEntries(Object.entries(DEFAULT_PRESET_CONTROLS).map(([key, ref]) => [key, controls[ref].value]));
    settings.customCameraFlowSequence = state.customCameraFlowSequence || [];
    const payload = { name: toolbar.name.value, settings: { ...settings, ...PRESET_BASELINE } };
    let data;
    try { data = await postJson("/vrgdg/music_builder/save_scene_defaults_preset", payload); }
    catch (error) {
      if (error.status !== 409) throw error;
      if (!window.confirm(`Replace the saved preset "${payload.name}"?`)) return;
      data = await postJson("/vrgdg/music_builder/save_scene_defaults_preset", { ...payload, overwrite: true });
    }
    await refresh(data.preset.name);
    createToast(`Saved Scene Defaults preset: ${data.preset.name}`);
  });
  refresh().catch((error) => createToast(`Could not load Scene Defaults presets: ${String(error?.message || error)}`, true));
  return { syncFromState };
}
