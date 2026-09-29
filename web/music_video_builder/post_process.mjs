import { api } from "../../../scripts/api.js";
import { postJson } from "./comfy_api.mjs";
import { escapeHtml, makeButton, makeCheckbox, makeMiniButton, toast } from "./controls.mjs";
import { showInfoModal, showLargeTextModal, showTextInputModal } from "./dialogs.mjs";
import { selectedSegmentVideoPath } from "./selection_preview.mjs";
import { activateSegmentVideoPath, mediaPathKey, normalizeSegmentVideoHistory } from "./timeline_state.mjs";

export function lutLabelFromName(name = "") {
  return String(name || "").replace(/\.cube$/i, "").replace(/_/g, " ").trim();
}

export function normalizeSceneLut(lut = {}, existing = {}) {
  const name = String(lut?.name || lut?.lut_name || "").trim();
  if (!name) return null;
  return {
    name,
    label: String(lut?.label || lutLabelFromName(name)),
    strength: Number.isFinite(Number(lut?.strength ?? existing?.strength)) ? Math.max(0, Math.min(10, Number(lut?.strength ?? existing?.strength))) : 10,
    enabled: lut?.enabled !== false,
  };
}

export function sceneLutBadgeHtml(segment) {
  const lut = normalizeSceneLut(segment?.lut || {});
  if (!lut || lut.enabled === false) return "";
  const label = escapeHtml(lut.label || lutLabelFromName(lut.name));
  return `<span title="LUT: ${label}" style="border:1px solid #f0abfc;border-radius:4px;padding:2px 5px;font-size:10px;font-weight:900;color:#f5d0fe;background:rgba(88,28,135,.42);">LUT</span>`;
}

export function normalizeSceneFilmGrain(settings = {}, existing = {}) {
  const settingsObject = settings && typeof settings === "object" ? settings : {};
  const existingObject = existing && typeof existing === "object" ? existing : {};
  const hasSettings = Object.keys(settingsObject).length > 0;
  const hasExisting = Object.keys(existingObject).length > 0;
  if (!hasSettings && !hasExisting) return null;
  const source = hasSettings ? settingsObject : existingObject;
  if (!source || typeof source !== "object") return null;
  const intensityValue = Number(source.grain_intensity ?? source.intensity ?? existingObject.grain_intensity);
  const saturationValue = Number(source.saturation_mix ?? source.saturation ?? existingObject.saturation_mix);
  return {
    enabled: source.enabled !== false,
    grain_intensity: Number.isFinite(intensityValue) ? Math.max(0, Math.min(1, intensityValue)) : 0.04,
    saturation_mix: Number.isFinite(saturationValue) ? Math.max(0, Math.min(1, saturationValue)) : 0.5,
  };
}

function defaultFilmGrainSettings() {
  return { enabled: true, grain_intensity: 0.04, saturation_mix: 0.5 };
}

export function filmGrainLabel(grain) {
  const settings = normalizeSceneFilmGrain(grain || defaultFilmGrainSettings()) || defaultFilmGrainSettings();
  return `Intensity ${settings.grain_intensity.toFixed(3)} / Color ${settings.saturation_mix.toFixed(2)}`;
}

export function sceneFilmGrainBadgeHtml(segment) {
  const grain = normalizeSceneFilmGrain(segment?.film_grain || {});
  if (!grain || grain.enabled === false) return "";
  return `<span title="Film Grain: ${escapeHtml(filmGrainLabel(grain))}" style="border:1px solid #fbbf24;border-radius:4px;padding:2px 5px;font-size:10px;font-weight:900;color:#fde68a;background:rgba(113,63,18,.42);">GRAIN</span>`;
}

function adjustPresetLabel(preset) {
  return String(preset?.label || preset?.name || "").replace(/_/g, " ");
}

function restoreSegmentBeforeAdjust(segment) {
  if (!segment) return;
  const sourcePath = String(segment.adjust_last_applied?.source_video_path || "").trim();
  if (!sourcePath) return;
  activateSegmentVideoPath(segment, sourcePath, segment.adjust_last_applied?.source_thumbnail_path || segment.video_thumbnail_path || "");
  segment.video_cache_bust = Date.now();
}

function restoreSegmentBeforeFilmGrain(segment) {
  if (!segment) return;
  const sourcePath = String(segment.film_grain_last_applied?.source_video_path || "").trim();
  if (!sourcePath) return;
  activateSegmentVideoPath(segment, sourcePath, segment.film_grain_last_applied?.source_thumbnail_path || segment.video_thumbnail_path || "");
  segment.video_cache_bust = Date.now();
}

function makeFilmGrainRange(label, value, min, max, step, decimals) {
  const wrapper = document.createElement("div");
  wrapper.style.cssText = "display:flex;flex-direction:column;gap:6px;";
  const caption = document.createElement("div");
  caption.style.cssText = "display:flex;justify-content:space-between;gap:8px;font-size:11px;color:#d4d4d8;font-weight:900;";
  const name = document.createElement("span");
  name.textContent = label;
  const amount = document.createElement("span");
  amount.style.color = "#fde68a";
  const input = document.createElement("input");
  input.type = "range";
  input.min = String(min);
  input.max = String(max);
  input.step = String(step);
  input.value = String(value);
  input.style.width = "100%";
  const sync = () => { amount.textContent = Number(input.value || 0).toFixed(decimals); };
  input.oninput = sync;
  sync();
  caption.append(name, amount);
  wrapper.append(caption, input);
  return { wrapper, input, amount };
}

export function createPostProcess({
  activeSegment, allEditableSegments, applySceneAdjustToRenderedVideo, applySceneFilmGrainToRenderedVideo,
  applySceneLutToRenderedVideo, autoSaveSessionQuiet, createProgressWindow, filmGrainPane, fxOverlaysPane,
  fxOverlaysTab, fxPane, lutsTabButton, lutsTools, postProcessFxTab, postProcessGrainTab, postProcessLutsTab,
  postProcessPane, projectInput, pushHistory, render, sceneAdjustPanel, sceneListPane, sceneToolsPanel,
  scenesTabButton, segmentIndexInfo, segmentTrack, selectedSegmentImagePath, showAdjustPreviewImage,
  showFilmGrainPreviewImage, showLutPreviewImage, state, syncInspector, syncPreview, toolsPane,
  toolsTabButton,
}) {
  function syncLeftPanelTabs() {
    const active = state.leftPanelTab === "tools" || state.leftPanelTab === "luts" ? state.leftPanelTab : "scenes";
    state.leftPanelTab = active;
    const styleTab = (button, selected) => {
      button.style.background = selected ? "#0e7490" : "#18181b";
      button.style.borderColor = selected ? "#22d3ee" : "#3f3f46";
      button.style.color = selected ? "#ecfeff" : "#fafafa";
      button.style.fontWeight = "900";
    };
    styleTab(scenesTabButton, active === "scenes");
    styleTab(toolsTabButton, active === "tools");
    styleTab(lutsTabButton, active === "luts");
    sceneListPane.style.display = active === "scenes" ? "block" : "none";
    toolsPane.style.display = active === "tools" ? "block" : "none";
    postProcessPane.style.display = active === "luts" ? "flex" : "none";
    if (active === "luts") {
      syncPostProcessTabs();
    }
  }

  function syncPostProcessTabs() {
    const active = ["film_grain", "fx"].includes(state.postProcessTab) ? state.postProcessTab : "luts";
    state.postProcessTab = active;
    const styleTab = (button, selected) => {
      button.style.background = selected ? "#0e7490" : "#18181b";
      button.style.borderColor = selected ? "#22d3ee" : "#3f3f46";
      button.style.color = selected ? "#ecfeff" : "#fafafa";
      button.style.fontWeight = "900";
    };
    styleTab(postProcessLutsTab, active === "luts");
    styleTab(postProcessGrainTab, active === "film_grain");
    styleTab(postProcessFxTab, active === "fx");
    lutsTools.element.style.display = active === "luts" ? "block" : "none";
    filmGrainPane.style.display = active === "film_grain" ? "flex" : "none";
    fxPane.style.display = active === "fx" ? "flex" : "none";
    if (active === "luts") {
      lutsTools.loadLuts().catch(() => null);
      lutsTools.render();
    } else if (active === "film_grain") {
      renderFilmGrainPostProcessPanel();
    } else {
      renderFxPostProcessPanel();
    }
  }

  function applyLutToSegment(lut, segment = activeSegment()) {
    if (!segment) {
      toast("Select a scene first, then choose a LUT.", true);
      return false;
    }
    if (segmentTrack(segment) === "overlay") {
      toast("LUTs can be applied to base scenes for now.", true);
      return false;
    }
    const nextLut = normalizeSceneLut(lut, segment.lut || {});
    if (!nextLut) {
      toast("That LUT could not be applied.", true);
      return false;
    }
    pushHistory();
    clearSegmentLutPreview(segment);
    segment.lut = nextLut;
    delete segment.lut_last_applied;
    delete segment.lut_rendered_path;
    delete segment.lut_rendered_thumbnail_path;
    if (String(segment.video_original_path || "").trim()) {
      activateSegmentVideoPath(segment, segment.video_original_path, segment.video_original_thumbnail_path || "");
      segment.video_cache_bust = Date.now();
    }
    state.activeId = segment.id;
    state.activeTrack = segmentTrack(segment);
    render();
    syncInspector();
    autoSaveSessionQuiet("scene LUT applied").catch(() => null);
    toast(`Applied LUT to ${segment.label || "scene"}.`);
    return true;
  }

  function applyLutNameToSegment(lutName, segment = activeSegment()) {
    const name = String(lutName || "").trim();
    if (!name) return false;
    const lut = (lutsTools.state.luts || []).find((item) => item.name === name) || { name, label: lutLabelFromName(name) };
    return applyLutToSegment(lut, segment);
  }

  const SCENE_ADJUST_FIELDS = [
    { key: "temperature", label: "Temp", group: "Color", min: -100, max: 100, step: 1, gradient: "linear-gradient(90deg,#2563eb,#f8fafc,#facc15)" },
    { key: "tint", label: "Tint", group: "Color", min: -100, max: 100, step: 1, gradient: "linear-gradient(90deg,#22c55e,#f8fafc,#d946ef)" },
    { key: "saturation", label: "Saturation", group: "Color", min: -100, max: 100, step: 1, gradient: "linear-gradient(90deg,#64748b,#f8fafc,#ef4444)" },
    { key: "exposure", label: "Exposure", group: "Lightness", min: -100, max: 100, step: 1 },
    { key: "contrast", label: "Contrast", group: "Lightness", min: -100, max: 100, step: 1 },
    { key: "highlights", label: "Highlights", group: "Lightness", min: -100, max: 100, step: 1 },
    { key: "shadows", label: "Shadows", group: "Lightness", min: -100, max: 100, step: 1 },
    { key: "whites", label: "Whites", group: "Lightness", min: -100, max: 100, step: 1 },
    { key: "blacks", label: "Blacks", group: "Lightness", min: -100, max: 100, step: 1 },
    { key: "sharpen", label: "Sharpen", group: "Effects", min: 0, max: 100, step: 1 },
    { key: "clarity", label: "Clarity", group: "Effects", min: -100, max: 100, step: 1 },
    { key: "vignette", label: "Vignette", group: "Effects", min: 0, max: 100, step: 1 },
    { key: "fade", label: "Fade", group: "Effects", min: 0, max: 100, step: 1 },
  ];
  const adjustPresetState = {
    loaded: false,
    loading: false,
    presets: [],
    selectedName: "",
  };
  let adjustLivePreviewStatusElement = null;
  let adjustLivePreviewStatusBar = null;

  function defaultSceneAdjustSettings() {
    const values = { enabled: false };
    for (const field of SCENE_ADJUST_FIELDS) values[field.key] = 0;
    return values;
  }

  function normalizeSceneAdjust(settings = {}, options = {}) {
    const source = settings && typeof settings === "object" ? settings : {};
    const hasValues = Object.keys(source).some((key) => key !== "enabled" && SCENE_ADJUST_FIELDS.some((field) => field.key === key));
    if (!hasValues && source.enabled !== true && !options.keepEmpty) return null;
    const next = defaultSceneAdjustSettings();
    next.enabled = source.enabled === true || (options.enableWhenChanged && hasValues);
    for (const field of SCENE_ADJUST_FIELDS) {
      const value = Number(source[field.key]);
      next[field.key] = Number.isFinite(value) ? Math.max(field.min, Math.min(field.max, value)) : 0;
    }
    return next;
  }

  function sceneAdjustHasChanges(adjust) {
    const settings = normalizeSceneAdjust(adjust || {}, { keepEmpty: true });
    return Boolean(settings?.enabled) || SCENE_ADJUST_FIELDS.some((field) => Number(settings?.[field.key] || 0) !== 0);
  }

  function sceneAdjustHasRenderableChanges(adjust) {
    const settings = normalizeSceneAdjust(adjust || {}, { keepEmpty: true });
    return Boolean(settings?.enabled) && SCENE_ADJUST_FIELDS.some((field) => Number(settings?.[field.key] || 0) !== 0);
  }

  function sceneAdjustSignature(adjust) {
    const settings = normalizeSceneAdjust(adjust || {}, { keepEmpty: true }) || defaultSceneAdjustSettings();
    return JSON.stringify({
      enabled: settings.enabled === true,
      values: SCENE_ADJUST_FIELDS.map((field) => [field.key, Number(settings[field.key] || 0)]),
    });
  }

  async function loadAdjustPresets(force = false) {
    if (adjustPresetState.loading) return adjustPresetState.presets;
    if (adjustPresetState.loaded && !force) return adjustPresetState.presets;
    adjustPresetState.loading = true;
    try {
      const response = await api.fetchApi("/vrgdg/music_builder/post_process/adjust/presets");
      const data = await response.json();
      if (!response.ok || data?.ok === false) throw new Error(data?.error || `Adjust preset list failed (${response.status})`);
      adjustPresetState.presets = Array.isArray(data.presets) ? data.presets : [];
      adjustPresetState.loaded = true;
    } catch (error) {
      toast(error?.message || "Could not load Adjust presets.", true);
    } finally {
      adjustPresetState.loading = false;
    }
    return adjustPresetState.presets;
  }

  async function applyAdjustPresetToSegment(preset, segment = activeSegment()) {
    if (!segment || !preset) return;
    const next = normalizeSceneAdjust(preset.settings || preset.adjust || {}, { keepEmpty: true }) || defaultSceneAdjustSettings();
    next.enabled = true;
    pushHistory();
    clearSegmentAdjustPreview(segment);
    restoreSegmentBeforeAdjust(segment);
    segment.adjust = next;
    delete segment.adjust_last_applied;
    delete segment.adjust_rendered_path;
    delete segment.adjust_rendered_thumbnail_path;
    adjustPresetState.selectedName = String(preset.name || "");
    render();
    syncInspector();
    autoSaveSessionQuiet("scene Adjust preset applied").catch(() => null);
    toast(`Applied Adjust preset: ${adjustPresetLabel(preset)}.`);
    const progress = createProgressWindow("Applying Adjust Preset Preview");
    progress.set(`Creating Adjust preview for ${segment.label || "scene"}...`, 35);
    try {
      await previewAdjustForSegment(next, segment, null, { quiet: true });
      progress.set("Adjust preset preview ready.", 100);
      progress.close(1200);
      render();
      syncInspector();
    } catch (error) {
      progress.set(`Adjust preset preview failed:\n${error?.message || error}`, 100);
      progress.close(4000);
      toast(error?.message || "Could not create Adjust preset preview.", true);
    }
  }

  async function saveCurrentAdjustPreset(segment = activeSegment()) {
    if (!segment) {
      toast("Select a scene before saving an Adjust preset.", true);
      return;
    }
    const adjust = normalizeSceneAdjust(segment.adjust || {}, { keepEmpty: true }) || defaultSceneAdjustSettings();
    const name = await showTextInputModal({
      title: "Save Adjust Preset",
      label: "Preset name",
      value: "My Adjust Preset",
      placeholder: "Moody contrast, warm concert, etc.",
      confirmLabel: "Save Preset",
    });
    if (!name) return;
    try {
      const data = await postJson("/vrgdg/music_builder/post_process/adjust/presets/save", {
        name,
        settings: adjust,
      }, 30000);
      adjustPresetState.presets = Array.isArray(data.presets) ? data.presets : adjustPresetState.presets;
      adjustPresetState.loaded = true;
      adjustPresetState.selectedName = data.preset?.name || "";
      renderSceneAdjustPanel();
      toast(`Saved global Adjust preset: ${name}.`);
    } catch (error) {
      toast(error?.message || "Could not save Adjust preset.", true);
    }
  }

  async function importAdjustPresetFromText() {
    const text = await showLargeTextModal({
      title: "Import Adjust Preset",
      label: "Paste preset JSON",
      value: "",
      placeholder: '{"name":"Warm Contrast","adjust":{"enabled":true,"contrast":12}}',
      confirmLabel: "Import Preset",
    });
    if (!text) return;
    try {
      const preset = JSON.parse(text);
      const data = await postJson("/vrgdg/music_builder/post_process/adjust/presets/import", {
        preset,
        name: preset?.name || "Imported Adjust Preset",
      }, 30000);
      adjustPresetState.presets = Array.isArray(data.presets) ? data.presets : adjustPresetState.presets;
      adjustPresetState.loaded = true;
      adjustPresetState.selectedName = data.preset?.name || "";
      renderSceneAdjustPanel();
      toast(`Imported global Adjust preset: ${data.preset?.label || preset?.name || "preset"}.`);
    } catch (error) {
      toast(error?.message || "Could not import Adjust preset JSON.", true);
    }
  }

  function setAdjustLivePreviewStatus(message = "") {
    state.adjustLivePreviewStatus = message;
    if (adjustLivePreviewStatusElement) {
      adjustLivePreviewStatusElement.textContent = message || "Live preview idle";
      const lower = String(message || "").toLowerCase();
      const busy = lower.includes("waiting") || lower.includes("updating");
      const failed = lower.includes("failed");
      const ready = lower.includes("ready");
      adjustLivePreviewStatusElement.style.color = failed ? "#fecaca" : busy ? "#67e8f9" : ready ? "#bbf7d0" : "#a1a1aa";
      adjustLivePreviewStatusElement.style.borderColor = failed ? "#7f1d1d" : busy ? "#0891b2" : ready ? "#15803d" : "#27272a";
      adjustLivePreviewStatusElement.style.background = failed ? "#2f1212" : busy ? "#082f49" : ready ? "#052e16" : "#18181b";
    }
    if (adjustLivePreviewStatusBar) {
      const lower = String(message || "").toLowerCase();
      const busy = lower.includes("waiting") || lower.includes("updating");
      const failed = lower.includes("failed");
      const ready = lower.includes("ready");
      adjustLivePreviewStatusBar.style.display = (busy || failed || ready) ? "block" : "none";
      adjustLivePreviewStatusBar.style.width = lower.includes("waiting") ? "35%" : lower.includes("updating") ? "72%" : "100%";
      adjustLivePreviewStatusBar.style.background = failed ? "#ef4444" : ready ? "#22c55e" : "#22d3ee";
    }
  }

  function scheduleAdjustLivePreview(segment = activeSegment()) {
    if (!state.adjustLivePreview || !segment) return;
    const adjust = normalizeSceneAdjust(segment.adjust || {}, { keepEmpty: true });
    if (!adjust || !sceneAdjustHasRenderableChanges(adjust)) return;
    state.adjustLivePreviewPending = true;
    setAdjustLivePreviewStatus("Preview waiting...");
    if (state.adjustLivePreviewTimer) clearTimeout(state.adjustLivePreviewTimer);
    state.adjustLivePreviewTimer = setTimeout(() => {
      state.adjustLivePreviewTimer = null;
      runAdjustLivePreview(segment).catch((error) => {
        setAdjustLivePreviewStatus("Preview failed");
        toast(error?.message || "Could not update Adjust live preview.", true);
      });
    }, 550);
  }

  async function runAdjustLivePreview(segment = activeSegment()) {
    if (!state.adjustLivePreview || !segment) return;
    if (state.adjustLivePreviewBusy) {
      state.adjustLivePreviewPending = true;
      return;
    }
    const adjust = normalizeSceneAdjust(segment.adjust || {}, { keepEmpty: true });
    if (!adjust || !sceneAdjustHasRenderableChanges(adjust)) return;
    const token = (state.adjustLivePreviewToken || 0) + 1;
    state.adjustLivePreviewToken = token;
    state.adjustLivePreviewBusy = true;
    state.adjustLivePreviewPending = false;
    setAdjustLivePreviewStatus("Preview updating...");
    const progress = createProgressWindow("Updating Adjust Preview");
    progress.set(`Updating Adjust preview for ${segment.label || "scene"}...\nThis updates a still frame only.`, 35);
    try {
      await previewAdjustForSegment(adjust, segment, null, { quiet: true, token });
      if (state.adjustLivePreviewToken === token) {
        setAdjustLivePreviewStatus("Preview ready");
        progress.set("Adjust preview ready.", 100);
        progress.close(1200);
      } else {
        progress.close(0);
      }
    } catch (error) {
      progress.set(`Adjust preview failed:\n${error?.message || error}`, 100);
      progress.close(4000);
      throw error;
    } finally {
      state.adjustLivePreviewBusy = false;
      if (state.adjustLivePreview && state.adjustLivePreviewPending) {
        state.adjustLivePreviewPending = false;
        scheduleAdjustLivePreview(segment);
      }
    }
  }

  function setSceneAdjustValue(segment, key, value) {
    if (!segment) return;
    const field = SCENE_ADJUST_FIELDS.find((item) => item.key === key);
    if (!field) return;
    const current = normalizeSceneAdjust(segment.adjust || {}, { keepEmpty: true }) || defaultSceneAdjustSettings();
    current.enabled = true;
    current[key] = Math.max(field.min, Math.min(field.max, Number(value) || 0));
    clearSegmentAdjustPreview(segment);
    restoreSegmentBeforeAdjust(segment);
    segment.adjust = current;
    delete segment.adjust_last_applied;
    delete segment.adjust_rendered_path;
    delete segment.adjust_rendered_thumbnail_path;
    scheduleAdjustLivePreview(segment);
    autoSaveSessionQuiet("scene adjust changed").catch(() => null);
  }

  function resetSceneAdjust(segment = activeSegment()) {
    if (!segment) return;
    pushHistory();
    state.adjustLivePreview = false;
    if (state.adjustLivePreviewTimer) clearTimeout(state.adjustLivePreviewTimer);
    state.adjustLivePreviewTimer = null;
    state.adjustLivePreviewPending = false;
    state.adjustLivePreviewBusy = false;
    setAdjustLivePreviewStatus("Live preview off");
    clearSegmentAdjustPreview(segment);
    restoreSegmentBeforeAdjust(segment);
    delete segment.adjust;
    delete segment.adjust_last_applied;
    delete segment.adjust_rendered_path;
    delete segment.adjust_rendered_thumbnail_path;
    if (selectedSegmentVideoPath(segment)) segment.preview_mode = "video";
    else segment.preview_mode = "image";
    syncPreview(segment);
    render();
    syncInspector();
    autoSaveSessionQuiet("scene adjust reset").catch(() => null);
    toast(`Reset Adjust for ${segment.label || "scene"}.`);
  }

  async function applySceneAdjustToAllScenes(segment = activeSegment()) {
    const adjust = normalizeSceneAdjust(segment?.adjust || {}, { keepEmpty: true });
    if (!segment || !adjust) return;
    const targets = allEditableSegments();
    if (!targets.length) return;
    const progress = createProgressWindow("Applying Adjust to All Scenes");
    progress.set(`Applying Adjust settings to ${targets.length} scene${targets.length === 1 ? "" : "s"}...`, 10);
    pushHistory();
    try {
      targets.forEach((target, index) => {
        progress.set(`Applying Adjust ${index + 1}/${targets.length}...\n${target.label || `Scene ${index + 1}`}`, Math.round(10 + ((index + 1) / targets.length) * 65));
        clearSegmentAdjustPreview(target);
        restoreSegmentBeforeAdjust(target);
        target.adjust = { ...adjust };
        delete target.adjust_last_applied;
        delete target.adjust_rendered_path;
        delete target.adjust_rendered_thumbnail_path;
      });
      render();
      syncInspector();
      progress.set("Saving scene Adjust settings...\nStitch will be ready after this completes.", 86);
      await autoSaveSessionQuiet("scene adjust applied to all");
      progress.set(`Adjust is ready for stitching on ${targets.length} scene${targets.length === 1 ? "" : "s"}.`, 100);
      progress.close(1400);
      toast(`Applied Adjust to ${targets.length} scene${targets.length === 1 ? "" : "s"}. Ready to stitch.`);
    } catch (error) {
      progress.set(`Apply Adjust to all scenes failed:\n${error?.message || error}`, 100);
      progress.close(5000);
      toast(error?.message || "Could not apply Adjust to all scenes.", true);
    }
  }

  function makeSceneAdjustRow(segment, field, adjust) {
    const row = document.createElement("div");
    row.style.cssText = "display:grid;grid-template-columns:72px minmax(0,1fr) 56px 28px;gap:8px;align-items:center;";
    const label = document.createElement("div");
    label.textContent = field.label;
    label.style.cssText = "font-size:11px;color:#d4d4d8;font-weight:800;";
    const range = document.createElement("input");
    range.type = "range";
    range.min = String(field.min);
    range.max = String(field.max);
    range.step = String(field.step);
    range.value = String(adjust[field.key] || 0);
    range.style.cssText = `width:100%;accent-color:#e5e7eb;${field.gradient ? `background:${field.gradient};` : ""}`;
    const value = document.createElement("input");
    value.type = "number";
    value.min = String(field.min);
    value.max = String(field.max);
    value.step = String(field.step);
    value.value = String(adjust[field.key] || 0);
    value.style.cssText = "width:56px;box-sizing:border-box;border:1px solid #27272a;border-radius:5px;background:#111113;color:#f4f4f5;padding:4px 5px;font-size:11px;text-align:right;";
    const reset = makeMiniButton("0");
    reset.title = `Reset ${field.label}`;
    reset.style.width = "28px";
    const commit = (nextValue) => {
      setSceneAdjustValue(segment, field.key, nextValue);
      value.value = String(segment.adjust?.[field.key] ?? 0);
      range.value = value.value;
    };
    range.oninput = () => {
      value.value = range.value;
      setSceneAdjustValue(segment, field.key, range.value);
    };
    value.onchange = () => commit(value.value);
    reset.onclick = () => commit(0);
    row.append(label, range, value, reset);
    return row;
  }

  function clearSegmentAdjustPreview(segment = activeSegment(), options = {}) {
    if (!segment) return;
    const path = String(segment.adjust_preview_image_path || "").trim();
    if (options.deleteFile !== false) deletePostProcessSourcePreview(segment, "adjust");
    delete segment.adjust_preview_image_path;
    delete segment.adjust_preview_source_path;
    delete segment.adjust_preview_source_preview_path;
    delete segment.adjust_preview_source_temporary;
    delete segment.adjust_preview_settings;
    if (path && options.deleteFile !== false) deletePostProcessPreviewFile(path);
    if (options.refresh) {
      if (selectedSegmentVideoPath(segment)) segment.preview_mode = "video";
      else segment.preview_mode = "image";
      syncPreview(segment);
      render();
      syncInspector();
    }
  }

  async function previewAdjustForSegment(settings, segment = activeSegment(), button = null, options = {}) {
    const current = segment || activeSegment();
    if (!current) {
      if (options.quiet) throw new Error("Select a scene first, then preview Adjust.");
      toast("Select a scene first, then preview Adjust.", true);
      return false;
    }
    const adjust = normalizeSceneAdjust(settings, { keepEmpty: true }) || defaultSceneAdjustSettings();
    if (!sceneAdjustHasRenderableChanges(adjust)) {
      if (options.quiet) throw new Error("Enable scene adjust and set at least one non-zero adjustment before previewing it.");
      toast("Enable scene adjust and set at least one non-zero adjustment before previewing it.", true);
      return false;
    }
    const sceneIndex = segmentIndexInfo(current).index;
    const videoPath = String(current.video_original_path || selectedSegmentVideoPath(current) || "").trim();
    const imagePath = selectedSegmentImagePath(current);
    const mediaPath = videoPath || imagePath;
    const mediaType = videoPath ? "video" : "image";
    if (!mediaPath) {
      if (options.quiet) throw new Error("This scene needs an image or video before previewing Adjust.");
      toast("This scene needs an image or video before previewing Adjust.", true);
      return false;
    }
    const projectFolder = String(projectInput.value || state.projectFolder || "").trim();
    if (!projectFolder) {
      if (options.quiet) throw new Error("Set a project folder before previewing Adjust.");
      toast("Set a project folder before previewing Adjust.", true);
      return false;
    }
    const previousText = button?.textContent || "";
    try {
      if (button) {
        button.disabled = true;
        button.textContent = "Previewing...";
      }
      clearSegmentLutPreview(current);
      clearSegmentAdjustPreview(current);
      clearSegmentFilmGrainPreview(current);
      const data = await postJson("/vrgdg/music_builder/post_process/adjust/preview", {
        input_path: mediaPath,
        media_type: mediaType,
        settings: adjust,
        device: "auto",
        scene_id: current.id || `scene_${sceneIndex}`,
        project_folder: projectFolder,
      }, 120000);
      if (options.token && state.adjustLivePreviewToken !== options.token) return;
      current.adjust_preview_image_path = data.preview_path || data.output || "";
      current.adjust_preview_source_path = mediaPath;
      current.adjust_preview_source_preview_path = data.source_preview_path || mediaPath;
      current.adjust_preview_source_temporary = Boolean(data.source_preview_temporary);
      current.adjust_preview_settings = { ...adjust, created_at: Date.now() };
      current.preview_mode = "image";
      if (!options.quiet) {
        render();
        syncInspector();
      }
      showAdjustPreviewImage(current);
      if (!options.quiet) toast(`Previewing Adjust on ${current.label || "scene"}.`);
      return true;
    } catch (error) {
      if (options.quiet) throw error;
      toast(error?.message || "Could not preview Adjust for this scene.", true);
      render();
      syncInspector();
    } finally {
      if (button) {
        button.disabled = false;
        button.textContent = previousText;
      }
    }
  }

  function renderSceneAdjustPanel() {
    if (!sceneAdjustPanel) return;
    sceneAdjustPanel.textContent = "";
    const segment = activeSegment();
    if (!segment) return;
    const adjust = normalizeSceneAdjust(segment.adjust || {}, { keepEmpty: true }) || defaultSceneAdjustSettings();
    const card = document.createElement("div");
    card.style.cssText = "display:flex;flex-direction:column;gap:10px;border:1px solid #303038;border-radius:7px;background:#18181b;padding:10px;";
    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:center;gap:8px;";
    const enabled = makeCheckbox("Enable scene adjust", adjust.enabled === true);
    enabled.input.onchange = () => {
      pushHistory();
      const current = normalizeSceneAdjust(segment.adjust || {}, { keepEmpty: true }) || defaultSceneAdjustSettings();
      current.enabled = enabled.input.checked;
      if (!current.enabled) {
        state.adjustLivePreview = false;
        if (state.adjustLivePreviewTimer) clearTimeout(state.adjustLivePreviewTimer);
        state.adjustLivePreviewTimer = null;
        state.adjustLivePreviewPending = false;
        setAdjustLivePreviewStatus("Live preview off");
      }
      clearSegmentAdjustPreview(segment);
      restoreSegmentBeforeAdjust(segment);
      segment.adjust = current;
      delete segment.adjust_last_applied;
      delete segment.adjust_rendered_path;
      delete segment.adjust_rendered_thumbnail_path;
      render();
      syncInspector();
      autoSaveSessionQuiet("scene adjust enabled").catch(() => null);
    };
    const hasRenderableAdjust = sceneAdjustHasRenderableChanges(adjust);
    const status = document.createElement("div");
    status.textContent = hasRenderableAdjust ? "Scene color settings" : (adjust.enabled === true ? "Enabled, no changes" : "Defaults");
    status.style.cssText = "margin-left:auto;color:#a1a1aa;font-size:11px;font-weight:800;";
    header.append(enabled.wrapper, status);
    card.append(header);

    const liveRow = document.createElement("div");
    liveRow.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) 30px;gap:8px;align-items:center;border:1px solid #27272a;border-radius:6px;background:#111113;padding:8px;";
    const livePreview = makeCheckbox("Live preview", Boolean(state.adjustLivePreview));
    livePreview.input.disabled = !hasRenderableAdjust;
    livePreview.wrapper.style.opacity = !hasRenderableAdjust ? ".55" : "1";
    livePreview.input.onchange = () => {
      state.adjustLivePreview = livePreview.input.checked;
      if (!state.adjustLivePreview) {
        if (state.adjustLivePreviewTimer) clearTimeout(state.adjustLivePreviewTimer);
        state.adjustLivePreviewTimer = null;
        state.adjustLivePreviewPending = false;
        setAdjustLivePreviewStatus("Live preview off");
      } else {
        setAdjustLivePreviewStatus("Live preview on");
        scheduleAdjustLivePreview(segment);
      }
    };
    const liveHint = makeMiniButton("?");
    liveHint.title = "How live preview works";
    liveHint.onclick = () => showInfoModal({
      title: "Adjust Live Preview",
      lines: [
        "Live preview updates a still frame only.",
        "It uses the scene image or the first frame of the scene video, waits briefly after slider movement, then refreshes the preview image.",
        "It does not render the full video until you click Render Adjust or build the final video.",
      ],
    });
    adjustLivePreviewStatusElement = document.createElement("div");
    adjustLivePreviewStatusElement.textContent = state.adjustLivePreviewStatus || (state.adjustLivePreview ? "Live preview on" : "Live preview off");
    adjustLivePreviewStatusElement.style.cssText = "grid-column:1 / -1;border:1px solid #27272a;border-radius:5px;background:#18181b;color:#a1a1aa;padding:6px 8px;font-size:10px;font-weight:900;";
    const liveProgressTrack = document.createElement("div");
    liveProgressTrack.style.cssText = "grid-column:1 / -1;height:5px;border-radius:999px;background:#27272a;overflow:hidden;";
    adjustLivePreviewStatusBar = document.createElement("div");
    adjustLivePreviewStatusBar.style.cssText = "display:none;width:0%;height:100%;border-radius:999px;background:#22d3ee;transition:width .18s ease,background .18s ease;";
    liveProgressTrack.append(adjustLivePreviewStatusBar);
    liveRow.append(livePreview.wrapper, liveHint, adjustLivePreviewStatusElement, liveProgressTrack);
    card.append(liveRow);
    setAdjustLivePreviewStatus(adjustLivePreviewStatusElement.textContent);

    if (!adjustPresetState.loaded && !adjustPresetState.loading) {
      loadAdjustPresets().then(() => renderSceneAdjustPanel()).catch(() => null);
    }
    const presetBox = document.createElement("div");
    presetBox.style.cssText = "display:flex;flex-direction:column;gap:8px;border:1px solid #27272a;border-radius:6px;background:#111113;padding:8px;";
    const presetTitle = document.createElement("div");
    presetTitle.textContent = "Global presets (output/VRGDG_AdjustPresets)";
    presetTitle.style.cssText = "font-size:11px;font-weight:900;color:#cffafe;";
    const presetRow = document.createElement("div");
    presetRow.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) 74px;gap:8px;";
    const presetSelect = document.createElement("select");
    presetSelect.style.cssText = "width:100%;box-sizing:border-box;border:1px solid #3f3f46;border-radius:5px;background:#18181b;color:#f4f4f5;padding:6px 8px;font-size:11px;font-weight:800;";
    const emptyOption = document.createElement("option");
    emptyOption.value = "";
    emptyOption.textContent = adjustPresetState.loading ? "Loading presets..." : "Choose preset...";
    presetSelect.append(emptyOption);
    for (const preset of adjustPresetState.presets) {
      const option = document.createElement("option");
      option.value = preset.name || "";
      option.textContent = adjustPresetLabel(preset);
      presetSelect.append(option);
    }
    presetSelect.value = adjustPresetState.selectedName || "";
    presetSelect.onchange = () => {
      adjustPresetState.selectedName = presetSelect.value;
    };
    const applyPreset = makeButton("Apply", "primary");
    applyPreset.disabled = !adjustPresetState.presets.length;
    applyPreset.style.opacity = applyPreset.disabled ? ".55" : "1";
    applyPreset.onclick = () => {
      const preset = adjustPresetState.presets.find((item) => item.name === presetSelect.value);
      if (!preset) {
        toast("Choose an Adjust preset first.", true);
        return;
      }
      applyAdjustPresetToSegment(preset, segment).catch((error) => {
        toast(error?.message || "Could not apply Adjust preset.", true);
      });
    };
    presetRow.append(presetSelect, applyPreset);
    const presetActions = document.createElement("div");
    presetActions.style.cssText = "display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;";
    const savePreset = makeButton("Save preset", "secondary");
    const importPreset = makeButton("Import preset", "secondary");
    const refreshPresets = makeButton("Refresh", "secondary");
    savePreset.onclick = () => saveCurrentAdjustPreset(segment);
    importPreset.onclick = () => importAdjustPresetFromText();
    refreshPresets.onclick = async () => {
      await loadAdjustPresets(true);
      renderSceneAdjustPanel();
      toast("Refreshed Adjust presets.");
    };
    presetActions.append(savePreset, importPreset, refreshPresets);
    presetBox.append(presetTitle, presetRow, presetActions);
    card.append(presetBox);

    let activeGroup = "";
    for (const field of SCENE_ADJUST_FIELDS) {
      if (field.group !== activeGroup) {
        activeGroup = field.group;
        const group = document.createElement("div");
        group.textContent = activeGroup;
        group.style.cssText = "margin-top:4px;border-top:1px solid #303038;padding-top:9px;color:#a5f3fc;font-size:11px;font-weight:900;";
        card.append(group);
      }
      card.append(makeSceneAdjustRow(segment, field, adjust));
    }

    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:2px;";
    const previewActions = document.createElement("div");
    previewActions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:2px;";
    const preview = makeButton("Preview Adjust", "primary");
    const stopPreview = makeButton("Stop Preview", "secondary");
    const renderAdjust = makeButton("Render Adjust for Scene", "primary");
    const applyAll = makeButton("Apply to all scenes", "primary");
    const reset = makeButton("Reset", "secondary");
    preview.disabled = !hasRenderableAdjust;
    preview.style.opacity = !hasRenderableAdjust ? ".55" : "1";
    renderAdjust.disabled = !hasRenderableAdjust;
    renderAdjust.style.opacity = !hasRenderableAdjust ? ".55" : "1";
    const canStopPreview = Boolean(
      String(segment.adjust_preview_image_path || "").trim()
      || state.adjustLivePreview
      || state.adjustLivePreviewPending
      || state.adjustLivePreviewBusy
    );
    stopPreview.disabled = !canStopPreview;
    stopPreview.style.opacity = stopPreview.disabled ? ".55" : "1";
    preview.onclick = () => previewAdjustForSegment(adjust, segment, preview);
    stopPreview.onclick = () => {
      state.adjustLivePreview = false;
      if (state.adjustLivePreviewTimer) clearTimeout(state.adjustLivePreviewTimer);
      state.adjustLivePreviewTimer = null;
      state.adjustLivePreviewPending = false;
      setAdjustLivePreviewStatus("Live preview off");
      clearSegmentAdjustPreview(segment, { refresh: true });
      toast("Stopped Adjust preview.");
    };
    renderAdjust.onclick = async () => {
      const current = activeSegment();
      if (!current) {
        toast("Select a scene before rendering Adjust.", true);
        return;
      }
      const currentAdjust = normalizeSceneAdjust(current.adjust || {}, { keepEmpty: true });
      if (!currentAdjust || !sceneAdjustHasRenderableChanges(currentAdjust)) {
        toast("Enable scene adjust and set at least one non-zero adjustment before rendering it.", true);
        return;
      }
      const sceneIndex = segmentIndexInfo(current).index;
      const videoPath = selectedSegmentVideoPath(current);
      if (!videoPath) {
        toast("This scene needs a video before rendering Adjust.", true);
        return;
      }
      const progress = createProgressWindow("Rendering Scene Adjust");
      try {
        pushHistory();
        clearSegmentAdjustPreview(current);
        renderAdjust.disabled = true;
        renderAdjust.textContent = "Rendering Adjust...";
        progress.set(`Applying Adjust to ${current.label || "scene"}...\nThis processes the video frame by frame.`, 12);
        const result = await applySceneAdjustToRenderedVideo(
          current,
          sceneIndex,
          videoPath,
          current.video_thumbnail_path || "",
          progress,
          (value) => value,
          "",
        );
        progress.set("Updating scene video preview...", 92);
        activateSegmentVideoPath(current, result.video_path || videoPath, result.thumbnail_path || current.video_thumbnail_path || "");
        current.video_cache_bust = Date.now();
        render();
        syncInspector();
        await autoSaveSessionQuiet("scene Adjust rendered");
        const applied = current.adjust_last_applied || {};
        const encoder = applied.encoder || result.encoder || "unknown";
        const browserFriendly = Boolean(applied.browser_friendly ?? result.browser_friendly);
        const sourceHadAudio = applied.source_had_audio ?? result.source_had_audio;
        const audioLine = sourceHadAudio === false
          ? "Source audio: none detected"
          : `Audio preserved in scene clip: ${applied.audio_preserved ? "yes" : "no"}`;
        progress.set(`Scene Adjust render complete.\n\nOutput:\n${current.video_path || result.video_path || ""}\n\nFrames processed: ${applied.processed_frames || "unknown"}\nEncoder: ${encoder}\nBrowser preview: ${browserFriendly ? "expected to work" : "fallback codec, may not preview in browser"}\n${audioLine}\n\nIf browser preview still fails, the scene stays rendered and the player will show the thumbnail fallback.`, 100);
        progress.close(2500);
        toast(`Rendered Adjust for ${current.label || "scene"}.`);
      } catch (error) {
        progress.set(`Adjust render failed:\n${error?.message || error}`, 100);
        toast(error?.message || "Could not render Adjust for this scene.", true);
        render();
        syncInspector();
      }
    };
    applyAll.onclick = () => {
      applySceneAdjustToAllScenes(segment).catch((error) => {
        toast(error?.message || "Could not apply Adjust to all scenes.", true);
      });
    };
    reset.onclick = () => resetSceneAdjust(segment);
    previewActions.append(preview, stopPreview);
    actions.append(applyAll, reset);
    card.append(previewActions, renderAdjust, actions);
    sceneAdjustPanel.append(card);
  }

  function applyFilmGrainToSegment(settings, segment = activeSegment()) {
    if (!segment) {
      toast("Select a scene first, then apply film grain.", true);
      return false;
    }
    if (segmentTrack(segment) === "overlay") {
      toast("Film grain can be applied to base scenes for now.", true);
      return false;
    }
    const next = normalizeSceneFilmGrain(settings, segment.film_grain || {});
    if (!next) {
      toast("Those film grain settings could not be applied.", true);
      return false;
    }
    pushHistory();
    clearSegmentLutPreview(segment);
    clearSegmentFilmGrainPreview(segment);
    restoreSegmentBeforeFilmGrain(segment);
    segment.film_grain = next;
    delete segment.film_grain_last_applied;
    delete segment.film_grain_rendered_path;
    delete segment.film_grain_rendered_thumbnail_path;
    state.activeId = segment.id;
    state.activeTrack = segmentTrack(segment);
    render();
    syncInspector();
    autoSaveSessionQuiet("scene film grain applied").catch(() => null);
    toast(`Applied film grain to ${segment.label || "scene"}.`);
    return true;
  }

  function removeFilmGrainFromSegment(segment = activeSegment()) {
    if (!segment?.film_grain) return;
    pushHistory();
    clearSegmentLutPreview(segment);
    clearSegmentFilmGrainPreview(segment);
    restoreSegmentBeforeFilmGrain(segment);
    delete segment.film_grain;
    delete segment.film_grain_last_applied;
    delete segment.film_grain_rendered_path;
    delete segment.film_grain_rendered_thumbnail_path;
    render();
    syncInspector();
    autoSaveSessionQuiet("scene film grain removed").catch(() => null);
    toast(`Removed film grain from ${segment.label || "scene"}.`);
  }

  function applySceneFilmGrainToAllScenes(segment = activeSegment()) {
    const grain = normalizeSceneFilmGrain(segment?.film_grain || {});
    if (!grain) {
      toast("Apply film grain to this scene first.", true);
      return;
    }
    const targets = allEditableSegments();
    if (!targets.length) return;
    pushHistory();
    for (const target of targets) {
      clearSegmentLutPreview(target);
      clearSegmentFilmGrainPreview(target);
      restoreSegmentBeforeFilmGrain(target);
      target.film_grain = { ...grain };
      delete target.film_grain_last_applied;
      delete target.film_grain_rendered_path;
      delete target.film_grain_rendered_thumbnail_path;
    }
    render();
    syncInspector();
    autoSaveSessionQuiet("scene film grain applied to all").catch(() => null);
    toast(`Applied film grain to ${targets.length} scene${targets.length === 1 ? "" : "s"}.`);
  }

  function clearSegmentFilmGrainPreview(segment = activeSegment(), options = {}) {
    if (!segment) return;
    const path = String(segment.film_grain_preview_image_path || "").trim();
    if (options.deleteFile !== false) deletePostProcessSourcePreview(segment, "film_grain");
    delete segment.film_grain_preview_image_path;
    delete segment.film_grain_preview_source_path;
    delete segment.film_grain_preview_source_preview_path;
    delete segment.film_grain_preview_source_temporary;
    delete segment.film_grain_preview_settings;
    if (path && options.deleteFile !== false) deletePostProcessPreviewFile(path);
    if (options.refresh) {
      if (selectedSegmentVideoPath(segment)) segment.preview_mode = "video";
      else segment.preview_mode = "image";
      syncPreview(segment);
      render();
      syncInspector();
    }
  }

  async function previewFilmGrainForSegment(settings, segment = activeSegment(), button = null) {
    const current = segment || activeSegment();
    if (!current) {
      toast("Select a scene first, then preview film grain.", true);
      return;
    }
    const grain = normalizeSceneFilmGrain(settings, current.film_grain || {}) || defaultFilmGrainSettings();
    if (grain.enabled === false) {
      toast("Enable film grain before previewing it.", true);
      return;
    }
    const sceneIndex = segmentIndexInfo(current).index;
    const videoPath = String(current.video_original_path || selectedSegmentVideoPath(current) || "").trim();
    const imagePath = selectedSegmentImagePath(current);
    const mediaPath = videoPath || imagePath;
    const mediaType = videoPath ? "video" : "image";
    if (!mediaPath) {
      toast("This scene needs an image or video before previewing film grain.", true);
      return;
    }
    const projectFolder = String(projectInput.value || state.projectFolder || "").trim();
    if (!projectFolder) {
      toast("Set a project folder before previewing film grain.", true);
      return;
    }
    const previousText = button?.textContent || "";
    try {
      if (button) {
        button.disabled = true;
        button.textContent = "Previewing...";
      }
      clearSegmentLutPreview(current);
      clearSegmentAdjustPreview(current);
      clearSegmentFilmGrainPreview(current);
      const data = await postJson("/vrgdg/music_builder/post_process/film_grain/preview", {
        input_path: mediaPath,
        media_type: mediaType,
        grain_intensity: grain.grain_intensity,
        saturation_mix: grain.saturation_mix,
        device: "auto",
        scene_id: current.id || `scene_${sceneIndex}`,
        project_folder: projectFolder,
      }, 120000);
      current.film_grain_preview_image_path = data.preview_path || data.output || "";
      current.film_grain_preview_source_path = mediaPath;
      current.film_grain_preview_source_preview_path = data.source_preview_path || mediaPath;
      current.film_grain_preview_source_temporary = Boolean(data.source_preview_temporary);
      current.film_grain_preview_settings = { ...grain, created_at: Date.now() };
      current.preview_mode = "image";
      render();
      syncInspector();
      showFilmGrainPreviewImage(current);
      toast(`Previewing film grain on ${current.label || "scene"}.`);
    } catch (error) {
      toast(error?.message || "Could not preview film grain for this scene.", true);
      render();
      syncInspector();
    } finally {
      if (button) {
        button.disabled = false;
        button.textContent = previousText;
      }
    }
  }

  function renderFilmGrainPostProcessPanel() {
    filmGrainPane.textContent = "";
    const tile = document.createElement("button");
    tile.type = "button";
    tile.draggable = true;
    tile.title = "Click to apply Film Grain to the selected scene, or drag it onto the scene you want.";
    tile.style.cssText = "width:100%;text-align:left;border:1px solid #fbbf24;border-radius:7px;background:#18181b;color:#f4f4f5;padding:12px;cursor:grab;display:flex;flex-direction:column;gap:6px;";
    tile.innerHTML = `
      <div style="display:flex;align-items:center;justify-content:space-between;gap:8px;">
        <span style="font-size:13px;font-weight:900;color:#fde68a;">Film Grain</span>
        <span style="border:1px solid #fbbf24;border-radius:4px;padding:2px 5px;font-size:10px;font-weight:900;color:#fde68a;background:rgba(113,63,18,.42);">FX</span>
      </div>
      <div style="font-size:11px;line-height:1.35;color:#a1a1aa;">Click to apply to the selected scene, or drag onto the scene you want. Settings appear in Scene Tools.</div>
    `;
    tile.ondragstart = (event) => {
      event.dataTransfer?.setData("application/x-vrgdg-post-effect", "film_grain");
      event.dataTransfer?.setData("text/plain", "Film Grain");
      if (event.dataTransfer) event.dataTransfer.effectAllowed = "copy";
    };
    tile.onclick = () => {
      const segment = activeSegment();
      if (!segment) {
        toast("Select a scene or drag Film Grain onto one.", true);
        return;
      }
      applyFilmGrainToSegment(defaultFilmGrainSettings(), segment);
    };
    filmGrainPane.append(tile);
  }

  function renderFxPostProcessPanel() {
    fxOverlaysPane.textContent = "";
    const styleFxSubTab = (button, selected) => {
      button.style.background = selected ? "#0e7490" : "#18181b";
      button.style.borderColor = selected ? "#22d3ee" : "#3f3f46";
      button.style.color = selected ? "#ecfeff" : "#fafafa";
      button.style.fontWeight = "900";
    };
    styleFxSubTab(fxOverlaysTab, true);

    const intro = document.createElement("div");
    intro.style.cssText = "border:1px solid #3f3f46;border-radius:7px;background:#18181b;color:#f4f4f5;padding:12px;display:flex;flex-direction:column;gap:7px;";
    intro.innerHTML = `
      <div style="display:flex;align-items:center;justify-content:space-between;gap:8px;">
        <span style="font-size:13px;font-weight:900;color:#cffafe;">Overlay FX</span>
        <span style="border:1px solid #22d3ee;border-radius:4px;padding:2px 5px;font-size:10px;font-weight:900;color:#cffafe;background:rgba(14,116,144,.35);">NEW</span>
      </div>
      <div style="font-size:11px;line-height:1.35;color:#a1a1aa;">Overlay packs will live here, starting with Photoshop-generated light leaks. Next step is loading the pack thumbnails and applying them to scenes.</div>
      <div style="font-size:10px;line-height:1.35;color:#71717a;word-break:break-word;">Test pack: scripts/photoshop/output/LightLeak_WarmDream</div>
    `;
    fxOverlaysPane.append(intro);
  }

  function lutExampleUrl(lutName = "") {
    const lut = (lutsTools.state.luts || []).find((item) => item.name === lutName);
    return lut?.example_url || "";
  }

  function deletePostProcessPreviewFile(path) {
    const cleanPath = String(path || "").trim();
    if (!cleanPath) return;
    postJson("/vrgdg/music_builder/luts/delete_preview", {
      path: cleanPath,
      project_folder: projectInput.value || state.projectFolder || "",
    }).catch(() => null);
  }

  function deletePostProcessSourcePreview(segment, prefix) {
    const path = String(segment?.[`${prefix}_preview_source_preview_path`] || "").trim();
    if (path && segment?.[`${prefix}_preview_source_temporary`]) deletePostProcessPreviewFile(path);
  }

  function clearSegmentLutPreview(segment = activeSegment(), options = {}) {
    if (!segment) return;
    const path = String(segment.lut_preview_image_path || "").trim();
    if (options.deleteFile !== false) deletePostProcessSourcePreview(segment, "lut");
    delete segment.lut_preview_image_path;
    delete segment.lut_preview_source_path;
    delete segment.lut_preview_source_preview_path;
    delete segment.lut_preview_source_temporary;
    delete segment.lut_preview_lut;
    if (path && options.deleteFile !== false) deletePostProcessPreviewFile(path);
    if (options.refresh) {
      if (selectedSegmentVideoPath(segment)) segment.preview_mode = "video";
      else segment.preview_mode = "image";
      syncPreview(segment);
      render();
      syncInspector();
    }
  }

  function removeLutFromSegment(segment = activeSegment()) {
    if (!segment?.lut) return;
    pushHistory();
    clearSegmentLutPreview(segment);
    delete segment.lut;
    delete segment.lut_last_applied;
    delete segment.lut_rendered_path;
    delete segment.lut_rendered_thumbnail_path;
    if (String(segment.video_original_path || "").trim()) {
      segment.video_path = segment.video_original_path;
      if (String(segment.video_original_thumbnail_path || "").trim()) {
        segment.video_thumbnail_path = segment.video_original_thumbnail_path;
      }
      normalizeSegmentVideoHistory(segment);
      const originalIndex = segment.video_history.findIndex((item) => mediaPathKey(item) === mediaPathKey(segment.video_original_path));
      if (originalIndex >= 0) segment.video_history_index = originalIndex;
      segment.video_thumbnail_path = segment.video_thumbnail_history[segment.video_history_index] || segment.video_thumbnail_path || "";
      segment.video_cache_bust = Date.now();
    }
    render();
    syncInspector();
    autoSaveSessionQuiet("scene LUT removed").catch(() => null);
    toast(`Removed LUT from ${segment.label || "scene"}.`);
  }

  function applySceneLutToAllScenes(segment = activeSegment()) {
    const lut = normalizeSceneLut(segment?.lut || {});
    if (!lut) {
      toast("Apply a LUT to this scene first.", true);
      return;
    }
    const targets = allEditableSegments();
    if (!targets.length) return;
    pushHistory();
    for (const target of targets) {
      clearSegmentLutPreview(target);
      target.lut = { ...lut };
      delete target.lut_last_applied;
      delete target.lut_rendered_path;
      delete target.lut_rendered_thumbnail_path;
      if (String(target.video_original_path || "").trim()) {
        activateSegmentVideoPath(target, target.video_original_path, target.video_original_thumbnail_path || "");
        target.video_cache_bust = Date.now();
      }
    }
    render();
    syncInspector();
    autoSaveSessionQuiet("scene LUT applied to all").catch(() => null);
    toast(`Applied LUT to ${targets.length} scene${targets.length === 1 ? "" : "s"}.`);
  }

  function appendFilmGrainSceneToolCard(segment) {
    const grain = normalizeSceneFilmGrain(segment?.film_grain || {});
    if (!grain) return;
    const card = document.createElement("div");
    card.style.cssText = "display:flex;flex-direction:column;gap:9px;border:1px solid #3f3f46;border-radius:7px;background:#18181b;padding:9px;";
    const header = document.createElement("div");
    header.innerHTML = `<div style="font-size:12px;font-weight:900;color:#fde68a;">Film Grain</div><div style="font-size:11px;line-height:1.35;color:#e4e4e7;margin-top:4px;">${escapeHtml(filmGrainLabel(grain))}</div>`;
    const enabled = makeCheckbox("Enable film grain", grain.enabled !== false);
    enabled.input.onchange = () => {
      const current = activeSegment();
      if (!current?.film_grain) return;
      pushHistory();
      restoreSegmentBeforeFilmGrain(current);
      clearSegmentAdjustPreview(current);
      clearSegmentFilmGrainPreview(current);
      current.film_grain = normalizeSceneFilmGrain({ ...current.film_grain, enabled: enabled.input.checked }, current.film_grain);
      delete current.film_grain_last_applied;
      delete current.film_grain_rendered_path;
      delete current.film_grain_rendered_thumbnail_path;
      if (String(current.video_original_path || "").trim()) {
        activateSegmentVideoPath(current, current.video_original_path, current.video_original_thumbnail_path || "");
        current.video_cache_bust = Date.now();
      }
      render();
      syncInspector();
      autoSaveSessionQuiet("scene film grain enabled").catch(() => null);
    };
    const intensity = makeFilmGrainRange("Intensity", grain.grain_intensity, 0, 0.25, 0.005, 3);
    intensity.input.onchange = () => {
      const current = activeSegment();
      if (!current?.film_grain) return;
      pushHistory();
      restoreSegmentBeforeFilmGrain(current);
      current.film_grain = normalizeSceneFilmGrain({ ...current.film_grain, grain_intensity: Number(intensity.input.value || 0) }, current.film_grain);
      delete current.film_grain_last_applied;
      delete current.film_grain_rendered_path;
      delete current.film_grain_rendered_thumbnail_path;
      render();
      syncInspector();
      autoSaveSessionQuiet("scene film grain intensity").catch(() => null);
    };
    const saturation = makeFilmGrainRange("Color grain mix", grain.saturation_mix, 0, 1, 0.01, 2);
    saturation.input.onchange = () => {
      const current = activeSegment();
      if (!current?.film_grain) return;
      pushHistory();
      restoreSegmentBeforeFilmGrain(current);
      current.film_grain = normalizeSceneFilmGrain({ ...current.film_grain, saturation_mix: Number(saturation.input.value || 0) }, current.film_grain);
      delete current.film_grain_last_applied;
      delete current.film_grain_rendered_path;
      delete current.film_grain_rendered_thumbnail_path;
      render();
      syncInspector();
      autoSaveSessionQuiet("scene film grain color mix").catch(() => null);
    };
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
    const previewActions = document.createElement("div");
    previewActions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
    const preview = makeButton("Preview Grain", "primary");
    const stopPreview = makeButton("Stop Preview", "secondary");
    const grainEnabled = grain.enabled !== false;
    preview.disabled = !grainEnabled;
    preview.style.opacity = grainEnabled ? "1" : ".55";
    stopPreview.disabled = !String(segment.film_grain_preview_image_path || "").trim();
    stopPreview.style.opacity = stopPreview.disabled ? ".55" : "1";
    const renderGrain = makeButton("Render Grain for Scene", "primary");
    renderGrain.disabled = !grainEnabled;
    renderGrain.style.opacity = grainEnabled ? "1" : ".55";
    const applyAll = makeButton("Apply to all scenes", "primary");
    const remove = makeButton("Remove Grain", "danger");
    preview.onclick = () => previewFilmGrainForSegment(grain, segment, preview);
    stopPreview.onclick = () => {
      clearSegmentFilmGrainPreview(segment, { refresh: true });
      toast("Stopped film grain preview.");
    };
    renderGrain.onclick = async () => {
      const current = activeSegment();
      if (!current) {
        toast("Select a scene before rendering film grain.", true);
        return;
      }
      const currentGrain = normalizeSceneFilmGrain(current.film_grain || {});
      if (!currentGrain || currentGrain.enabled === false) {
        toast("Enable film grain before rendering it.", true);
        return;
      }
      const sceneIndex = segmentIndexInfo(current).index;
      const videoPath = selectedSegmentVideoPath(current);
      if (!videoPath) {
        toast("This scene needs a video before rendering film grain.", true);
        return;
      }
      const progress = createProgressWindow("Rendering Scene Film Grain");
      try {
        pushHistory();
        clearSegmentAdjustPreview(current);
        clearSegmentFilmGrainPreview(current);
        renderGrain.disabled = true;
        renderGrain.textContent = "Rendering Grain...";
        progress.set(`Applying film grain to ${current.label || "scene"}...\nThis processes the video frame by frame.`, 12);
        const result = await applySceneFilmGrainToRenderedVideo(
          current,
          sceneIndex,
          videoPath,
          current.video_thumbnail_path || "",
          progress,
          (value) => value,
          "",
        );
        progress.set("Updating scene video preview...", 92);
        activateSegmentVideoPath(current, result.video_path || videoPath, result.thumbnail_path || current.video_thumbnail_path || "");
        current.video_cache_bust = Date.now();
        render();
        syncInspector();
        await autoSaveSessionQuiet("scene film grain rendered");
        const applied = current.film_grain_last_applied || {};
        const encoder = applied.encoder || result.encoder || "unknown";
        const browserFriendly = Boolean(applied.browser_friendly ?? result.browser_friendly);
        const sourceHadAudio = applied.source_had_audio ?? result.source_had_audio;
        const audioLine = sourceHadAudio === false
          ? "Source audio: none detected"
          : `Audio preserved in scene clip: ${applied.audio_preserved ? "yes" : "no"}`;
        progress.set(`Scene film grain render complete.\n\nOutput:\n${current.video_path || result.video_path || ""}\n\nFrames processed: ${applied.processed_frames || "unknown"}\nEncoder: ${encoder}\nBrowser preview: ${browserFriendly ? "expected to work" : "fallback codec, may not preview in browser"}\n${audioLine}\n\nIf browser preview still fails, the scene stays rendered and the player will show the thumbnail fallback.`, 100);
        toast(`Rendered film grain for ${current.label || "scene"}.`);
      } catch (error) {
        progress.set(`Film grain render failed:\n${error?.message || error}`, 100);
        toast(error?.message || "Could not render film grain for this scene.", true);
        render();
        syncInspector();
      }
    };
    applyAll.onclick = () => applySceneFilmGrainToAllScenes(segment);
    remove.onclick = () => removeFilmGrainFromSegment(segment);
    previewActions.append(preview, stopPreview);
    actions.append(applyAll, remove);
    card.append(header, enabled.wrapper, intensity.wrapper, saturation.wrapper, previewActions, renderGrain, actions);
    sceneToolsPanel.append(card);
  }

  function renderSceneToolsPanel() {
    if (!sceneToolsPanel) return;
    sceneToolsPanel.textContent = "";
    const segment = activeSegment();
    if (!segment) return;
    const lut = normalizeSceneLut(segment.lut || {});
    const grain = normalizeSceneFilmGrain(segment.film_grain || {});
    if (!lut && !grain) {
      const empty = document.createElement("div");
      empty.textContent = "No tools added to scene.";
      empty.style.cssText = "border:1px dashed #3f3f46;border-radius:7px;background:#18181b;color:#a1a1aa;padding:16px;text-align:center;font-size:12px;font-weight:800;";
      sceneToolsPanel.append(empty);
      return;
    }
    if (!lut) {
      appendFilmGrainSceneToolCard(segment);
      return;
    }

    const card = document.createElement("div");
    card.style.cssText = "display:flex;flex-direction:column;gap:9px;border:1px solid #3f3f46;border-radius:7px;background:#18181b;padding:9px;";
    const header = document.createElement("div");
    header.style.cssText = "display:grid;grid-template-columns:96px minmax(0,1fr);gap:9px;align-items:center;";
    const previewWrap = document.createElement("div");
    previewWrap.style.cssText = "width:96px;aspect-ratio:16/10;border:1px solid #3f3f46;border-radius:5px;background:#09090b;overflow:hidden;";
    const exampleUrl = lutExampleUrl(lut.name);
    if (exampleUrl) {
      const image = document.createElement("img");
      image.src = exampleUrl;
      image.alt = "";
      image.style.cssText = "width:100%;height:100%;object-fit:cover;display:block;";
      previewWrap.append(image);
    }
    const text = document.createElement("div");
    text.innerHTML = `<div style="font-size:12px;font-weight:900;color:#f5d0fe;">LUT</div><div style="font-size:12px;line-height:1.25;color:#e4e4e7;overflow-wrap:anywhere;margin-top:3px;">${escapeHtml(lut.label || lutLabelFromName(lut.name))}</div><div style="font-size:10px;color:#a1a1aa;margin-top:5px;overflow-wrap:anywhere;">${escapeHtml(lut.name)}</div>`;
    header.append(previewWrap, text);

    const enabled = makeCheckbox("Enable LUT", lut.enabled !== false);
    enabled.input.onchange = () => {
      const current = activeSegment();
      if (!current?.lut) return;
      pushHistory();
      clearSegmentLutPreview(current);
      clearSegmentAdjustPreview(current);
      clearSegmentFilmGrainPreview(current);
      current.lut = normalizeSceneLut({ ...current.lut, enabled: enabled.input.checked }, current.lut);
      delete current.lut_last_applied;
      delete current.lut_rendered_path;
      delete current.lut_rendered_thumbnail_path;
      if (String(current.video_original_path || "").trim()) {
        activateSegmentVideoPath(current, current.video_original_path, current.video_original_thumbnail_path || "");
        current.video_cache_bust = Date.now();
      }
      render();
      syncInspector();
      autoSaveSessionQuiet("scene LUT enabled").catch(() => null);
    };

    const strength = document.createElement("input");
    strength.type = "range";
    strength.min = "0";
    strength.max = "10";
    strength.step = "0.1";
    strength.value = String(lut.strength ?? 10);
    const strengthValue = document.createElement("div");
    strengthValue.textContent = `Strength: ${Number(strength.value).toFixed(1)}`;
    strengthValue.style.cssText = "font-size:11px;color:#c4b5fd;font-weight:900;";
    strength.oninput = () => {
      strengthValue.textContent = `Strength: ${Number(strength.value).toFixed(1)}`;
    };
    strength.onchange = () => {
      const current = activeSegment();
      if (!current?.lut) return;
      pushHistory();
      clearSegmentLutPreview(current);
      clearSegmentAdjustPreview(current);
      current.lut = normalizeSceneLut({ ...current.lut, strength: Number(strength.value) }, current.lut);
      delete current.lut_last_applied;
      delete current.lut_rendered_path;
      delete current.lut_rendered_thumbnail_path;
      if (String(current.video_original_path || "").trim()) {
        activateSegmentVideoPath(current, current.video_original_path, current.video_original_thumbnail_path || "");
        current.video_cache_bust = Date.now();
      }
      render();
      syncInspector();
      autoSaveSessionQuiet("scene LUT strength").catch(() => null);
    };

    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
    const previewActions = document.createElement("div");
    previewActions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
    const previewLut = makeButton("Preview LUT", "primary");
    const stopPreview = makeButton("Stop Preview", "secondary");
    const lutEnabled = lut.enabled !== false;
    previewLut.disabled = !lutEnabled;
    previewLut.style.opacity = lutEnabled ? "1" : ".55";
    stopPreview.disabled = !String(segment.lut_preview_image_path || "").trim();
    stopPreview.style.opacity = stopPreview.disabled ? ".55" : "1";
    const renderLut = makeButton("Render LUT for Scene", "primary");
    renderLut.disabled = !lutEnabled;
    renderLut.style.opacity = lutEnabled ? "1" : ".55";
    const applyAll = makeButton("Apply to all scenes", "primary");
    const remove = makeButton("Remove LUT", "danger");
    previewLut.onclick = async () => {
      const current = activeSegment();
      if (!current?.lut) return;
      const currentLut = normalizeSceneLut(current.lut || {});
      if (!currentLut || currentLut.enabled === false) {
        toast("Enable the LUT before previewing it.", true);
        return;
      }
      const sceneIndex = segmentIndexInfo(current).index;
      const videoPath = String(current.video_original_path || selectedSegmentVideoPath(current) || "").trim();
      const imagePath = selectedSegmentImagePath(current);
      const mediaPath = videoPath || imagePath;
      const mediaType = videoPath ? "video" : "image";
      if (!mediaPath) {
        toast("This scene needs an image or video before previewing a LUT.", true);
        return;
      }
      const projectFolder = String(projectInput.value || state.projectFolder || "").trim();
      if (!projectFolder) {
        toast("Set a project folder before previewing LUTs.", true);
        return;
      }
      try {
        previewLut.disabled = true;
        previewLut.textContent = "Previewing...";
        clearSegmentLutPreview(current);
        clearSegmentAdjustPreview(current);
        clearSegmentFilmGrainPreview(current);
        const data = await postJson("/vrgdg/music_builder/luts/preview", {
          input_path: mediaPath,
          media_type: mediaType,
          lut_name: lut.name,
          strength: lut.strength,
          device: "auto",
          scene_id: current.id || `scene_${sceneIndex}`,
          project_folder: projectFolder,
        }, 120000);
        current.lut_preview_image_path = data.preview_path || data.output || "";
        current.lut_preview_source_path = mediaPath;
        current.lut_preview_source_preview_path = data.source_preview_path || mediaPath;
        current.lut_preview_source_temporary = Boolean(data.source_preview_temporary);
        current.lut_preview_lut = { name: lut.name, strength: lut.strength, created_at: Date.now() };
        current.preview_mode = "image";
        render();
        syncInspector();
        showLutPreviewImage(current);
        toast(`Previewing LUT on ${current.label || "scene"}.`);
      } catch (error) {
        toast(error?.message || "Could not preview LUT for this scene.", true);
        render();
        syncInspector();
      }
    };
    stopPreview.onclick = () => {
      const current = activeSegment();
      clearSegmentLutPreview(current, { refresh: true });
      toast("Stopped LUT preview.");
    };
    renderLut.onclick = async () => {
      const current = activeSegment();
      if (!current) {
        toast("Select a scene before rendering a LUT.", true);
        return;
      }
      const currentLut = normalizeSceneLut(current.lut || {});
      if (!currentLut || currentLut.enabled === false) {
        toast("Enable the LUT before rendering it.", true);
        return;
      }
      const sceneIndex = segmentIndexInfo(current).index;
      const videoPath = selectedSegmentVideoPath(current);
      if (!videoPath) {
        toast("This scene needs a video before rendering a LUT.", true);
        return;
      }
      const progress = createProgressWindow("Rendering Scene LUT");
      try {
        pushHistory();
        clearSegmentLutPreview(current);
        clearSegmentAdjustPreview(current);
        renderLut.disabled = true;
        renderLut.textContent = "Rendering LUT...";
        progress.set(`Applying LUT to ${current.label || "scene"}...\nThis can take a bit because it processes the video frame by frame.`, 12);
        const result = await applySceneLutToRenderedVideo(
          current,
          sceneIndex,
          videoPath,
          current.video_thumbnail_path || "",
          progress,
          (value) => value,
          "",
        );
        progress.set("Updating scene video preview...", 92);
        activateSegmentVideoPath(current, result.video_path || videoPath, result.thumbnail_path || current.video_thumbnail_path || "");
        current.video_cache_bust = Date.now();
        render();
        syncInspector();
        await autoSaveSessionQuiet("scene LUT rendered");
        const applied = current.lut_last_applied || {};
        const encoder = applied.encoder || result.encoder || "unknown";
        const browserFriendly = Boolean(applied.browser_friendly ?? result.browser_friendly);
        const sourceHadAudio = applied.source_had_audio ?? result.source_had_audio;
        const audioLine = sourceHadAudio === false
          ? "Source audio: none detected"
          : `Audio preserved in scene clip: ${applied.audio_preserved ? "yes" : "no"}`;
        progress.set(`Scene LUT render complete.\n\nOutput:\n${current.video_path || result.video_path || ""}\n\nFrames processed: ${applied.processed_frames || "unknown"}\nEncoder: ${encoder}\nBrowser preview: ${browserFriendly ? "expected to work" : "fallback codec, may not preview in browser"}\n${audioLine}\n\nIf browser preview still fails, the scene stays rendered and the player will show the thumbnail fallback.`, 100);
        toast(`Rendered LUT for ${current.label || "scene"}.`);
      } catch (error) {
        progress.set(`LUT render failed:\n${error?.message || error}`, 100);
        toast(error?.message || "Could not render LUT for this scene.", true);
        render();
        syncInspector();
      }
    };
    previewActions.append(previewLut, stopPreview);
    applyAll.onclick = () => applySceneLutToAllScenes(segment);
    remove.onclick = () => removeLutFromSegment(segment);
    actions.append(applyAll, remove);

    card.append(header, enabled.wrapper, strengthValue, strength, previewActions, renderLut, actions);
    sceneToolsPanel.append(card);
    appendFilmGrainSceneToolCard(segment);
  }

  function enableLutDrop(element, segment) {
    if (!element || !segment || segmentTrack(segment) === "overlay") return;
    element.addEventListener("dragover", (event) => {
      const types = Array.from(event.dataTransfer?.types || []).map((item) => String(item).toLowerCase());
      if (!types.includes("application/x-vrgdg-lut-name")) return;
      event.preventDefault();
      event.stopPropagation();
      if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
      element.style.outline = "2px solid #f0abfc";
    });
    element.addEventListener("dragleave", (event) => {
      event.stopPropagation();
      element.style.outline = "";
    });
    element.addEventListener("drop", (event) => {
      const lutName = event.dataTransfer?.getData("application/x-vrgdg-lut-name") || "";
      if (!lutName) return;
      event.preventDefault();
      event.stopPropagation();
      element.style.outline = "";
      applyLutNameToSegment(lutName, segment);
    });
  }

  function enablePostEffectDrop(element, segment) {
    if (!element || !segment || segmentTrack(segment) === "overlay") return;
    element.addEventListener("dragover", (event) => {
      const types = Array.from(event.dataTransfer?.types || []).map((item) => String(item).toLowerCase());
      if (!types.includes("application/x-vrgdg-post-effect")) return;
      event.preventDefault();
      event.stopPropagation();
      if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
      element.style.outline = "2px solid #fbbf24";
    });
    element.addEventListener("dragleave", (event) => {
      event.stopPropagation();
      element.style.outline = "";
    });
    element.addEventListener("drop", (event) => {
      const effectName = event.dataTransfer?.getData("application/x-vrgdg-post-effect") || "";
      if (!effectName) return;
      event.preventDefault();
      event.stopPropagation();
      element.style.outline = "";
      if (effectName === "film_grain") {
        applyFilmGrainToSegment(defaultFilmGrainSettings(), segment);
      }
    });
  }

  return {
    applyLutToSegment, clearSegmentAdjustPreview, clearSegmentFilmGrainPreview, clearSegmentLutPreview,
    enableLutDrop, enablePostEffectDrop, normalizeSceneAdjust, renderFilmGrainPostProcessPanel,
    renderSceneAdjustPanel, renderSceneToolsPanel, sceneAdjustHasRenderableChanges, sceneAdjustSignature,
    syncLeftPanelTabs, syncPostProcessTabs,
  };
}

export function buildPostProcessPane({ lutsTools }) {
  const postProcessPane = document.createElement("div");
  postProcessPane.style.cssText = "display:none;flex-direction:column;overflow:hidden;min-height:0;flex:1 1 auto;";
  postProcessPane.className = "vrgdg-builder-post-process-pane";
  const postProcessTabBar = document.createElement("div");
  postProcessTabBar.style.cssText = "display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px;padding:10px 10px 0;background:#202024;flex:0 0 auto;";
  const postProcessLutsTab = makeButton("LUTS");
  const postProcessGrainTab = makeButton("Film Grain");
  const postProcessFxTab = makeButton("FX");
  postProcessLutsTab.title = "Browse LUT looks and apply them to the selected scene.";
  postProcessGrainTab.title = "Apply film grain settings to the selected scene.";
  postProcessFxTab.title = "Browse overlay and visual FX packs.";
  const filmGrainPane = document.createElement("div");
  filmGrainPane.style.cssText = "display:none;overflow:auto;padding:10px;min-height:0;flex:1 1 auto;flex-direction:column;gap:10px;";
  const fxPane = document.createElement("div");
  fxPane.style.cssText = "display:none;overflow:hidden;min-height:0;flex:1 1 auto;flex-direction:column;";
  const fxSubTabBar = document.createElement("div");
  fxSubTabBar.style.cssText = "display:grid;grid-template-columns:1fr;gap:6px;padding:10px 10px 0;background:#202024;flex:0 0 auto;";
  const fxOverlaysTab = makeButton("Overlays");
  fxOverlaysTab.title = "Animated overlay FX packs like light leaks, dust, fog, and sparks.";
  const fxOverlaysPane = document.createElement("div");
  fxOverlaysPane.style.cssText = "display:flex;overflow:auto;padding:10px;min-height:0;flex:1 1 auto;flex-direction:column;gap:10px;";
  fxSubTabBar.append(fxOverlaysTab);
  fxPane.append(fxSubTabBar, fxOverlaysPane);
  postProcessTabBar.append(postProcessLutsTab, postProcessGrainTab, postProcessFxTab);
  postProcessPane.append(postProcessTabBar, lutsTools.element, filmGrainPane, fxPane);

  return {
    filmGrainPane, fxOverlaysPane, fxOverlaysTab, fxPane, postProcessFxTab, postProcessGrainTab,
    postProcessLutsTab, postProcessPane,
  };
}
