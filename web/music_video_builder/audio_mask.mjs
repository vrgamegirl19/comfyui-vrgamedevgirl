import {
  STEM_COLORS, STEM_LABELS, STEM_ORDER, audioMaskInUse, clearStemLanes, normalizeAudioMaskAuto, onOpenAudioMaskRequest, onStemEditRequest, refreshStemLanes,
  setStemLanes,
} from "./audio_mask_store.mjs";
import { audioUrl, postJson } from "./comfy_api.mjs";
import { makeButton, makeCheckbox, makeInput, makeSelect, toast } from "./controls.mjs";
import { audioSourceStart, timelineSegmentDuration } from "./timeline_state.mjs";
import { prepareEditedAudio, speakingAudioEditsActive } from "./audio_clip_editor.mjs";

// Audio Mask. A scene's audio is split into stems (vocals, drums, bass, other, and with the 6 stem model also guitar and
// piano). For every stem the scene keeps: a mask switch (only its regions are audible), the regions, a level and mute. The
// masked mix, the stems added together, is what the video model is rendered with. The finished clip gets the real audio
// back. The settings live on the scene as segment.audio_mask. The timeline draws one track per stem and edits the regions;
// this window holds the split, the per stem switches and exact numbers, the build, the preview and the render switch.

const MODELS = [
  { value: "htdemucs", label: "htdemucs — 4 stems (vocals, drums, bass, other)" },
  { value: "htdemucs_ft", label: "htdemucs_ft — 4 stems, slower, often cleaner" },
  { value: "mdx_extra", label: "mdx_extra — 4 stems, different tuning" },
  { value: "htdemucs_6s", label: "htdemucs_6s — 6 stems, adds guitar and piano" },
];
const FOUR_STEMS = ["vocals", "drums", "bass", "other"];
const STEMS_BY_MODEL = { htdemucs: FOUR_STEMS, htdemucs_ft: FOUR_STEMS, mdx_extra: FOUR_STEMS, htdemucs_6s: STEM_ORDER };
const MIN_REGION_SECONDS = 0.02;
const DEFAULT_FADE_MS = 30;

const round3 = (value) => Math.round(Number(value) * 1000) / 1000;
const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
const finite = (value, fallback) => (Number.isFinite(Number(value)) && value !== "" && value !== null ? Number(value) : fallback);

function cleanRegions(list) {
  return (Array.isArray(list) ? list : [])
    .map((item) => ({ start: round3(finite(item?.start, 0)), end: round3(finite(item?.end, 0)), fade_ms: clamp(finite(item?.fade_ms, DEFAULT_FADE_MS), 0, 1000) }))
    .filter((item) => item.end - item.start >= MIN_REGION_SECONDS);
}

// Sort and join regions that touch or overlap.
export function mergeRegions(regions) {
  const sorted = cleanRegions(regions).sort((a, b) => a.start - b.start);
  const merged = [];
  for (const item of sorted) {
    const last = merged[merged.length - 1];
    if (last && item.start <= last.end) {
      last.end = Math.max(last.end, item.end);
      last.fade_ms = Math.max(last.fade_ms, item.fade_ms);
    } else {
      merged.push(item);
    }
  }
  return merged;
}

// Nothing is masked or muted until the user does it.
function defaultStem() {
  return { mask: false, regions: [], db: 0, mute: false };
}

// Version 2 settings. Masks saved before this turned the vocal mask on with no regions as soon as a scene was split, which
// silenced every vocal by default.
const MASK_VERSION = 2;

const isDefaultStem = (item) => !item.mask && !item.regions.length && !item.db && !item.mute;

// The scene's settings with every field present. Everything is on until the user turns something off. A mask saved in the
// first format (regions, vocal_db, bed_db) is converted, and a mask that only has the leftover "vocals masked, no regions"
// default from before version 2 gets its vocals back.
function maskSettings(segment) {
  const saved = segment?.audio_mask && typeof segment.audio_mask === "object" ? segment.audio_mask : {};
  const savedStems = saved.stems && typeof saved.stems === "object" ? saved.stems : null;
  const firstFormat = !savedStems && (Array.isArray(saved.regions) || "vocal_db" in saved || "bed_db" in saved);
  const stems = {};
  for (const name of STEM_ORDER) {
    const item = savedStems?.[name];
    if (item && typeof item === "object") {
      stems[name] = { mask: Boolean(item.mask), regions: mergeRegions(item.regions), db: clamp(finite(item.db, 0), -100, 24), mute: Boolean(item.mute) };
    } else if (firstFormat && name === "vocals") {
      stems[name] = { mask: true, regions: mergeRegions(saved.regions), db: clamp(finite(saved.vocal_db, 0), -100, 24), mute: false };
    } else if (firstFormat) {
      stems[name] = { mask: false, regions: [], db: clamp(finite(saved.bed_db, 0), -100, 24), mute: false };
    } else {
      stems[name] = defaultStem();
    }
  }
  if (savedStems && saved.version !== MASK_VERSION && stems.vocals.mask && !stems.vocals.regions.length && !stems.vocals.db && !stems.vocals.mute
    && STEM_ORDER.every((name) => name === "vocals" || isDefaultStem(stems[name]))) {
    stems.vocals = defaultStem();
  }
  return {
    version: MASK_VERSION,
    // Masking, muting or changing a stem means the scene renders with that audio. Only an explicit "off" keeps it from doing so.
    user_off: Boolean(saved.user_off),
    enabled: !saved.user_off && (Boolean(saved.enabled) || STEM_ORDER.some((name) => !isDefaultStem(stems[name]))),
    model_name: STEMS_BY_MODEL[saved.model_name] ? saved.model_name : "htdemucs_6s",
    input_gain_db: clamp(finite(saved.input_gain_db, 0), -24, 24),
    stems,
  };
}

// Text that is equal for equal stem settings, to tell whether the built mix matches what is set now.
function stemSignature(stems, names) {
  return names.map((name) => {
    const item = stems?.[name] || defaultStem();
    const regions = cleanRegions(item.regions).map((region) => `${round3(region.start)}-${round3(region.end)}-${Math.round(region.fade_ms)}`).join(",");
    return `${name}:${item.mask ? 1 : 0}${item.mute ? 1 : 0}|${Math.round(finite(item.db, 0) * 100) / 100}|${regions}`;
  }).join(";");
}

export function createAudioMask({
  button, monitorButton, visibilityButton, overlay, state, activeSegment, currentProjectAudioPath, getProjectFolder, autoSaveSessionQuiet,
  currentGlobalTime, isTimelinePlaying, audio: mainAudio, sceneAudio: mainSceneAudio,
}) {
  button.title = "Split a scene's audio into stems (vocals, drums, bass, guitar...), keep only the parts you want, and render the scene with that audio.";

  // ---- shared scene logic (works for any scene, with or without the window open) ----
  const cache = new Map(); // scene id -> last backend state
  let busy = false;
  let saveTimer = 0;
  const rebuildTimers = new Map();

  const folder = () => String(getProjectFolder() || "").trim();
  const sceneById = (id) => [...(state.segments || []), ...(state.overlaySegments || [])].find((item) => item?.id === id) || null;
  const sceneSource = (item) => {
    const custom = speakingAudioEditsActive(state) ? "" : String(item?.custom_audio_path || "").trim();
    return {
      audioPath: speakingAudioEditsActive(state) ? String(state.audioClipGenerationMixPath || "") : custom || String(currentProjectAudioPath() || "").trim(),
      start: custom ? audioSourceStart(item) : Number(item?.start || 0),
      duration: timelineSegmentDuration(item),
    };
  };
  const stemNames = (data) => (Array.isArray(data?.stem_names) && data.stem_names.length ? data.stem_names : FOUR_STEMS);

  function splitStale(item, data) {
    const separation = data?.separation;
    if (!data?.exists || !separation) return false;
    const source = sceneSource(item);
    return Math.abs(Number(separation.duration_seconds) - source.duration) > 0.02
      || Math.abs(Number(separation.start_seconds) - source.start) > 0.02
      || String(separation.source_path || "") !== source.audioPath;
  }

  function mixStale(item, data) {
    if (!data?.mix) return true;
    const names = stemNames(data);
    return stemSignature(data.mix.stems, names) !== stemSignature(maskSettings(item).stems, names);
  }

  function entryFor(item, data) {
    // Stems made for a different length or start would be drawn stretched, so they are not drawn at all.
    if (!data?.exists || splitStale(item, data)) return null;
    const settings = maskSettings(item);
    const current = !mixStale(item, data);
    return {
      duration: Number(data.duration) || 0,
      enabled: settings.enabled,
      stems: stemNames(data).map((name) => ({ name, peaks: data.peaks?.[name] || null, ...settings.stems[name] })),
      mix: { peaks: current ? data.peaks?.masked_mix || null : null, current },
    };
  }

  function publish(item) {
    if (item) setStemLanes(item.id, entryFor(item, cache.get(item.id)));
  }

  function saveSoon() {
    window.clearTimeout(saveTimer);
    saveTimer = window.setTimeout(() => autoSaveSessionQuiet("Audio Mask").catch(() => null), 600);
  }

  async function fetchScene(item) {
    const data = await postJson("/vrgdg/music_builder/audio_mask/state", { project_folder: folder(), scene_id: item.id }, 60000);
    cache.set(item.id, data);
    return data;
  }

  async function buildScene(item) {
    await prepareEditedAudio(state, folder(), { generation: true });
    if (splitStale(item, cache.get(item.id))) {
      throw new Error("Audio changed after splitting stems. Split the scene again before building its mask.");
    }
    const settings = maskSettings(item);
    const data = await postJson("/vrgdg/music_builder/audio_mask/render", { project_folder: folder(), scene_id: item.id, stems: settings.stems }, 120000);
    cache.set(item.id, data);
    publish(item);
    return data;
  }

  // A scene that renders with its mask must never use a mix older than its settings, so edits rebuild it.
  function scheduleRebuild(item) {
    window.clearTimeout(rebuildTimers.get(item.id));
    const data = cache.get(item.id);
    if (!(maskSettings(item).enabled || normalizeAudioMaskAuto(state.audioMaskAuto).enabled) || !data?.exists || splitStale(item, data)) return;
    rebuildTimers.set(item.id, window.setTimeout(async () => {
      const latest = cache.get(item.id);
      if (!mixStale(item, latest)) return;
      if (busy) {
        scheduleRebuild(item);
        return;
      }
      await run("Rebuilding the masked mix...", () => buildScene(item));
    }, 700));
  }

  const sceneName = (item) => {
    const index = (state.segments || []).indexOf(item);
    return index >= 0 ? `Scene ${index + 1}` : "An overlay scene";
  };

  // The stems no longer fit the scene (its length, start or audio changed, as when scenes are merged, split or resized):
  // remove them as if the mask had been deleted.
  async function removeStems(item, reason) {
    cache.delete(item.id);
    delete item.audio_mask;
    setStemLanes(item.id, null);
    saveSoon();
    refreshButton();
    try {
      await postJson("/vrgdg/music_builder/audio_mask/delete", { project_folder: folder(), scene_id: item.id }, 60000);
    } catch {
      // Files left behind are removed by the orphan check below.
    }
    if (reason) toast(`${sceneName(item)}: stems removed (${reason}).`);
    if (item.id === loadedId) refreshWindow();
  }

  // A scene must stay out of date for a moment before its stems go, so dragging an edge does not remove them on the way.
  const STALE_GRACE_MS = 1500;
  const staleSince = new Map();
  function watchStale() {
    if (busy) return;
    const now = Date.now();
    for (const [id, data] of cache) {
      const item = sceneById(id);
      if (!item || !data?.exists || !sceneSource(item).audioPath || !splitStale(item, data)) {
        staleSince.delete(id);
        continue;
      }
      const since = staleSince.get(id) ?? now;
      staleSince.set(id, since);
      if (now - since >= STALE_GRACE_MS) {
        staleSince.delete(id);
        removeStems(item, "the scene's length or audio changed");
      }
    }
  }

  // Stems of scenes that no longer exist (deleted or merged away) are removed from the project folder. A scene has to be
  // gone for a while first, so Undo can bring it back with its stems.
  const ORPHAN_GRACE_MS = 20000;
  const ORPHAN_CHECK_MS = 10000;
  const orphanSince = new Map();
  let orphanCheckedAt = 0;
  let orphanChecking = false;
  async function collectOrphans() {
    const projectFolder = folder();
    if (orphanChecking || !projectFolder || projectFolder !== hydratedFolder || !(state.segments || []).length) return;
    if (Date.now() - orphanCheckedAt < ORPHAN_CHECK_MS) return;
    orphanCheckedAt = Date.now();
    orphanChecking = true;
    try {
      const listed = await postJson("/vrgdg/music_builder/audio_mask/list", { project_folder: projectFolder }, 60000);
      const ids = Array.isArray(listed.scene_ids) ? listed.scene_ids : [];
      const current = new Set([...(state.segments || []), ...(state.overlaySegments || [])].map((item) => item.id));
      const now = Date.now();
      for (const id of [...orphanSince.keys()]) if (!ids.includes(id) || current.has(id)) orphanSince.delete(id);
      for (const id of ids) {
        if (current.has(id)) continue;
        const since = orphanSince.get(id) ?? now;
        orphanSince.set(id, since);
        if (now - since < ORPHAN_GRACE_MS) continue;
        orphanSince.delete(id);
        cache.delete(id);
        setStemLanes(id, null);
        await postJson("/vrgdg/music_builder/audio_mask/delete", { project_folder: projectFolder, scene_id: id }, 60000);
      }
    } catch {
      // Tried again at the next check.
    } finally {
      orphanChecking = false;
    }
  }

  // Auto stems: with the project option on, every scene without current stems is split in the background, one at a time.
  // It waits while a scene is rendering, so the video render keeps the GPU to itself.
  const autoFailed = new Set();
  const anySceneRendering = () => [...(state.segments || []), ...(state.overlaySegments || [])].some((item) => item.video_status === "running");
  // Splitting is heavy (GPU or CPU work on the same machine and server that stream the video), so it waits while the
  // timeline plays and for a few seconds after, instead of stalling playback.
  const PLAYBACK_QUIET_MS = 4000;
  let lastPlayingAt = 0;
  function playbackBusy() {
    if (isTimelinePlaying()) lastPlayingAt = Date.now();
    return Date.now() - lastPlayingAt < PLAYBACK_QUIET_MS;
  }

  let preparingEditedAudio = false;
  async function autoSplitTick() {
    const auto = normalizeAudioMaskAuto(state.audioMaskAuto);
    if (!auto.enabled || busy || !folder() || anySceneRendering() || playbackBusy()) return;
    if (speakingAudioEditsActive(state) && !state.audioClipGenerationMixPath) {
      if (preparingEditedAudio) return;
      preparingEditedAudio = true;
      try { await prepareEditedAudio(state, folder(), { generation: true }); }
      catch (error) { setStatus(String(error?.message || error), true); return; }
      finally { preparingEditedAudio = false; }
    }
    for (const item of state.segments || []) {
      if (!item?.id || autoFailed.has(item.id)) continue;
      const source = sceneSource(item);
      if (!source.audioPath || source.duration < 0.25) continue;
      const data = cache.get(item.id);
      if (!data) {
        try {
          await fetchScene(item);
          publish(item);
        } catch {
          autoFailed.add(item.id);
        }
        return;
      }
      if (data.exists && !splitStale(item, data)) continue;
      await run(`Auto stems: splitting ${sceneName(item)}...`, async () => {
        try {
          const result = await postJson("/vrgdg/music_builder/audio_mask/separate", {
            project_folder: folder(), scene_id: item.id, audio_path: source.audioPath, start_seconds: source.start,
            duration_seconds: source.duration, model_name: auto.model_name, input_gain_db: auto.input_gain_db,
          }, 900000);
          cache.set(item.id, result);
          changeSettings(item, (next) => {
            next.model_name = auto.model_name;
            next.input_gain_db = auto.input_gain_db;
          });
        } catch (error) {
          autoFailed.add(item.id);
          throw error;
        }
      });
      return;
    }
  }

  // Auto stems also builds each scene's masked mix (all stems added together, with the masks, levels and mutes applied), so
  // the Masked mix track shows for every scene and Hear Stems has something to play. It is quick: a few files added together.
  const autoMixFailed = new Set();
  async function autoMixTick() {
    const auto = normalizeAudioMaskAuto(state.audioMaskAuto);
    if (!auto.enabled || busy || !folder() || anySceneRendering() || playbackBusy()) return;
    for (const item of state.segments || []) {
      if (!item?.id || autoMixFailed.has(item.id)) continue;
      const data = cache.get(item.id);
      if (!data?.exists || splitStale(item, data) || !mixStale(item, data)) continue;
      await run(`Auto stems: building the mix for ${sceneName(item)}...`, async () => {
        try {
          await buildScene(item);
        } catch (error) {
          autoMixFailed.add(item.id);
          throw error;
        }
      });
      return;
    }
  }

  // Change one scene's settings from anywhere (the timeline, the window), then save, redraw and rebuild.
  function changeSettings(item, mutate) {
    const settings = maskSettings(item);
    const before = stemSignature(settings.stems, STEM_ORDER);
    mutate(settings);
    // Editing a stem (mask, regions, level, mute) switches the render on again, even after it was turned off by hand.
    if (stemSignature(settings.stems, STEM_ORDER) !== before && STEM_ORDER.some((name) => !isDefaultStem(settings.stems[name]))) {
      settings.user_off = false;
      settings.enabled = true;
    }
    item.audio_mask = settings;
    saveSoon();
    publish(item);
    scheduleRebuild(item);
    refreshButton();
    if (item.id === loadedId) refreshWindow();
  }

  onStemEditRequest((sceneId, stem, patch) => {
    const item = sceneById(sceneId);
    if (!item || !STEM_ORDER.includes(stem)) return;
    changeSettings(item, (settings) => {
      const target = settings.stems[stem];
      if ("mask" in patch) target.mask = Boolean(patch.mask);
      if ("regions" in patch) target.regions = mergeRegions(patch.regions);
      if ("db" in patch) target.db = clamp(finite(patch.db, 0), -100, 24);
      if ("mute" in patch) target.mute = Boolean(patch.mute);
    });
  });

  // ---- window shell ----
  const win = document.createElement("div");
  win.style.cssText = "position:fixed;z-index:100004;display:none;width:600px;max-width:calc(100vw - 16px);max-height:calc(100vh - 16px);overflow:auto;box-sizing:border-box;border:1px solid #155e75;border-radius:8px;background:#18181b;box-shadow:0 12px 36px rgba(0,0,0,.6);color:#f4f4f5;font-size:12px;";
  const header = document.createElement("div");
  header.style.cssText = "display:flex;align-items:center;justify-content:space-between;padding:8px 10px;background:#083344;cursor:move;user-select:none;position:sticky;top:0;z-index:2;";
  const title = document.createElement("strong");
  const closeButton = makeButton("×");
  closeButton.style.padding = "2px 8px";
  header.append(title, closeButton);
  const body = document.createElement("div");
  body.style.cssText = "display:flex;flex-direction:column;gap:10px;padding:12px;";
  win.append(header, body);
  overlay.append(win);

  const section = (text) => {
    const wrap = document.createElement("div");
    wrap.style.cssText = "display:flex;flex-direction:column;gap:6px;border:1px solid #27272a;border-radius:6px;padding:8px;";
    const label = document.createElement("div");
    label.textContent = text;
    label.style.cssText = "font-weight:700;color:#67e8f9;";
    wrap.append(label);
    body.append(wrap);
    return wrap;
  };
  const line = (...nodes) => {
    const row = document.createElement("div");
    row.style.cssText = "display:flex;gap:8px;align-items:center;flex-wrap:wrap;";
    row.append(...nodes);
    return row;
  };
  const labelled = (text, input, width = "90px") => {
    const wrap = document.createElement("label");
    wrap.style.cssText = "display:flex;gap:5px;align-items:center;";
    input.style.width = width;
    wrap.append(document.createTextNode(text), input);
    return wrap;
  };
  const numberInput = (value, min, max, step) => {
    const input = makeInput(String(value), "number");
    input.min = String(min);
    input.max = String(max);
    input.step = String(step);
    return input;
  };
  const note = (text) => {
    const el = document.createElement("div");
    el.style.cssText = "color:#a1a1aa;line-height:1.4;";
    el.textContent = text;
    return el;
  };

  const statusLine = document.createElement("div");
  statusLine.style.cssText = "min-height:16px;color:#a1a1aa;line-height:1.4;white-space:pre-wrap;";
  const showLanesCheck = makeCheckbox("Show the stem tracks under the scenes on the timeline", state.showTimelineStems !== false);
  showLanesCheck.input.addEventListener("change", () => {
    state.showTimelineStems = showLanesCheck.input.checked;
    refreshStemLanes();
  });
  body.append(statusLine, showLanesCheck.wrapper);

  // 1. Split
  const splitBox = section("1. Split this scene's audio into stems");
  const modelSelect = makeSelect(MODELS, "htdemucs_6s");
  const gainInput = numberInput(0, -24, 24, 0.5);
  gainInput.title = "Raise this to amplify the scene audio before it is split, which helps when the vocals are quiet. The stems are saved at this level.";
  const splitButton = makeButton("Split into stems", "primary");
  const splitInfo = note("");
  const autoCheck = makeCheckbox("Auto stems: split every scene into stems in the background (uses the model and input gain above)", state.audioMaskAuto?.enabled !== false);
  const autoInfo = note("");
  splitBox.append(line(labelled("Model", modelSelect, "330px")), line(labelled("Input gain (dB)", gainInput, "70px"), splitButton), splitInfo, autoCheck.wrapper, autoInfo);

  // 2. Stems
  const stemBox = section("2. What to keep from each stem");
  const stemRows = document.createElement("div");
  stemRows.style.cssText = "display:flex;flex-direction:column;gap:8px;";
  stemBox.append(
    note("Drag on a stem's track in the timeline to keep that part. With Only keep the regions on, everything outside the regions of that stem is muted. With it off the whole stem plays."),
    stemRows,
  );

  // 3. Build and render
  const buildBox = section("3. Build the masked mix and render with it");
  const buildButton = makeButton("Build masked mix", "primary");
  const previewSelect = makeSelect([], "masked_mix");
  previewSelect.style.width = "240px";
  const previewPlayButton = makeButton("Play");
  const previewStopButton = makeButton("Stop");
  const useCheck = makeCheckbox("Render this scene with the masked mix (the finished clip gets the real audio back)", false);
  const buildInfo = note("");
  const removeButton = makeButton("Remove stems");
  buildBox.append(line(buildButton), line(previewSelect, previewPlayButton, previewStopButton), useCheck.wrapper, buildInfo, line(removeButton));

  // ---- window state ----
  let open = false;
  let loadedId = "";
  let audio = null;
  const loaded = () => (loadedId ? sceneById(loadedId) : null);
  const loadedData = () => cache.get(loadedId) || { exists: false };

  const setStatus = (text, isError = false) => {
    statusLine.textContent = text;
    statusLine.style.color = isError ? "#fca5a5" : "#a1a1aa";
  };
  const sceneLabel = () => {
    const index = (state.segments || []).findIndex((item) => item.id === loadedId);
    return index >= 0 ? `Scene ${index + 1}` : "Overlay scene";
  };

  function refreshStatus() {
    if (!open) return;
    const item = loaded();
    const data = loadedData();
    if (!item) setStatus("Select a scene on the timeline to work on its audio.");
    else if (!sceneSource(item).audioPath) setStatus("This project has no audio loaded, so there is nothing to split.", true);
    else if (!data.exists) setStatus("Step 1: split this scene's audio into stems.");
    else if (splitStale(item, data)) setStatus("This scene's timing or audio changed after it was split. Split it again before building.", true);
    else if (mixStale(item, data)) setStatus("The masked mix is out of date. Press Build masked mix.");
    else setStatus("The masked mix matches the settings.");
  }

  function renderStemRows() {
    const item = loaded();
    stemRows.replaceChildren();
    if (!item) return;
    const data = loadedData();
    const names = data.exists ? stemNames(data) : STEMS_BY_MODEL[modelSelect.value];
    const settings = maskSettings(item);
    const duration = Math.max(0.05, Number(data.duration) || sceneSource(item).duration || 1);
    for (const name of names) {
      const stem = settings.stems[name];
      const card = document.createElement("div");
      card.style.cssText = `border-left:4px solid ${STEM_COLORS[name]};padding:4px 8px;display:flex;flex-direction:column;gap:5px;background:#111113;border-radius:4px;`;
      const maskCheck = makeCheckbox("Only keep the regions", stem.mask);
      const muteCheck = makeCheckbox("Mute", stem.mute);
      const dbInput = numberInput(stem.db, -100, 24, 1);
      dbInput.title = "Level in dB. 0 leaves it as it is. -100 is silent.";
      const head = document.createElement("strong");
      head.textContent = STEM_LABELS[name];
      head.style.cssText = `min-width:54px;color:${STEM_COLORS[name]};`;
      card.append(line(head, maskCheck.wrapper, labelled("Level (dB)", dbInput, "60px"), muteCheck.wrapper));
      maskCheck.input.addEventListener("change", () => changeSettings(item, (next) => { next.stems[name].mask = maskCheck.input.checked; }));
      muteCheck.input.addEventListener("change", () => changeSettings(item, (next) => { next.stems[name].mute = muteCheck.input.checked; }));
      dbInput.addEventListener("change", () => changeSettings(item, (next) => { next.stems[name].db = clamp(finite(dbInput.value, 0), -100, 24); }));
      if (stem.mask) {
        stem.regions.forEach((region, index) => {
          const startInput = numberInput(region.start, 0, duration, 0.001);
          const endInput = numberInput(region.end, 0, duration, 0.001);
          const fadeInput = numberInput(region.fade_ms, 0, 1000, 5);
          const remove = makeButton("Delete");
          remove.style.padding = "3px 8px";
          const apply = () => {
            const start = clamp(finite(startInput.value, region.start), 0, duration);
            const end = clamp(finite(endInput.value, region.end), 0, duration);
            if (end - start < MIN_REGION_SECONDS) {
              toast("A region must be at least 0.02 seconds long.", true);
              refreshWindow();
              return;
            }
            changeSettings(item, (next) => {
              next.stems[name].regions[index] = { start: round3(start), end: round3(end), fade_ms: clamp(finite(fadeInput.value, DEFAULT_FADE_MS), 0, 1000) };
              next.stems[name].regions = mergeRegions(next.stems[name].regions);
            });
          };
          for (const input of [startInput, endInput, fadeInput]) input.addEventListener("change", apply);
          remove.onclick = () => changeSettings(item, (next) => { next.stems[name].regions.splice(index, 1); });
          card.append(line(document.createTextNode(`Region ${index + 1}`), labelled("Start (s)", startInput, "80px"), labelled("End (s)", endInput, "80px"), labelled("Fade (ms)", fadeInput, "60px"), remove));
        });
        if (!stem.regions.length) card.append(note("No regions: this stem is muted for the whole scene."));
        const add = makeButton("Add region");
        const keepAll = makeButton("Keep all");
        add.onclick = () => changeSettings(item, (next) => {
          const list = next.stems[name].regions;
          const start = round3(Math.min(duration - 0.5, list.length ? list[list.length - 1].end : 0));
          list.push({ start: Math.max(0, start), end: round3(Math.min(duration, Math.max(0, start) + 0.5)), fade_ms: DEFAULT_FADE_MS });
          next.stems[name].regions = mergeRegions(list);
        });
        keepAll.onclick = () => changeSettings(item, (next) => { next.stems[name].regions = [{ start: 0, end: round3(duration), fade_ms: 0 }]; });
        card.append(line(add, keepAll));
      }
      stemRows.append(card);
    }
  }

  function refreshInfo() {
    if (!open) return;
    const item = loaded();
    const data = loadedData();
    const separation = data.separation;
    splitInfo.textContent = data.exists && separation
      ? `Last split: ${separation.model_name}, input gain ${Number(separation.input_gain_db).toFixed(1)} dB, ${Number(separation.seconds_taken || 0).toFixed(1)} s${item && splitStale(item, data) ? " (out of date)" : ""}.`
      : "Not split yet.";
    buildInfo.textContent = data.mix
      ? `Masked mix built${item && mixStale(item, data) ? " — out of date" : ""}.`
      : "No masked mix built yet.";
    autoCheck.input.checked = Boolean(state.audioMaskAuto?.enabled);
    if (autoCheck.input.checked) {
      const scenes = (state.segments || []).filter((scene) => scene?.id && sceneSource(scene).audioPath && sceneSource(scene).duration >= 0.25);
      const done = scenes.filter((scene) => cache.get(scene.id)?.exists && !splitStale(scene, cache.get(scene.id))).length;
      const failed = scenes.filter((scene) => autoFailed.has(scene.id)).length;
      autoInfo.textContent = `Auto stems on: ${done} of ${scenes.length} scenes have stems${failed ? `, ${failed} failed (turn it off and on to retry)` : ""}. New, merged and resized scenes are split by themselves. It pauses while a scene renders.`;
    } else {
      autoInfo.textContent = "";
    }
    const ready = Boolean(data.exists) && !busy;
    buildButton.disabled = !ready || Boolean(item && splitStale(item, data));
    previewPlayButton.disabled = !data.exists;
    removeButton.disabled = !data.exists || busy;
    splitButton.disabled = busy || !item || !sceneSource(item).audioPath;
    const options = data.exists
      ? [{ value: "original", label: "Original scene audio" }, ...stemNames(data).map((name) => ({ value: name, label: `${STEM_LABELS[name]} stem` })), { value: "masked_mix", label: "Masked mix (what the video model hears)" }]
      : [];
    const previous = previewSelect.value;
    previewSelect.replaceChildren(...options.map((option) => new Option(option.label, option.value)));
    previewSelect.value = options.some((option) => option.value === previous) ? previous : "masked_mix";
  }

  function refreshWindow() {
    if (!open) return;
    const item = loaded();
    title.textContent = item ? `Audio Mask — ${sceneLabel()}` : "Audio Mask";
    if (item) useCheck.input.checked = maskSettings(item).enabled;
    renderStemRows();
    refreshStatus();
    refreshInfo();
  }

  function refreshButton() {
    const active = activeSegment();
    const on = audioMaskInUse(active);
    button.textContent = on ? "Audio Mask ●" : "Audio Mask";
    button.style.borderColor = on ? "#22d3ee" : "";
  }

  async function run(label, work) {
    if (busy) return;
    busy = true;
    refreshInfo();
    setStatus(label);
    try {
      await work();
    } catch (error) {
      setStatus(String(error?.message || error), true);
      toast(String(error?.message || error), true);
    } finally {
      busy = false;
      refreshWindow();
    }
  }

  splitButton.onclick = () => {
    const item = loaded();
    if (!item) return;
    run("Splitting the scene audio into stems (the first time also loads the Demucs model)...", async () => {
      await prepareEditedAudio(state, folder(), { generation: true });
      const source = sceneSource(item);
      const data = await postJson("/vrgdg/music_builder/audio_mask/separate", {
        project_folder: folder(), scene_id: item.id, audio_path: source.audioPath, start_seconds: source.start,
        duration_seconds: source.duration, model_name: modelSelect.value, input_gain_db: clamp(finite(gainInput.value, 0), -24, 24),
      }, 900000);
      cache.set(item.id, data);
      changeSettings(item, (next) => {
        next.model_name = modelSelect.value;
        next.input_gain_db = clamp(finite(gainInput.value, 0), -24, 24);
      });
    });
  };
  buildButton.onclick = () => {
    const item = loaded();
    if (item) run("Building the masked mix...", () => buildScene(item));
  };
  removeButton.onclick = () => {
    const item = loaded();
    if (!item || !window.confirm("Remove this scene's stems and masked mix? The settings stay saved.")) return;
    run("Removing stems...", async () => {
      await postJson("/vrgdg/music_builder/audio_mask/delete", { project_folder: folder(), scene_id: item.id }, 60000);
      changeSettings(item, (next) => { next.enabled = false; next.user_off = true; });
      await fetchScene(item);
      publish(item);
    });
  };
  useCheck.input.addEventListener("change", () => {
    const item = loaded();
    if (!item) return;
    const data = loadedData();
    if (useCheck.input.checked && (!data.exists || splitStale(item, data))) {
      useCheck.input.checked = false;
      toast("Split this scene's audio first, then build the masked mix.", true);
      return;
    }
    changeSettings(item, (next) => { next.enabled = useCheck.input.checked; next.user_off = !useCheck.input.checked; });
    if (useCheck.input.checked && mixStale(item, loadedData())) run("Building the masked mix...", () => buildScene(item));
  });
  // The project's auto stems option takes the model and input gain shown here.
  function syncAutoOption(enabled = state.audioMaskAuto?.enabled) {
    state.audioMaskAuto = normalizeAudioMaskAuto({ enabled, model_name: modelSelect.value, input_gain_db: gainInput.value });
    autoFailed.clear();
    autoMixFailed.clear();
    saveSoon();
    refreshInfo();
  }
  autoCheck.input.addEventListener("change", () => syncAutoOption(autoCheck.input.checked));
  modelSelect.addEventListener("change", () => {
    const item = loaded();
    if (item) changeSettings(item, (next) => { next.model_name = modelSelect.value; });
    if (state.audioMaskAuto?.enabled) syncAutoOption();
  });
  gainInput.addEventListener("change", () => {
    const item = loaded();
    if (item) changeSettings(item, (next) => { next.input_gain_db = clamp(finite(gainInput.value, 0), -24, 24); });
    if (state.audioMaskAuto?.enabled) syncAutoOption();
  });

  // ---- preview ----
  function stopPreview() {
    if (audio) {
      audio.pause();
      audio = null;
    }
  }
  previewStopButton.onclick = stopPreview;
  previewPlayButton.onclick = async () => {
    stopPreview();
    const item = loaded();
    if (!item) return;
    const which = previewSelect.value;
    if (which === "masked_mix" && mixStale(item, loadedData())) await run("Building the masked mix...", () => buildScene(item));
    const path = loadedData().files?.[which];
    if (!path) {
      toast("That audio has not been built yet.", true);
      return;
    }
    audio = new Audio(audioUrl(path));
    audio.addEventListener("ended", stopPreview);
    try {
      await audio.play();
    } catch (error) {
      toast(`Could not play the audio: ${String(error?.message || error)}`, true);
    }
  };

  // ---- loading the selected scene ----
  async function loadWindow() {
    stopPreview();
    try { await prepareEditedAudio(state, folder(), { generation: true }); }
    catch (error) { setStatus(String(error?.message || error), true); return; }
    const item = activeSegment();
    loadedId = item?.id || "";
    if (item) {
      const settings = maskSettings(item);
      modelSelect.value = settings.model_name;
      gainInput.value = String(settings.input_gain_db);
      useCheck.input.checked = settings.enabled;
      if (folder() && sceneSource(item).audioPath) {
        try {
          await fetchScene(item);
          publish(item);
        } catch (error) {
          setStatus(String(error?.message || error), true);
        }
      }
    }
    refreshWindow();
  }

  // ---- window position ----
  const clampPosition = (value, maximum) => Math.max(8, Math.min(value, Math.max(8, maximum)));
  // Opens beside the tool rail at the top of the screen, clear of the stem tracks, and stays until closed.
  function openWindow() {
    open = true;
    const anchor = button.getBoundingClientRect();
    win.style.display = "block";
    win.style.left = `${clampPosition(anchor.right + 10, window.innerWidth - win.offsetWidth - 8)}px`;
    win.style.top = "8px";
    showLanesCheck.input.checked = state.showTimelineStems !== false;
    loadWindow();
  }
  function closeWindow() {
    open = false;
    stopPreview();
    win.style.display = "none";
  }
  button.addEventListener("click", () => (open ? closeWindow() : openWindow()));
  closeButton.onclick = closeWindow;
  header.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 || closeButton.contains(event.target)) return;
    event.preventDefault();
    const startX = event.clientX;
    const startY = event.clientY;
    const startLeft = win.offsetLeft;
    const startTop = win.offsetTop;
    const move = (moveEvent) => {
      win.style.left = `${clampPosition(startLeft + moveEvent.clientX - startX, window.innerWidth - win.offsetWidth - 8)}px`;
      win.style.top = `${clampPosition(startTop + moveEvent.clientY - startY, window.innerHeight - 60)}px`;
    };
    const stop = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", stop);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", stop);
  });

  // A click on a stem track in the timeline selects that scene, then asks for this window.
  onOpenAudioMaskRequest(() => {
    if (open) loadWindow();
    else openWindow();
  });

  // Saved scenes get their stem tracks on the timeline when the project opens, one scene per tick.
  const HYDRATE_BATCH = 12;
  let bulkLoadFailed = false;
  const hydrated = new Set();
  let hydratedFolder = "";
  let hydrating = false;
  async function hydrateSavedScenes() {
    const projectFolder = folder();
    if (projectFolder !== hydratedFolder) {
      hydratedFolder = projectFolder;
      hydrated.clear();
      cache.clear();
      staleSince.clear();
      orphanSince.clear();
      autoFailed.clear();
      autoMixFailed.clear();
      clearStemLanes();
    }
    if (hydrating || !projectFolder || playbackBusy()) return;
    // Up to a dozen scenes in one request, so a project opens with a few requests, not one per scene.
    const batch = (state.segments || [])
      .filter((item) => item?.id && item.audio_mask && typeof item.audio_mask === "object" && !hydrated.has(item.id))
      .slice(0, HYDRATE_BATCH);
    if (!batch.length) return;
    hydrating = true;
    try {
      let states = null;
      if (!bulkLoadFailed) {
        try {
          states = (await postJson("/vrgdg/music_builder/audio_mask/states", { project_folder: projectFolder, scene_ids: batch.map((item) => item.id) }, 120000)).states;
        } catch {
          bulkLoadFailed = true; // an older server without the batch route: load scene by scene instead
        }
      }
      for (const item of batch) {
        hydrated.add(item.id);
        let data = states?.[item.id];
        if (!data && bulkLoadFailed) {
          try {
            data = await fetchScene(item);
          } catch {
            continue; // the track just does not show; opening the window reports the problem
          }
        }
        if (!data) continue;
        cache.set(item.id, data);
        if (item.audio_mask.version !== MASK_VERSION) {
          item.audio_mask = maskSettings(item);
          saveSoon();
        }
        publish(item);
      }
    } finally {
      hydrating = false;
    }
  }

  // ---- hear the stem mix from the timeline ----
  // With "Hear Stems" on, a timeline that plays through a scene with a built masked mix plays that mix, in step with the
  // playhead, and silences the main audio for that scene only. Other scenes keep their main audio.
  const monitor = { sceneId: "", builtAt: 0, element: null };
  const building = new Set();
  let savedVolumes = null;

  function restoreMainAudio() {
    if (!savedVolumes) return;
    mainAudio.volume = savedVolumes.audio;
    mainSceneAudio.volume = savedVolumes.sceneAudio;
    savedVolumes = null;
  }

  function stopMonitor() {
    if (monitor.element && !monitor.element.paused) monitor.element.pause();
    restoreMainAudio();
  }

  function refreshMonitorButton() {
    const on = Boolean(state.monitorStems);
    monitorButton.textContent = on ? "Hear Stems ●" : "Hear Stems";
    monitorButton.style.borderColor = on ? "#22d3ee" : "";
    monitorButton.style.background = on ? "#0e7490" : "";
  }

  function monitorTick() {
    if (!state.monitorStems || !isTimelinePlaying()) {
      stopMonitor();
      return;
    }
    const now = Number(currentGlobalTime());
    const item = (state.segments || []).find((candidate) => now >= candidate.start && now < candidate.end && cache.get(candidate.id)?.exists);
    const data = item ? cache.get(item.id) : null;
    if (!item || splitStale(item, data)) {
      stopMonitor();
      return;
    }
    if (mixStale(item, data)) {
      // The stems changed since the mix was built: rebuild it in the background and keep the main audio until it is ready.
      if (!building.has(item.id) && !busy) {
        building.add(item.id);
        buildScene(item).catch(() => null).finally(() => building.delete(item.id));
      }
      stopMonitor();
      return;
    }
    const path = data.files?.masked_mix;
    const builtAt = Number(data.mix?.built_at) || 0;
    if (!path) {
      stopMonitor();
      return;
    }
    if (!monitor.element || monitor.sceneId !== item.id || monitor.builtAt !== builtAt) {
      if (monitor.element) monitor.element.pause();
      monitor.element = new Audio(audioUrl(path));
      monitor.sceneId = item.id;
      monitor.builtAt = builtAt;
    }
    if (!savedVolumes) savedVolumes = { audio: mainAudio.volume, sceneAudio: mainSceneAudio.volume };
    mainAudio.volume = 0;
    mainSceneAudio.volume = 0;
    const offset = now - item.start;
    if (monitor.element.paused) {
      monitor.element.currentTime = offset;
      monitor.element.play().catch(() => null);
    } else if (Math.abs(monitor.element.currentTime - offset) > 0.15) {
      monitor.element.currentTime = offset;
    }
  }

  // Hide or show every stem track on the timeline. The window's checkbox does the same.
  function refreshVisibilityButton() {
    const shown = state.showTimelineStems !== false;
    visibilityButton.textContent = shown ? "Hide Stems" : "Show Stems";
    visibilityButton.style.borderColor = shown ? "" : "#f59e0b";
    showLanesCheck.input.checked = shown;
  }
  visibilityButton.addEventListener("click", () => {
    state.showTimelineStems = state.showTimelineStems === false;
    refreshVisibilityButton();
    refreshStemLanes();
  });
  showLanesCheck.input.addEventListener("change", refreshVisibilityButton);
  refreshVisibilityButton();

  monitorButton.addEventListener("click", () => {
    state.monitorStems = !state.monitorStems;
    refreshMonitorButton();
    if (!state.monitorStems) stopMonitor();
  });
  refreshMonitorButton();
  window.setInterval(monitorTick, 80);

  // The window follows the selected scene, and the button shows a dot while the selected scene renders with its mask.
  window.setInterval(() => {
    refreshButton();
    refreshVisibilityButton();
    hydrateSavedScenes();
    watchStale();
    collectOrphans();
    autoSplitTick();
    autoMixTick();
    if (open && !busy && (activeSegment()?.id || "") !== loadedId) loadWindow();
  }, 400);
  refreshButton();

  return { button, openWindow, closeWindow };
}
