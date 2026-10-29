// The overlay library panel (Post Process, Overlays): the dust, particle, smoke and light leak clips a person downloaded, with a still of each
// that plays a short loop under the pointer. Clips are imported from a file or a download folder, checked by the app, and kept in its
// overlay library. The library is only there when the build can draw overlays (`state.overlayAvailable`), so the pack keeps its placeholder until then.

import { api } from "../../../scripts/api.js";
import { postJson } from "./comfy_api.mjs";
import { makeButton, makeCheckbox, toast } from "./controls.mjs";
import { addOverlayLayer } from "./overlay_lane.mjs";
import { OVERLAY_BLEND_MODES, normalizeOverlayLayers } from "./overlay_layers.mjs";
import { openPackImport } from "./overlay_pack_import.mjs";
import { OVERLAY_PRESETS, appendOverlayOptions, overlayFontOptions, overlayPresetClips, renderOverlayInspector } from "./overlay_inspector.mjs";

const IMPORT_TIMEOUT_MS = 15 * 60 * 1000;
const FREE_SOURCES = [
  ["FX Elements free clips", "https://www.fxelements.com/free"],
  ["Pexels videos", "https://www.pexels.com/search/videos/dust%20particles/"],
  ["Pixabay videos", "https://pixabay.com/videos/search/dust%20particles/"],
];
const OVERLAY_CATEGORY_LABELS = { dust: "Dust", particles: "Particles", smoke: "Smoke", flares: "Lens flares", light_leaks: "Light leaks", snow: "Snow", other: "Other" };

// "4096x2304 - 7.9 s - 24 fps - 197 MB" for a library clip.
export function overlayLayerMeta(item) {
  const parts = [];
  if (item?.width && item?.height) parts.push(`${item.width}x${item.height}`);
  if (Number(item?.duration) > 0) parts.push(`${Number(item.duration).toFixed(1)} s`);
  if (Number(item?.fps) > 0) parts.push(`${Math.round(Number(item.fps))} fps`);
  const bytes = Number(item?.size_bytes) || 0;
  if (bytes > 0) parts.push(bytes >= 1024 * 1024 * 1024 ? `${(bytes / 1024 ** 3).toFixed(1)} GB` : `${Math.max(1, Math.round(bytes / 1024 ** 2))} MB`);
  return parts.join(" · ");
}

// The clips of one category, or all of them.
export function filterOverlayItems(items, category) {
  const list = Array.isArray(items) ? items : [];
  return !category || category === "all" ? list : list.filter((item) => item.category === category);
}

// Categories that hold clips, in the library's order, with how many each holds.
export function overlayCategoryCounts(items) {
  const counts = new Map();
  for (const item of Array.isArray(items) ? items : []) counts.set(item.category, (counts.get(item.category) || 0) + 1);
  return [...counts.entries()].map(([category, count]) => ({ category, count, label: OVERLAY_CATEGORY_LABELS[category] || category }));
}

// Where a new clip goes. Grain and vignette cover the selected range of the timeline, or all of it. A media clip starts at the playhead and
// is as long as its file (at least a second), pulled back from the end of the timeline when it would run past it.
export function overlayLayerPlacement(kind, { playhead = 0, total = 0, range = null, duration = 0 } = {}) {
  const length = Math.max(1, Number(duration) || 0);
  const whole = Math.max(1, Number(total) || 0);
  if (kind === "media") {
    const size = Math.min(length, whole);
    const start = Math.min(Math.max(0, Number(playhead) || 0), Math.max(0, whole - size));
    return { start: Math.round(start * 1000) / 1000, end: Math.round((start + size) * 1000) / 1000 };
  }
  const a = Number(range?.in ?? range?.start);
  const b = Number(range?.out ?? range?.end);
  if (Number.isFinite(a) && Number.isFinite(b) && b > a) return { start: Math.max(0, a), end: b };
  return { start: 0, end: whole };
}

// The selected range of the timeline as seconds, or null when there is none.
export function overlayCaptionSpan(range) {
  const from = Number(range?.in ?? range?.start);
  const to = Number(range?.out ?? range?.end);
  return Number.isFinite(from) && Number.isFinite(to) && to > from ? { from: Math.max(0, from), to } : null;
}

// The clips without the lyric captions, and how many were taken out.
export function overlayWithoutCaptions(clips) {
  const list = Array.isArray(clips) ? clips : [];
  const kept = list.filter((clip) => clip?.group !== "lyric_captions");
  return { clips: kept, removed: list.length - kept.length };
}

// The request for the exact picture at the playhead: the scene video under it, how far into it, and where that video starts on the timeline.
// `{ error }` says what is missing.
export function overlayFramePayload({ playhead, segment, videoPath, clips, track, projectFolder }) {
  if (!segment) return { error: "Put the playhead on a scene first." };
  if (!String(videoPath || "").trim()) return { error: "This scene has no rendered video yet, so there is no picture to check." };
  const start = Number(segment.start) || 0;
  return {
    payload: {
      input_path: String(videoPath),
      seek: Math.max(0, Math.round(((Number(playhead) || 0) - start) * 1000) / 1000),
      time_offset: start,
      overlay_layers: Array.isArray(clips) ? clips : [],
      overlay_layer_track: track || { enabled: true },
      project_folder: projectFolder || "",
    },
  };
}

// The picture in a window of its own over the Builder, closed with a click, Escape or the button.
function showOverlayFrame(image, caption) {
  const overlay = document.createElement("div");
  overlay.style.cssText = "position:fixed;inset:0;z-index:100020;background:rgba(0,0,0,.82);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px;padding:24px;";
  const picture = document.createElement("img");
  picture.src = image;
  picture.alt = caption;
  picture.style.cssText = "max-width:100%;max-height:calc(100% - 70px);object-fit:contain;border:1px solid #3f3f46;background:#000;";
  const line = document.createElement("div");
  line.textContent = caption;
  line.style.cssText = "font-size:12px;color:#d4d4d8;text-align:center;max-width:900px;line-height:1.4;";
  const close = makeButton("Close", "primary");
  const done = () => {
    window.removeEventListener("keydown", onKey, true);
    overlay.remove();
  };
  const onKey = (event) => {
    if (event.key === "Escape") done();
  };
  close.onclick = done;
  overlay.onclick = (event) => {
    if (event.target === overlay) done();
  };
  window.addEventListener("keydown", onKey, true);
  overlay.append(picture, line, close);
  document.body.append(overlay);
}

const CHIP_STYLE = "border:1px solid #3f3f46;border-radius:999px;background:#18181b;color:#f4f4f5;font-size:11px;font-weight:700;padding:3px 9px;cursor:pointer;";
const CHIP_ON_STYLE = "border:1px solid #22d3ee;border-radius:999px;background:#0e7490;color:#ecfeff;font-size:11px;font-weight:800;padding:3px 9px;cursor:pointer;";

const OVERLAY_FONT_KEY = "vrgdg_overlay_font";

// The font chosen for new text and for lyric captions is kept for the next time the page opens. A page that cannot keep it starts with Default.
function savedOverlayFont() {
  try {
    return String(window.localStorage?.getItem(OVERLAY_FONT_KEY) || "");
  } catch {
    return "";
  }
}

function saveOverlayFont(name) {
  try {
    window.localStorage?.setItem(OVERLAY_FONT_KEY, name);
  } catch {
    // The choice is just not remembered.
  }
}

// The font a new text or caption gets: the chosen one when the fonts list has it, and "" (the default of each) when it does not (a font that was removed).
export function overlayChosenFont(fonts, name) {
  const wanted = String(name || "").trim().toLowerCase();
  if (!wanted) return "";
  return overlayFontOptions(fonts, "").find((option) => option.value.toLowerCase() === wanted)?.value || "";
}

export function createOverlayLibraryPanel({ state, pushHistory, playheadTime, timelineSeconds, projectFolder, commit, luts, lutsFolder, activeSceneRange, inspectorCommit, segmentAt, videoPathOf }) {
  const captions = { style: "clean", font: savedOverlayFont() };
  const element = document.createElement("div");
  element.style.cssText = "display:flex;flex-direction:column;gap:10px;";
  const panel = { items: [], filter: "all", busy: "", error: "", loaded: false, fonts: [], folder: "" };
  // The settings of the selected clip have a place of their own that is redrawn alone, so a click on the lane does not rebuild the library.
  const inspectorHost = document.createElement("div");
  const renderSelected = () => renderOverlayInspector(inspectorHost, { state, pushHistory, commit: inspectorCommit, luts: () => luts?.() || [], fonts: () => panel.fonts, activeSceneRange: () => activeSceneRange?.() || null, total: () => Number(timelineSeconds?.()) || 0 });

  async function loadFonts() {
    try {
      const response = await api.fetchApi("/vrgdg/music_builder/overlay_layers/fonts", { timeoutMs: null });
      const data = await response.json().catch(() => ({}));
      if (response.ok && data.ok !== false && Array.isArray(data.fonts)) panel.fonts = data.fonts;
    } catch {
      // The text still draws in the default font. The list is only for choosing.
    }
  }

  async function load() {
    await loadFonts();
    try {
      const response = await api.fetchApi("/vrgdg/music_builder/overlay_library", { timeoutMs: null });
      const data = await response.json().catch(() => ({}));
      if (!response.ok || data.ok === false) throw new Error(String(data.error || `Request failed (${response.status})`));
      panel.items = Array.isArray(data.items) ? data.items : [];
      panel.folder = typeof data.folder === "string" ? data.folder : "";
      panel.error = "";
      panel.loaded = true;
    } catch (error) {
      panel.error = String(error?.message || error);
    }
    render();
  }

  async function importFrom(kind) {
    let path = "";
    try {
      const picked = await postJson("/vrgdg/music_builder/pick_path", { kind }, 6 * 60 * 1000);
      path = String(picked?.path || "");
    } catch (error) {
      toast(String(error?.message || error), true);
      return;
    }
    if (!path) return;
    if (kind === "folder") {
      // A folder is read first and the person chooses what to import from it.
      await openPackImport({ folder: path, onImported: (items) => { panel.items = items; render(); } });
      return;
    }
    panel.busy = "Importing, checking and making a preview. A 4K clip can take a few seconds...";
    panel.error = "";
    render();
    try {
      const reply = await postJson("/vrgdg/music_builder/overlay_library/import", { path }, IMPORT_TIMEOUT_MS);
      const added = reply?.added?.length || 0;
      const already = reply?.already_there?.length || 0;
      panel.items = Array.isArray(reply?.items) ? reply.items : panel.items;
      toast(added ? `Added ${added} clip${added === 1 ? "" : "s"} to the overlay library.` : already ? "That clip is already in the overlay library." : "Nothing was added.", !added && !already);
    } catch (error) {
      panel.error = String(error?.message || error);
      toast(panel.error, true);
    }
    panel.busy = "";
    render();
  }

  async function remove(item) {
    if (!window.confirm(`Remove "${item.name}" from the overlay library? Projects that already use it keep their own copy.`)) return;
    try {
      await postJson("/vrgdg/music_builder/overlay_library/delete", { id: item.id });
      panel.items = panel.items.filter((other) => other.id !== item.id);
    } catch (error) {
      toast(String(error?.message || error), true);
    }
    render();
  }

  function placement(kind, duration = 0) {
    return overlayLayerPlacement(kind, { playhead: Number(playheadTime?.()) || 0, total: Number(timelineSeconds?.()) || 0, range: state.selectedTimelineRange, duration });
  }

  function put(fields, reason) {
    pushHistory();
    // A clip put on the track from here can be dragged and trimmed straight away. (A clip saved without a lock setting is locked.)
    state.overlayLayers = addOverlayLayer(Array.isArray(state.overlayLayers) ? state.overlayLayers : [], { locked: false, ...fields });
    state.overlayLaneOpen = true;
    state.activeOverlayId = state.overlayLayers[state.overlayLayers.length - 1]?.id || "";
    commit(reason);
    render();
  }

  function addEffect(kind) {
    put({ kind, ...placement(kind) }, "Overlay added");
  }

  function addText() {
    const start = Number(playheadTime?.()) || 0;
    const total = Number(timelineSeconds?.()) || 0;
    const length = 3;
    const from = total > length ? Math.min(start, total - length) : start;
    const font = overlayChosenFont(panel.fonts, captions.font);
    put({ kind: "text", text: "Your text", start: from, end: from + length, ...(font ? { font } : {}) }, "Overlay added");
  }

  // One text clip for each lyric line, with the time it is sung. A second run replaces the first one's captions (over the selected range, or all).
  async function addCaptions() {
    const folder = String(projectFolder?.() || "");
    if (!folder) {
      toast("Open or create a project first.", true);
      return;
    }
    const span = overlayCaptionSpan(state.selectedTimelineRange);
    panel.busy = "Reading the lyrics...";
    render();
    try {
      const font = overlayChosenFont(panel.fonts, captions.font);
      const reply = await postJson("/vrgdg/music_builder/overlay_layers/lyric_captions", { project_folder: folder, style: captions.style, ...(font ? { font } : {}), from: span?.from, to: span?.to });
      pushHistory();
      const overlaps = (clip) => !span || (clip.end > span.from && clip.start < span.to);
      const kept = (Array.isArray(state.overlayLayers) ? state.overlayLayers : []).filter((clip) => !(clip.group === "lyric_captions" && overlaps(clip)));
      state.overlayLayers = normalizeOverlayLayers([...kept, ...reply.clips]);
      state.overlayLaneOpen = true;
      state.activeOverlayId = "";
      commit("Overlay captions added");
      toast(`Added ${reply.count} caption${reply.count === 1 ? "" : "s"} ${reply.source === "timestamped_lyrics" ? "timed to the words of the song" : "from the scene lyrics (the song has no word timing yet, so each caption covers its scene)"}.`);
    } catch (error) {
      toast(String(error?.message || error), true);
    }
    panel.busy = "";
    render();
  }

  // Every lyric caption at once, for when the captions are not wanted. Undo brings them back.
  function removeCaptions() {
    const { clips, removed } = overlayWithoutCaptions(state.overlayLayers);
    if (!removed) {
      toast("There are no lyric captions to remove.", true);
      return;
    }
    pushHistory();
    state.overlayLayers = normalizeOverlayLayers(clips);
    if (!state.overlayLayers.some((clip) => clip.id === state.activeOverlayId)) state.activeOverlayId = "";
    commit("Overlay captions removed");
    toast(`Removed ${removed} caption${removed === 1 ? "" : "s"}. Undo brings them back.`);
    render();
  }

  function addLut() {
    const names = luts?.() || [];
    if (!names.length) {
      const folder = String(lutsFolder?.() || "");
      toast(folder ? `No LUTs are installed. Put .cube files in ${folder}, then open the Overlays tab again.` : "No LUTs are installed. Put .cube files in the LUTS folder, then open the Overlays tab again.", true);
      return;
    }
    put({ kind: "lut", lut: names[0], ...placement("lut") }, "Overlay added");
  }

  // The exact picture: the same render Export uses, for one frame of the scene under the playhead.
  async function checkFrame() {
    const playhead = Number(playheadTime?.()) || 0;
    const segment = segmentAt?.(playhead) || null;
    const built = overlayFramePayload({ playhead, segment, videoPath: segment ? videoPathOf?.(segment) : "", clips: state.overlayLayers, track: state.overlayLayerTrack, projectFolder: projectFolder?.() || "" });
    if (built.error) {
      toast(built.error, true);
      return;
    }
    panel.busy = "Making the exact picture...";
    render();
    try {
      const reply = await postJson("/vrgdg/music_builder/overlay_layers/preview_frame", built.payload, 5 * 60 * 1000);
      const left = Array.isArray(reply.skipped) && reply.skipped.length ? ` Left out: ${reply.skipped.map((item) => `${item.id} (${item.reason})`).join("; ")}.` : "";
      showOverlayFrame(reply.image, `The exact picture at ${playhead.toFixed(2)} s, made by the same render as Export.${reply.track_off ? " The overlays are off, so this is the plain picture." : ""}${left}`);
    } catch (error) {
      toast(String(error?.message || error), true);
    }
    panel.busy = "";
    render();
  }

  function addPreset(name) {
    const clips = overlayPresetClips(name, { total: Number(timelineSeconds?.()) || 0, range: state.selectedTimelineRange });
    pushHistory();
    let list = Array.isArray(state.overlayLayers) ? state.overlayLayers : [];
    for (const fields of clips) list = addOverlayLayer(list, fields);
    state.overlayLayers = list;
    state.overlayLaneOpen = true;
    state.activeOverlayId = list[list.length - 1]?.id || "";
    commit("Overlay preset added");
    render();
  }

  async function addMedia(item) {
    const folder = String(projectFolder?.() || "");
    if (!folder) {
      toast("Open or create a project first, then add the clip.", true);
      return;
    }
    try {
      const used = await postJson("/vrgdg/music_builder/overlay_library/use", { id: item.id, project_folder: folder });
      put({ kind: "media", path: used.path, blend_mode: OVERLAY_BLEND_MODES.includes(item.suggested_blend) ? item.suggested_blend : "screen", ...placement("media", item.duration) }, "Overlay added");
    } catch (error) {
      toast(String(error?.message || error), true);
    }
  }

  function tile(item) {
    const box = document.createElement("div");
    box.style.cssText = "border:1px solid #3f3f46;border-radius:7px;background:#18181b;overflow:hidden;display:flex;flex-direction:column;min-width:0;";
    box.dataset.overlayId = item.id;
    const preview = document.createElement("div");
    preview.style.cssText = "position:relative;aspect-ratio:16/9;background:#000;overflow:hidden;";
    const still = document.createElement("img");
    still.alt = item.name;
    still.style.cssText = "width:100%;height:100%;object-fit:cover;display:block;";
    if (item.thumb_url) still.src = api.apiURL(item.thumb_url);
    preview.append(still);
    let loop = null;
    preview.onmouseenter = () => {
      if (!item.loop_url || loop) return;
      loop = document.createElement("video");
      loop.src = api.apiURL(item.loop_url);
      loop.muted = true;
      loop.loop = true;
      loop.autoplay = true;
      loop.playsInline = true;
      loop.style.cssText = "position:absolute;inset:0;width:100%;height:100%;object-fit:cover;";
      preview.append(loop);
      loop.play().catch(() => {});
    };
    preview.onmouseleave = () => {
      loop?.remove();
      loop = null;
    };
    const badge = document.createElement("span");
    badge.textContent = item.has_alpha ? "ALPHA" : "BLACK BG";
    badge.title = item.has_alpha ? "This clip has transparency. It is laid over the video as it is." : "This clip has a black background. It is laid over the video with a blend mode that drops the black.";
    badge.style.cssText = "position:absolute;left:4px;bottom:4px;border-radius:3px;padding:1px 4px;font-size:9px;font-weight:900;background:rgba(0,0,0,.65);color:#cffafe;";
    preview.append(badge);
    const remover = makeButton("×");
    remover.title = "Remove this clip from the overlay library.";
    remover.style.cssText += "position:absolute;right:4px;top:4px;padding:0 6px;font-size:13px;line-height:18px;background:rgba(0,0,0,.65);";
    remover.onclick = (event) => {
      event.stopPropagation();
      remove(item);
    };
    preview.append(remover);
    const caption = document.createElement("div");
    caption.style.cssText = "padding:5px 7px 6px;display:flex;flex-direction:column;gap:2px;min-width:0;";
    const name = document.createElement("div");
    name.textContent = item.name;
    name.title = item.file;
    name.style.cssText = "font-size:11px;font-weight:800;color:#f4f4f5;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;";
    const meta = document.createElement("div");
    meta.textContent = overlayLayerMeta(item);
    meta.style.cssText = "font-size:10px;color:#a1a1aa;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;";
    meta.title = meta.textContent;
    caption.append(name, meta);
    const add = makeButton("Add at playhead", "primary");
    add.title = "Add this clip as an overlay at the playhead. It is copied into the project.";
    add.style.cssText += "margin:0 6px 6px;padding:4px 6px;font-size:11px;";
    add.onclick = () => addMedia(item);
    box.append(preview, caption, add);
    return box;
  }

  function render() {
    element.textContent = "";
    const head = document.createElement("div");
    head.style.cssText = "border:1px solid #3f3f46;border-radius:7px;background:#18181b;padding:10px;display:flex;flex-direction:column;gap:8px;";
    const title = document.createElement("div");
    title.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:8px;";
    title.innerHTML = `<span style="font-size:13px;font-weight:900;color:#cffafe;">Overlay library</span><span style="font-size:11px;color:#a1a1aa;">${panel.items.length} clip${panel.items.length === 1 ? "" : "s"}</span>`;
    const buttons = document.createElement("div");
    buttons.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:6px;";
    const importClip = makeButton("Import clip...", "primary");
    importClip.title = "Choose a .mov, .mp4, .webm, .mkv or .avi clip to add to the library.";
    importClip.onclick = () => importFrom("video");
    const importFolder = makeButton("Import folder...");
    importFolder.title = "Choose a folder, such as an unzipped FX Elements download or a folder of effect packs. You then choose which of what it holds to import.";
    importFolder.onclick = () => importFrom("folder");
    importClip.disabled = importFolder.disabled = Boolean(panel.busy);
    buttons.append(importClip, importFolder);
    const note = document.createElement("div");
    note.textContent = "Use the ProRes or MP4 download of a clip. RED RAW (.R3D) and EXR frames cannot be used.";
    note.style.cssText = "font-size:10px;line-height:1.4;color:#a1a1aa;";
    head.append(title, buttons, note);
    element.append(head);

    const track = document.createElement("div");
    track.style.cssText = "border:1px solid #3f3f46;border-radius:7px;background:#18181b;padding:10px;display:flex;flex-direction:column;gap:8px;";
    const trackTitle = document.createElement("div");
    trackTitle.textContent = "Overlays";
    trackTitle.style.cssText = "font-size:13px;font-weight:900;color:#cffafe;";
    const on = makeCheckbox("Draw the overlays in Render All and Export", state.overlayLayerTrack?.enabled !== false);
    on.wrapper.style.cssText += "font-size:11px;";
    on.input.onchange = () => {
      pushHistory();
      state.overlayLayerTrack = { ...state.overlayLayerTrack, enabled: on.input.checked };
      commit("Overlays switched");
    };
    const effects = document.createElement("div");
    effects.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:6px;";
    const grain = makeButton("+ Film grain");
    grain.title = "Add film grain over the selected range of the timeline, or all of it.";
    grain.onclick = () => addEffect("grain");
    const vignette = makeButton("+ Vignette");
    vignette.title = "Add a vignette (darker corners) over the selected range of the timeline, or all of it.";
    vignette.onclick = () => addEffect("vignette");
    const text = makeButton("+ Text");
    text.title = "Add a title at the playhead. Change its words, font and look in the settings below.";
    text.onclick = addText;
    const lut = makeButton("+ LUT");
    lut.title = "Add a LUT look over the selected range of the timeline, or all of it.";
    lut.onclick = addLut;
    effects.append(grain, vignette, text, lut);
    const fontRow = document.createElement("label");
    fontRow.style.cssText = "display:flex;flex-direction:column;gap:4px;font-size:11px;font-weight:700;color:#d4d4d8;";
    const fontSelect = document.createElement("select");
    fontSelect.title = "The font of a new text and of the lyric captions. Default keeps the look of each. The fonts installed on this computer are in the list.";
    fontSelect.style.cssText = "width:100%;min-width:0;box-sizing:border-box;border:1px solid #3f3f46;border-radius:6px;background:#09090b;color:#fafafa;padding:6px;font-size:12px;";
    appendOverlayOptions(fontSelect, [{ value: "", label: "Default" }, ...overlayFontOptions(panel.fonts, "")]);
    fontSelect.value = overlayChosenFont(panel.fonts, captions.font);
    fontSelect.onchange = () => {
      captions.font = fontSelect.value;
      saveOverlayFont(captions.font);
    };
    fontRow.append(document.createTextNode("Font for text and captions"), fontSelect);
    const captionRow = document.createElement("div");
    captionRow.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:6px;";
    const style = document.createElement("select");
    style.title = "How the lyric captions look.";
    style.style.cssText = "box-sizing:border-box;border:1px solid #3f3f46;border-radius:6px;background:#09090b;color:#fafafa;padding:6px;font-size:12px;";
    for (const [value, label] of [["clean", "Clean"], ["bold", "Bold"], ["lower", "Lower third"]]) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = label;
      style.append(option);
    }
    style.value = captions.style;
    style.onchange = () => { captions.style = style.value; };
    const captionButton = makeButton("Add lyric captions");
    captionButton.title = "Add one caption for each line of the song's lyrics, timed to the words when the project has word timing (selected range only, when there is one).";
    captionButton.onclick = addCaptions;
    captionButton.disabled = Boolean(panel.busy);
    const removeCaptionsButton = makeButton("Remove captions");
    removeCaptionsButton.title = "Take every lyric caption off the timeline at once.";
    removeCaptionsButton.onclick = removeCaptions;
    removeCaptionsButton.style.gridColumn = "1 / -1";
    captionRow.append(style, captionButton, removeCaptionsButton);
    const presets = document.createElement("div");
    presets.style.cssText = "display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:6px;";
    for (const [name, preset] of Object.entries(OVERLAY_PRESETS)) {
      const button = makeButton(preset.label);
      button.title = "Add grain and a vignette set up like this, over the selected range of the timeline or all of it.";
      button.onclick = () => addPreset(name);
      presets.append(button);
    }
    const trackNote = document.createElement("div");
    trackNote.textContent = "On the timeline, drag a clip to move it, drag its edges to trim or extend it, right click for more.";
    trackNote.style.cssText = "font-size:10px;line-height:1.4;color:#a1a1aa;";
    const frame = makeButton("Check this frame");
    frame.title = "Make the exact picture Export would make at the playhead, from the scene video under it, and show it. Use it to judge the look before a long render.";
    frame.onclick = checkFrame;
    frame.disabled = Boolean(panel.busy);
    track.append(trackTitle, on.wrapper, effects, fontRow, captionRow, presets, frame, trackNote);
    element.append(track);
    element.append(inspectorHost);
    renderSelected();

    if (panel.busy) {
      const busy = document.createElement("div");
      busy.textContent = panel.busy;
      busy.style.cssText = "font-size:11px;line-height:1.4;color:#67e8f9;border:1px solid #155e75;border-radius:6px;padding:7px 9px;background:#083344;";
      element.append(busy);
    }
    if (panel.error) {
      const error = document.createElement("div");
      error.textContent = panel.error;
      error.style.cssText = "font-size:11px;line-height:1.4;color:#fecaca;border:1px solid #7f1d1d;border-radius:6px;padding:7px 9px;background:#2a1010;white-space:pre-wrap;word-break:break-word;";
      element.append(error);
    }

    const categories = overlayCategoryCounts(panel.items);
    if (categories.length > 1) {
      const chips = document.createElement("div");
      chips.style.cssText = "display:flex;flex-wrap:wrap;gap:5px;";
      for (const { category, count, label } of [{ category: "all", count: panel.items.length, label: "All" }, ...categories]) {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.textContent = `${label} ${count}`;
        chip.style.cssText = panel.filter === category ? CHIP_ON_STYLE : CHIP_STYLE;
        chip.onclick = () => {
          panel.filter = category;
          render();
        };
        chips.append(chip);
      }
      element.append(chips);
    } else {
      panel.filter = "all";
    }

    const shown = filterOverlayItems(panel.items, panel.filter);
    if (shown.length) {
      const grid = document.createElement("div");
      grid.style.cssText = "display:grid;grid-template-columns:repeat(auto-fill,minmax(112px,1fr));gap:8px;";
      for (const item of shown) grid.append(tile(item));
      element.append(grid);
    } else if (panel.loaded && !panel.busy) {
      const empty = document.createElement("div");
      empty.textContent = "No clips yet. Download a free dust, smoke or light leak clip, then import it here.";
      empty.style.cssText = "font-size:11px;line-height:1.45;color:#a1a1aa;border:1px dashed #3f3f46;border-radius:7px;padding:12px;";
      element.append(empty);
    }

    if (panel.folder) {
      const where = document.createElement("div");
      where.textContent = `Library folder: ${panel.folder}`;
      where.title = "Clips copied into a category folder here (dust, smoke, ...) show up in the library.";
      where.style.cssText = "font-size:10px;line-height:1.4;color:#71717a;overflow-wrap:anywhere;user-select:text;";
      element.append(where);
    }

    const sources = document.createElement("div");
    sources.style.cssText = "display:flex;flex-direction:column;gap:5px;border-top:1px solid #27272a;padding-top:8px;";
    const sourcesTitle = document.createElement("div");
    sourcesTitle.textContent = "Free clips (check each site's license)";
    sourcesTitle.style.cssText = "font-size:10px;font-weight:800;color:#a1a1aa;";
    sources.append(sourcesTitle);
    for (const [label, url] of FREE_SOURCES) {
      const link = document.createElement("button");
      link.type = "button";
      link.textContent = label;
      link.title = url;
      link.style.cssText = "text-align:left;border:0;background:none;color:#67e8f9;font-size:11px;padding:0;cursor:pointer;text-decoration:underline;";
      link.onclick = () => window.open(url, "_blank", "noopener,noreferrer");
      sources.append(link);
    }
    element.append(sources);
  }

  return { element, render, load, renderSelected };
}
