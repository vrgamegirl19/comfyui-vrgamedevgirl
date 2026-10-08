// Stem tracks shown under the scenes on the timeline. The Audio Mask feature publishes them; the timeline draws them and
// asks for edits. Each entry:
//   { duration, enabled, stems: [{ name, peaks, mask, regions: [{ start, end, fade_ms }], db, mute }],
//     mix: { peaks, current } }
// ``mask`` means only the regions of that stem are audible. Times are seconds from the start of the scene.

export const STEM_ORDER = ["vocals", "drums", "bass", "guitar", "piano", "other"];
export const STEM_LABELS = { vocals: "Vocals", drums: "Drums", bass: "Bass", guitar: "Guitar", piano: "Piano", other: "Other" };
export const STEM_COLORS = { vocals: "#67e8f9", drums: "#fbbf24", bass: "#fb7185", guitar: "#a3e635", piano: "#f0abfc", other: "#a78bfa", mix: "#86efac" };

const lanesBySceneId = new Map();
let changeListener = () => {};
let openListener = () => {};
let editListener = () => {};

export function getStemLanes(sceneId) {
  return lanesBySceneId.get(String(sceneId || "")) || null;
}

export function hasAnyStemLanes() {
  return lanesBySceneId.size > 0;
}

// The stem names shown as rows, in a fixed order, for the given scenes: every stem any of them has.
export function stemRowNames(sceneIds) {
  const present = new Set();
  for (const id of sceneIds) {
    for (const stem of getStemLanes(id)?.stems || []) present.add(stem.name);
  }
  return STEM_ORDER.filter((name) => present.has(name));
}

// Pass null to remove a scene's lanes.
export function setStemLanes(sceneId, lanes) {
  const key = String(sceneId || "");
  if (!key) return;
  if (lanes) lanesBySceneId.set(key, lanes);
  else if (!lanesBySceneId.delete(key)) return;
  changeListener();
}

export function clearStemLanes() {
  if (!lanesBySceneId.size) return;
  lanesBySceneId.clear();
  changeListener();
}

// Ask the timeline to draw again, for example after the "show stems" switch changed.
export function refreshStemLanes() {
  changeListener();
}

export function onStemLanesChange(listener) {
  changeListener = typeof listener === "function" ? listener : () => {};
}

// The timeline asks for the Audio Mask window to open on a scene; the window registers how.
export function onOpenAudioMaskRequest(listener) {
  openListener = typeof listener === "function" ? listener : () => {};
}

export function requestOpenAudioMask(sceneId) {
  openListener(String(sceneId || ""));
}

// A drag on a stem track: ``patch`` is any of { mask, regions, db, mute } for that stem.
export function onStemEditRequest(listener) {
  editListener = typeof listener === "function" ? listener : () => {};
}

export function requestStemEdit(sceneId, stemName, patch) {
  editListener(String(sceneId || ""), String(stemName || ""), patch || {});
}

// The project option that splits every scene into stems in the background. It is saved with the project.
// Defaults: on, with the 6 stem model (vocals, drums, bass, guitar, piano, other).
export function normalizeAudioMaskAuto(value) {
  const item = value && typeof value === "object" ? value : {};
  const gain = Number(item.input_gain_db);
  return {
    enabled: "enabled" in item ? Boolean(item.enabled) : true,
    model_name: ["htdemucs", "htdemucs_ft", "mdx_extra", "htdemucs_6s"].includes(item.model_name) ? item.model_name : "htdemucs_6s",
    input_gain_db: Number.isFinite(gain) ? Math.max(-24, Math.min(24, gain)) : 0,
  };
}
