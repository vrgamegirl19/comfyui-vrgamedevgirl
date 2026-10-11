import { cloneMiniMaxH3Settings } from "./minimax_h3.mjs";

const PREFIX = "vrgdg:video-mode:";
const storageKey = folder => PREFIX + String(folder || "").trim().replace(/\\/g, "/").replace(/\/+$/, "").toLowerCase();

function modeSelection(state) {
  const settings = cloneMiniMaxH3Settings(state.miniMaxH3Settings);
  return {
    engine: state.projectVideoEngine || "ltx",
    videoModelMode: state.projectVideoEngine === "minimax_h3" ? "" : state.videoModelMode || "i2v",
    miniMax: { video_mode: settings.video_mode, render_pass: settings.render_pass, pipeline: settings.pipeline },
    scenes: [...(state.segments || []), ...(state.overlaySegments || [])]
      .filter(scene => scene.use_scene_minimax_h3_settings)
      .map(scene => ({ id: scene.id,
        video_mode: scene.minimax_h3_settings?.video_mode || scene.minimax_h3_mode || settings.video_mode,
        render_pass: scene.minimax_h3_settings?.render_pass ?? settings.render_pass }))
      .sort((a, b) => String(a.id).localeCompare(String(b.id))),
  };
}

export function recordSavedVideoMode(state, detail, storage) {
  if (!detail?.session || storageKey(detail.projectFolder) !== storageKey(state.projectFolder)) return;
  const session = detail.session;
  state.videoModeResumeBase = JSON.stringify(modeSelection({
    projectVideoEngine: session.video_engine, videoModelMode: session.video_model_mode,
    miniMaxH3Settings: session.minimax_h3_settings,
    segments: session.segments, overlaySegments: session.overlay_segments,
  }));
  try { storage?.removeItem(storageKey(state.projectFolder)); } catch { /* Storage may be disabled. */ }
}

// Closing the browser cannot depend on an asynchronous request to ComfyUI.
// Keep only the mode selection locally; all other project edits still use Save/Autosave.
export function rememberVideoMode(storage, state) {
  if (!String(state.projectFolder || "").trim()) return;
  try {
    storage.setItem(storageKey(state.projectFolder), JSON.stringify({
      base: state.videoModeResumeBase,
      selection: modeSelection(state), activeId: state.activeId,
    }));
  } catch { /* Storage may be disabled; the saved project remains authoritative. */ }
}

export function restoreVideoMode(storage, state) {
  const saved = modeSelection(state);
  state.videoModeResumeBase = JSON.stringify(saved);
  try {
    const resume = JSON.parse(storage.getItem(storageKey(state.projectFolder)) || "null");
    if (!resume?.selection) return "";
    // A newer project save from another browser/API must not be overwritten by an old local mode.
    if (resume.base !== state.videoModeResumeBase
      && JSON.stringify(resume.selection) !== state.videoModeResumeBase) return "";
    if (resume.selection.videoModelMode) state.videoModelMode = resume.selection.videoModelMode;
    state.miniMaxH3Settings = cloneMiniMaxH3Settings({ ...state.miniMaxH3Settings, ...resume.selection.miniMax });
    for (const savedScene of resume.selection.scenes || []) {
      const scene = [...state.segments, ...state.overlaySegments].find(item => item.id === savedScene.id);
      if (!scene?.use_scene_minimax_h3_settings) continue;
      scene.minimax_h3_mode = savedScene.video_mode;
      scene.minimax_h3_settings = cloneMiniMaxH3Settings({ ...state.miniMaxH3Settings,
        ...scene.minimax_h3_settings, video_mode: savedScene.video_mode, render_pass: savedScene.render_pass });
    }
    return String(resume.activeId || "");
  } catch { return ""; }
}
