import { postJson } from "./comfy_api.mjs";
import { normalizeProjectVideoEngine, toast } from "./controls.mjs";
import { showFinalVideoReadyModal } from "./dialogs.mjs";
import { selectedSegmentVideoPath } from "./selection_preview.mjs";

// The stitch the Export button makes is named with this prefix, so the server deletes it once it is copied to the place the
// user chose and never anything else. Keep it equal to EXPORT_STITCH_PREFIX in builder/export_video.py.
export const EXPORT_STITCH_PREFIX = "EXPORT_VIDEO";
const EXPORT_LISTED_SCENES = 4;
let exportInProgress = false;

function exportSceneList(labels) {
  const shown = labels.slice(0, EXPORT_LISTED_SCENES).join(", ");
  return labels.length > EXPORT_LISTED_SCENES ? `${shown} and ${labels.length - EXPORT_LISTED_SCENES} more` : shown;
}

// Whether every scene has its final video, and if not, what is missing. A MiniMax scene that only has a pass-1 draft does not
// count (Render All finishes those first), and neither does a scene that is rendering right now.
export function exportReadiness(state, sceneLabel = (segment, index) => `Scene ${index + 1}`) {
  const scenes = Array.isArray(state?.segments) ? state.segments : [];
  const miniMax = normalizeProjectVideoEngine(state?.projectVideoEngine) === "minimax_h3";
  const rendering = [];
  const withoutVideo = [];
  const drafts = [];
  scenes.forEach((segment, index) => {
    if (segment?.video_status === "running") rendering.push(sceneLabel(segment, index));
    else if (!String(selectedSegmentVideoPath(segment) || "").trim()) withoutVideo.push(sceneLabel(segment, index));
    else if (miniMax && segment.minimax_h3_draft?.id) drafts.push(sceneLabel(segment, index));
  });
  const total = scenes.length;
  const rendered = total - rendering.length - withoutVideo.length - drafts.length;
  let reason = "";
  if (!total) reason = "Add scenes and render them before exporting.";
  else if (rendering.length) reason = `Export is ready when rendering is done: ${exportSceneList(rendering)}.`;
  else if (withoutVideo.length) reason = `Render every scene to export (${rendered} of ${total} done). Still without a video: ${exportSceneList(withoutVideo)}.`;
  else if (drafts.length) reason = `Finalize the drafts to export: ${exportSceneList(drafts)}.`;
  return { ready: !reason, total, rendered, reason };
}

// The name the save dialog starts with: the project's folder name.
export function exportFileName(projectFolder) {
  const name = String(projectFolder || "").split(/[\\/]+/).filter(Boolean).at(-1) || "";
  const safe = name.replace(/[<>:"/\\|?*\u0000-\u001f]/g, "_").replace(/[. ]+$/, "").trim();
  return `${safe || "video"}.mp4`;
}

export function syncExportButton(button, state) {
  if (!button) return;
  const { ready, total, reason } = exportReadiness(state);
  const enabled = ready && !exportInProgress;
  button.disabled = !enabled;
  button.textContent = exportInProgress ? "Exporting..." : "Export";
  button.title = exportInProgress
    ? "Exporting. Choose where to save in the dialog, or wait for the progress window to finish."
    : ready
      ? `Stitch all ${total} scene videos into one video and save it where you choose.`
      : reason;
  button.style.background = enabled ? "#06b6d4" : "#27272a";
  button.style.borderColor = enabled ? "#0891b2" : "#3f3f46";
  button.style.color = enabled ? "#082f49" : "#a1a1aa";
  button.style.opacity = enabled ? "1" : "0.6";
  button.style.cursor = enabled ? "pointer" : "not-allowed";
}

// Export: choose where to save, stitch every scene with the project's audio exactly as Render All does, copy the result there.
// The place is asked for first, so nothing is stitched for a dialog that is closed, and the stitch is deleted after the copy.
export function wireExportVideo({ button, state, projectInput, createProgressWindow, stitchRenderedScenes }) {
  if (!button) return;
  button.onclick = async () => {
    if (exportInProgress) return;
    const readiness = exportReadiness(state);
    if (!readiness.ready) {
      toast(readiness.reason, true);
      return;
    }
    const projectFolder = String(projectInput.value || state.projectFolder || "").trim();
    if (!projectFolder) {
      toast("Create or load a project before exporting.", true);
      return;
    }
    exportInProgress = true;
    syncExportButton(button, state);
    let progress = null;
    let stitchedPath = "";
    try {
      const picked = await postJson("/vrgdg/music_builder/pick_path", { kind: "save_video", file_name: exportFileName(projectFolder) }, 6 * 60 * 1000);
      const destination = String(picked?.path || "").trim();
      if (!destination) return;
      progress = createProgressWindow("Export Video");
      progress.set(`Stitching ${readiness.total} scene video${readiness.total === 1 ? "" : "s"}...`, 8);
      const stitched = await stitchRenderedScenes(progress, { outputPrefix: EXPORT_STITCH_PREFIX });
      stitchedPath = String(stitched?.final_video_path || "");
      progress.set(`Saving the video to:\n${destination}`, 97);
      const saved = await postJson("/vrgdg/music_builder/export_final_video", {
        source_path: stitchedPath,
        destination_path: destination,
        project_folder: projectFolder,
        remove_source: true,
      }, 10 * 60 * 1000);
      progress.set(`Export complete.\n\nSaved to:\n${saved.destination_path}`, 100);
      progress.close(4500);
      toast(`Exported:\n${saved.destination_path}`);
      showFinalVideoReadyModal(saved.destination_path);
    } catch (error) {
      const kept = stitchedPath ? `\n\nThe stitched video is still in the project folder:\n${stitchedPath}` : "";
      const message = `${String(error?.message || error)}${kept}`;
      progress?.set(`Error:\n${message}`, 100);
      toast(message, true);
    } finally {
      exportInProgress = false;
      syncExportButton(button, state);
    }
  };
}
