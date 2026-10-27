// The overlay pass after the final stitch. The scenes are stitched into one video as before, and when the project has overlays and this
// build can draw them, the stitched video is sent through `POST /vrgdg/music_builder/overlay_layers/render`, which writes the finished video over
// it. The scene clips are never touched, so turning the overlays off and stitching again gives the plain video with nothing to re-render.
// Render All and Export both stitch through `stitchRenderedScenes`, so both get the overlays from this one place.

import { api } from "../../../scripts/api.js";
import { postJson } from "./comfy_api.mjs";
import { toast } from "./controls.mjs";
import { normalizeOverlayLayers, normalizeOverlayLayerTrack } from "./overlay_layers.mjs";

const OVERLAY_RENDER_TIMEOUT_MS = 30 * 60 * 1000;

// Asks whether this build can draw overlays. The app answers; a pack without the overlay render has no such route, so the overlays are simply not
// applied there and nothing else changes. It asks with a plain request, since a missing route is an answer here and not an error
// for the project's error log.
export async function loadOverlayCapabilities(state) {
  try {
    const response = await api.fetchApi("/vrgdg/music_builder/overlay_layers/capabilities", { timeoutMs: null });
    const data = response.ok ? await response.json().catch(() => null) : null;
    state.overlayAvailable = Boolean(data && data.ok !== false && data.version);
  } catch {
    state.overlayAvailable = false;
  }
  return state.overlayAvailable;
}

// The clips the stitch should draw: none when this build cannot, when the overlays are off or when every layer is off.
export function overlayLayersForStitch(state) {
  if (!state?.overlayAvailable) return [];
  if (normalizeOverlayLayerTrack(state.overlayLayerTrack).enabled === false) return [];
  return normalizeOverlayLayers(state.overlayLayers).filter((clip) => clip.enabled);
}

function overlaySkippedText(skipped) {
  return skipped.map((item) => `${item.id}: ${item.reason}`).join("; ");
}

// `stitched` is what the stitch route returned. It comes back unchanged (its `final_video_path` now holds the video with the overlays on it)
// with the render's answer under `overlay_layers`. When the overlays cannot be applied the plain stitched video stays and the reason is shown.
export async function applyOverlayAfterStitch(state, stitched, { progress = null, timelineOffset = 0, projectFolder = "" } = {}) {
  const clips = overlayLayersForStitch(state);
  const path = String(stitched?.final_video_path || "");
  if (!clips.length || !path) return stitched;
  progress?.set("Applying the overlays...", 97);
  let reply;
  try {
    reply = await postJson("/vrgdg/music_builder/overlay_layers/render", {
      input_path: path,
      replace_source: true,
      overlay_layers: clips,
      overlay_layer_track: normalizeOverlayLayerTrack(state.overlayLayerTrack),
      time_offset: Number(timelineOffset) || 0,
      project_folder: projectFolder,
    }, OVERLAY_RENDER_TIMEOUT_MS);
  } catch (error) {
    toast(`The overlays could not be applied, so the video has none. ${String(error?.message || error)}`, true);
    return stitched;
  }
  const skipped = Array.isArray(reply?.skipped) ? reply.skipped : [];
  if (skipped.length) toast(`Some overlays were left out. ${overlaySkippedText(skipped)}`, true);
  return { ...stitched, overlay_layers: reply };
}
