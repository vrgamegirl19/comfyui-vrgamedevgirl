// The Reference Builder's scene_map is the single source of location assignments.
export function mappedLocation(refs, segment, index = -1) {
  const locations = Array.isArray(refs?.locations) ? refs.locations : [];
  const map = refs?.scene_map || {};
  const id = String(map[segment?.id] || (index >= 0 ? map[String(index + 1)] : "") || "").trim();
  return locations.find((location) => String(location?.id || "") === id) || null;
}

export function hasMappedSceneLocation(refs, segments) {
  return (segments || []).some((segment, index) => Boolean(mappedLocation(refs, segment, index)));
}

export function sharedLocationRuns(refs, segments) {
  const ordered = (segments || []).slice().sort((a, b) => Number(a.start || 0) - Number(b.start || 0));
  const runs = [];
  let current = [];
  let previous = null;
  for (const [index, segment] of ordered.entries()) {
    const location = mappedLocation(refs, segment, index);
    const touchesPrevious = previous && Math.abs(Number(segment.start || 0) - Number(previous.end || 0)) <= 0.05;
    if (!location || !current.length || location.id !== mappedLocation(refs, previous, index - 1)?.id || !touchesPrevious) {
      if (current.length > 1) runs.push(current);
      current = [];
    }
    if (location) current.push(segment);
    previous = segment;
  }
  if (current.length > 1) runs.push(current);
  return runs;
}

export function eligibleSharedLocationRuns(refs, segments, projectSettings) {
  return sharedLocationRuns(refs, segments).filter((run) => run.slice(1).every((scene) => {
    const settings = scene.use_scene_minimax_h3_settings && scene.minimax_h3_settings
      ? scene.minimax_h3_settings : projectSettings;
    return isMiniMaxH3ContinuityAllowedForMode(
      "latent_continuation_masked", settings?.video_mode, settings?.render_pass,
    );
  }));
}
import { isMiniMaxH3ContinuityAllowedForMode } from "./minimax_h3.mjs";
