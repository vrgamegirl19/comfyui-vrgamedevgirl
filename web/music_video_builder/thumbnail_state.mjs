// What the Thumbnail intro keeps in the project (`thumbnail_intro` in the session), and when the stitch uses it.
// No browser code in here, so tests/thumbnail_state.cjs can run it.

export const THUMBNAIL_MIN_SECONDS = 0.5;
export const THUMBNAIL_MAX_SECONDS = 10;
export const THUMBNAIL_DEFAULT_SECONDS = 2;
export const THUMBNAIL_ENGINES = ["nano_banana", "flux_klein"];
const THUMBNAIL_HISTORY_LIMIT = 12;

export function clampThumbnailSeconds(value) {
  const seconds = Number(value);
  if (!Number.isFinite(seconds)) return THUMBNAIL_DEFAULT_SECONDS;
  return Math.max(THUMBNAIL_MIN_SECONDS, Math.min(THUMBNAIL_MAX_SECONDS, seconds));
}

function thumbnailText(value, limit) {
  return String(value ?? "").slice(0, limit);
}

// The saved settings in a known shape. Anything missing or wrong falls back to its default, so an old or hand-edited session loads.
export function normalizeThumbnailIntro(value) {
  const data = value && typeof value === "object" ? value : {};
  const history = (Array.isArray(data.history) ? data.history : [])
    .filter((item) => item && typeof item === "object" && String(item.path || "").trim())
    .map((item) => ({
      path: String(item.path).trim(),
      prompt: thumbnailText(item.prompt, 4000),
      engine: THUMBNAIL_ENGINES.includes(item.engine) ? item.engine : "nano_banana",
      scene_id: thumbnailText(item.scene_id, 200),
    }))
    .slice(-THUMBNAIL_HISTORY_LIMIT);
  return {
    enabled: data.enabled === true,
    image_path: thumbnailText(data.image_path, 2000).trim(),
    seconds: clampThumbnailSeconds(data.seconds ?? THUMBNAIL_DEFAULT_SECONDS),
    scene_id: thumbnailText(data.scene_id, 200),
    engine: THUMBNAIL_ENGINES.includes(data.engine) ? data.engine : "nano_banana",
    title: thumbnailText(data.title, 120),
    style_notes: thumbnailText(data.style_notes, 1200),
    prompt: thumbnailText(data.prompt, 4000),
    history,
  };
}

// The picture and length the stitch puts in front of the video, or null when it should not: this build cannot, it is switched off
// or there is no picture yet.
export function thumbnailIntroForStitch(state) {
  if (!state?.thumbnailAvailable) return null;
  const intro = normalizeThumbnailIntro(state.thumbnailIntro);
  if (!intro.enabled || !intro.image_path) return null;
  return { image_path: intro.image_path, seconds: intro.seconds };
}

// Only the full video gets the picture. A preview of a few scenes, or of a slice of the song, is left as it is.
export function stitchIsFullVideo(options = {}) {
  const sliced = Array.isArray(options.segments) && options.segments.length > 0;
  return !sliced && !Number(options.audioDuration || 0) && !Number(options.audioStart || 0) && !Number(options.timelineOffset || 0);
}

// The history after a new picture: newest last, no duplicates, limited in length.
export function thumbnailHistoryWith(history, entry) {
  const kept = (Array.isArray(history) ? history : []).filter((item) => item?.path !== entry.path);
  return normalizeThumbnailIntro({ history: [...kept, entry] }).history;
}
