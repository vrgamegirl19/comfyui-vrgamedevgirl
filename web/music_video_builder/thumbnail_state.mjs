// What the Thumbnail intro keeps in the project (`thumbnail_intro` in the session), and when the stitch uses it.
// No browser code in here, so tests/thumbnail_state.cjs can run it.

export const THUMBNAIL_MIN_SECONDS = 0.5;
export const THUMBNAIL_MAX_SECONDS = 10;
export const THUMBNAIL_DEFAULT_SECONDS = 2;
export const THUMBNAIL_ENGINES = ["nano_banana", "flux_klein"];
// The frame rate of the timeline: scene lengths are whole frames of it, and the frame picker steps by it.
export const THUMBNAIL_FPS = 24;
const THUMBNAIL_HISTORY_LIMIT = 12;
const THUMBNAIL_SOURCE_KINDS = ["frame", "image"];

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
  // The picture the thumbnail is made from: one exact frame of a scene video ("frame") or a scene's image ("image"). The frame
  // number and seconds are where on the timeline it was taken, so the picker opens on the same moment next time.
  const sourcePath = thumbnailText(data.source_path, 2000).trim();
  const frame = Number(data.source_frame);
  const sourceSeconds = Number(data.source_seconds);
  return {
    enabled: data.enabled === true,
    image_path: thumbnailText(data.image_path, 2000).trim(),
    seconds: clampThumbnailSeconds(data.seconds ?? THUMBNAIL_DEFAULT_SECONDS),
    source_path: sourcePath,
    source_kind: sourcePath && THUMBNAIL_SOURCE_KINDS.includes(data.source_kind) ? data.source_kind : (sourcePath ? "image" : ""),
    source_frame: Number.isFinite(frame) && frame > 0 ? Math.floor(frame) : 0,
    source_seconds: Number.isFinite(sourceSeconds) && sourceSeconds > 0 ? sourceSeconds : 0,
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

// The scenes the frame picker moves through, in timeline order, each with the whole frames it covers. A scene needs a length and
// something to show: `video` (a rendered scene video) or `image` (its scene image). The rest of the scene is kept as given.
export function thumbnailTimelineItems(scenes, fps = THUMBNAIL_FPS) {
  const items = [];
  for (const scene of Array.isArray(scenes) ? scenes : []) {
    const start = Number(scene?.start);
    const end = Number(scene?.end);
    if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start || start < 0) continue;
    if (!String(scene.video || "").trim() && !String(scene.image || "").trim()) continue;
    const startFrame = Math.round(start * fps);
    items.push({ ...scene, start, end, startFrame, endFrame: Math.max(startFrame + 1, Math.round(end * fps)) });
  }
  return items.sort((a, b) => a.startFrame - b.startFrame);
}

// The first and last frame the picker can stop on.
export function thumbnailFrameRange(items) {
  if (!items.length) return { min: 0, max: 0 };
  return { min: items[0].startFrame, max: Math.max(...items.map((item) => item.endFrame)) - 1 };
}

// The scene that shows `frame`. In a gap between scenes it is the next scene, and past the last one the last scene.
export function thumbnailItemAt(items, frame) {
  return items.find((item) => frame >= item.startFrame && frame < item.endFrame)
    || items.find((item) => item.startFrame > frame)
    || items[items.length - 1]
    || null;
}

// How far into the scene's own video `frame` is, in seconds. The scene video starts at the scene's first frame.
export function thumbnailLocalSeconds(item, frame, fps = THUMBNAIL_FPS) {
  const inside = Math.min(Math.max(frame, item.startFrame), item.endFrame - 1);
  return (inside - item.startFrame) / fps;
}

// `1:05.50`: minutes, then seconds with hundredths.
export function formatThumbnailTime(seconds) {
  const total = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(total / 60);
  return `${minutes}:${(total - minutes * 60).toFixed(2).padStart(5, "0")}`;
}
