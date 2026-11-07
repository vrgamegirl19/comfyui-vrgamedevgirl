// What the Thumbnail intro keeps in the project (`thumbnail_intro` in the session), and when the stitch uses it.
// No browser code in here, so tests/thumbnail_state.cjs can run it.

// The image workflows the thumbnail can be made with, and how each one uses the chosen frame. "edit" workflows (NanoBanana,
// Flux/Klein) take the frame as a reference picture and follow an instruction such as "recrop the attached image". "moss_edit"
// (MOSS-Image Edit) also follows an instruction, but one in the short form its edit LoRA was trained on ("Add ... Keep the ...
// unchanged."). "restyle" workflows (Krea 2, Z-Image, Ernie, MOSS-Image) start from the frame as an image-to-image source and need a
// prompt that describes the finished picture. The LLM writes a different kind of prompt for each kind.
export const THUMBNAIL_ENGINE_KINDS = {
  nano_banana: "edit",
  flux_klein: "edit",
  moss_image_edit: "moss_edit",
  krea2_2pass: "restyle",
  zimage: "restyle",
  ernie_image: "restyle",
  moss_image: "restyle",
};
export const THUMBNAIL_ENGINES = Object.keys(THUMBNAIL_ENGINE_KINDS);
export const THUMBNAIL_DEFAULT_ENGINE = "nano_banana";
export const THUMBNAIL_DEFAULT_CHANGE = 0.65;
// The frame rate of the timeline: scene lengths are whole frames of it, and the frame picker steps by it.
export const THUMBNAIL_FPS = 24;
const THUMBNAIL_HISTORY_LIMIT = 12;
const THUMBNAIL_SOURCE_KINDS = ["frame", "image"];

// "edit", "moss_edit" or "restyle". An unknown engine is treated like the default one.
export function thumbnailEngineKind(engine) {
  return THUMBNAIL_ENGINE_KINDS[engine] || THUMBNAIL_ENGINE_KINDS[THUMBNAIL_DEFAULT_ENGINE];
}

// How the title text is to look, in the words every image workflow gets. The sentence that asks for it is mandatory: see
// ensureThumbnailTitle.
const THUMBNAIL_TITLE_LOOK = "in huge bold sans-serif capital letters with a thick dark outline and a slight glow, large enough to read on a phone, in the calmest part of the frame and never covering the face";

function thumbnailTitleText(title) {
  return String(title ?? "").replace(/"/g, "'").replace(/\s+/g, " ").trim();
}

// The sentence that makes an image workflow draw the title: an instruction for the edit workflows, a statement of the finished
// picture for the restyle ones. Empty when there is no title.
export function thumbnailTitleClause(title, kind) {
  const text = thumbnailTitleText(title);
  if (!text) return "";
  return kind === "restyle"
    ? `The title text "${text}" is written exactly as spelled ${THUMBNAIL_TITLE_LOOK}. No other text.`
    : `Add the title text "${text}" exactly as written and spelled ${THUMBNAIL_TITLE_LOOK}. Add no other text.`;
}

function escapeForPattern(text) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// The prompt with the title text in it, always. An image workflow only draws a title that its prompt asks for, and an LLM can leave
// it out or a prompt can be edited, so this runs before every generation: a title that is already in the prompt in quotes is put
// back to exactly the spelling and case that was typed, and one that is missing gets the mandatory sentence at the end. No title or
// no prompt changes nothing.
export function ensureThumbnailTitle(prompt, title, kind) {
  const body = String(prompt ?? "").trim();
  const text = thumbnailTitleText(title);
  if (!text || !body) return body;
  const words = text.split(" ").map(escapeForPattern).join("\\s+");
  const quoted = new RegExp(`["\\u201C\\u201D'\\u2018\\u2019]\\s*${words}\\s*["\\u201C\\u201D'\\u2018\\u2019]`, "i");
  if (quoted.test(body)) return body.replace(quoted, () => `"${text}"`);
  return `${body} ${thumbnailTitleClause(text, kind)}`;
}

// How much a restyle workflow may change the frame, 0 to 1 (0.65 by default).
export function clampThumbnailChange(value) {
  const change = Number(value);
  if (value === "" || value === null || value === undefined || !Number.isFinite(change)) return THUMBNAIL_DEFAULT_CHANGE;
  return Math.max(0, Math.min(1, change));
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
      engine: THUMBNAIL_ENGINES.includes(item.engine) ? item.engine : THUMBNAIL_DEFAULT_ENGINE,
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
    source_path: sourcePath,
    source_kind: sourcePath && THUMBNAIL_SOURCE_KINDS.includes(data.source_kind) ? data.source_kind : (sourcePath ? "image" : ""),
    source_frame: Number.isFinite(frame) && frame > 0 ? Math.floor(frame) : 0,
    source_seconds: Number.isFinite(sourceSeconds) && sourceSeconds > 0 ? sourceSeconds : 0,
    scene_id: thumbnailText(data.scene_id, 200),
    engine: THUMBNAIL_ENGINES.includes(data.engine) ? data.engine : THUMBNAIL_DEFAULT_ENGINE,
    change: clampThumbnailChange(data.change),
    title: thumbnailText(data.title, 120),
    style_notes: thumbnailText(data.style_notes, 1200),
    prompt: thumbnailText(data.prompt, 4000),
    history,
  };
}

// The picture the stitch makes the first frame of the video, or null when it should not: this build cannot, it is switched off or
// there is no picture yet. It is one frame, never a length of time, so there is nothing else to say about it.
export function thumbnailIntroForStitch(state) {
  if (!state?.thumbnailAvailable) return null;
  const intro = normalizeThumbnailIntro(state.thumbnailIntro);
  if (!intro.enabled || !intro.image_path) return null;
  return { image_path: intro.image_path };
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
