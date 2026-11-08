// What the Thumbnail intro keeps in the project (`thumbnail_intro` in the session), and when the stitch uses it.
// No browser code in here, so tests/thumbnail_state.cjs can run it.

// The ways the thumbnail can be made, and how each one uses the chosen frame. "edit" (NanoBanana, API image) takes the frame as a
// reference picture and follows an instruction such as "recrop the attached image". "qwen_edit" (Qwen Image Edit) also follows an
// instruction, but one about the change only: the title lettering, then what to keep. The LLM writes a different kind of prompt for
// each kind. "overlay" is not an image workflow at all: it keeps every pixel of the frame and puts the grade and the title on top
// (no prompt, no LLM, no ComfyUI), the only way to be sure the person and the scene do not change. "api_image" sends the frame to
// the image API of the LLM Runner's provider (Gemini, Grok or OpenAI) with that provider's key. The order is the order of the list.
export const THUMBNAIL_ENGINE_KINDS = {
  overlay: "overlay",
  qwen_image_edit: "qwen_edit",
  nano_banana: "edit",
  api_image: "edit",
};
export const THUMBNAIL_ENGINES = Object.keys(THUMBNAIL_ENGINE_KINDS);
export const THUMBNAIL_DEFAULT_ENGINE = "nano_banana";
// The frame rate of the timeline: scene lengths are whole frames of it, and the frame picker steps by it.
export const THUMBNAIL_FPS = 24;
const THUMBNAIL_HISTORY_LIMIT = 12;
const THUMBNAIL_SOURCE_KINDS = ["frame", "image"];

// The Overlay engine's choices (the pack's `builder/thumbnail_overlay.py` has the same lists): the color of the title, where it goes
// ("auto" is the calmest part of the frame) and how strong the grade, the glow and the vignette are, 0 to 100.
export const THUMBNAIL_OVERLAY_COLORS = ["yellow", "white", "orange", "cyan", "red"];
export const THUMBNAIL_OVERLAY_POSITIONS = ["auto", "top", "bottom", "left", "right"];
export const THUMBNAIL_OVERLAY_DEFAULT_STRENGTH = 60;

export function normalizeThumbnailOverlay(value) {
  const data = value && typeof value === "object" ? value : {};
  const strength = Number(data.strength);
  return {
    color: THUMBNAIL_OVERLAY_COLORS.includes(data.color) ? data.color : THUMBNAIL_OVERLAY_COLORS[0],
    position: THUMBNAIL_OVERLAY_POSITIONS.includes(data.position) ? data.position : THUMBNAIL_OVERLAY_POSITIONS[0],
    strength: data.strength === "" || data.strength === null || data.strength === undefined || !Number.isFinite(strength)
      ? THUMBNAIL_OVERLAY_DEFAULT_STRENGTH : Math.max(0, Math.min(100, Math.round(strength))),
  };
}

// "edit", "qwen_edit" or "overlay". An unknown engine is treated like the default one.
export function thumbnailEngineKind(engine) {
  return THUMBNAIL_ENGINE_KINDS[engine] || THUMBNAIL_ENGINE_KINDS[THUMBNAIL_DEFAULT_ENGINE];
}

// How the title text is to look, in the words every image workflow gets. The sentence that asks for it is mandatory: see
// ensureThumbnailTitle.
const THUMBNAIL_TITLE_LOOK = "in huge bold sans-serif capital letters with a thick dark outline and a slight glow, large enough to read on a phone, in the calmest part of the frame and never covering the face";

function thumbnailTitleText(title) {
  return String(title ?? "").replace(/"/g, "'").replace(/\s+/g, " ").trim();
}

// The sentence that makes an image workflow draw the title, as an instruction. Empty when there is no title. Qwen Image Edit's carries
// the lettering style from the style notes (a graffiti style must not be asked for as "bold sans-serif capitals"), and falls back to
// bold lettering with an outline and a glow.
export function thumbnailTitleClause(title, kind, styleNotes = "") {
  const text = thumbnailTitleText(title);
  if (!text) return "";
  if (kind === "qwen_edit") {
    const look = String(styleNotes ?? "").replace(/\s+/g, " ").trim().replace(/[.\s]+$/, "");
    return `Add the text "${text}" as ${look ? `lettering in this style: ${look}` : "bold, high-contrast lettering with a thick outline and a glow"}, large enough to read on a phone, in a calm part of the frame, fully inside the frame and never covering the face.`;
  }
  return `Add the title text "${text}" exactly as written and spelled ${THUMBNAIL_TITLE_LOOK}. Add no other text.`;
}

function escapeForPattern(text) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// The prompt with the title text in it, always. An image workflow only draws a title that its prompt asks for, and an LLM can leave
// it out or a prompt can be edited, so this runs before every generation: a title that is already in the prompt in quotes is put
// back to exactly the spelling and case that was typed, and one that is missing gets the mandatory sentence at the end. No title or
// no prompt changes nothing.
export function ensureThumbnailTitle(prompt, title, kind, styleNotes = "") {
  const body = String(prompt ?? "").trim();
  const text = thumbnailTitleText(title);
  if (!text || !body) return body;
  const words = text.split(" ").map(escapeForPattern).join("\\s+");
  const quoted = new RegExp(`["\\u201C\\u201D'\\u2018\\u2019]\\s*${words}\\s*["\\u201C\\u201D'\\u2018\\u2019]`, "i");
  if (quoted.test(body)) return body.replace(quoted, () => `"${text}"`);
  return `${body} ${thumbnailTitleClause(text, kind, styleNotes)}`;
}

function thumbnailText(value, limit) {
  return String(value ?? "").slice(0, limit);
}

// The API image engine's one choice: the image model, typed or picked from the provider's list. Empty means the provider's first one.
export function normalizeThumbnailApiImage(value) {
  const data = value && typeof value === "object" ? value : {};
  return { model: thumbnailText(data.model, 120).trim() };
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
    overlay: normalizeThumbnailOverlay(data.overlay),
    api_image: normalizeThumbnailApiImage(data.api_image),
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
