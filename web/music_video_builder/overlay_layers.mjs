// Overlay data: the layers of the overlay lane (media overlays such as particles and light leaks, film grain, vignette, text, a
// whole-timeline LUT) and the track switch. Both are saved in the project session as `overlay_layers` and `overlay_layer_track`.
// Every value is clamped and every missing one gets its default, so a clip from an old project, an agent or a typo is
// corrected instead of rejected. The Rust twin is `h3-core/src/overlay_layers.rs` in the app. Both run the cases in
// `tests/overlay_layer_cases.json`, so change one and the cases together.

export const OVERLAY_KINDS = ["media", "grain", "vignette", "text", "lut"];
export const MIN_OVERLAY_SECONDS = 0.05;
export const MAX_OVERLAY_SECONDS = 86400;
export const MAX_OVERLAY_LAYERS = 500;
export const OVERLAY_BLEND_MODES = ["normal", "screen", "add", "lighten", "overlay", "softlight", "multiply"];
export const OVERLAY_SCALE_MODES = ["cover", "contain", "stretch"];
export const OVERLAY_GRAIN_SIZES = ["fine", "medium", "coarse"];
export const OVERLAY_TEXT_ANIMATIONS = ["none", "slide_up"];

const OVERLAY_COMMON_FIELDS = ["id", "kind", "start", "end", "enabled", "locked", "opacity", "fade_in", "fade_out"];
const OVERLAY_KIND_FIELDS = {
  media: ["path", "source_start", "blend_mode", "scale", "loop", "speed", "flip_h", "brightness", "contrast", "tint"],
  grain: ["strength", "chroma", "size", "animated"],
  vignette: ["strength"],
  text: ["text", "font", "size", "color", "outline_width", "outline_color", "shadow", "x", "y", "animation"],
  lut: ["lut", "strength"],
};

// Half away from zero to three decimals, so the JS, Rust and Python twins agree on every value.
function overlayRound3(value) {
  const rounded = Math.sign(value) * Math.round(Math.abs(value) * 1000) / 1000;
  return rounded === 0 ? 0 : rounded;
}

function overlayNumber(value, fallback, low, high) {
  let parsed = fallback;
  if (typeof value === "number") parsed = value;
  else if (typeof value === "string" && /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/.test(value.trim())) parsed = Number(value);
  if (!Number.isFinite(parsed)) parsed = fallback;
  return overlayRound3(Math.min(high, Math.max(low, parsed)));
}

function overlayFlag(value, fallback) {
  return typeof value === "boolean" ? value : fallback;
}

function overlayChoice(value, allowed, fallback) {
  const text = typeof value === "string" ? value.trim().toLowerCase() : "";
  return allowed.includes(text) ? text : fallback;
}

function overlayColor(value, fallback) {
  const text = typeof value === "string" ? value.trim().toLowerCase() : "";
  if (/^#[0-9a-f]{6}$/.test(text)) return text;
  if (/^#[0-9a-f]{3}$/.test(text)) return `#${text[1]}${text[1]}${text[2]}${text[2]}${text[3]}${text[3]}`;
  return fallback;
}

function overlayWords(value, limit, fallback = "") {
  if (typeof value !== "string") return fallback;
  return Array.from(value).slice(0, limit).join("");
}

function overlayTrimmedWords(value, limit, fallback = "") {
  const text = overlayWords(value, limit, fallback).trim();
  return text || fallback;
}

function overlayIsObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

// One clip, or null when it is not an object or has no kind. `index` is its place in the saved list and `usedIds`
// the ids already taken, so a missing or repeated id becomes overlay_<n> or <id>_2, <id>_3.
export function normalizeOverlayLayer(source, index = 0, usedIds = new Set()) {
  if (!overlayIsObject(source)) return null;
  const kind = typeof source.kind === "string" ? source.kind.trim().toLowerCase() : "";
  if (!kind) return null;
  const clip = {};
  const known = new Set([...OVERLAY_COMMON_FIELDS, ...(OVERLAY_KIND_FIELDS[kind] || [])]);
  for (const [key, value] of Object.entries(source)) {
    if (!known.has(key)) clip[key] = value;
  }
  let id = overlayTrimmedWords(source.id, 80, `overlay_${index + 1}`);
  for (let n = 2; usedIds.has(id); n += 1) id = `${overlayTrimmedWords(source.id, 80, `overlay_${index + 1}`)}_${n}`;
  usedIds.add(id);
  const start = overlayNumber(source.start, 0, 0, MAX_OVERLAY_SECONDS);
  const end = Math.max(overlayRound3(start + MIN_OVERLAY_SECONDS), overlayNumber(source.end, start + MIN_OVERLAY_SECONDS, 0, MAX_OVERLAY_SECONDS));
  const duration = overlayRound3(end - start);
  let fadeIn = overlayNumber(source.fade_in, 0.3, 0, 5);
  let fadeOut = overlayNumber(source.fade_out, 0.3, 0, 5);
  if (fadeIn + fadeOut > duration) {
    const scale = duration / (fadeIn + fadeOut);
    fadeIn = overlayRound3(fadeIn * scale);
    fadeOut = overlayRound3(fadeOut * scale);
  }
  Object.assign(clip, {
    id, kind, start, end,
    enabled: overlayFlag(source.enabled, true),
    locked: overlayFlag(source.locked, true),
    opacity: overlayNumber(source.opacity, kind === "media" ? 0.9 : 1, 0, 1),
    fade_in: fadeIn,
    fade_out: fadeOut,
  });
  if (kind === "media") {
    Object.assign(clip, {
      path: overlayTrimmedWords(source.path, 1024),
      source_start: overlayNumber(source.source_start, 0, 0, MAX_OVERLAY_SECONDS),
      blend_mode: overlayChoice(source.blend_mode, OVERLAY_BLEND_MODES, "screen"),
      scale: overlayChoice(source.scale, OVERLAY_SCALE_MODES, "cover"),
      loop: overlayFlag(source.loop, true),
      speed: overlayNumber(source.speed, 1, 0.5, 2),
      flip_h: overlayFlag(source.flip_h, false),
      brightness: overlayNumber(source.brightness, 0, -1, 1),
      contrast: overlayNumber(source.contrast, 1, 0, 2),
      tint: overlayColor(source.tint, ""),
    });
  } else if (kind === "grain") {
    Object.assign(clip, {
      strength: overlayNumber(source.strength, 28, 0, 60),
      chroma: overlayNumber(source.chroma, 0, 0, 20),
      size: overlayChoice(source.size, OVERLAY_GRAIN_SIZES, "fine"),
      animated: overlayFlag(source.animated, true),
    });
  } else if (kind === "vignette") {
    clip.strength = overlayNumber(source.strength, 0.7, 0, 1);
  } else if (kind === "text") {
    Object.assign(clip, {
      text: overlayWords(source.text, 500),
      font: overlayTrimmedWords(source.font, 80, "Anton"),
      size: overlayNumber(source.size, 96, 8, 400),
      color: overlayColor(source.color, "#ffffff"),
      outline_width: overlayNumber(source.outline_width, 3, 0, 20),
      outline_color: overlayColor(source.outline_color, "#000000"),
      shadow: overlayNumber(source.shadow, 4, 0, 20),
      x: overlayNumber(source.x, 0.5, 0, 1),
      y: overlayNumber(source.y, 0.78, 0, 1),
      animation: overlayChoice(source.animation, OVERLAY_TEXT_ANIMATIONS, "none"),
    });
  } else if (kind === "lut") {
    Object.assign(clip, {
      lut: overlayTrimmedWords(source.lut, 260),
      strength: overlayNumber(source.strength, 1, 0, 1),
    });
  }
  return clip;
}

// The saved list, in track order (later clips draw on top). Anything that is not a clip is dropped.
export function normalizeOverlayLayers(source) {
  if (!Array.isArray(source)) return [];
  const usedIds = new Set();
  const clips = [];
  source.forEach((item, index) => {
    if (clips.length >= MAX_OVERLAY_LAYERS) return;
    const clip = normalizeOverlayLayer(item, index, usedIds);
    if (clip) clips.push(clip);
  });
  return clips;
}

// The switch for all overlays. They are on until it is turned off, since adding one means it should show.
export function normalizeOverlayLayerTrack(source) {
  const track = overlayIsObject(source) ? { ...source } : {};
  track.enabled = overlayFlag(track.enabled, true);
  return track;
}
