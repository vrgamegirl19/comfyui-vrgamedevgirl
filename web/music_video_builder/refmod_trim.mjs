// Pure helpers for the RefMods Studio trim: find the subject's box in an image, keep a crop box inside the image,
// and work out the canvas and token count. No imports, so they can be tested with Node.
// Tokens follow the RefMod pack: frames x (canvas width / 32) x (canvas height / 32).
// The Python twin is `canvas_for` in minimax/refmod_studio.py. Keep the two in step.

export const TRIM_BACKGROUND_THRESHOLD = 245;
export const REF_TOKEN_CAP = 5120;
export const MIN_CROP_SIZE = 32;

export function snap32(value) {
  return Math.max(32, Math.floor(Number(value) / 32 + 0.5) * 32);
}

// Tight box (x1 and y1 are exclusive) around the pixels that are not background, or null when none are found.
// `data` is RGBA. A pixel is background when it is transparent or when every colour channel is at or above
// `threshold` (near white). A row or column only counts when enough of its pixels are subject, so JPEG noise in
// a white background does not stretch the box.
export function trimBoxFromPixels(data, width, height, { threshold = TRIM_BACKGROUND_THRESHOLD, minFraction = 0.004 } = {}) {
  if (!data || width < 1 || height < 1) return null;
  const rows = new Uint32Array(height);
  const cols = new Uint32Array(width);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const i = (y * width + x) * 4;
      if (data[i + 3] < 16) continue;
      if (data[i] < threshold || data[i + 1] < threshold || data[i + 2] < threshold) {
        rows[y] += 1;
        cols[x] += 1;
      }
    }
  }
  const minRow = Math.max(2, Math.round(width * minFraction));
  const minCol = Math.max(2, Math.round(height * minFraction));
  let y0 = 0;
  while (y0 < height && rows[y0] < minRow) y0 += 1;
  let y1 = height;
  while (y1 > y0 && rows[y1 - 1] < minRow) y1 -= 1;
  let x0 = 0;
  while (x0 < width && cols[x0] < minCol) x0 += 1;
  let x1 = width;
  while (x1 > x0 && cols[x1 - 1] < minCol) x1 -= 1;
  if (x1 <= x0 || y1 <= y0) return null;
  return { x0, y0, x1, y1 };
}

// Grow a box by `margin` pixels on every side, kept inside the image.
export function expandBox(box, margin, width, height) {
  const m = Math.max(0, Number(margin) || 0);
  const source = box || { x0: 0, y0: 0, x1: width, y1: height };
  return {
    x0: Math.max(0, source.x0 - m),
    y0: Math.max(0, source.y0 - m),
    x1: Math.min(width, source.x1 + m),
    y1: Math.min(height, source.y1 + m),
  };
}

// Keep a crop box inside the image and at least `minSize` wide and tall.
export function clampBox(box, width, height, minSize = MIN_CROP_SIZE) {
  const size = Math.min(minSize, width, height);
  let x0 = Math.min(Math.max(0, box.x0), width - size);
  let y0 = Math.min(Math.max(0, box.y0), height - size);
  let x1 = Math.max(Math.min(width, box.x1), x0 + size);
  let y1 = Math.max(Math.min(height, box.y1), y0 + size);
  x0 = Math.min(x0, x1 - size);
  y0 = Math.min(y0, y1 - size);
  return { x0, y0, x1, y1 };
}

// Quality presets. A preset is how much of the original detail is kept: every image is scaled by `scale` (never
// above 1, and never past the 1024 px canvas limit). Users see the name and the description, not the numbers.
// The Python twin is QUALITY_SCALES in minimax/refmod_studio.py. Keep the two in step.
export const REF_CANVAS_LIMIT = 1024;
// The video VAE needs at least 320 px on both sides, and the RefMod pack scales smaller images up to reach it. The canvas
// is padded to this size instead, so the image keeps the chosen scale.
export const MIN_CANVAS_SIDE = 320;
export const QUALITY_PRESETS = [
  { key: "maximum", label: "Maximum", scale: 1, detail: "Keeps every pixel (up to the 1024 px limit). Best for fine print, patterns and face detail. Uses the most tokens." },
  { key: "high", label: "High", scale: 0.8, detail: "Very close to the originals. Fine print stays readable." },
  { key: "balanced", label: "Balanced", scale: 0.6, detail: "Faces and clothing keep their detail. Very small text may soften. Recommended." },
  { key: "compact", label: "Compact", scale: 0.45, detail: "Faces stay recognisable. Small patterns and text soften. Good when a scene has several characters." },
  { key: "draft", label: "Draft", scale: 0.3, detail: "Soft detail, smallest size. For quick tests." },
];
export const DEFAULT_QUALITY = "balanced";

export function qualityScale(key) {
  return (QUALITY_PRESETS.find((preset) => preset.key === key) || QUALITY_PRESETS.find((preset) => preset.key === DEFAULT_QUALITY)).scale;
}

// One canvas that holds every image without cutting any: the widest width by the tallest height, each scaled by the
// quality (never up, never past the canvas limit), then snapped to multiples of 32 and padded up to the 320 px minimum.
export function canvasFor(sizes, scale = 1, limit = REF_CANVAS_LIMIT) {
  const widest = Math.max(...sizes.map((size) => size.width));
  const tallest = Math.max(...sizes.map((size) => size.height));
  const fit = Math.min(1, scale, limit / Math.max(widest, tallest));
  return { width: Math.max(MIN_CANVAS_SIDE, snap32(widest * fit)), height: Math.max(MIN_CANVAS_SIDE, snap32(tallest * fit)), fit };
}

function summary(sizes, scale) {
  const canvas = canvasFor(sizes, scale);
  const perFrame = (canvas.width / 32) * (canvas.height / 32);
  return { frames: sizes.length, canvas: [canvas.width, canvas.height], tokens: sizes.length * perFrame, fit: canvas.fit };
}

// What creating the mod costs.
//   sizes:     natural { width, height } of each image, in load order
//   cropSizes: { width, height } of each crop box, same order (the whole image where there is no crop)
//   quality:   a preset key
// `whole` is the images uncropped and `trimmed` is the crop boxes, both at the chosen quality. `byQuality` lists the
// trimmed token count at every preset so the trade-off is visible.
export function estimateTokens({ sizes, cropSizes, quality = DEFAULT_QUALITY }) {
  if (!sizes?.length) return null;
  const crops = cropSizes?.length === sizes.length ? cropSizes : sizes;
  const scale = qualityScale(quality);
  return {
    whole: summary(sizes, scale),
    trimmed: summary(crops, scale),
    byQuality: QUALITY_PRESETS.map((preset) => ({ key: preset.key, label: preset.label, tokens: summary(crops, preset.scale).tokens })),
    cap: REF_TOKEN_CAP,
  };
}
