// The rules of the effect lane and the filter lane, without the page: what a saved block is, which frames it covers, how far it can be moved and stretched, and
// what the stitch is sent. look_lane.mjs draws and listens, this file decides. The H3 Studio app's `plan` (crates/h3-media, looks.rs) is the same frame rule for the
// final video, so the player shows the frames the export draws: both run `tests/look_frame_cases.json`.
//
// A block is `{id, look, start, seconds, intensity, params}`: a stretch of the timeline like an overlay, `start` seconds in and `seconds` long. `look` names an effect
// or a filter, `intensity` (0 to 1) is how much of its change is mixed in and `params` are the settings changed from their defaults. Blocks may overlap: a frame
// gets every filter that covers it, then every effect, each in the order of the timeline.

export const LOOK_KINDS = {
  filter: {
    kind: "filter", plural: "filters", singular: "filter", label: "Filters", laneLabel: "FILTERS", sessionKey: "video_filters", listKey: "videoFilters", availableKey: "filtersAvailable",
    activeKey: "activeFilterId", idPrefix: "f", color: "#047857", border: "#6ee7b7", text: "#a7f3d0", defaultSeconds: 3,
    blurb: "A filter grades the colours of the video where its block lies: a look such as teal and orange, black and white or faded film, with the settings of a grade.",
  },
  effect: {
    kind: "effect", plural: "effects", singular: "effect", label: "Effects", laneLabel: "EFFECTS", sessionKey: "video_effects", listKey: "videoEffects", availableKey: "effectsAvailable",
    activeKey: "activeEffectId", idPrefix: "e", color: "#c2410c", border: "#fdba74", text: "#fed7aa", defaultSeconds: 2,
    blurb: "An effect adds or bends something where its block lies: a glitch, a shake, a blur, a flash, a zoom pulse.",
  },
};
// The order the kinds are applied: filters first.
export const LOOK_ORDER = ["filter", "effect"];

export const LOOK_FPS = 24;
export const LOOK_MIN_SECONDS = 0.05;
export const LOOK_MAX_SECONDS = 86400;

const lookRound3 = (value) => Math.round(value * 1000) / 1000;
const lookClamp = (value, low, high) => Math.min(high, Math.max(low, value));
const lookNumber = (value, fallback) => (Number.isFinite(Number(value)) && value !== null && value !== "" ? Number(value) : fallback);
export const lookFrameOf = (seconds) => Math.floor(Math.max(0, Number(seconds) || 0) * LOOK_FPS + 0.5);
export const lookSnapToFrame = (seconds) => Math.round((Number(seconds) || 0) * LOOK_FPS) / LOOK_FPS;

// The list as it is kept: every block has its own id, a length and an intensity in range and a start, in the order of the timeline. What cannot be read is
// dropped. `kind` is a key of LOOK_KINDS.
export function normalizeLookEntries(list, kind) {
  const config = LOOK_KINDS[kind];
  const read = [];
  for (const raw of Array.isArray(list) ? list : []) {
    if (!raw || typeof raw !== "object") continue;
    const look = String(raw.look || "").trim();
    const start = raw.start === undefined || raw.start === null ? NaN : lookNumber(raw.start, NaN);
    if (!look || !Number.isFinite(start)) continue;
    read.push({
      id: String(raw.id || "").trim(),
      look,
      start: lookRound3(Math.max(0, start)),
      seconds: lookRound3(lookClamp(lookNumber(raw.seconds, 1), LOOK_MIN_SECONDS, LOOK_MAX_SECONDS)),
      intensity: lookRound3(lookClamp(lookNumber(raw.intensity, 1), 0, 1)),
      params: raw.params && typeof raw.params === "object" ? { ...raw.params } : {},
    });
  }
  const used = new Set();
  for (const entry of read) {
    if (entry.id && !used.has(entry.id)) used.add(entry.id);
    else entry.id = "";
  }
  let counter = 0;
  for (const entry of read) {
    if (entry.id) continue;
    do counter += 1; while (used.has(`${config.idPrefix}${counter}`));
    entry.id = `${config.idPrefix}${counter}`;
    used.add(entry.id);
  }
  return read.sort((a, b) => a.start - b.start);
}

export const lookEnd = (entry) => lookRound3(entry.start + entry.seconds);

// Which row each block sits in. Blocks that overlap in time are stacked in rows so every one can be read and grabbed: a block takes the first row whose last block has
// ended.
export function lookRows(list) {
  const rows = {};
  const ends = [];
  for (const entry of list) {
    let row = ends.findIndex((end) => end <= entry.start + 0.001);
    if (row < 0) row = ends.length;
    ends[row] = lookEnd(entry);
    rows[entry.id] = row;
  }
  return { rows, count: Math.max(1, ends.length) };
}

// The frames of the joined video a block covers, `[first, last)`, with `total` frames in the video (0 for no limit), or null when it covers none.
export function lookFrames(entry, total = 0) {
  const first = lookFrameOf(entry.start);
  const wanted = Math.max(first + 1, lookFrameOf(entry.start + entry.seconds));
  const last = total > 0 ? Math.min(total, wanted) : wanted;
  return total > 0 && first >= total || last <= first ? null : { first, last };
}

// How far a block has run on `frame` of the timeline, as the final video gets it: `progress` runs over (0, 1) across the block and `time` is the seconds since it
// started. Null when the block does not cover the frame.
export function lookStateAt(entry, frame, total = 0) {
  const frames = lookFrames(entry, total);
  if (!frames || frame < frames.first || frame >= frames.last) return null;
  return { progress: (frame - frames.first + 0.5) / (frames.last - frames.first), time: (frame - frames.first) / LOOK_FPS };
}

// The start `wanted` for a block, kept inside the timeline (`total` seconds, 0 for no limit).
export function moveLook(entry, wanted, total = 0) {
  const limit = total > 0 ? Math.max(0, total - entry.seconds) : Infinity;
  return lookRound3(lookClamp(wanted, 0, limit));
}

// The start edge of a block dragged to `wanted`: the length changes, the end stays.
export function trimLookStart(entry, wanted) {
  const end = lookEnd(entry);
  const start = lookClamp(wanted, Math.max(0, end - LOOK_MAX_SECONDS), end - LOOK_MIN_SECONDS);
  return { start: lookRound3(start), seconds: lookRound3(end - start) };
}

// The end edge of a block dragged to `wanted`: the start stays.
export function trimLookEnd(entry, wanted, total = 0) {
  const limit = Math.min(total > 0 ? total : Infinity, entry.start + LOOK_MAX_SECONDS);
  const end = lookClamp(wanted, entry.start + LOOK_MIN_SECONDS, Math.max(entry.start + LOOK_MIN_SECONDS, limit));
  return { start: entry.start, seconds: lookRound3(end - entry.start) };
}

// Where a new block of `seconds` goes: over the selected range of the timeline when there is one, else over `span` (a scene: `{start, end}`) when one is asked for,
// else at the playhead. Kept inside the timeline (`total` seconds, 0 for no limit).
export function placeNewLook({ playhead = 0, seconds = 2, total = 0, range = null, span = null } = {}) {
  let start = Math.max(0, playhead);
  let length = lookClamp(seconds, LOOK_MIN_SECONDS, LOOK_MAX_SECONDS);
  const area = range && Number(range.end) - Number(range.start) >= LOOK_MIN_SECONDS ? range : span;
  if (area && Number(area.end) - Number(area.start) >= LOOK_MIN_SECONDS) {
    start = Number(area.start);
    length = Number(area.end) - Number(area.start);
  } else if (total > 0 && start + length > total) {
    start = Math.max(0, total - length);
    length = Math.min(length, total);
  }
  return { start: lookRound3(lookSnapToFrame(Math.max(0, start))), seconds: lookRound3(Math.max(LOOK_MIN_SECONDS, lookSnapToFrame(length))) };
}

// The blocks of a stitch of `baseSegments`, on the clock of its scene timing (`offset` is the whole-frame shift the Builder takes off every time of the stitch). A
// block that lies on none of the stitched scenes is left out, and one that starts before the first scene is cut at its start.
export function lookStitchItems(list, baseSegments, offset = 0) {
  const items = [];
  for (const entry of list) {
    const end = entry.start + entry.seconds;
    if (!baseSegments.some((segment) => entry.start < lookNumber(segment.end, 0) && end > lookNumber(segment.start, 0))) continue;
    const start = Math.max(0, entry.start - offset);
    const seconds = end - offset - start;
    if (seconds < LOOK_MIN_SECONDS - 0.0005) continue;
    items.push({ look: entry.look, start: lookRound3(start), seconds: lookRound3(Math.max(LOOK_MIN_SECONDS, seconds)), intensity: entry.intensity, params: entry.params || {} });
  }
  return items;
}
