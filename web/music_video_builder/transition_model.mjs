// The rules of the transition lane, without the page: what a saved entry is, where a block lies on the timeline, how far it can be moved and
// stretched, and what the stitch is sent. transition_lane.mjs draws and listens, this file decides. The H3 Studio app's `plan` (crates/h3-media,
// transition.rs) is the same window rule for the final video, so the player shows the frames the export draws: both run
// `tests/transition_window_cases.json`.
//
// An entry is `{id, transition, start, seconds, params}`: a block of the timeline like an overlay, `start` seconds in and `seconds` long. An entry of an
// earlier version, `{after_scene, transition, seconds, align, params}`, is read as the block it stood for.

export const TRANSITION_FPS = 24;
export const TRANSITION_MIN_SECONDS = 0.1;
export const TRANSITION_MAX_SECONDS = 10;
export const TRANSITION_DEFAULT_SECONDS = 0.5;
// A block placed within this many seconds of a cut is centered on the cut.
const TRANSITION_CUT_REACH_SECONDS = 0.5;
const TRANSITION_SNAP_PIXELS = 8;

const transitionRound3 = (value) => Math.round(value * 1000) / 1000;
const transitionClamp = (value, low, high) => Math.min(high, Math.max(low, value));
export const transitionFrameOf = (seconds) => Math.floor(Math.max(0, Number(seconds) || 0) * TRANSITION_FPS + 0.5);
export const transitionSnapToFrame = (seconds) => Math.round((Number(seconds) || 0) * TRANSITION_FPS) / TRANSITION_FPS;
const transitionNumber = (value, fallback) => (Number.isFinite(Number(value)) && value !== null && value !== "" ? Number(value) : fallback);

// -- reading entries ---------------------------------------------------------------------------------------------------------

// Where an earlier version's entry starts: its length placed around the end of the scene before the cut the way `align` said.
function transitionStartOnCut(segment, seconds, align) {
  const frames = Math.round(seconds * TRANSITION_FPS);
  const before = align === "end" ? frames : align === "start" ? 0 : Math.floor(frames / 2);
  return Math.max(0, transitionNumber(segment.end, 0) - before / TRANSITION_FPS);
}

// The list as it is kept: every entry has its own id, a length in range and a start, in the order of the timeline. What cannot be read is dropped.
export function normalizeTransitionEntries(list, segments = []) {
  const read = [];
  for (const raw of Array.isArray(list) ? list : []) {
    if (!raw || typeof raw !== "object") continue;
    const transition = String(raw.transition || "").trim();
    if (!transition) continue;
    const seconds = transitionClamp(transitionNumber(raw.seconds, TRANSITION_DEFAULT_SECONDS), TRANSITION_MIN_SECONDS, TRANSITION_MAX_SECONDS);
    let start = raw.start === undefined || raw.start === null ? NaN : transitionNumber(raw.start, NaN);
    if (!Number.isFinite(start)) {
      const segment = segments.find((item) => String(item.id) === String(raw.after_scene));
      if (!segment) continue;
      start = transitionStartOnCut(segment, seconds, String(raw.align || "").toLowerCase());
    }
    read.push({ id: String(raw.id || "").trim(), transition, start: transitionRound3(Math.max(0, start)), seconds: transitionRound3(seconds), params: raw.params && typeof raw.params === "object" ? { ...raw.params } : {} });
  }
  const used = new Set();
  for (const entry of read) {
    if (entry.id && !used.has(entry.id)) used.add(entry.id);
    else entry.id = "";
  }
  let counter = 0;
  for (const entry of read) {
    if (entry.id) continue;
    do counter += 1; while (used.has(`t${counter}`));
    entry.id = `t${counter}`;
    used.add(entry.id);
  }
  return read.sort((a, b) => a.start - b.start);
}

export const transitionEnd = (entry) => transitionRound3(entry.start + entry.seconds);

// Which row each block sits in. Blocks that overlap (agents can save them so) are stacked in rows so every one can be read and grabbed.
export function transitionRows(list) {
  const rows = {};
  const ends = [];
  for (const entry of list) {
    let row = ends.findIndex((end) => end <= entry.start + 0.001);
    if (row < 0) row = ends.length;
    ends[row] = transitionEnd(entry);
    rows[entry.id] = row;
  }
  return { rows, count: Math.max(1, ends.length) };
}

// -- where a block lies ------------------------------------------------------------------------------------------------------

// The frames of the joined video a block replaces, and what it draws from. `segments` are the scenes of the base track in timeline order.
//   cut: the index of the cut it sits on (the cut between scene `cut` and scene `cut + 1`), or null when it lies inside one scene
//   before / after: its frames before and after that cut
//   scene: the scene it lies in (inside), or the scene before the cut
// It sits on the cut nearest its middle among those it covers or touches, and is cut down to the two scenes around that cut.
export function transitionWindow(entry, segments) {
  // The frames past the end of the last scene are not part of the video.
  const total = segments.length ? transitionFrameOf(segments[segments.length - 1].end) : Infinity;
  let first = Math.min(total, transitionFrameOf(entry.start));
  let last = Math.min(total, Math.max(first, transitionFrameOf(entry.start + entry.seconds)));
  let cut = null;
  let nearest = Infinity;
  for (let i = 0; i + 1 < segments.length; i += 1) {
    const at = transitionFrameOf(segments[i].end);
    if (at < first || at > last) continue;
    const distance = Math.abs(2 * at - (first + last));
    if (distance <= nearest) {
      nearest = distance;
      cut = i;
    }
  }
  if (cut !== null) {
    first = Math.max(first, transitionFrameOf(segments[cut].start));
    last = Math.min(last, transitionFrameOf(segments[cut + 1].end));
    const at = transitionFrameOf(segments[cut].end);
    return { first, last, frames: last - first, cut, before: at - first, after: last - at, scene: cut };
  }
  let scene = segments.findIndex((item) => first >= transitionFrameOf(item.start) && first < transitionFrameOf(item.end));
  if (scene < 0) scene = Math.max(0, segments.length - 1);
  return { first, last, frames: last - first, cut: null, before: 0, after: 0, scene };
}

// The progress the final video has at `frame` of the timeline in a window (it never shows exactly 0 or 1: those frames are the scenes' own), or null.
export function transitionProgressAt(window, frame) {
  if (!window || frame < window.first || frame >= window.last || window.frames < 2) return null;
  return (frame - window.first + 1) / (window.frames + 1);
}

// -- moving and stretching ---------------------------------------------------------------------------------------------------

// The blocks other than `id`, in order.
const transitionOthers = (list, id) => list.filter((entry) => entry.id !== id);

// The room around block `id` while it moves: the end of the block before it and the start of the block after it (the order of the blocks
// does not change by dragging).
function transitionNeighbours(list, entry) {
  const rest = transitionOthers(list, entry.id);
  const before = rest.filter((item) => item.start < entry.start || (item.start === entry.start && item.id < entry.id));
  const after = rest.filter((item) => !before.includes(item));
  return { low: before.length ? Math.max(...before.map(transitionEnd)) : 0, high: after.length ? Math.min(...after.map((item) => item.start)) : Infinity };
}

// The start `wanted` for block `id`, kept between its neighbours and inside the timeline (`total` seconds, 0 for no limit).
export function moveTransition(list, id, wanted, total = 0) {
  const entry = list.find((item) => item.id === id);
  if (!entry) return wanted;
  const { low, high } = transitionNeighbours(list, entry);
  const limit = total > 0 ? Math.min(high, total) : high;
  return transitionRound3(transitionClamp(wanted, low, Math.max(low, limit - entry.seconds)));
}

// The start edge of block `id` dragged to `wanted`: the length changes, the end stays.
export function trimTransitionStart(list, id, wanted) {
  const entry = list.find((item) => item.id === id);
  if (!entry) return { start: wanted, seconds: 0 };
  const { low } = transitionNeighbours(list, entry);
  const end = transitionEnd(entry);
  const start = transitionClamp(wanted, Math.max(low, end - TRANSITION_MAX_SECONDS), end - TRANSITION_MIN_SECONDS);
  return { start: transitionRound3(start), seconds: transitionRound3(end - start) };
}

// The end edge of block `id` dragged to `wanted`: the start stays.
export function trimTransitionEnd(list, id, wanted, total = 0) {
  const entry = list.find((item) => item.id === id);
  if (!entry) return { start: wanted, seconds: 0 };
  const { high } = transitionNeighbours(list, entry);
  const limit = Math.min(high, total > 0 ? total : Infinity, entry.start + TRANSITION_MAX_SECONDS);
  const end = transitionClamp(wanted, entry.start + TRANSITION_MIN_SECONDS, Math.max(entry.start + TRANSITION_MIN_SECONDS, limit));
  return { start: entry.start, seconds: transitionRound3(end - entry.start) };
}

// A time pulled to the nearest point within a few pixels: the scene edges and the edges of other blocks always, the beats when Snap beats is on.
export function snapTransitionTime(time, { points = [], beats = [], snapBeats = false, pxPerSecond = 40 } = {}) {
  const reach = TRANSITION_SNAP_PIXELS / Math.max(1, pxPerSecond);
  let best = time;
  let bestDistance = Infinity;
  for (const point of [0, ...points, ...(snapBeats ? beats : [])]) {
    const distance = Math.abs(Number(point) - time);
    if (Number.isFinite(distance) && distance < bestDistance) {
      bestDistance = distance;
      best = Number(point);
    }
  }
  return bestDistance <= reach ? best : time;
}

// -- adding ------------------------------------------------------------------------------------------------------------------

// Where a new block of `seconds` goes: over the selected range of the timeline when there is one, else centered on the cut within reach of the
// playhead, else at the playhead. It is moved to the first free place after that (or before, near the end of the timeline). `null` when there is no room.
export function placeNewTransition(list, segments, { playhead = 0, seconds = TRANSITION_DEFAULT_SECONDS, total = 0, range = null } = {}) {
  let length = transitionClamp(seconds, TRANSITION_MIN_SECONDS, TRANSITION_MAX_SECONDS);
  let start = Math.max(0, playhead);
  if (range && Number(range.end) - Number(range.start) >= TRANSITION_MIN_SECONDS) {
    start = Number(range.start);
    length = transitionClamp(Number(range.end) - Number(range.start), TRANSITION_MIN_SECONDS, TRANSITION_MAX_SECONDS);
  } else {
    const cuts = segments.slice(0, -1).map((segment) => transitionNumber(segment.end, 0));
    const near = cuts.filter((cut) => Math.abs(cut - playhead) <= TRANSITION_CUT_REACH_SECONDS).sort((a, b) => Math.abs(a - playhead) - Math.abs(b - playhead))[0];
    if (near !== undefined) start = near - Math.floor((length * TRANSITION_FPS) / 2) / TRANSITION_FPS;
  }
  start = transitionSnapToFrame(Math.max(0, start));
  length = Math.max(TRANSITION_MIN_SECONDS, transitionSnapToFrame(length));
  const limit = total > 0 ? total : Infinity;
  const sorted = [...list].sort((a, b) => a.start - b.start);
  const fits = (at) => at >= 0 && at + length <= limit + 0.0005 && sorted.every((item) => at + length <= item.start + 0.0005 || at >= transitionEnd(item) - 0.0005);
  if (fits(start)) return { start: transitionRound3(start), seconds: transitionRound3(length) };
  // After the blocks that are in the way, then before them.
  for (const item of sorted) {
    const at = transitionSnapToFrame(transitionEnd(item));
    if (at >= start - 0.0005 && fits(at)) return { start: transitionRound3(at), seconds: transitionRound3(length) };
  }
  for (const item of [...sorted].reverse()) {
    const at = transitionSnapToFrame(item.start - length);
    if (fits(at)) return { start: transitionRound3(at), seconds: transitionRound3(length) };
  }
  return null;
}

// -- the stitch --------------------------------------------------------------------------------------------------------------

// The transitions of a stitch of `baseSegments`, on the clock of its scene timing (`offset` is the whole-frame shift the Builder takes off every time).
// A block that lies on none of the stitched scenes is left out, and one that starts before the first scene is cut at its start.
export function transitionStitchItems(list, baseSegments, offset = 0) {
  const items = [];
  for (const entry of list) {
    const end = entry.start + entry.seconds;
    if (!baseSegments.some((segment) => entry.start < transitionNumber(segment.end, 0) && end > transitionNumber(segment.start, 0))) continue;
    const start = Math.max(0, entry.start - offset);
    const seconds = end - offset - start;
    if (seconds < TRANSITION_MIN_SECONDS - 0.0005) continue;
    items.push({ transition: entry.transition, start: transitionRound3(start), seconds: transitionRound3(Math.max(TRANSITION_MIN_SECONDS, seconds)), params: entry.params || {} });
  }
  return items;
}
