// What the transition screens do to the project's blocks (`state.sceneTransitions`): read them, save a change, select one, remove, duplicate, center on a cut.
// Every change makes a new list, since the undo history keeps the old one. `transitionShared` is what the lane, the tab and the player tell each other.

import { toast } from "./controls.mjs";
import {
  TRANSITION_FPS, moveTransition, normalizeTransitionEntries, placeNewTransition, transitionEnd, transitionRows, transitionSnapToFrame, transitionStitchItems, transitionWindow,
} from "./transition_model.mjs";
import { transitionSceneName } from "./transition_dom.mjs";

export const TRANSITION_ROW = 30;
export const TRANSITION_BLOCK_STYLE = {
  cut: { color: "#6d28d9", border: "#c4b5fd" },
  inside: { color: "#a21caf", border: "#f0abfc" },
};

// `lane` is what the lane was last drawn with (state, total, edges, playheadTime, pushHistory, commit), `clock()` the playhead in seconds,
// `refreshPanel()` redraws the Transitions tab when it is showing and `deselectOthers(state)` clears the selection of the effect and filter lanes.
export const transitionShared = { lane: null, clock: () => 0, refreshPanel: () => {}, deselectOthers: () => {} };

export function notifyTransition(text, isError = true) {
  toast(text, isError);
}

// The saved blocks, read into the shape the lane works with. A list saved by an earlier version is converted once, here.
export function transitionEntries(state) {
  const saved = Array.isArray(state.sceneTransitions) ? state.sceneTransitions : [];
  if (saved.every((item) => item && item.id && typeof item.start === "number")) return saved;
  // An entry of the earlier version is placed by its scene, so it waits until the scenes are loaded.
  if (!(state.segments || []).length && saved.some((item) => item && typeof item.start !== "number")) return [];
  state.sceneTransitions = normalizeTransitionEntries(saved, state.segments || []);
  return state.sceneTransitions;
}

// Every change makes a new list: the undo history keeps the old one.
export function saveTransitions(state, next, reason, { pushHistory, commit }) {
  pushHistory?.();
  state.sceneTransitions = normalizeTransitionEntries(next, state.segments || []);
  commit?.(reason);
}

export const transitionTimelineEnd = (state) => Math.max(0, ...(state.segments || []).map((segment) => Number(segment.end) || 0), transitionShared.lane?.total || 0);

export const transitionCutTimes = (state) => (state.segments || []).slice(0, -1).map((segment) => Number(segment.end) || 0);

export function freshTransitionId(list) {
  const used = new Set(list.map((item) => item.id));
  for (let n = 1; ; n += 1) {
    if (!used.has(`t${n}`)) return `t${n}`;
  }
}

// Where a block lies in words, and which kind of block it is.
export function describeTransition(entry, segments) {
  const w = transitionWindow(entry, segments);
  if (w.cut !== null) return { kind: "cut", text: `On the cut between ${transitionSceneName(segments[w.cut], w.cut)} and ${transitionSceneName(segments[w.cut + 1], w.cut + 1)}` };
  return { kind: "inside", text: segments[w.scene] ? `Inside ${transitionSceneName(segments[w.scene], w.scene)}, an effect on its footage` : "Outside the scenes" };
}

// The transitions of a stitch of `baseSegments`: none in a build that cannot draw them. `offset` is the whole-frame shift the Builder takes off every time of the
// stitch.
export function transitionsForStitch(state, baseSegments, offset = 0) {
  return state.transitionsAvailable ? transitionStitchItems(transitionEntries(state), baseSegments, offset) : [];
}

// The list to save in the session. A build that cannot draw transitions keeps what the project has, as it was saved.
export function transitionsForSession(state) {
  if (!state.transitionsAvailable) return Array.isArray(state.sceneTransitions) ? state.sceneTransitions : [];
  const list = transitionEntries(state);
  return list.length || !Array.isArray(state.sceneTransitions) ? list : state.sceneTransitions;
}

// Whether the lane shows: in a build that can draw transitions, once the project has one or the Transitions tab has been opened.
export function transitionLaneVisible(state) {
  return Boolean(state.transitionsAvailable) && (transitionEntries(state).length > 0 || Boolean(state.transitionLaneOpen));
}

// The height of the lane for its blocks, 0 while it is not shown.
export function transitionLaneHeight(state) {
  return transitionLaneVisible(state) ? transitionRows(transitionEntries(state)).count * TRANSITION_ROW + 4 : 0;
}

export function paintTransitionSelection(state) {
  for (const node of document.querySelectorAll(".h3-transition-block")) {
    const on = node.dataset.transitionId === state.activeTransitionId;
    const kind = TRANSITION_BLOCK_STYLE[node.dataset.kind] || TRANSITION_BLOCK_STYLE.cut;
    node.style.borderColor = on ? "#f8fafc" : kind.border;
    node.style.boxShadow = on ? "0 0 0 2px rgba(248,250,252,.6)" : "none";
  }
}

export function selectTransition(state, id) {
  state.activeTransitionId = id;
  if (id) transitionShared.deselectOthers(state);
  paintTransitionSelection(state);
  transitionShared.refreshPanel();
}

// An overlay was chosen: the transition stops being the one the Delete key acts on.
export function deselectTransition(state) {
  if (!state.activeTransitionId) return;
  selectTransition(state, "");
}

export function removeTransition(state, id, context) {
  if (state.activeTransitionId === id) state.activeTransitionId = "";
  saveTransitions(state, transitionEntries(state).filter((item) => item.id !== id), "Transition deleted", context);
}

// The selected range of the timeline (the Builder keeps it as `in` and `out`), or null.
export function transitionSelectedRange(state) {
  const range = state.selectedTimelineRange;
  return Number.isFinite(range?.in) && Number.isFinite(range?.out) && range.out > range.in ? { start: range.in, end: range.out } : null;
}

export function duplicateTransition(state, id, context) {
  const list = transitionEntries(state);
  const entry = list.find((item) => item.id === id);
  if (!entry) return;
  const spot = placeNewTransition(list, state.segments || [], { playhead: transitionEnd(entry), seconds: entry.seconds, total: transitionTimelineEnd(state) });
  if (!spot) {
    notifyTransition("There is no free room on the timeline for a copy.");
    return;
  }
  // The copy is placed after the block (centered on a cut near its end, if there is one).
  const copy = { ...entry, id: freshTransitionId(list), params: { ...entry.params }, ...spot };
  state.activeTransitionId = copy.id;
  saveTransitions(state, [...list, copy], "Transition duplicated", context);
}

// The block moved so it is centered on the cut nearest to it.
export function centerTransitionOnCut(state, id, context) {
  const list = transitionEntries(state);
  const entry = list.find((item) => item.id === id);
  const cuts = transitionCutTimes(state);
  if (!entry || !cuts.length) return;
  const middle = entry.start + entry.seconds / 2;
  const cut = cuts.sort((a, b) => Math.abs(a - middle) - Math.abs(b - middle))[0];
  const target = transitionSnapToFrame(cut - Math.floor((entry.seconds * TRANSITION_FPS) / 2) / TRANSITION_FPS);
  const start = moveTransition(list, id, target, transitionTimelineEnd(state));
  if (Math.abs(start - target) > 0.0005) notifyTransition("Another transition is in the way, so it moved only as far as it could.");
  if (start === entry.start) return;
  saveTransitions(state, list.map((item) => (item.id === id ? { ...item, start } : item)), "Transition moved", context);
}
