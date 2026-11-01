// What the effect and filter screens do to the project's blocks (`state.videoEffects`, `state.videoFilters`): read them, save a change, select one, remove,
// duplicate. Every change makes a new list, since the undo history keeps the old one. `lookShared` is what the lanes, the tabs and the player tell each other.
// Every function takes the kind (a key of LOOK_KINDS) first after the state.

import { toast } from "./controls.mjs";
import { LOOK_KINDS, LOOK_ORDER, lookEnd, lookRows, lookStitchItems, moveLook, normalizeLookEntries } from "./look_model.mjs";
import { deselectTransition } from "./transition_actions.mjs";

export const LOOK_ROW = 26;

// `lanes[kind]` is what the lane was last drawn with (state, total, edges, playheadTime, pushHistory, commit) and `panels[kind]` redraws the tab of the kind when
// it is showing.
export const lookShared = { lanes: {}, panels: {} };

export function notifyLook(text, isError = true) {
  toast(text, isError);
}

// The saved blocks of a kind, read into the shape the lane works with.
export function lookEntries(state, kind) {
  const { listKey } = LOOK_KINDS[kind];
  const saved = Array.isArray(state[listKey]) ? state[listKey] : [];
  if (saved.every((item) => item && item.id && typeof item.start === "number")) return saved;
  state[listKey] = normalizeLookEntries(saved, kind);
  return state[listKey];
}

// Every change makes a new list: the undo history keeps the old one.
export function saveLooks(state, kind, next, reason, { pushHistory, commit }) {
  pushHistory?.();
  state[LOOK_KINDS[kind].listKey] = normalizeLookEntries(next, kind);
  commit?.(reason);
}

export const lookTimelineEnd = (state, kind) => Math.max(0, ...(state.segments || []).map((segment) => Number(segment.end) || 0), lookShared.lanes[kind]?.total || 0);

export function freshLookId(list, kind) {
  const used = new Set(list.map((item) => item.id));
  for (let n = 1; ; n += 1) {
    if (!used.has(`${LOOK_KINDS[kind].idPrefix}${n}`)) return `${LOOK_KINDS[kind].idPrefix}${n}`;
  }
}

// The blocks of a stitch of `baseSegments`: none in a build that cannot draw them. `offset` is the whole-frame shift the Builder takes off every time of the stitch.
export function looksForStitch(state, kind, baseSegments, offset = 0) {
  return state[LOOK_KINDS[kind].availableKey] ? lookStitchItems(lookEntries(state, kind), baseSegments, offset) : [];
}

// The list to save in the session. A build that cannot draw them keeps what the project has, as it was saved.
export function looksForSession(state, kind) {
  const { listKey, availableKey } = LOOK_KINDS[kind];
  if (!state[availableKey]) return Array.isArray(state[listKey]) ? state[listKey] : [];
  const list = lookEntries(state, kind);
  return list.length || !Array.isArray(state[listKey]) ? list : state[listKey];
}

// Whether the lane shows: in a build that can draw the kind, while the project has a block of it. An empty lane is not drawn.
export function lookLaneVisible(state, kind) {
  return Boolean(state[LOOK_KINDS[kind].availableKey]) && lookEntries(state, kind).length > 0;
}

// The height of the lane for its blocks, 0 while it is not shown.
export function lookLaneHeight(state, kind) {
  return lookLaneVisible(state, kind) ? lookRows(lookEntries(state, kind)).count * LOOK_ROW + 4 : 0;
}

export function paintLookSelection(state) {
  for (const node of document.querySelectorAll(".h3-look-block")) {
    const config = LOOK_KINDS[node.dataset.lookKind];
    const on = config && state[config.activeKey] === node.dataset.lookId;
    node.style.borderColor = on ? "#f8fafc" : config?.border || "";
    node.style.boxShadow = on ? "0 0 0 2px rgba(248,250,252,.6)" : "none";
  }
}

// Chooses a block (an empty id chooses none). Only one block of the timeline is the chosen one: the Delete key acts on it.
export function selectLook(state, kind, id) {
  for (const other of LOOK_ORDER) {
    if (other !== kind) state[LOOK_KINDS[other].activeKey] = "";
  }
  state[LOOK_KINDS[kind].activeKey] = id;
  if (id) deselectTransition(state);
  paintLookSelection(state);
  for (const refresh of Object.values(lookShared.panels)) refresh();
}

// An overlay or a transition was chosen: no effect or filter stays the one the Delete key acts on.
export function deselectLooks(state) {
  const chosen = LOOK_ORDER.find((kind) => state[LOOK_KINDS[kind].activeKey]);
  if (chosen) selectLook(state, chosen, "");
}

export function removeLook(state, kind, id, context) {
  const config = LOOK_KINDS[kind];
  if (state[config.activeKey] === id) state[config.activeKey] = "";
  saveLooks(state, kind, lookEntries(state, kind).filter((item) => item.id !== id), `${config.label.slice(0, -1)} deleted`, context);
}

// The selected range of the timeline (the Builder keeps it as `in` and `out`), or null.
export function lookSelectedRange(state) {
  const range = state.selectedTimelineRange;
  return Number.isFinite(range?.in) && Number.isFinite(range?.out) && range.out > range.in ? { start: range.in, end: range.out } : null;
}

// A copy placed right after the block.
export function duplicateLook(state, kind, id, context) {
  const config = LOOK_KINDS[kind];
  const list = lookEntries(state, kind);
  const entry = list.find((item) => item.id === id);
  if (!entry) return;
  const total = lookTimelineEnd(state, kind);
  const copy = { ...entry, id: freshLookId(list, kind), params: { ...entry.params }, start: moveLook(entry, lookEnd(entry), total) };
  state[config.activeKey] = copy.id;
  saveLooks(state, kind, [...list, copy], `${config.label.slice(0, -1)} duplicated`, context);
}
