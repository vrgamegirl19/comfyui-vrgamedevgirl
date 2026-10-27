// The overlay lane of the timeline: one block per overlay, on a row of its own. A block is dragged to move it, its edges are dragged to trim or
// extend it, and a right click opens a menu (split at the playhead, duplicate, off or on, lock, delete). With Snap beats on, edges snap to
// beats and to scene edges. The editing rules are plain functions at the top of this file so they can be tested without a page; the part
// that draws the blocks and reads the pointer follows.

import { makeButton } from "./controls.mjs";
import { normalizeOverlayLayers } from "./overlay_layers.mjs";

export const OVERLAY_ROW_HEIGHT = 30;
const OVERLAY_MAX_ROWS = 6;
export const OVERLAY_MIN_LENGTH = 0.2;
const OVERLAY_SNAP_PIXELS = 8;

const OVERLAY_KIND_STYLE = {
  media: { color: "#0e7490", border: "#22d3ee", label: "Clip" },
  grain: { color: "#92400e", border: "#fbbf24", label: "Grain" },
  vignette: { color: "#581c87", border: "#c084fc", label: "Vignette" },
  text: { color: "#9d174d", border: "#f472b6", label: "Text" },
  lut: { color: "#166534", border: "#4ade80", label: "LUT" },
};

function overlayLaneFileStem(path) {
  const name = String(path || "").split(/[\\/]/).pop() || "";
  return name.replace(/\.[^.]+$/, "");
}

// What a block says: the clip's file for a media clip, its text for text, its LUT for a LUT, otherwise the effect's name.
export function overlayLayerLabel(clip) {
  const style = OVERLAY_KIND_STYLE[clip?.kind] || { label: String(clip?.kind || "Overlay") };
  if (clip?.kind === "media") return overlayLaneFileStem(clip.path) || style.label;
  if (clip?.kind === "text") return String(clip.text || "").trim().split("\n")[0] || style.label;
  if (clip?.kind === "lut") return String(clip.lut || "").replace(/\.cube$/i, "") || style.label;
  return style.label;
}

// A time pulled to the nearest beat or scene edge within a few pixels, when Snap beats is on. Zero and the end of the timeline count too.
export function snapOverlayTime(time, { snap = false, beats = [], edges = [], pxPerSecond = 40, total = 0 } = {}) {
  const value = Number(time) || 0;
  if (!snap) return value;
  const reach = OVERLAY_SNAP_PIXELS / Math.max(1, pxPerSecond);
  let best = value;
  let bestDelta = Infinity;
  for (const candidate of [0, total, ...edges, ...beats]) {
    const point = Number(candidate);
    if (!Number.isFinite(point)) continue;
    const delta = Math.abs(point - value);
    if (delta < bestDelta) {
      bestDelta = delta;
      best = point;
    }
  }
  return bestDelta <= reach ? best : value;
}

const overlayLaneRound3 = (value) => Math.round(value * 1000) / 1000;

function overlayLaneReplace(clips, id, change) {
  return clips.map((clip) => (clip.id === id ? change(clip) : clip));
}

// The clip moved so it starts at `start`, keeping its length, and kept inside the timeline (`total` is its length in seconds, 0 for no limit).
export function moveOverlayLayer(clips, id, start, total = 0) {
  return overlayLaneReplace(clips, id, (clip) => {
    const length = clip.end - clip.start;
    const limit = total > 0 ? Math.max(0, total - length) : Infinity;
    const next = overlayLaneRound3(Math.min(limit, Math.max(0, start)));
    return { ...clip, start: next, end: overlayLaneRound3(next + length) };
  });
}

// The start edge dragged to `start`. A media clip keeps showing the same picture under the pointer, so its source start moves with the edge.
export function trimOverlayLayerStart(clips, id, start) {
  return overlayLaneReplace(clips, id, (clip) => {
    const next = overlayLaneRound3(Math.min(clip.end - OVERLAY_MIN_LENGTH, Math.max(0, start)));
    const change = { ...clip, start: next };
    if (clip.kind === "media") change.source_start = overlayLaneRound3(Math.max(0, (clip.source_start || 0) + (next - clip.start) * (clip.speed || 1)));
    return change;
  });
}

// The end edge dragged to `end`. A media clip longer than its file loops (the clip's own Loop setting decides what happens past the end).
export function trimOverlayLayerEnd(clips, id, end, total = 0) {
  return overlayLaneReplace(clips, id, (clip) => {
    const limit = total > 0 ? total : Infinity;
    return { ...clip, end: overlayLaneRound3(Math.min(limit, Math.max(clip.start + OVERLAY_MIN_LENGTH, end))) };
  });
}

function overlayLaneFreshId(clips, base) {
  const used = new Set(clips.map((clip) => clip.id));
  const stem = String(base || "overlay").replace(/_\d+$/, "");
  for (let n = 2; ; n += 1) {
    if (!used.has(`${stem}_${n}`)) return `${stem}_${n}`;
  }
}

// One clip cut in two at `time`. Does nothing when the cut is too close to an edge. The second half starts its media where the first one
// stopped, and the fade out belongs to the second half and the fade in to the first.
export function splitOverlayLayer(clips, id, time) {
  const clip = clips.find((item) => item.id === id);
  if (!clip || time <= clip.start + OVERLAY_MIN_LENGTH || time >= clip.end - OVERLAY_MIN_LENGTH) return clips;
  const cut = overlayLaneRound3(time);
  const first = { ...clip, end: cut, fade_out: 0 };
  const second = { ...clip, id: overlayLaneFreshId(clips, clip.id), start: cut, fade_in: 0 };
  if (clip.kind === "media") second.source_start = overlayLaneRound3((clip.source_start || 0) + (cut - clip.start) * (clip.speed || 1));
  const index = clips.indexOf(clip);
  return [...clips.slice(0, index), first, second, ...clips.slice(index + 1)];
}

// A copy placed straight after the clip (or before it when there is no room before the end of the timeline). It lands on top, at the end of the list.
export function duplicateOverlayLayer(clips, id, total = 0) {
  const clip = clips.find((item) => item.id === id);
  if (!clip) return clips;
  const length = clip.end - clip.start;
  const fits = total <= 0 || clip.end + length <= total + 0.0005;
  const start = fits ? clip.end : Math.max(0, clip.start - length);
  return [...clips, { ...clip, id: overlayLaneFreshId(clips, clip.id), start: overlayLaneRound3(start), end: overlayLaneRound3(start + length) }];
}

export function deleteOverlayLayer(clips, id) {
  return clips.filter((clip) => clip.id !== id);
}

// Whether a key press goes to a place where the person types, so Delete and Backspace keep their text editing job there.
export function overlayKeyIsTyping(target) {
  if (!target) return false;
  if (target.isContentEditable) return true;
  const tag = String(target.tagName || "").toUpperCase();
  if (tag === "TEXTAREA" || tag === "SELECT") return true;
  if (tag !== "INPUT") return false;
  return !["range", "checkbox", "radio", "button", "submit", "reset", "file", "image"].includes(String(target.type || "text").toLowerCase());
}

// The Delete (or Backspace) key removes the selected overlay, the way the Delete item of its right click menu does. It leaves the key alone while
// the person is typing, when no clip is selected and when the key is pressed in a dialog that is not part of the Builder. `root` is the Builder's page.
export function createOverlayDeleteKey({ state, root, pushHistory, commit }) {
  return (event) => {
    if (event.key !== "Delete" && event.key !== "Backspace") return;
    if (event.ctrlKey || event.metaKey || event.altKey || event.defaultPrevented) return;
    if (!state.overlayAvailable || !state.activeOverlayId || !(state.overlayLayers || []).some((clip) => clip.id === state.activeOverlayId)) return;
    const target = event.target;
    const page = typeof document === "undefined" ? null : document;
    const inBuilder = !target || target === page?.body || target === page?.documentElement || Boolean(root?.contains?.(target));
    if (!inBuilder || overlayKeyIsTyping(target)) return;
    event.preventDefault();
    event.stopPropagation();
    pushHistory();
    state.overlayLayers = deleteOverlayLayer(state.overlayLayers, state.activeOverlayId);
    state.activeOverlayId = "";
    commit("Overlay deleted");
  };
}

// A new clip added on top of the others. `fields` need only the kind and the times, the rest gets its defaults when the list is normalized.
export function addOverlayLayer(clips, fields) {
  const base = String(fields.kind || "overlay");
  const id = clips.some((clip) => clip.id === base) ? overlayLaneFreshId(clips, base) : base;
  return normalizeOverlayLayers([...clips, { ...fields, id }]);
}

// Which row each clip sits in. Clips that overlap in time are stacked in rows so every one can be read and grabbed: a clip takes the first
// row whose last clip has ended. (The order of the list, which is the order they are drawn onto the video, is not changed.)
export function overlayLaneRowsFor(clips) {
  const list = Array.isArray(clips) ? clips : [];
  const order = list.map((clip, index) => index).sort((a, b) => (list[a].start - list[b].start) || (a - b));
  const rowEnds = [];
  const rows = {};
  for (const index of order) {
    const clip = list[index];
    let row = rowEnds.findIndex((end) => end <= clip.start + 0.001);
    if (row < 0) row = rowEnds.length < OVERLAY_MAX_ROWS ? rowEnds.length : OVERLAY_MAX_ROWS - 1;
    rowEnds[row] = Math.max(rowEnds[row] ?? 0, clip.end);
    rows[clip.id] = row;
  }
  return { rows, count: Math.max(1, rowEnds.length) };
}

// The height of the lane for its clips.
export function overlayLaneHeight(clips) {
  return overlayLaneRowsFor(clips).count * OVERLAY_ROW_HEIGHT + 4;
}

// Whether the overlay lane shows: in a build that can draw overlays, once the project has one or the library has been opened for it.
export function overlayLaneVisible(state) {
  return Boolean(state?.overlayAvailable) && (Array.isArray(state?.overlayLayers) && state.overlayLayers.length > 0 || Boolean(state?.overlayLaneOpen));
}

const OVERLAY_MENU_STYLE = "position:fixed;z-index:100010;min-width:180px;border:1px solid #155e75;border-radius:7px;background:#111827;color:#f8fafc;box-shadow:0 16px 50px rgba(0,0,0,.55);padding:6px;display:flex;flex-direction:column;gap:4px;";

// Draws the lane into `layer` (the timeline's segment layer, which does not take the pointer itself). `ctx` gives it the page:
//   state, top, total (seconds), edges (scene edges to snap to), playheadTime(), pushHistory(), commit(reason) (redraw and save), select(id)
export function renderOverlayLane({ layer, state, top, total, edges, playheadTime, pushHistory, commit, select }) {
  const lane = document.createElement("div");
  lane.className = "vrgdg-builder-overlay-lane";
  lane.style.cssText = `position:absolute;left:0;top:${top}px;width:${Math.max(900, Math.ceil(total * state.pxPerSecond))}px;height:${overlayLaneHeight(state.overlayLayers)}px;pointer-events:auto;background:rgba(14,116,144,.08);border-top:1px solid #164e63;border-bottom:1px solid #164e63;`;
  lane.onpointerdown = (event) => {
    if (event.target === lane) choose("");
  };
  const label = document.createElement("div");
  label.textContent = "OVERLAYS";
  label.style.cssText = `position:absolute;left:4px;top:${top - 12}px;color:#67e8f9;font-size:10px;font-weight:900;letter-spacing:.08em;pointer-events:none;text-shadow:0 1px 2px #020617;`;
  layer.append(label);

  const paintSelection = () => {
    for (const element of lane.querySelectorAll(".vrgdg-builder-overlay-block")) {
      const on = element.dataset.overlayId === state.activeOverlayId;
      const kind = state.overlayLayers.find((item) => item.id === element.dataset.overlayId)?.kind;
      element.style.borderColor = on ? "#f8fafc" : (OVERLAY_KIND_STYLE[kind] || { border: "#a1a1aa" }).border;
      element.style.boxShadow = on ? "0 0 0 2px rgba(248,250,252,.6)" : "none";
    }
  };
  const choose = (id) => {
    select(id);
    paintSelection();
  };
  const { rows } = overlayLaneRowsFor(state.overlayLayers);
  const snapOptions = () => ({ snap: Boolean(state.snapToBeats), beats: state.showBeatMarkers === false ? [] : state.beats || [], edges, pxPerSecond: state.pxPerSecond, total });
  const secondsAt = (event, origin, startTime) => startTime + (event.clientX - origin) / state.pxPerSecond;

  for (const clip of state.overlayLayers) {
    const style = OVERLAY_KIND_STYLE[clip.kind] || { color: "#3f3f46", border: "#a1a1aa" };
    const block = document.createElement("div");
    block.className = "vrgdg-builder-overlay-block";
    block.dataset.overlayId = clip.id;
    block.tabIndex = 0;
    const selected = state.activeOverlayId === clip.id;
    block.style.cssText = `position:absolute;left:${clip.start * state.pxPerSecond}px;top:${3 + (rows[clip.id] || 0) * OVERLAY_ROW_HEIGHT}px;width:${Math.max(18, (clip.end - clip.start) * state.pxPerSecond)}px;height:${OVERLAY_ROW_HEIGHT - 6}px;box-sizing:border-box;border:1px solid ${selected ? "#f8fafc" : style.border};border-radius:5px;background:${style.color};color:#f8fafc;overflow:hidden;cursor:${clip.locked ? "pointer" : "grab"};opacity:${clip.enabled ? 1 : 0.45};${selected ? "box-shadow:0 0 0 2px rgba(248,250,252,.6);" : ""}user-select:none;touch-action:none;`;
    const text = document.createElement("div");
    text.textContent = overlayLayerLabel(clip);
    text.title = `${overlayLayerLabel(clip)} (${OVERLAY_KIND_STYLE[clip.kind]?.label || clip.kind}) ${clip.start.toFixed(2)} s to ${clip.end.toFixed(2)} s`;
    text.style.cssText = "position:absolute;left:10px;right:50px;top:0;bottom:0;display:flex;align-items:center;font-size:11px;font-weight:800;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;pointer-events:none;";
    block.append(text);

    const eye = document.createElement("button");
    eye.type = "button";
    eye.textContent = clip.enabled ? "◉" : "◎";
    eye.title = clip.enabled ? "Turn this overlay off (it stays on the timeline)" : "Turn this overlay on";
    eye.style.cssText = "position:absolute;right:26px;top:3px;width:15px;height:15px;padding:0;border:0;background:none;color:#f8fafc;font-size:11px;line-height:15px;cursor:pointer;z-index:2;";
    eye.onpointerdown = (event) => event.stopPropagation();
    eye.onclick = (event) => {
      event.stopPropagation();
      pushHistory();
      state.overlayLayers = overlayLaneReplace(state.overlayLayers, clip.id, (item) => ({ ...item, enabled: !item.enabled }));
      commit("Overlay turned on or off");
    };
    const lock = document.createElement("button");
    lock.type = "button";
    lock.textContent = clip.locked ? "🔒" : "🔓";
    lock.title = clip.locked ? "Unlock this overlay so it can be moved and trimmed" : "Lock this overlay";
    lock.style.cssText = "position:absolute;right:10px;top:3px;width:15px;height:15px;padding:0;border:0;background:none;font-size:10px;line-height:15px;cursor:pointer;z-index:2;";
    lock.onpointerdown = (event) => event.stopPropagation();
    lock.onclick = (event) => {
      event.stopPropagation();
      pushHistory();
      state.overlayLayers = overlayLaneReplace(state.overlayLayers, clip.id, (item) => ({ ...item, locked: !item.locked }));
      commit("Overlay locked or unlocked");
    };
    // The buttons sit inside the edge handles. A block too narrow to hold them leaves them out (right click has the same actions).
    if ((clip.end - clip.start) * state.pxPerSecond >= 90) block.append(eye, lock);

    for (const side of ["start", "end"]) {
      const handle = document.createElement("div");
      handle.dataset.edge = side;
      handle.style.cssText = `position:absolute;${side === "start" ? "left" : "right"}:0;top:0;bottom:0;width:7px;cursor:${clip.locked ? "default" : "ew-resize"};background:rgba(248,250,252,.18);z-index:1;`;
      block.append(handle);
    }

    // One pointer gesture: a drag of the body moves the clip, a drag of an edge trims it. The block follows the pointer without a redraw,
    // and the change is made once when the pointer is released.
    block.onpointerdown = (event) => {
      if (event.button !== 0) return;
      choose(clip.id);
      if (clip.locked) return;
      const edge = event.target?.dataset?.edge || "";
      const origin = event.clientX;
      const startTime = edge === "end" ? clip.end : clip.start;
      let moved = false;
      let result = state.overlayLayers;
      event.preventDefault();
      block.setPointerCapture?.(event.pointerId);
      const apply = (moveEvent) => {
        const wanted = secondsAt(moveEvent, origin, startTime);
        if (Math.abs(moveEvent.clientX - origin) > 2) moved = true;
        const options = snapOptions();
        if (edge === "start") result = trimOverlayLayerStart(state.overlayLayers, clip.id, snapOverlayTime(wanted, options));
        else if (edge === "end") result = trimOverlayLayerEnd(state.overlayLayers, clip.id, snapOverlayTime(wanted, options), total);
        else {
          // Snap whichever end of the clip is nearer a beat.
          const length = clip.end - clip.start;
          const snappedStart = snapOverlayTime(wanted, options);
          const snappedEnd = snapOverlayTime(wanted + length, options) - length;
          const useStart = Math.abs(snappedStart - wanted) <= Math.abs(snappedEnd - wanted);
          result = moveOverlayLayer(state.overlayLayers, clip.id, useStart ? snappedStart : snappedEnd, total);
        }
        const now = result.find((item) => item.id === clip.id);
        block.style.left = `${now.start * state.pxPerSecond}px`;
        block.style.width = `${Math.max(18, (now.end - now.start) * state.pxPerSecond)}px`;
      };
      const finish = () => {
        block.removeEventListener("pointermove", apply);
        block.removeEventListener("pointerup", finish);
        block.removeEventListener("pointercancel", finish);
        if (!moved) return;
        pushHistory();
        state.overlayLayers = normalizeOverlayLayers(result);
        commit(edge ? "Overlay trimmed" : "Overlay moved");
      };
      block.addEventListener("pointermove", apply);
      block.addEventListener("pointerup", finish);
      block.addEventListener("pointercancel", finish);
    };
    block.oncontextmenu = (event) => {
      event.preventDefault();
      event.stopPropagation();
      choose(clip.id);
      openOverlayLayerMenu(event, clip, { state, total, playheadTime, pushHistory, commit });
    };
    lane.append(block);
  }
  layer.append(lane);
  return lane;
}

function openOverlayLayerMenu(event, clip, { state, total, playheadTime, pushHistory, commit }) {
  document.querySelector(".vrgdg-builder-context-menu")?.remove();
  const menu = document.createElement("div");
  menu.className = "vrgdg-builder-context-menu";
  menu.style.cssText = OVERLAY_MENU_STYLE;
  menu.style.left = `${Math.min(window.innerWidth - 200, event.clientX)}px`;
  menu.style.top = `${Math.min(window.innerHeight - 220, event.clientY)}px`;
  const close = () => {
    menu.remove();
    window.removeEventListener("pointerdown", closeOnPointer, true);
    window.removeEventListener("keydown", closeOnKey, true);
  };
  const closeOnPointer = (closeEvent) => {
    if (!menu.contains(closeEvent.target)) close();
  };
  const closeOnKey = (keyEvent) => {
    if (keyEvent.key === "Escape") close();
  };
  const add = (text, action, disabled = false) => {
    const button = makeButton(text);
    button.disabled = disabled;
    button.style.justifyContent = "flex-start";
    button.style.textAlign = "left";
    button.onclick = () => {
      close();
      action();
    };
    menu.append(button);
  };
  const time = playheadTime();
  add("Split at the playhead", () => {
    const next = splitOverlayLayer(state.overlayLayers, clip.id, time);
    if (next === state.overlayLayers) return;
    pushHistory();
    state.overlayLayers = normalizeOverlayLayers(next);
    commit("Overlay split");
  }, !(time > clip.start + OVERLAY_MIN_LENGTH && time < clip.end - OVERLAY_MIN_LENGTH));
  add("Duplicate", () => {
    pushHistory();
    state.overlayLayers = normalizeOverlayLayers(duplicateOverlayLayer(state.overlayLayers, clip.id, total));
    commit("Overlay duplicated");
  });
  add(clip.enabled ? "Turn off" : "Turn on", () => {
    pushHistory();
    state.overlayLayers = overlayLaneReplace(state.overlayLayers, clip.id, (item) => ({ ...item, enabled: !item.enabled }));
    commit("Overlay turned on or off");
  });
  add(clip.locked ? "Unlock" : "Lock", () => {
    pushHistory();
    state.overlayLayers = overlayLaneReplace(state.overlayLayers, clip.id, (item) => ({ ...item, locked: !item.locked }));
    commit("Overlay locked or unlocked");
  });
  add("Delete", () => {
    pushHistory();
    state.overlayLayers = deleteOverlayLayer(state.overlayLayers, clip.id);
    if (state.activeOverlayId === clip.id) state.activeOverlayId = "";
    commit("Overlay deleted");
  });
  document.body.append(menu);
  window.addEventListener("pointerdown", closeOnPointer, true);
  window.addEventListener("keydown", closeOnKey, true);
}

