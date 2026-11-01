// The transition lane of the timeline: one block per transition, on a row of its own under the overlay lane. A block is dragged to move it, its edges are
// dragged to stretch it, a double click changes the transition and a right click opens a menu (change, center on the nearest cut, duplicate, delete). The
// Delete key removes the selected block. The block snaps to the scene edges and to the edges of other blocks, and to the beats when Snap beats is on.
// Blocks cannot overlap. The rules are in transition_model.mjs; this file draws the blocks and reads the pointer.

import {
  TRANSITION_BLOCK_STYLE, TRANSITION_ROW, centerTransitionOnCut, describeTransition, duplicateTransition, removeTransition, saveTransitions, selectTransition,
  transitionCutTimes, transitionEntries, transitionLaneHeight, transitionLaneVisible, transitionShared,
} from "./transition_actions.mjs";
import { transitionElement, transitionKeyIsTyping, transitionName, transitionSeconds } from "./transition_dom.mjs";
import { moveTransition, snapTransitionTime, transitionEnd, transitionRows, transitionSnapToFrame, transitionWindow, trimTransitionEnd, trimTransitionStart } from "./transition_model.mjs";
import { openTransitionPicker } from "./transition_picker.mjs";

const TRANSITION_MENU_STYLE = "position:fixed;z-index:100010;min-width:200px;border:1px solid #5b21b6;border-radius:7px;background:#111827;color:#f8fafc;box-shadow:0 16px 50px rgba(0,0,0,.55);padding:6px;display:flex;flex-direction:column;gap:4px;";

function openTransitionMenu(event, items) {
  document.querySelector(".vrgdg-builder-context-menu")?.remove();
  const menu = transitionElement("div", TRANSITION_MENU_STYLE, { className: "vrgdg-builder-context-menu" });
  menu.style.left = `${Math.min(window.innerWidth - 220, event.clientX)}px`;
  menu.style.top = `${Math.min(window.innerHeight - 40 * items.length - 20, event.clientY)}px`;
  const close = () => {
    menu.remove();
    window.removeEventListener("pointerdown", onPointer, true);
    window.removeEventListener("keydown", onKey, true);
  };
  const onPointer = (closeEvent) => {
    if (!menu.contains(closeEvent.target)) close();
  };
  const onKey = (keyEvent) => {
    if (keyEvent.key === "Escape") close();
  };
  for (const [text, action, disabled] of items) {
    const item = transitionElement("button", "text-align:left;border:0;border-radius:4px;background:transparent;color:inherit;padding:6px 8px;cursor:pointer;font:inherit;", { type: "button", textContent: text, disabled: Boolean(disabled) });
    if (disabled) item.style.opacity = "0.4";
    item.onclick = () => {
      close();
      action();
    };
    menu.append(item);
  }
  document.body.append(menu);
  window.addEventListener("pointerdown", onPointer, true);
  window.addEventListener("keydown", onKey, true);
}

let deleteKeyInstalled = false;

// The Delete (or Backspace) key removes the selected block, as the Delete item of its menu does. It leaves the key alone while the person types.
function installTransitionDeleteKey() {
  if (deleteKeyInstalled) return;
  deleteKeyInstalled = true;
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Delete" && event.key !== "Backspace") return;
    const context = transitionShared.lane;
    if (event.ctrlKey || event.metaKey || event.altKey || event.defaultPrevented || !context) return;
    const { state } = context;
    if (!state.activeTransitionId || transitionKeyIsTyping(event.target) || document.querySelector(".h3-transition-picker") || !document.querySelector(".h3-transition-lane")) return;
    if (!transitionEntries(state).some((item) => item.id === state.activeTransitionId)) return;
    event.preventDefault();
    event.stopPropagation();
    removeTransition(state, state.activeTransitionId, context);
  }, true);
}

function openTransitionBlockMenu(event, entry, context) {
  const { state } = context;
  openTransitionMenu(event, [
    ["Change the transition...", () => openTransitionPicker({ ...context, mode: "edit", id: entry.id })],
    ["Center on the nearest cut", () => centerTransitionOnCut(state, entry.id, context), transitionCutTimes(state).length === 0],
    ["Duplicate", () => duplicateTransition(state, entry.id, context)],
    ["Delete", () => removeTransition(state, entry.id, context)],
  ]);
}

// Draws the lane into `layer` (the timeline's segment layer, which does not take the pointer itself). `ctx` gives it the page:
//   state, top, total (seconds), edges (the scene edges to snap to), playheadTime(), pushHistory(), commit(reason) (redraw and save)
export function renderTransitionLane({ layer, state, top, total, edges, playheadTime, pushHistory, commit }) {
  const context = { state, total, edges, playheadTime, pushHistory, commit };
  transitionShared.lane = context;
  if (playheadTime) transitionShared.clock = playheadTime;
  transitionShared.refreshPanel();
  if (!transitionLaneVisible(state)) return null;
  installTransitionDeleteKey();
  const list = transitionEntries(state);
  const segments = state.segments || [];
  const { rows } = transitionRows(list);
  const lane = transitionElement(
    "div",
    `position:absolute;left:0;top:${top}px;width:${Math.max(900, Math.ceil(total * state.pxPerSecond))}px;height:${transitionLaneHeight(state)}px;pointer-events:auto;background:rgba(124,58,237,.08);border-top:1px solid #4c1d95;border-bottom:1px solid #4c1d95;`,
    { className: "h3-transition-lane" },
  );
  lane.onpointerdown = (event) => {
    if (event.target === lane) selectTransition(state, "");
  };
  lane.oncontextmenu = (event) => {
    if (event.target !== lane) return;
    event.preventDefault();
    const at = Math.max(0, (event.clientX - lane.getBoundingClientRect().left) / state.pxPerSecond);
    openTransitionMenu(event, [["Add a transition here...", () => openTransitionPicker({ ...context, mode: "add", at })]]);
  };
  layer.append(transitionElement("div", `position:absolute;left:4px;top:${top - 12}px;color:#c4b5fd;font-size:10px;font-weight:900;letter-spacing:.08em;pointer-events:none;text-shadow:0 1px 2px #020617;`, { textContent: "TRANSITIONS" }));

  const secondsAt = (event, origin, startTime) => startTime + (event.clientX - origin) / state.pxPerSecond;
  for (const entry of list) {
    const where = describeTransition(entry, segments);
    const style = TRANSITION_BLOCK_STYLE[where.kind];
    const selected = state.activeTransitionId === entry.id;
    const block = transitionElement(
      "div",
      `position:absolute;left:${entry.start * state.pxPerSecond}px;top:${3 + (rows[entry.id] || 0) * TRANSITION_ROW}px;width:${Math.max(18, entry.seconds * state.pxPerSecond)}px;height:${TRANSITION_ROW - 6}px;box-sizing:border-box;`
        + `border:1px solid ${selected ? "#f8fafc" : style.border};border-radius:5px;background:${style.color};color:#f8fafc;overflow:hidden;cursor:grab;user-select:none;touch-action:none;${selected ? "box-shadow:0 0 0 2px rgba(248,250,252,.6);" : ""}`,
      { className: "h3-transition-block", tabIndex: 0 },
    );
    block.dataset.transitionId = entry.id;
    block.dataset.kind = where.kind;
    const label = transitionElement("div", "position:absolute;left:10px;right:10px;top:0;bottom:0;display:flex;align-items:center;font-size:11px;font-weight:800;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;pointer-events:none;", {
      textContent: `${where.kind === "cut" ? "⇆ " : ""}${transitionName(entry.transition)}`,
      title: `${transitionName(entry.transition)}, ${transitionSeconds(entry.start)} to ${transitionSeconds(transitionEnd(entry))}, ${transitionWindow(entry, segments).frames} frames. ${where.text}.`,
    });
    block.append(label);
    for (const side of ["start", "end"]) {
      const handle = transitionElement("div", `position:absolute;${side === "start" ? "left" : "right"}:0;top:0;bottom:0;width:7px;cursor:ew-resize;background:rgba(248,250,252,.18);z-index:1;`);
      handle.dataset.edge = side;
      block.append(handle);
    }

    // One pointer gesture: a drag of the body moves the block, a drag of an edge stretches it. The block follows the pointer without a redraw and the
    // change is made once when the pointer is released.
    block.onpointerdown = (event) => {
      if (event.button !== 0) return;
      selectTransition(state, entry.id);
      const edge = event.target?.dataset?.edge || "";
      const origin = event.clientX;
      const base = edge === "end" ? transitionEnd(entry) : entry.start;
      const snapOptions = {
        points: [...edges, ...list.filter((item) => item.id !== entry.id).flatMap((item) => [item.start, transitionEnd(item)])],
        beats: state.showBeatMarkers === false ? [] : state.beats || [],
        snapBeats: Boolean(state.snapToBeats),
        pxPerSecond: state.pxPerSecond,
      };
      let result = { start: entry.start, seconds: entry.seconds };
      let moved = false;
      event.preventDefault();
      try {
        block.setPointerCapture(event.pointerId);
      } catch {
        /* the pointer is gone already: the events still reach the block */
      }
      const apply = (moveEvent) => {
        if (Math.abs(moveEvent.clientX - origin) > 2) moved = true;
        const wanted = secondsAt(moveEvent, origin, base);
        if (edge === "start") result = trimTransitionStart(list, entry.id, transitionSnapToFrame(snapTransitionTime(wanted, snapOptions)));
        else if (edge === "end") result = trimTransitionEnd(list, entry.id, transitionSnapToFrame(snapTransitionTime(wanted, snapOptions)), total);
        else {
          // Snap whichever end of the block is nearer a snap point.
          const start = snapTransitionTime(wanted, snapOptions);
          const end = snapTransitionTime(wanted + entry.seconds, snapOptions) - entry.seconds;
          result = { start: moveTransition(list, entry.id, transitionSnapToFrame(Math.abs(start - wanted) <= Math.abs(end - wanted) ? start : end), total), seconds: entry.seconds };
        }
        block.style.left = `${result.start * state.pxPerSecond}px`;
        block.style.width = `${Math.max(18, result.seconds * state.pxPerSecond)}px`;
      };
      const finish = () => {
        block.removeEventListener("pointermove", apply);
        block.removeEventListener("pointerup", finish);
        block.removeEventListener("pointercancel", finish);
        if (!moved) return;
        saveTransitions(state, list.map((item) => (item.id === entry.id ? { ...item, ...result } : item)), edge ? "Transition stretched" : "Transition moved", context);
      };
      block.addEventListener("pointermove", apply);
      block.addEventListener("pointerup", finish);
      block.addEventListener("pointercancel", finish);
    };
    block.ondblclick = (event) => {
      event.stopPropagation();
      openTransitionPicker({ ...context, mode: "edit", id: entry.id });
    };
    block.oncontextmenu = (event) => {
      event.preventDefault();
      event.stopPropagation();
      selectTransition(state, entry.id);
      openTransitionBlockMenu(event, entry, context);
    };
    lane.append(block);
  }
  layer.append(lane);
  return lane;
}
